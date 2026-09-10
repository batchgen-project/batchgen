# `batchgen/models/moonshotai/kimi_k3/`

Kimi-K3 tokenizer + its vendored, md5-verified checkpoint assets. The K3 *model*
(weights, layers, parallel strategy) lives in `kimi_linear/` and `kimi_linear/k3/`
— K3's text tower is the Kimi-Linear architecture. Only the tokenizer is
genuinely K3-specific, and it is genuinely different.

## Why this package exists at all

K3 and Kimi-Linear-48B share a byte-identical `tiktoken.model`. They do **not**
share special tokens. The 256 reserved slots are *named* by
`tokenizer_config.json`, and six names differ:

| id | Kimi-K3 | Kimi-Linear-48B |
|----|---------|-----------------|
| 163586 | `<\|end_of_msg\|>` | `<\|im_end\|>` |
| 163587 | `<\|open\|>` | `<\|im_user\|>` |
| 163588 | `<\|close\|>` | `<\|im_assistant\|>` |
| 163589 | `<\|sep\|>` | *(unnamed)* |
| 163590 | `[start_header_id]` | `<\|start_header_id\|>` |
| 163591 | `[end_header_id]` | `<\|end_header_id\|>` |

Serving K3 with the 48B's config renders a 12-token K3 prompt fragment as 32
marker-free BPE tokens, and `decode()` round-trips identically either way. There
is no error and no warning. That is bug_log.md 2026-07-31, and everything in
`tokenizer.py` that looks paranoid is aimed at it.

K3 also has **no Jinja chat template and no `chat_template` key**. Its format is
XTML, implemented in Python in `assets/encoding_k3.py`. `tokenizer.py` imports
that renderer verbatim rather than porting it.

## Files

| Path | What it is |
|------|------------|
| `tokenizer.py` | `KimiK3Tokenizer` — the only BatchGen-authored file here |
| `assets/` | byte-for-byte copies of the served checkpoint; **never edit** |
| `assets/__init__.py` | load-bearing: `find_packages()` is what ships `assets/*.py` into the wheel |

## Serving gates — decide these before running K3 in production

1. **`--parse-thinking` is effectively required.** With the server defaults
   (`parse_thinking=False`), the API `content` field is raw XTML, e.g.
   `'thoughts<|close|>think<|sep|><|open|>response<|sep|>Hello!<|close|>response<|sep|>'`.
   That decode is *correct* — `<|open|>/<|close|>/<|sep|>` are `"special": false`
   so `skip_special_tokens=True` must not strip them — but no other model in the
   repo leaks structure this way.
2. **Thinking is on by default and costs 67 tokens per prompt.** HuggingFace
   parity is `thinking=True, thinking_effort="max"`, which injects a
   thinking-effort system message: a 1-message chat is 22 ids without it and 89
   with it. Decide explicitly whether BatchGen's OpenAI seam should default
   `enable_thinking=false`.
3. **`thinking_effort` is unreachable from the API today.** The scheduler
   forwards only `enable_thinking`/`thinking`/`tools`/`preserve_thinking`.
   `ChatCompletionRequest.reasoning_effort` is a different field and is injected
   for gpt-oss only, so a K3 client's `reasoning_effort` is dropped. Its
   `Literal` is also `low|medium|high` while K3 accepts `low|high|max`.
4. **Which stop id the model actually emits is unverified.** `{163585, 163586}`
   are both configured, with 163586 (`<|end_of_msg|>`) operative per
   `generation_config.json` and `config.json`. No generation has been run.
5. **Tool calling is round-trip tested against the vendored renderer, not
   against real model output.** The renderer defines the grammar the model was
   trained to emit, so it should hold; the first real tool-calling run is the
   proof.
6. **Multimodal is refused, not supported.** `image_prompts` and image content
   parts raise. The media tokens (163602-163605) are in the vocabulary and the
   path is untested. `<osagent_mode>` (163649) is in the vocabulary and is
   referenced nowhere in `encoding_k3.py`.
7. **`/v1/completions` bypasses every check in this module.** Raw prompts go
   straight to the worker and are encoded with the structural markers enabled,
   so a completions client can inject XTML structure. This is true for every
   model in the repo, not a K3 regression, but it is the one hole
   `apply_chat_template` cannot close.

## The string seam (read before changing anything in `apply_chat_template`)

BatchGen has no pre-tokenized prompt path: the scheduler renders chat to a
string and the worker re-encodes it. That flattening loses K3's per-segment
`allow_special` split, and the re-encoded ids can differ from HuggingFace in two
ways — forged control markers in caller text, and BPE merges drifting across the
four-way segment split `_attr` makes around every attribute value (an argument
key of `'  spaced  '` is enough; no special token required).

So `apply_chat_template(tokenize=False)` renders the string and then re-encodes
it with this tokenizer's own `encode()` — the exact function the worker will
call — and requires the result to equal the reference segment ids. Exact, no
marker heuristics, no blind spots. Measured: 400/400 realistic conversations
pass; verification costs ~58% of one encode (8 ms on an 88 KB prompt).

Two consequences worth remembering:

- `encode()` narrows tiktoken's allowlist to the four structural markers rather
  than `"all"`. Every structural position encodes identically to upstream; the
  settings diverge only where *caller* text holds a non-structural spelling, and
  there the narrow form is the HuggingFace-correct one. Without this, "What does
  `[EOS]` mean in a tokenizer?" silently injects stop token 163585 into the
  prompt body — or, under a marker-scanning design, gets rejected outright.
- A rejection currently fails the **whole batch**:
  `_convert_requests_to_worker_inputs` has no per-request try/except, and
  `_process_batch` has already set `IN_PROGRESS`, so the exception lands in
  `_run`'s bare `except` and the batch never reaches a terminal status. The core
  PR turns that into a loud batch failure naming the offending `custom_id`;
  per-request isolation is still a follow-up.

## Tests

`tests/test_kimi_k3_tokenizer.py` — CPU only, no GPU, no weights, no JIT.

```bash
pytest tests/test_kimi_k3_tokenizer.py -q

# with the real checkpoint (h20-instance-1), oracle enabled and skips banned:
KIMI_K3_CHECKPOINT=/taijifs_zw35/share_304153846/hunyuan/tairanxu/models/Kimi-K3 \
  KIMI_K3_STRICT=1 pytest tests/test_kimi_k3_tokenizer.py -q
```

`KIMI_K3_STRICT=1` turns every skip into a failure, so a CI image missing
`tokenizers` cannot report green with zero coverage.

## Re-vendoring procedure

1. Copy files wholesale from the served checkpoint; md5-verify each one.
2. Run the suite. The pinned tables (`KIMI_K3_ADDED_TOKENS`,
   `KIMI_K3_ADDITIONAL_SPECIAL_TOKENS`, `KIMI_K3_ALL_SPECIAL_IDS`,
   `KIMI_K3_STRUCTURAL_MARKERS`) fail loudly if the layout moved.
3. A failure is a decision point for a human, not something to "fix" by updating
   the constant reflexively. Special-token drift is silent prompt corruption.
4. If the renderer gained a fifth structural marker, guard G9 catches it — and
   `encode()`'s allowlist and the verification both need re-deriving.
