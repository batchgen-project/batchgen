"""Interface-stability pins for batchgen/server/.

These signatures are load-bearing for downstream callers; changing one is
deliberate -- update this pin in the same PR and say why in the PR
description. Adding or removing a public method, renaming a parameter, and
dropping a return annotation all count.

PINS below is the frozen copy: for a function, its parameter names and
unparsed return annotation; for a class, its base classes, annotated field
names, class-level members, the set of public method names, and each of
those methods' parameters and return annotation.

Nothing here imports batchgen. ``server/__init__.py`` still eagerly re-
exports the app factory, so any ``batchgen.server.*`` import pulls FastAPI,
uvicorn and torch. Every assertion is therefore made against the defining
module's own AST, so this suite passes on a machine with no GPU toolchain
and costs milliseconds.

The io_struct request and response models are pinned on their field names as
well: they are the wire shape external clients see, so a renamed or dropped
field is an API break.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Frozen by reading the defining sources' AST.
PINS = {
    "batchgen/server/http_server.py": {
        "launch_server": {
            "kind": "function",
            "params": ["server_args"],
            "returns": "None",
        },
    },
    "batchgen/server/io_struct.py": {
        "AbortSessionRequest": {
            "kind": "class",
            "bases": ["BaseModel"],
            "fields": ["reason"],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "BatchEndpoint": {
            "kind": "class",
            "bases": ["str", "Enum"],
            "fields": [],
            "members": {
                "CHAT_COMPLETIONS": "'/v1/chat/completions'",
                "COMPLETIONS": "'/v1/completions'",
            },
            "public_methods": [],
            "methods": {},
        },
        "BatchError": {
            "kind": "class",
            "bases": ["BaseModel"],
            "fields": ["code", "message"],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "BatchObject": {
            "kind": "class",
            "bases": ["BaseModel"],
            "fields": [
                "id",
                "object",
                "endpoint",
                "input_file_id",
                "output_file_id",
                "completion_window",
                "status",
                "created_at",
                "expires_at",
                "started_at",
                "completed_at",
                "cancelled_at",
                "cancelling_at",
                "error",
                "metadata",
                "max_decoding_length",
                "max_context_length",
                "temperature",
                "top_p",
                "top_k",
                "batchgen_debug",
            ],
            "members": {},
            "public_methods": ["default_timestamps"],
            "methods": {
                "default_timestamps": {
                    "params": ["cls", "values"],
                    "returns": "Dict[str, Any]",
                },
            },
        },
        "BatchRequestItem": {
            "kind": "class",
            "bases": ["BaseModel"],
            "fields": ["custom_id", "method", "url", "body"],
            "members": {},
            "public_methods": ["parse_body"],
            "methods": {
                "parse_body": {
                    "params": ["cls", "value", "values"],
                    "returns": "Union[ChatCompletionRequest, CompletionRequest]",
                },
            },
        },
        "BatchResponse": {
            "kind": "class",
            "bases": ["BaseModel"],
            "fields": ["status_code", "request_id", "body"],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "BatchResultItem": {
            "kind": "class",
            "bases": ["BaseModel"],
            "fields": ["id", "custom_id", "response", "error"],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "BatchStatus": {
            "kind": "class",
            "bases": ["str", "Enum"],
            "fields": [],
            "members": {
                "VALIDATING": "'validating'",
                "IN_PROGRESS": "'in_progress'",
                "COMPLETED": "'completed'",
                "FAILED": "'failed'",
                "CANCELLING": "'cancelling'",
                "CANCELLED": "'cancelled'",
            },
            "public_methods": [],
            "methods": {},
        },
        "ChatCompletionChoice": {
            "kind": "class",
            "bases": ["BaseModel"],
            "fields": ["index", "message", "logprobs", "finish_reason"],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "ChatCompletionChoiceMessage": {
            "kind": "class",
            "bases": ["BaseModel"],
            "fields": ["role", "content", "reasoning_content", "tool_calls"],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "ChatCompletionRequest": {
            "kind": "class",
            "bases": ["BaseModel"],
            "fields": [
                "model",
                "messages",
                "temperature",
                "top_p",
                "top_k",
                "n",
                "stream",
                "max_tokens",
                "max_completion_tokens",
                "presence_penalty",
                "frequency_penalty",
                "logit_bias",
                "user",
                "reasoning_effort",
                "tools",
                "thinking",
                "enable_thinking",
                "clear_thinking",
                "preserve_thinking",
                "ignore_eos",
            ],
            "members": {},
            "public_methods": ["validate_stream"],
            "methods": {
                "validate_stream": {
                    "params": ["cls", "value"],
                    "returns": "Optional[bool]",
                },
            },
        },
        "ChatCompletionResponse": {
            "kind": "class",
            "bases": ["BaseModel"],
            "fields": [
                "id",
                "object",
                "created",
                "model",
                "choices",
                "usage",
                "system_fingerprint",
            ],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "ChatMessage": {
            "kind": "class",
            "bases": ["BaseModel"],
            "fields": [
                "role",
                "content",
                "reasoning_content",
                "name",
                "tool_call_id",
                "tool_calls",
            ],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "CloseSessionRequest": {
            "kind": "class",
            "bases": ["BaseModel"],
            "fields": ["reason"],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "CompletionChoice": {
            "kind": "class",
            "bases": ["BaseModel"],
            "fields": ["index", "text", "logprobs", "finish_reason"],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "CompletionRequest": {
            "kind": "class",
            "bases": ["BaseModel"],
            "fields": [
                "model",
                "prompt",
                "max_tokens",
                "max_completion_tokens",
                "temperature",
                "top_p",
                "top_k",
                "n",
                "stream",
                "logprobs",
                "echo",
                "presence_penalty",
                "frequency_penalty",
                "user",
                "ignore_eos",
            ],
            "members": {},
            "public_methods": ["validate_stream"],
            "methods": {
                "validate_stream": {
                    "params": ["cls", "value"],
                    "returns": "Optional[bool]",
                },
            },
        },
        "CompletionResponse": {
            "kind": "class",
            "bases": ["BaseModel"],
            "fields": [
                "id",
                "object",
                "created",
                "model",
                "choices",
                "usage",
                "system_fingerprint",
            ],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "CompletionWindow": {
            "kind": "class",
            "bases": ["str", "Enum"],
            "fields": [],
            "members": {
                "ONE_DAY": "'24h'",
            },
            "public_methods": [],
            "methods": {},
        },
        "CreateBatchRequest": {
            "kind": "class",
            "bases": ["BaseModel"],
            "fields": [
                "input_file_id",
                "endpoint",
                "completion_window",
                "metadata",
                "max_decoding_length",
                "max_context_length",
                "temperature",
                "top_p",
                "top_k",
                "batchgen_debug",
            ],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "DeleteFileResponse": {
            "kind": "class",
            "bases": ["BaseModel"],
            "fields": ["id", "deleted", "object"],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "FileObject": {
            "kind": "class",
            "bases": ["BaseModel"],
            "fields": [
                "id",
                "object",
                "bytes",
                "created_at",
                "filename",
                "purpose",
                "status",
                "status_details",
                "checksum",
            ],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "FilePurpose": {
            "kind": "class",
            "bases": ["str", "Enum"],
            "fields": [],
            "members": {
                "BATCH": "'batch'",
                "BATCH_OUTPUT": "'batch_output'",
            },
            "public_methods": [],
            "methods": {},
        },
        "FileStatus": {
            "kind": "class",
            "bases": ["str", "Enum"],
            "fields": [],
            "members": {
                "PROCESSED": "'processed'",
                "FAILED": "'failed'",
            },
            "public_methods": [],
            "methods": {},
        },
        "ListBatchesRequest": {
            "kind": "class",
            "bases": ["BaseModel"],
            "fields": ["after", "limit"],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "ListBatchesResponse": {
            "kind": "class",
            "bases": ["BaseModel"],
            "fields": ["data", "first_id", "last_id", "has_more"],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "ListFilesRequest": {
            "kind": "class",
            "bases": ["BaseModel"],
            "fields": ["purpose", "limit", "order", "after"],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "ListFilesResponse": {
            "kind": "class",
            "bases": ["BaseModel"],
            "fields": ["data", "has_more"],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "ListModelsResponse": {
            "kind": "class",
            "bases": ["BaseModel"],
            "fields": ["object", "data"],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "ModelObject": {
            "kind": "class",
            "bases": ["BaseModel"],
            "fields": ["id", "object", "created", "owned_by", "max_context_length"],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "RawInferenceRequest": {
            "kind": "class",
            "bases": ["BaseModel"],
            "fields": [
                "prompts",
                "max_input_len",
                "max_output_len",
                "ignore_eos",
                "temperature",
                "top_p",
            ],
            "members": {},
            "public_methods": ["validate_prompts"],
            "methods": {
                "validate_prompts": {
                    "params": ["cls", "value"],
                    "returns": "List[str]",
                },
            },
        },
        "SessionIdentity": {
            "kind": "class",
            "bases": ["BaseModel"],
            "fields": ["traj_id", "session_id", "parent_session_id"],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "ToolCall": {
            "kind": "class",
            "bases": ["BaseModel"],
            "fields": ["id", "type", "function"],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "ToolCallFunction": {
            "kind": "class",
            "bases": ["BaseModel"],
            "fields": ["name", "arguments"],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "TurnRequest": {
            "kind": "class",
            "bases": ["ChatCompletionRequest"],
            "fields": ["session", "batchgen_debug"],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "Usage": {
            "kind": "class",
            "bases": ["BaseModel"],
            "fields": ["prompt_tokens", "completion_tokens", "total_tokens"],
            "members": {},
            "public_methods": [],
            "methods": {},
        },
        "build_batch_object_from_create_request": {
            "kind": "function",
            "params": ["batch_id", "body", "created_at", "expires_at"],
            "returns": "BatchObject",
        },
        "normalize_inference_results": {
            "kind": "function",
            "params": ["results"],
            "returns": "List[Any]",
        },
    },
    "batchgen/server/process_utils.py": {
        "die_with_parent": {
            "kind": "function",
            "params": [],
            "returns": "None",
        },
        "get_hugepage_size": {
            "kind": "function",
            "params": [],
            "returns": "int",
        },
        "get_model_byte_size": {
            "kind": "function",
            "params": ["model_name"],
            "returns": "int",
        },
        "install_worker_signal_handlers": {
            "kind": "function",
            "params": ["shutdown_callback"],
            "returns": "None",
        },
    },
    "batchgen/server/runtime_locks.py": {
        "hold_run_lock": {
            "kind": "function",
            "params": ["runtime_dir", "create"],
            "returns": "BinaryIO",
        },
    },
    "batchgen/server/server_args.py": {
        "prepare_server_args": {
            "kind": "function",
            "params": ["argv"],
            "returns": "ServerArgs",
        },
    },
    "batchgen/server/watchdog.py": {
        "Watchdog": {
            "kind": "class",
            "bases": [],
            "fields": [],
            "members": {},
            "public_methods": ["create", "disable", "feed"],
            "methods": {
                "create": {
                    "params": [
                        "debug_name",
                        "watchdog_timeout",
                        "soft",
                        "test_stuck_time",
                        "dump_info",
                    ],
                    "returns": "Watchdog",
                },
                "feed": {
                    "params": ["self"],
                    "returns": "None",
                },
                "disable": {
                    "params": ["self"],
                    "returns": None,
                },
            },
        },
    },
    "batchgen/server/worker_readiness.py": {
        "_signal_local_worker_manager_ready": {
            "kind": "function",
            "params": ["ready_event", "local_rank", "global_rank"],
            "returns": "bool",
        },
    },
}

# Named in the module's public surface but deliberately NOT pinned, and why:
SKIPPED = {
    "server/batch_scheduler.py, scheduling_pool.py, storage.py, gpu_arch.py": "imported freely by tests; not part of the stable surface",
}


# ---------------------------------------------------------------------------
# AST harness.  Duplicated verbatim in every tests/contracts/test_pin_*.py so
# that each pin file stands alone: reading one file tells you the whole frozen
# surface of one module, with no shared helper to chase.
# ---------------------------------------------------------------------------

def _params(fn) -> list:
    a = fn.args
    names = [p.arg for p in a.posonlyargs] + [p.arg for p in a.args]
    if a.vararg:
        names.append("*" + a.vararg.arg)
    names += [p.arg for p in a.kwonlyargs]
    if a.kwarg:
        names.append("**" + a.kwarg.arg)
    return names


def _returns(fn):
    return ast.unparse(fn.returns) if fn.returns is not None else None


def _class_actual(node: ast.ClassDef) -> dict:
    methods = {
        m.name: {"params": _params(m), "returns": _returns(m)}
        for m in node.body
        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not m.name.startswith("_")
    }
    return {
        "kind": "class",
        "bases": [ast.unparse(b) for b in node.bases],
        "fields": [
            m.target.id
            for m in node.body
            if isinstance(m, ast.AnnAssign)
            and isinstance(m.target, ast.Name)
            and not m.target.id.startswith("_")
        ],
        "members": {
            m.targets[0].id: ast.unparse(m.value)
            for m in node.body
            if isinstance(m, ast.Assign)
            and len(m.targets) == 1
            and isinstance(m.targets[0], ast.Name)
            and not m.targets[0].id.startswith("_")
        },
        "public_methods": sorted(methods),
        "methods": methods,
    }


def _actual(rel: str) -> dict:
    tree = ast.parse((REPO / rel).read_text())
    out: dict = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            out.setdefault(node.name, _class_actual(node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.setdefault(
                node.name,
                {
                    "kind": "function",
                    "params": _params(node),
                    "returns": _returns(node),
                },
            )
    return out


ACTUAL = {rel: _actual(rel) for rel in PINS}

_RULE = (
    "A pinned signature changed.  If that is intended, update this pin in "
    "the SAME pull request and say why in the PR description."
)


def test_every_pinned_symbol_is_still_defined_in_its_file():
    missing = [
        "%s::%s" % (rel, name)
        for rel, syms in PINS.items()
        for name in syms
        if name not in ACTUAL[rel]
    ]
    assert not missing, "pinned symbols are gone: %r\n%s" % (
        missing,
        _RULE,
    )


def test_pinned_kinds_and_base_classes_are_unchanged():
    bad = []
    for rel, syms in PINS.items():
        for name, pin in syms.items():
            got = ACTUAL[rel].get(name)
            if got is None:
                continue
            if got["kind"] != pin["kind"]:
                bad.append(
                    "%s::%s kind %s -> %s" % (rel, name, pin["kind"], got["kind"])
                )
            elif pin["kind"] == "class" and got["bases"] != pin["bases"]:
                bad.append(
                    "%s::%s bases %r -> %r"
                    % (rel, name, pin["bases"], got["bases"])
                )
    assert not bad, "%r\n%s" % (bad, _RULE)


def test_pinned_class_surfaces_are_unchanged():
    """Public method names, annotated field names, and enum/class members."""
    bad = []
    for rel, syms in PINS.items():
        for name, pin in syms.items():
            if pin["kind"] != "class":
                continue
            got = ACTUAL[rel].get(name)
            if got is None or got["kind"] != "class":
                continue
            for key in ("public_methods", "fields", "members"):
                if got[key] != pin[key]:
                    added = sorted(set(got[key]) - set(pin[key]))
                    removed = sorted(set(pin[key]) - set(got[key]))
                    bad.append(
                        "%s::%s %s added=%r removed=%r"
                        % (rel, name, key, added, removed)
                    )
    assert not bad, "%r\n%s" % (bad, _RULE)


def test_pinned_signatures_are_unchanged():
    """Parameter name tuples and unparsed return annotations."""
    bad = []
    for rel, syms in PINS.items():
        for name, pin in syms.items():
            got = ACTUAL[rel].get(name)
            if got is None or got["kind"] != pin["kind"]:
                continue
            if pin["kind"] == "function":
                pairs = [("", pin, got)]
            else:
                pairs = [
                    (".%s" % m, spec, got["methods"][m])
                    for m, spec in pin["methods"].items()
                    if m in got["methods"]
                ]
            for suffix, want, have in pairs:
                if have["params"] != want["params"]:
                    bad.append(
                        "%s::%s%s params %r -> %r"
                        % (rel, name, suffix, want["params"], have["params"])
                    )
                if have["returns"] != want["returns"]:
                    bad.append(
                        "%s::%s%s returns %r -> %r"
                        % (rel, name, suffix, want["returns"], have["returns"])
                    )
    assert not bad, "%r\n%s" % (bad, _RULE)
