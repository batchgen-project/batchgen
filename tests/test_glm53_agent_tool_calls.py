"""GLM-5.3 agent turns on CPU: tool rendering, tool-call parsing, the OpenAI path.

An agent harness on the OpenAI SDK sends ``tools``, gets
``message.tool_calls`` back, and resends the assistant message with
``function.arguments`` as a JSON string. Each step is pinned here:

* the GLM-5.3 template renders ``tools`` (it calls ``tojson(ensure_ascii=
  False)``, which Jinja2's builtin filter rejects);
* string arguments in the history are rendered as arg_key/arg_value pairs;
* a generated call is parsed into OpenAI ``tool_calls`` typed by the tool
  schema, and the completion reports ``finish_reason="tool_calls"``;
* the next turn's prompt starts with the previous prompt plus the generated
  tokens (what a session hold reuses);
* ``chat_template_kwargs`` is folded into the request, never dropped;
* ``POST /v1/chat/completions`` serves a session turn and refuses a body
  without ``session``.
"""

from __future__ import annotations

import asyncio
import json
import queue
from types import SimpleNamespace

import pytest

from tests.test_glm52_tokenizer import _bootstrap_glm52_tokenizer

TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a command.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer"},
            },
            "required": ["command"],
        },
    },
}]
GENERATED = (
    "I will list the files.</think>Let me check."
    "<tool_call>bash<arg_key>command</arg_key><arg_value>123</arg_value>"
    "<arg_key>timeout</arg_key><arg_value>30</arg_value></tool_call>"
)


@pytest.fixture(scope="module")
def glm53():
    pytest.importorskip("tokenizers")
    pytest.importorskip("jinja2")
    _, module = _bootstrap_glm52_tokenizer()
    return module.GLM53Tokenizer()


def _base():
    return [{"role": "system", "content": "sys"},
            {"role": "user", "content": "fix the bug"}]


def test_glm53_template_renders_tools_with_non_ascii_text(glm53):
    tools = [dict(TOOLS[0])]
    tools[0] = {"type": "function", "function": {
        **TOOLS[0]["function"], "description": "Führe einen Befehl aus <x>."}}
    prompt = glm53.apply_chat_template(
        _base(), tokenize=False, add_generation_prompt=True, tools=tools)
    assert "# Tools" in prompt
    # HF's tojson: no ASCII escaping and no HTML escaping.
    assert '"description": "Führe einen Befehl aus <x>."' in prompt
    assert prompt.endswith("<|user|>fix the bug<|assistant|><think>")


def test_string_tool_call_arguments_render_as_arg_pairs(glm53):
    history = _base() + [
        {"role": "assistant", "content": "Let me check.",
         "tool_calls": [{"id": "call_1", "type": "function", "function": {
             "name": "bash",
             "arguments": json.dumps({"command": "ls -la", "timeout": 30})}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "a b c"},
    ]
    original = json.dumps(history, sort_keys=True)
    prompt = glm53.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True, tools=TOOLS)
    assert ("<tool_call>bash<arg_key>command</arg_key><arg_value>ls -la"
            "</arg_value><arg_key>timeout</arg_key><arg_value>30</arg_value>"
            "</tool_call><|observation|><tool_response>a b c</tool_response>"
            ) in prompt
    # The caller's messages are not modified.
    assert json.dumps(history, sort_keys=True) == original


@pytest.mark.parametrize("arguments, match", [
    ("{not json", "not JSON"),
    ("[1, 2]", "must be a JSON object"),
])
def test_tool_call_arguments_that_are_not_a_json_object_are_refused(
    glm53, arguments, match,
):
    history = _base() + [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "c", "type": "function",
         "function": {"name": "bash", "arguments": arguments}}]}]
    with pytest.raises(ValueError, match=match):
        glm53.apply_chat_template(history, tokenize=False, tools=TOOLS)


def test_parsed_arguments_follow_the_tool_schema(glm53):
    _, visible = glm53.parse_thinking(GENERATED)
    calls, rest = glm53.parse_tool_calls_with_schema(visible, TOOLS)
    assert rest == "Let me check."
    assert [c["function"]["name"] for c in calls] == ["bash"]
    # "command" is a string parameter: "123" stays text; timeout is JSON.
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "command": "123", "timeout": 30}
    # Without a schema every value that parses as JSON is read as JSON.
    calls, _ = glm53.parse_tool_calls(visible)
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "command": 123, "timeout": 30}


def test_the_next_prompt_extends_the_held_tokens(glm53):
    """Turn k+1 starts with turn k's prompt and generated tokens.

    The hold keeps prompt + generated - 1 tokens (the stop token
    ``<|observation|>`` has no KV); the next prompt renders that stop token
    at the same position, so every held token is reused when the client
    returns ``reasoning_content``. Without it the template renders an empty
    think block and the reuse ends at the previous prompt.
    """
    prompt = glm53.apply_chat_template(
        _base(), tokenize=False, add_generation_prompt=True, tools=TOOLS)
    _, visible = glm53.parse_thinking(GENERATED)
    calls, content = glm53.parse_tool_calls_with_schema(visible, TOOLS)
    assistant = {"role": "assistant", "content": content,
                 "reasoning_content": "I will list the files.",
                 "tool_calls": [{"id": calls[0]["id"], "type": "function",
                                 "function": calls[0]["function"]}]}
    tool = {"role": "tool", "tool_call_id": calls[0]["id"], "content": "out"}
    held = glm53.encode(prompt + GENERATED, add_special_tokens=False)
    following = glm53.encode(glm53.apply_chat_template(
        _base() + [assistant, tool], tokenize=False,
        add_generation_prompt=True, tools=TOOLS), add_special_tokens=False)
    assert following[:len(held)] == held
    assert following[len(held)] == 154829  # <|observation|>, the stop token

    without_reasoning = {k: v for k, v in assistant.items()
                         if k != "reasoning_content"}
    following = glm53.encode(glm53.apply_chat_template(
        _base() + [without_reasoning, tool], tokenize=False,
        add_generation_prompt=True, tools=TOOLS), add_special_tokens=False)
    first = glm53.encode(prompt, add_special_tokens=False)
    lcp = next(i for i, (a, b) in enumerate(zip(held, following)) if a != b)
    assert lcp == len(first)


@pytest.mark.parametrize("reasoning_text", [
    "I will list the files.",          # canonical
    "\nI will list the files.\n",      # newlines around the reasoning
    "\n\nI will list\nthe files. ",     # inner and trailing whitespace
])
def test_turn2_strictly_extends_turn1_through_parsed_reasoning(glm53, reasoning_text):
    """Round trip: turn-1 output -> parse -> resend -> render turn 2.

    The held tokens are the turn-1 prompt ids followed by the generated ids;
    turn 2's prompt, rendered from the PARSED turn-1 message as an OpenAI
    client resends it, must start with every one of them, then the stop
    token. A stripped reasoning breaks this at the first output token.
    """
    prompt = glm53.apply_chat_template(
        _base(), tokenize=False, add_generation_prompt=True, tools=TOOLS)
    generated = (
        reasoning_text + "</think>"
        "<tool_call>bash<arg_key>command</arg_key><arg_value>ls -la</arg_value>"
        "</tool_call>"
    )
    reasoning, visible = glm53.parse_thinking(generated)
    assert reasoning == reasoning_text
    calls, content = glm53.parse_tool_calls_with_schema(visible, TOOLS)
    assistant = {"role": "assistant", "content": content,
                 "reasoning_content": reasoning,
                 "tool_calls": [{"id": calls[0]["id"], "type": "function",
                                 "function": calls[0]["function"]}]}
    tool = {"role": "tool", "tool_call_id": calls[0]["id"], "content": "out"}
    held = (glm53.encode(prompt, add_special_tokens=False)
            + glm53.encode(generated, add_special_tokens=False))
    following = glm53.encode(glm53.apply_chat_template(
        _base() + [assistant, tool], tokenize=False,
        add_generation_prompt=True, tools=TOOLS), add_special_tokens=False)
    assert following[:len(held)] == held
    assert following[len(held)] == 154829  # <|observation|>, the stop token

    if reasoning_text != reasoning_text.strip():
        # What the stripped parse used to send back: diverges inside the
        # turn-1 output, so the held output tokens would be recomputed.
        stripped = dict(assistant, reasoning_content=reasoning_text.strip())
        diverged = glm53.encode(glm53.apply_chat_template(
            _base() + [stripped, tool], tokenize=False,
            add_generation_prompt=True, tools=TOOLS), add_special_tokens=False)
        first = len(glm53.encode(prompt, add_special_tokens=False))
        lcp = next(i for i, (a, b) in enumerate(zip(held, diverged)) if a != b)
        assert first <= lcp < len(held)


# ----------------------------------------------------------- server paths

from tests import test_turn_api as turn_tests  # noqa: E402

bs = turn_tests.bs
io_struct = turn_tests.io_struct


class _GLMScheduler:
    """The scheduler's response builder with the GLM-5.3 tokenizer."""

    def __init__(self, tokenizer, parse_tool_call=True):
        self.server_args = SimpleNamespace(
            parse_thinking=True, parse_tool_call=parse_tool_call)
        self._tokenizer = tokenizer

    def _get_tokenizer(self, model):
        return self._tokenizer

    _parse_output = bs.BatchScheduler._parse_output
    _chat_completion_body = bs.BatchScheduler._chat_completion_body


def test_completion_body_reports_typed_tool_calls(glm53):
    scheduler = _GLMScheduler(glm53)
    result = {"text": GENERATED, "prompt_length": 10, "decoded_length": 40,
              "finish_reason": "stop"}
    body = scheduler._chat_completion_body("zai-org/GLM-5.3-FP8", result, 0, TOOLS)
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    message = choice["message"]
    assert message["reasoning_content"] == "I will list the files."
    assert message["content"] == "Let me check."
    call = message["tool_calls"][0]
    assert call["type"] == "function" and call["function"]["name"] == "bash"
    assert json.loads(call["function"]["arguments"]) == {
        "command": "123", "timeout": 30}
    # A length stop is still reported as such, and no call means "stop".
    body = scheduler._chat_completion_body(
        "zai-org/GLM-5.3-FP8", {**result, "finish_reason": "length"}, 0, TOOLS)
    assert body["choices"][0]["finish_reason"] == "length"
    body = scheduler._chat_completion_body(
        "zai-org/GLM-5.3-FP8", {**result, "text": "plain</think>answer"}, 0)
    assert body["choices"][0]["finish_reason"] == "stop"
    assert "tool_calls" not in body["choices"][0]["message"]


def _request(**extra):
    return io_struct.ChatCompletionRequest(
        model="zai-org/GLM-5.3-FP8",
        messages=[{"role": "user", "content": "hi"}], **extra)


def test_chat_template_kwargs_fold_into_the_request():
    folded = bs._apply_chat_template_kwargs(_request(
        chat_template_kwargs={"reasoning_effort": "high"}))
    assert folded.reasoning_effort == "high"
    assert folded.chat_template_kwargs is None
    same = bs._apply_chat_template_kwargs(_request(
        reasoning_effort="max", chat_template_kwargs={"reasoning_effort": "max"}))
    assert same.reasoning_effort == "max"
    with pytest.raises(ValueError, match="conflicts"):
        bs._apply_chat_template_kwargs(_request(
            reasoning_effort="low",
            chat_template_kwargs={"reasoning_effort": "max"}))
    with pytest.raises(ValueError, match="not supported"):
        bs._apply_chat_template_kwargs(_request(
            chat_template_kwargs={"tool_choice_mode": "x"}))
    with pytest.raises(Exception, match="reasoning_effort"):
        bs._apply_chat_template_kwargs(_request(
            chat_template_kwargs={"reasoning_effort": "extreme"}))


def test_chat_template_kwargs_reach_the_template_and_the_glm53_checks(tmp_path):
    seen = []

    class _Recorder:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt,
                                **kwargs):
            seen.append(kwargs)
            return "prompt"

    scheduler = turn_tests._scheduler(tmp_path, turn_tests._ns_worker())
    scheduler._get_tokenizer = lambda model: _Recorder()

    def convert(**extra):
        item = io_struct.BatchRequestItem.model_construct(
            custom_id="c", method="POST", url="/v1/chat/completions",
            body=_request(max_tokens=8, **extra))
        return scheduler._convert_requests_to_worker_inputs([item])

    convert(chat_template_kwargs={"reasoning_effort": "high"})
    assert seen[-1]["reasoning_effort"] == "high"
    with pytest.raises(ValueError, match="medium is unsupported"):
        convert(chat_template_kwargs={"reasoning_effort": "medium"})
    with pytest.raises(ValueError, match="enable_thinking=false"):
        convert(chat_template_kwargs={"enable_thinking": False})


def test_chat_completions_path_serves_a_session_turn(tmp_path):
    async def scenario(client, scheduler, worker, app):
        await client.post("/v1/sessions",
                          json={"traj_id": "traj-1", "session_id": "main"})
        served = await client.post(
            "/v1/chat/completions", json=turn_tests._turn_body("main", "one"))
        sessionless = turn_tests._turn_body("main", "two")
        sessionless.pop("session")
        refused = await client.post("/v1/chat/completions", json=sessionless)
        return served, refused, list(worker.admitted)

    served, refused, admitted = asyncio.run(turn_tests._serve(tmp_path, scenario))
    assert served.status_code == 200, served.text
    body = served.json()
    assert body["object"] == "chat.completion"
    assert body["batchgen"]["session"]["session_id"] == "main"
    assert [e["session_id"] for e in admitted] == ["main"]
    assert refused.status_code == 400
    assert refused.json()["detail"]["code"] == "session_required"
