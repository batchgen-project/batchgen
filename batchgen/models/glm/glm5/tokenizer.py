# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
# ---------------------------------------------------------------------------- #

"""GLM-5 tokenizer for BatchGen.

GLM-5 tokenizer specifications:
- Vocabulary size: 154,880 tokens
- EOS tokens: [154820, 154827, 154829] (multiple stop tokens)
- PAD token: 154820 (same as first EOS)
- BOS token: not used
- Uses HuggingFace tokenizer.json format (bundled in this directory)
"""

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union

from batchgen.config.fast_tokenizer import FastTokenizer
from batchgen.config.tokenizer_registry import register_tokenizer

logger = logging.getLogger(__name__)

TOKENIZER_DIR = Path(__file__).parent

GLM5_EOS_TOKEN_ID = 154820
GLM5_PAD_TOKEN_ID = 154820
GLM5_STOP_TOKEN_IDS = [154820, 154827, 154829]
GLM5_VOCAB_SIZE = 154880


@register_tokenizer("glm_moe_dsa")
class GLM5Tokenizer(FastTokenizer):
    """GLM-5 tokenizer.

    Loads tokenizer.json from package directory (not user cache).
    Requires tokenizers>=0.21 which natively supports ignore_merges.

    Attributes:
        eos_token_id: 154820 (primary EOS)
        pad_token_id: 154820 (same as EOS)
        stop_token_ids: [154820, 154827, 154829] (all stop tokens for generation)
        vocab_size: 154,880
    """

    def __init__(self):
        super().__init__(str(TOKENIZER_DIR))

        # Load chat template from separate jinja file (not inline in tokenizer_config.json)
        chat_template_file = TOKENIZER_DIR / "chat_template.jinja"
        if chat_template_file.exists():
            self.chat_template = chat_template_file.read_text()
        else:
            self.chat_template = self._config.get("chat_template")

        self.bos_token_id = None
        self.eos_token_id = GLM5_EOS_TOKEN_ID
        self.pad_token_id = GLM5_PAD_TOKEN_ID
        self.stop_token_ids = GLM5_STOP_TOKEN_IDS
        self.vocab_size = GLM5_VOCAB_SIZE

        # Find pad token string for padding setup
        vocab = self.tokenizer.get_vocab()
        id_to_token = {v: k for k, v in vocab.items()}

        self.eos_token = id_to_token.get(self.eos_token_id)
        self.pad_token = self.eos_token

        logger.info(
            f"GLM-5 tokenizer initialized: vocab_size={self.vocab_size}, "
            f"eos={self.eos_token_id}, stop_tokens={self.stop_token_ids}"
        )

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> Union[str, List[int]]:
        """Apply GLM-5 chat template.

        Overrides parent to use permissive Jinja2 undefined (not StrictUndefined),
        since the GLM-5 template checks optional variables like 'tools',
        'enable_thinking', 'clear_thinking' with {% if %} guards.
        """
        if not self.chat_template:
            raise ValueError("No chat template available for GLM-5 tokenizer.")

        from jinja2 import Template

        template = Template(self.chat_template)
        rendered = template.render(
            messages=messages,
            bos_token="",
            eos_token=self.eos_token or "",
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

        if tokenize:
            return self.encode(rendered, add_special_tokens=False)
        return rendered

    # ---- Output parsing ----

    _THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    _TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    _ARG_RE = re.compile(
        r"<arg_key>(.*?)</arg_key><arg_value>(.*?)</arg_value>", re.DOTALL
    )

    def parse_thinking(self, text: str) -> tuple[Optional[str], str]:
        m = self._THINK_RE.search(text)
        if not m:
            return None, text
        reasoning = m.group(1).strip()
        visible = self._THINK_RE.sub("", text, count=1).strip()
        return reasoning, visible

    def parse_tool_calls(self, text: str) -> tuple[Optional[list], str]:
        matches = self._TOOL_CALL_RE.findall(text)
        if not matches:
            return None, text
        tool_calls = []
        for raw in matches:
            raw = raw.strip()
            # Function name is the text before the first <arg_key>
            arg_start = raw.find("<arg_key>")
            if arg_start == -1:
                name = raw
                arguments = {}
            else:
                name = raw[:arg_start].strip()
                arguments = {}
                for am in self._ARG_RE.finditer(raw):
                    key = am.group(1).strip()
                    val = am.group(2).strip()
                    # Try to parse JSON values (numbers, booleans, objects)
                    try:
                        arguments[key] = json.loads(val)
                    except (ValueError, json.JSONDecodeError):
                        arguments[key] = val

            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments),
                },
            })
        visible = self._TOOL_CALL_RE.sub("", text).strip()
        return tool_calls, visible
