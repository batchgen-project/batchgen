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
from pathlib import Path
from typing import Dict, List, Optional, Union

from tokenizers import Tokenizer
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

    Attributes:
        eos_token_id: 154820 (primary EOS)
        pad_token_id: 154820 (same as EOS)
        stop_token_ids: [154820, 154827, 154829] (all stop tokens for generation)
        vocab_size: 154,880
    """

    def __init__(self):
        # Patch tokenizer.json for older tokenizers Rust library compatibility.
        # GLM-5 uses ignore_merges=True (tiktoken-style: direct vocab lookup, no BPE merges).
        # Old library doesn't support this field. Remove it AND clear merges so the old
        # library does direct vocab lookup too (BPE with no merges = same behavior).
        tokenizer_file = TOKENIZER_DIR / "tokenizer.json"
        with open(tokenizer_file) as f:
            tok_data = json.load(f)

        model = tok_data.get("model", {})
        if model.pop("ignore_merges", None):
            logger.info("Patching GLM-5 tokenizer: removing ignore_merges=True and clearing merges list")
            model["merges"] = []
        model.pop("byte_fallback", None)

        # Load from patched JSON string, then run parent setup (skip file load)
        json_str = json.dumps(tok_data)
        try:
            self.tokenizer = Tokenizer.from_str(json_str)
        except Exception as e:
            import re
            m = re.search(r"column (\d+)", str(e))
            if m:
                col = int(m.group(1))
                start = max(0, col - 200)
                end = min(len(json_str), col + 200)
                logger.error(f"Tokenizer parse failed near column {col}: ...{json_str[start:end]}...")
            raise
        self.tokenizer_path = TOKENIZER_DIR
        self._config = self._load_config()
        self._setup_special_tokens(self._config)

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

        if self.pad_token is not None:
            self.tokenizer.enable_padding(
                direction="left",  # GLM-5 uses left padding
                pad_id=self.pad_token_id,
                pad_token=self.pad_token,
            )

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
