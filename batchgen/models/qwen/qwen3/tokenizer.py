# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
#                                                                               #
#  you may obtain a copy of the license at                                      #
#                                                                               #
#                  http://www.apache.org/licenses/license-2.0                   #
#                                                                               #
#  unless required by applicable law or agreed to in writing, software          #
#  distributed under the license is distributed on an "as is" basis,            #
#  without warranties or conditions of any kind, either express or implied.     #
#  see the license for the specific language governing permissions and          #
#  limitations under the license.                                               #
# ---------------------------------------------------------------------------- #

"""Qwen3 tokenizer for BatchGen.

Uses HuggingFace's Qwen2Tokenizer (BBPE) as backend.
Qwen3 tokenizer specs:
- Vocabulary size: 151,936
- BOS token: <|endoftext|> (ID: 151643)
- EOS token: <|im_end|> (ID: 151645)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

from batchgen.config.base_tokenizer import BaseTokenizer
from batchgen.config.tokenizer_registry import register_tokenizer

logger = logging.getLogger(__name__)


@register_tokenizer("qwen3")
class Qwen3Tokenizer(BaseTokenizer):
    """Qwen3 tokenizer wrapping HuggingFace's tokenizer."""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize Qwen3 tokenizer.

        Args:
            model_path: Path to model directory or HF model identifier.
                       If None, uses the default Qwen3Guard-Gen-8B.
        """
        from transformers import AutoTokenizer

        if model_path is None:
            model_path = "Qwen/Qwen3Guard-Gen-8B"

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        self.eos_token_id = 151645  # <|im_end|>
        self.pad_token_id = 151643  # <|endoftext|>
        self.bos_token_id = 151643
        self.vocab_size = 151936
        self.padding_side = "left"

        # MUST define eos_token_ids set (convention)
        self.eos_token_ids = {151645}

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> List[int]:
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
    ) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def __call__(
        self,
        texts: Union[str, List[str]],
        return_tensors: Optional[str] = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        **kwargs,
    ) -> Dict:
        result = self._tokenizer(
            texts,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            **kwargs,
        )
        return result

    def apply_chat_template(
        self,
        messages: List[Dict],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> Union[str, List[int]]:
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
