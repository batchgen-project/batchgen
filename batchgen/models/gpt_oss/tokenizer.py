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

"""GPT-OSS-120B tokenizer for BatchGen.

This module provides the tokenizer implementation for GPT-OSS-120B model.

GPT-OSS tokenizer specifications:
- Based on o200k encoding with custom special tokens
- Vocabulary size: 201,088 tokens
- BOS token: <|startoftext|> (ID: 199998)
- EOS token: <|return|> (ID: 200002)
- PAD token: <|endoftext|> (ID: 199999)

The tokenizer.json file is bundled with BatchGen in this directory.
It is NOT loaded from user's cache directory.
"""

import logging
from pathlib import Path

from batchgen.config.fast_tokenizer import FastTokenizer
from batchgen.config.tokenizer_registry import register_tokenizer

logger = logging.getLogger(__name__)

# Tokenizer files are in the same directory as this module
TOKENIZER_DIR = Path(__file__).parent


# GPT-OSS-120B special token IDs
GPT_OSS_BOS_TOKEN_ID = 199998   # <|startoftext|>
GPT_OSS_EOS_TOKEN_ID = 200002   # <|return|>
GPT_OSS_PAD_TOKEN_ID = 199999   # <|endoftext|>
GPT_OSS_VOCAB_SIZE = 201088


@register_tokenizer("gpt_oss")
class GPTOssTokenizer(FastTokenizer):
    """GPT-OSS-120B tokenizer.

    Loads tokenizer.json from package directory (not user cache).

    Attributes:
        bos_token_id: 199998 (<|startoftext|>)
        eos_token_id: 200002 (<|return|>)
        pad_token_id: 199999 (<|endoftext|>)
        vocab_size: 201,088
    """

    def __init__(self):
        """Initialize the GPT-OSS tokenizer.

        Loads tokenizer.json from the package directory (TOKENIZER_DIR).
        """
        # Load from package directory, not user path
        super().__init__(str(TOKENIZER_DIR))

        # Override with GPT-OSS-specific special token IDs
        self.bos_token_id = GPT_OSS_BOS_TOKEN_ID
        self.eos_token_id = GPT_OSS_EOS_TOKEN_ID
        self.pad_token_id = GPT_OSS_PAD_TOKEN_ID
        self.vocab_size = GPT_OSS_VOCAB_SIZE

        # Get the actual token strings from vocabulary for padding setup
        vocab = self.tokenizer.get_vocab()
        self.bos_token = None
        self.eos_token = None
        self.pad_token = None

        # Find tokens by ID
        id_to_token = {v: k for k, v in vocab.items()}
        if self.bos_token_id in id_to_token:
            self.bos_token = id_to_token[self.bos_token_id]
        if self.eos_token_id in id_to_token:
            self.eos_token = id_to_token[self.eos_token_id]
        if self.pad_token_id in id_to_token:
            self.pad_token = id_to_token[self.pad_token_id]

        # Re-enable padding with correct pad token
        if self.pad_token is not None:
            self.tokenizer.enable_padding(
                direction="right",
                pad_id=self.pad_token_id,
                pad_token=self.pad_token,
            )

        logger.info(
            f"GPT-OSS tokenizer initialized: vocab_size={self.vocab_size}, "
            f"bos={self.bos_token_id}, eos={self.eos_token_id}, pad={self.pad_token_id}"
        )
