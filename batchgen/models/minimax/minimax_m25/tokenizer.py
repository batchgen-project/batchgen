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

"""MiniMax-M2.5 tokenizer for BatchGen.

MiniMax-M2.5 tokenizer specifications:
- Vocabulary size: 200,064 tokens
- Uses HuggingFace tokenizer.json format
- tokenizer.json must be bundled in this directory

To bundle the tokenizer files:
    huggingface-cli download MiniMaxAI/MiniMax-M2.5 tokenizer.json tokenizer_config.json \
        --local-dir batchgen/models/minimax/minimax_m25/
"""

import logging
from pathlib import Path

from batchgen.config.fast_tokenizer import FastTokenizer
from batchgen.config.tokenizer_registry import register_tokenizer

logger = logging.getLogger(__name__)

# Tokenizer files are in the same directory as this module
TOKENIZER_DIR = Path(__file__).parent

# MiniMax-M2.5 special token IDs (from tokenizer_config.json added_tokens_decoder)
MINIMAX_M25_BOS_TOKEN_ID = 200034
MINIMAX_M25_EOS_TOKEN_ID = 200020
MINIMAX_M25_VOCAB_SIZE = 200064


@register_tokenizer("minimax_m25")
class MiniMaxM25Tokenizer(FastTokenizer):
    """MiniMax-M2.5 tokenizer.

    Loads tokenizer.json from package directory (not user cache).

    Attributes:
        bos_token_id: BOS token ID
        eos_token_id: EOS token ID
        pad_token_id: PAD token ID (uses EOS)
        vocab_size: 200,064
    """

    def __init__(self):
        """Initialize the MiniMax-M2.5 tokenizer."""
        super().__init__(str(TOKENIZER_DIR))

        self.bos_token_id = MINIMAX_M25_BOS_TOKEN_ID
        self.eos_token_id = MINIMAX_M25_EOS_TOKEN_ID
        self.pad_token_id = MINIMAX_M25_EOS_TOKEN_ID  # Use EOS as pad
        self.vocab_size = MINIMAX_M25_VOCAB_SIZE

        # Get the actual token strings from vocabulary for padding setup
        vocab = self.tokenizer.get_vocab()
        self.bos_token = None
        self.eos_token = None
        self.pad_token = None

        id_to_token = {v: k for k, v in vocab.items()}
        if self.bos_token_id in id_to_token:
            self.bos_token = id_to_token[self.bos_token_id]
        if self.eos_token_id in id_to_token:
            self.eos_token = id_to_token[self.eos_token_id]
            self.pad_token = self.eos_token

        if self.pad_token is not None:
            self.tokenizer.enable_padding(
                direction="right",
                pad_id=self.pad_token_id,
                pad_token=self.pad_token,
            )

        # Load chat template from separate .jinja file (not in tokenizer_config.json)
        chat_template_path = TOKENIZER_DIR / "chat_template.jinja"
        if chat_template_path.exists():
            self.chat_template = chat_template_path.read_text()
            logger.info("Loaded chat template from chat_template.jinja")

        logger.info(
            f"MiniMax-M2.5 tokenizer initialized: vocab_size={self.vocab_size}, "
            f"bos={self.bos_token_id}, eos={self.eos_token_id}"
        )
