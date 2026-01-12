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

"""GPT-OSS-120B model support for BatchGen.

OpenAI's GPT-OSS-120B is a 117B parameter MoE model (5.1B active) with:
- 36 layers, hidden_size=2880
- GQA: 64 attention heads, 8 KV heads
- 128 experts, Top-4 routing
- MXFP4 quantization (~4.25 bits/param)
- Alternating sliding (128) / full attention
- YaRN RoPE with theta=150000, factor=32
"""

from .configuration_gpt_oss import GptOssConfig
from .modeling_gpt_oss import GptOssForCausalLM

__all__ = ["GptOssConfig", "GptOssForCausalLM"]
