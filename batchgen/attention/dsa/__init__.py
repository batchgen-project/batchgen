"""DeepSeek Sparse Attention (DSA) module for BatchGen.

DSA uses a lightweight Lightning Indexer to score cached tokens and select
the top-K most relevant ones, then runs full MLA attention only on the
sparse subset. Used by DeepSeek-V3.2 and GLM-5.

Components:
- indexer: Lightning Indexer module (Q/K projections + top-K selection)
- dsa_prefill: Prefill path (full attention via FA3 + indexer cache population)
- dsa_decode: Decode path (indexer scoring → top-K → sparse FlashMLA)
- sparse_gather: GPU kernel for gathering KV entries by token index
"""

from batchgen.attention.dsa.indexer import LightningIndexer
from batchgen.attention.dsa.sparse_gather import sparse_gather_from_paged_kv
from batchgen.attention.dsa.unified_selector import (
	make_flashmla_selected_block_table,
	select_mla_kv_for_flashmla_bf16,
	select_mla_kv_for_flashmla_bf16_out,
	view_selected_mla_kv_as_flashmla_pages,
)

__all__ = [
	"LightningIndexer",
	"make_flashmla_selected_block_table",
	"select_mla_kv_for_flashmla_bf16",
	"select_mla_kv_for_flashmla_bf16_out",
	"sparse_gather_from_paged_kv",
	"view_selected_mla_kv_as_flashmla_pages",
]
