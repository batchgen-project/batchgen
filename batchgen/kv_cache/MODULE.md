# GPU page-table capacity

Serving supplies the resolved deployment `max_model_len` on `BasicConfig` and
`GPUPagedKVConfig`. `KVCacheManager.apply_page_table_capacity` sets each region's
row width to `ceil(max_model_len / page_size_tokens)`. Prompt length, completion
budget, prepack hints and pool page count do not determine the row width.

Standalone managers can still supply an explicit
`cuda_graph_max_pages_per_sequence`; the default standalone capacity is fixed
and does not change with batch hints. The serving path always supplies
`max_model_len`, including eager decode.

GLM graph scoring and its kernel prewarm consume this same deployment length.
The existing graph-static-memory policy can cap the captured span; longer
legal sequences use the existing eager fallback without changing the advertised
deployment limit.

See [the server contract](../server/MODULE.md) for startup resource verification,
auto-fit, diagnostics, and the pool-budget input.
