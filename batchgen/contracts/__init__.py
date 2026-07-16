"""Model-support contracts the runtime core depends on.

Generic runtime code (batchgen_worker.py / decode.py / prefill.py / wrappers)
must stay model-agnostic; per-model behavior plugs in behind these contracts.
See batchgen_design/model_architecture_spec.md (section 2.1) and
batchgen_design/core_model_purity_audit.md.
"""
