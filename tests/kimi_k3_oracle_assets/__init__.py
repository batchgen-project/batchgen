"""Vendored Kimi-K3 oracle assets — TEST-ONLY, never imported by production.

Provenance (byte-verified; the pins are asserted by
tests/test_kimi_k3_model.py::test_oracle_md5_pins):

  * ``modeling_kimi_linear.py`` — verbatim from the released checkpoint
    ``/taijifs_zw35/share_304153846/hunyuan/tairanxu/models/Kimi-K3/``
    (fetch: ``ssh h20-instance-2 'cat .../modeling_kimi_linear.py'``).
    md5 4e3de36ab2a5de1232c05ce346a3426e.  Owns the ENTIRE K3 text model:
    the VLM wrapper ``modeling_kimi_k3.py`` (md5 d5b7e2e6d4f1263cc390c0f6476aeea2,
    deliberately NOT vendored — it drags vision imports) instantiates
    ``KimiLinearForCausalLM(config.text_config)`` and overrides nothing.
  * ``configuration_kimi_k3.py`` — verbatim from the same checkpoint,
    md5 3165dde7cebe8471fdf43aa9890d5c02; byte-identical to the copy already
    vendored at ``batchgen/models/moonshotai/kimi_k3/assets/`` (also asserted).
  * ``fla_cpu_shim.py`` — new code (BatchGen), providing torch stand-ins for
    the 7 fla symbols the oracle imports.  fla-core is triton/GPU-only and
    does not install on the macOS dev machine; the shim's KDA math delegates
    to the production-vendored ``kimi_k3/kda_reference.py`` (fla's own torch
    reference functions, fla-core 0.4.2), so on CPU the kernel interior
    CANCELS between the two stacks — it is validated exclusively by the staged
    GPU test (tests/gpu/test_kimi_k3_kda_fla_parity.py).

License note: ``modeling_kimi_linear.py`` / ``configuration_kimi_k3.py`` carry
the checkpoint's own license; the fla torch references inside
``kda_reference.py`` are MIT (fla-org/flash-linear-attention).  Vendored for
offline parity testing only.
"""

MODELING_KIMI_LINEAR_MD5 = "4e3de36ab2a5de1232c05ce346a3426e"
CONFIGURATION_KIMI_K3_MD5 = "3165dde7cebe8471fdf43aa9890d5c02"
MODELING_KIMI_K3_MD5 = "d5b7e2e6d4f1263cc390c0f6476aeea2"  # NOT vendored; recorded
