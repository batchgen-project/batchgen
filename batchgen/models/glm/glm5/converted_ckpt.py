"""Converted-checkpoint directory resolution for the GLM5 parameter server.

A caller-provided directory (the server's ``--converted-ckpt-dir``) is handed
to the converter as the conversion's output directory, so a pre-converted
checkpoint kept outside the cache dir — for example on node-local storage —
is validated and reused instead of being silently reconverted into
``<cache_dir>/converted_ckpt``. ``None`` keeps the converter's cache-dir
default. This lives in a leaf module with no native imports so the contract
stays testable without the engine runtime.
"""


def resolve_converted_ckpt_dir(cache_dir, converted_ckpt_dir):
    from batchgen.ckpt_converter.ckpt_converter import ckpt_converter

    converter = ckpt_converter()
    return converter.convert_model_directory(
        cache_dir,
        output_dir=converted_ckpt_dir,
    )
