"""The user docs name exactly the models the prefix-cache gate accepts.

Kimi-K3.1-Pro gained prefix-cache support in code while the flag reference,
the support matrix and its deployment guide still said a launch with
``--enable-prefix-cache`` was refused. These checks tie the docs to
``batchgen.prefix_reuse.config``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from batchgen.prefix_reuse import config

REPO = Path(__file__).resolve().parents[1]

# The display name the docs use for each model type the gate accepts.
_DOC_NAMES = {
    "gpt_oss": "GPT-OSS",
    "glm_moe_dsa_5_2": "GLM-5.2",
    "glm_moe_dsa_5_3": "GLM-5.3",
    "kimi_k31": "Kimi-K3.1",
    "kimi_k3": "Kimi-K3",
    "kimi_linear": "Kimi-Linear",
}

_ACCEPTED = (
    config._FULL_KV_MODEL_TYPES
    | config._DSA_MODEL_TYPES
    | config._MLA_LATENT_MODEL_TYPES
)


def _flag_row() -> str:
    text = (REPO / "docs" / "server-flags.md").read_text()
    return next(
        line for line in text.splitlines() if line.startswith("| `--enable-prefix-cache`")
    )


def test_every_accepted_model_type_has_a_doc_name():
    assert set(_DOC_NAMES) == set(_ACCEPTED)


@pytest.mark.parametrize("model_type", sorted(_DOC_NAMES))
def test_the_flag_reference_lists_every_accepted_model(model_type):
    row = _flag_row()
    supported = row.split("Supported model types:", 1)[1].split(" For any other", 1)[0]
    name = _DOC_NAMES[model_type]
    assert re.search(rf"{re.escape(name)}(?!\w|\.\d)", supported), (
        f"--enable-prefix-cache accepts {model_type} but the flag reference "
        f"does not list {name}: {supported!r}"
    )


@pytest.mark.parametrize(
    "doc", ["docs/support-matrix.md", "docs/deploy-kimi-k3.1-h20.md"]
)
def test_no_doc_says_an_accepted_model_is_refused(doc):
    config.require_prefix_cache_model_support("moonshotai/Kimi-K3.1-Pro")
    text = " ".join((REPO / doc).read_text().split())
    for sentence in re.split(r"(?<=[.;])\s", text):
        if "prefix cache" in sentence.lower() or "--enable-prefix-cache" in sentence:
            assert not re.search(
                r"not supported|does not support|unsupported|is refused", sentence
            ), f"{doc}: {sentence}"
