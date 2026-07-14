"""Resolve a model name to its Initializer class.

Thin wrapper over the single dispatch registry in `batchgen.model_dispatch`
(see batchgen_design/model_architecture_spec.md section 2.1 -- model->implementation
dispatch lives only in the registry layer; the runtime core must not branch on
model names).
"""
from batchgen.model_dispatch import resolve_model


def get_initializer(model_name: str):
	return resolve_model(model_name).initializer_loader()
