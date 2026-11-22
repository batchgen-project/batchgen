"""The debug-key registry of a server: core keys plus every model family's keys.

Each family declares its keys in its own package; this module is the one place
that assembles them, so the server validates a change against the full set.
"""
from batchgen.debug_state import CORE_KEYS, CORE_RETIRED, Registry
from batchgen.models.glm.glm5 import debug_keys as _glm5
from batchgen.models.moonshotai.kimi_linear import debug_keys as _kimi_linear

REGISTRY = Registry(
    CORE_KEYS + _glm5.DEBUG_KEYS + _kimi_linear.DEBUG_KEYS,
    retired={**CORE_RETIRED, **_glm5.RETIRED_DEBUG_KEYS, **_kimi_linear.RETIRED_DEBUG_KEYS},
)
EMPTY = REGISTRY.empty
