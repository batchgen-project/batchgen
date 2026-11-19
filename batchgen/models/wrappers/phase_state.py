# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                            #
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

"""Process-wide model phase state and its single owner.

The model wrappers keep their phase state (prefill/decode context) at process
scope, so it is only well defined while exactly one ``BatchGenWorker`` drives
it. ``claim_process_phase_state`` enforces that: a second live worker in the
same process fails loudly at construction instead of silently sharing, and
overwriting, the first worker's state.

The claim holds its owner weakly, so a worker that has been garbage collected
no longer blocks a successor. Workers are constructed once per process, on the
process's main thread, so the claim needs no lock.
"""

from __future__ import annotations

import weakref
from typing import Any, Optional

__all__ = [
    "claim_process_phase_state",
    "release_process_phase_state",
]

_owner_ref: Optional["weakref.ReferenceType[Any]"] = None


def _current_owner() -> Any:
    return None if _owner_ref is None else _owner_ref()


def claim_process_phase_state(owner: Any) -> None:
    """Record ``owner`` as the one worker driving this process's phase state.

    Raises ``RuntimeError`` if a different owner still holds the claim. The
    same owner may claim again.
    """
    global _owner_ref
    current = _current_owner()
    if current is not None and current is not owner:
        raise RuntimeError(
            "a BatchGenWorker already owns the model phase state of this process"
        )
    _owner_ref = weakref.ref(owner)


def release_process_phase_state(owner: Any) -> None:
    """Drop the claim if ``owner`` holds it; otherwise do nothing."""
    global _owner_ref
    if _current_owner() is owner:
        _owner_ref = None
