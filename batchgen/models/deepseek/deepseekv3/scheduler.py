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

"""
DEPRECATED: This module is deprecated. Use planner.py instead.

The Scheduler has been renamed to Planner to better reflect its purpose:
- Planners decide config values
- Workers execute based on those configs

Migration:
    # Old
    from .scheduler import Scheduler

    # New
    from .planner import DeepSeekV3Planner
"""

import warnings

from .planner import DeepSeekV3Planner

# Backward compatibility alias
Scheduler = DeepSeekV3Planner

warnings.warn(
    "scheduler.py is deprecated. Use 'from .planner import DeepSeekV3Planner' instead.",
    DeprecationWarning,
    stacklevel=2
)
