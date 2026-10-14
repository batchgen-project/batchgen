#!/usr/bin/env python3
# ---------------------------------------------------------------------------- #
#  BatchGen Tools - Module Entry Point                                          #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#                                                                               #
#  Enables: python -m batchgen.tools.convert_checkpoint                         #
# ---------------------------------------------------------------------------- #
"""
Module entry point for BatchGen tools.

This allows running tools as modules:
    python -m batchgen.tools.convert_checkpoint --input-dir /path/to/model
"""

from batchgen.tools.convert_checkpoint import main

if __name__ == "__main__":
    main()
