#!/usr/bin/env python3
# ---------------------------------------------------------------------------- #
#  BatchGen Tools - Module Entry Point                                          #
#  copyright (c) EfficientMoE team 2025                                         #
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
