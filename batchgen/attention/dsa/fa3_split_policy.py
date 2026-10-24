# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                            #
# ---------------------------------------------------------------------------- #

"""Split policy for the SPARSE selected-token FA3 decode call.

Dependency-free on purpose: both the eager selector and the captured graph
segment read this constant, and the segment module must stay importable in
CPU-only environments (the selector's import chain needs triton/FA3).

FA3 v2.8.2's auto-split (0) over-splits heterogeneous decode batches: at
production per-row selected lengths it adds a combine pass per layer while
the main kernel is no faster, and the integer knob behaves as a boolean in
this build (2/3/4/8 act like 0). 1 = no split, measured bitwise-identical
outputs. The dense all_short path keeps auto-split: its geometry was not
covered by the measurement, so it is deliberately not changed.
"""

GLM5_SPARSE_FA3_NUM_SPLITS = 1
