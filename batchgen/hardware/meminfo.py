# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                         #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
#  see the license at                                                           #
#                                                                               #
#                  http://www.apache.org/licenses/license-2.0                   #
# ---------------------------------------------------------------------------- #

"""Host physical memory facts, read from ``/proc/meminfo``."""


def get_physical_memory_info():
    """
    Get physical memory information, focusing on actually free memory
    that can be pinned. Returns values in GB.
    """
    try:
        with open("/proc/meminfo", "r") as f:
            mem_info = {}
            for line in f:
                if line.startswith(
                    ("MemTotal:", "MemFree:", "Cached:", "Buffers:")
                ):
                    key, value = line.split(":")
                    # Convert kB to GB
                    mem_info[key] = float(value.strip().split()[0]) / (
                        1024 * 1024
                    )

            # Actually free physical memory = MemFree + Cached + Buffers
            # This memory can be safely pinned
            actually_free = (
                mem_info.get("MemFree", 0)
                + mem_info.get("Cached", 0)
                + mem_info.get("Buffers", 0)
            )

            return {
                "total_physical": mem_info.get("MemTotal", 0),
                "actually_free": actually_free,
            }
    except Exception as e:
        return f"Error reading memory info: {str(e)}"
