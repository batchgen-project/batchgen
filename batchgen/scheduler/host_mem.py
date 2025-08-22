# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
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

# def get_simple_memory_info():
#     """
#     Get basic memory information in a simple format
#     """
#     mem = psutil.virtual_memory()
#     return {
#         'total_gb': mem.total / (1024**3),
#         'available_gb': mem.available / (1024**3),
#         'used_percent': mem.percent
#     }

# # Usage
# mem = get_simple_memory_info()
# print(f"Total Memory: {mem['total_gb']:.1f} GB")
# print(f"Available Memory: {mem['available_gb']:.1f} GB")
# print(f"Memory Usage: {mem['used_percent']}%")


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


# Example usage
if __name__ == "__main__":
    mem = get_physical_memory_info()
    if isinstance(mem, dict):
        print(f"Total Physical Memory: {mem['total_physical']:.2f} GB")
        print(f"Actually Free Memory: {mem['actually_free']:.2f} GB")
