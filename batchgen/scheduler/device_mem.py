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

import torch


def get_gpu_memory_info():
    """
    Get memory information for all available CUDA devices.
    Returns a list of dictionaries containing device information.
    """
    if not torch.cuda.is_available():
        return "No CUDA devices available"

    gpu_info = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        info = {
            "device_index": i,
            "name": props.name,
            "total_memory": props.total_memory / (1024**3),  # Convert to GB
            "cuda_capability": f"{props.major}.{props.minor}",
        }
        gpu_info.append(info)

    return gpu_info


# Example usage
if __name__ == "__main__":
    devices = get_gpu_memory_info()
    if isinstance(devices, str):
        print(devices)
    else:
        for device in devices:
            print(f"GPU {device['device_index']}: {device['name']}")
            print(f"Total Memory: {device['total_memory']:.2f} GB")
            print(f"CUDA Capability: {device['cuda_capability']}")
            print("-" * 50)
