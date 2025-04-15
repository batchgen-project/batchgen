import torch
import logging
from typing import Optional
def copy_tensor_p2p_by_pointer(dst_ptr: int, dst_device: int, 
                                src_ptr: int, src_device: int, 
                                size_in_bytes: int,
                                stream: Optional[torch.cuda.Stream] = None) -> bool:
    """
    Copy data directly between device pointers using P2P.
    This is a lower-level function for advanced usage.
    
    Args:
        dst_ptr: Destination pointer (integer representation)
        dst_device: Destination device ID
        src_ptr: Source pointer (integer representation)
        src_device: Source device ID
        size_in_bytes: Size of the data to copy in bytes
        stream: Optional CUDA stream to use for the copy
    
    Returns:
        True if copy is successful, False otherwise
    """
    # if not self.is_p2p_available(src_device, dst_device):
    #     self.logger.error(f"P2P access not available from device {src_device} to device {dst_device}")
    #     return False
    
    torch.cuda.set_device(dst_device)
    
    # Create source and destination tensors from pointers
    src_tensor = torch.cuda.ByteTensor(size=torch.Size([size_in_bytes]), 
                                        device=torch.device(f"cuda:{src_device}"),
                                        storage_offset=0)
    dst_tensor = torch.cuda.ByteTensor(size=torch.Size([size_in_bytes]), 
                                        device=torch.device(f"cuda:{dst_device}"),
                                        storage_offset=0)
    
    # Set tensor storage pointers (this is a bit hacky but works)
    src_tensor.storage().data_ptr = src_ptr
    dst_tensor.storage().data_ptr = dst_ptr
    
    # Use provided stream or create a new one
    with torch.cuda.stream(stream) if stream else torch.cuda.stream(torch.cuda.Stream(device=dst_device)):
        try:
            # Perform the copy
            dst_tensor.copy_(src_tensor)
            return True
        except RuntimeError as e:
            logging.error(f"Failed to copy data from device {src_device} to {dst_device}: {e}")
            return False