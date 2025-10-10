from typing import Optional

from pydantic import BaseModel


class LoadModelRequest(BaseModel):
    huggingface_ckpt_name: str
    hf_cache_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    pt_ckpt_dir: Optional[str] = None

class ModelInfoResponse(BaseModel):
    status: str
    message: Optional[str] = None
    shm_name: Optional[str] = None
    tensor_meta_shm_name: Optional[str] = None
    parameter_server_size: Optional[int] = None
    huggingface_ckpt_name: Optional[str] = None
    pt_ckpt_dir: Optional[str] = None

