import os

import sitecustomize  # noqa: F401  triggers tracer install when V4_COLL_TRACE=1

import torch
import torch.distributed as dist  # noqa: F401

print("torch", torch.__version__, "ndev", torch.cuda.device_count())

import v4_collective_tracer as t

print("tracer installed:", bool(t._WRAPPED), "wrapped:", sorted(t._WRAPPED)[:4])

import batchgen

print("batchgen from:", os.path.dirname(batchgen.__file__))
