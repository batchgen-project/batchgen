import os
import threading
import time
import traceback

import torch.distributed as dist

_LOCK = threading.Lock()
_STATE = {"counter": 0, "fh": None, "rank": -1}
_WRAPPED = {}
_TRACED = (
    "all_gather_object",
    "broadcast_object_list",
    "all_gather_into_tensor",
    "all_gather",
    "all_reduce",
    "reduce_scatter_tensor",
    "broadcast",
    "barrier",
    "gather_object",
    "scatter_object_list",
)


def _caller_site(skip=3):
    stack = traceback.extract_stack()
    for frame in reversed(stack[:-skip]):
        if "v4_collective_tracer" in frame.filename:
            continue
        if frame.filename.endswith("distributed/distributed_c10d.py"):
            continue
        return f"{os.path.basename(frame.filename)}:{frame.lineno}:{frame.name}"
    return "unknown"


def _open_for_rank():
    rank = (
        dist.get_rank()
        if dist.is_initialized()
        else int(os.getenv("RANK", "-1"))
    )
    if _STATE["fh"] is not None and _STATE["rank"] == rank:
        return
    out_dir = os.getenv("V4_COLL_TRACE_DIR", "/tmp")
    path = os.path.join(out_dir, f"v4_coll_trace_rank{rank}.log")
    _STATE["fh"] = open(path, "a", buffering=1)
    _STATE["rank"] = rank


def _make_wrapper(name, orig):
    def wrapper(*args, **kwargs):
        with _LOCK:
            _open_for_rank()
            _STATE["counter"] += 1
            idx = _STATE["counter"]
            site = _caller_site()
            _STATE["fh"].write(f"{idx}\t{name}\t{site}\t{time.time():.6f}\n")
        return orig(*args, **kwargs)

    return wrapper


def install():
    if _WRAPPED:
        return
    for name in _TRACED:
        orig = getattr(dist, name, None)
        if orig is None:
            continue
        _WRAPPED[name] = orig
        setattr(dist, name, _make_wrapper(name, orig))


def uninstall():
    for name, orig in _WRAPPED.items():
        setattr(dist, name, orig)
    _WRAPPED.clear()


if os.getenv("V4_COLL_TRACE", "0") == "1":
    install()
