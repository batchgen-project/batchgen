import logging
import os
from typing import Any


def apply_worker_env_overrides(args: Any) -> None:
    inherited = os.environ.get("BATCHGEN_V4_RESIDENT_EXPERTS")
    effective = "1" if getattr(args, "v4_resident_experts", False) else "0"
    os.environ["BATCHGEN_V4_RESIDENT_EXPERTS"] = effective
    logging.info(
        "Worker env applied: BATCHGEN_V4_RESIDENT_EXPERTS=%s "
        "(worker_args=%s, inherited=%s)",
        effective,
        getattr(args, "v4_resident_experts", False),
        inherited if inherited is not None else "<unset>",
    )
