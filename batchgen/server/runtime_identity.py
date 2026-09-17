"""Immutable identity and names for one BatchGen server run."""

from __future__ import annotations

import re
import secrets
import tempfile
from dataclasses import dataclass
from pathlib import Path


_INSTANCE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,47}\Z")
_RUN_ID_RE = re.compile(r"[0-9a-f]{32}\Z")


@dataclass(frozen=True)
class RuntimeIdentity:
    """Names one immutable server run without enabling shared-host mode."""

    mode: str
    instance_id: str
    run_id: str

    @classmethod
    def create(
        cls,
        instance_id: str,
        *,
        mode: str = "exclusive",
        run_id: str | None = None,
    ) -> "RuntimeIdentity":
        if mode != "exclusive":
            raise ValueError(
                "shared runtime mode is not available until scoped lifecycle "
                "and lane admission are implemented"
            )
        if not isinstance(instance_id, str) or not _INSTANCE_ID_RE.fullmatch(
            instance_id
        ):
            raise ValueError(
                "instance_id must match [a-z0-9][a-z0-9_-]{0,47}"
            )
        resolved_run_id = secrets.token_hex(16) if run_id is None else run_id
        if not isinstance(resolved_run_id, str) or not _RUN_ID_RE.fullmatch(
            resolved_run_id
        ):
            raise ValueError(
                "run_id must be exactly 32 lowercase hexadecimal characters"
            )
        return cls(
            mode=mode,
            instance_id=instance_id,
            run_id=resolved_run_id,
        )

    @property
    def resource_prefix(self) -> str:
        return f"batchgen_{self.instance_id}_{self.run_id}"

    @property
    def host_kv_shm_name(self) -> str:
        return f"{self.resource_prefix}_host_kv"

    @property
    def host_kv_aux_shm_name(self) -> str:
        return f"{self.resource_prefix}_host_kv_aux"

    @property
    def query_book_shm_prefix(self) -> str:
        return f"{self.resource_prefix}_input_ids"

    @property
    def runtime_dir(self) -> Path:
        return Path(tempfile.gettempdir()) / self.resource_prefix

    @property
    def reload_status_dir(self) -> Path:
        return self.runtime_dir / "reload_status"
