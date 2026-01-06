"""Disk-backed storage for batch files, outputs, and metadata."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

from batchgen.server.io_struct import (
    BatchObject,
    BatchResultItem,
    BatchStatus,
    FileObject,
)

logger = logging.getLogger(__name__)


class StorageManager:
    """Manage on-disk state for uploaded files and batches."""

    def __init__(self, storage_root: Path):
        self.storage_root = storage_root
        self.files_dir = storage_root / "files"
        self.files_meta_dir = storage_root / "files_meta"
        self.batches_dir = storage_root / "batches"
        self.output_dir = storage_root / "outputs"
        for directory in (
            self.files_dir,
            self.files_meta_dir,
            self.batches_dir,
            self.output_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    # ---------------------- File Metadata ----------------------
    def save_metadata(self, file_id: str, metadata: Dict) -> None:
        path = self.files_meta_dir / f"{file_id}.json"
        self._write_json(path, metadata)

    def load_metadata(self, file_id: str) -> Optional[Dict]:
        path = self.files_meta_dir / f"{file_id}.json"
        return self._read_json(path)

    def delete_file_metadata(self, file_id: str) -> None:
        meta_path = self.files_meta_dir / f"{file_id}.json"
        if meta_path.exists():
            meta_path.unlink()

    def find_file_by_checksum(self, checksum: str) -> Optional[Dict]:
        for meta in self.list_all_metadata():
            if meta.get("checksum") == checksum:
                return meta
        return None

    def list_all_metadata(self) -> List[Dict]:
        records: List[Dict] = []
        for meta_file in self.files_meta_dir.glob("*.json"):
            meta = self._read_json(meta_file)
            if meta:
                records.append(meta)
        return records

    # ---------------------- Batch Metadata ----------------------
    def save_batch(self, batch: BatchObject) -> None:
        path = self.batches_dir / f"{batch.id}.json"
        self._write_json(path, batch.dict())

    def load_batch(self, batch_id: str) -> Optional[BatchObject]:
        path = self.batches_dir / f"{batch_id}.json"
        data = self._read_json(path)
        if not data:
            return None
        return BatchObject(**data)

    def list_all_batches(self) -> List[BatchObject]:
        batches: List[BatchObject] = []
        for batch_file in self.batches_dir.glob("*.json"):
            data = self._read_json(batch_file)
            if data:
                batches.append(BatchObject(**data))
        batches.sort(key=lambda b: b.created_at, reverse=True)
        return batches

    def get_active_batch_for_file(self, file_id: str) -> Optional[BatchObject]:
        active_statuses = {
            BatchStatus.VALIDATING,
            BatchStatus.IN_PROGRESS,
            BatchStatus.CANCELLING,
        }
        for batch in self.list_all_batches():
            if (
                batch.input_file_id == file_id
                and batch.status in active_statuses
            ):
                return batch
        return None

    def update_batch_status(
        self, batch_id: str, status: BatchStatus, **updates
    ) -> Optional[BatchObject]:
        batch = self.load_batch(batch_id)
        if not batch:
            return None
        data = batch.dict()
        data.update({"status": status.value, **updates})
        updated = BatchObject(**data)
        self.save_batch(updated)
        return updated

    # ---------------------- Outputs ----------------------
    def write_output_file(
        self, file_id: str, items: List[BatchResultItem]
    ) -> Path:
        output_path = self.files_dir / file_id
        with output_path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item.dict(), default=str))
                handle.write("\n")
        return output_path

    # ---------------------- Helpers ----------------------
    def _write_json(self, path: Path, data: Dict) -> None:
        with self._lock:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(data, handle)

    def _read_json(self, path: Path) -> Optional[Dict]:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            try:
                return json.load(handle)
            except json.JSONDecodeError:
                logger.warning("Failed to decode JSON from %s", path)
                return None
