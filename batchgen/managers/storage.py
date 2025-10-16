import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from batchgen.server_args import ServerArgs
import multiprocessing as mp
from transformers import AutoTokenizer

from .batch_schema import BatchObject, BatchStatus
from .completion_schema import (
    ChatCompletion,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRole,
    FinishReason,
    CompletionUsage
)

logger = logging.getLogger(__name__)

class StorageManager:
    def __init__(self, server_args: ServerArgs):
        '''
        Manages storage for batch processing.
        '''
        self.server_args = server_args
        
        # Setup storage directories
        self.storage_path = Path(server_args.file_path)
        self.metadata_path = self.storage_path / "metadata"
        self.batches_path = self.storage_path / "batches"
        self.outputs_path = self.storage_path / "outputs"
        
        # Create all necessary directories
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        self.batches_path.mkdir(parents=True, exist_ok=True)
        self.outputs_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Storage initialized at {self.storage_path}")
        
    # ============================================================================
    # File Metadata Management
    # ============================================================================
    
    def save_metadata(self, file_id: str, metadata: dict) -> None:
        """Save file metadata to disk"""
        metadata_file = self.metadata_path / f"{file_id}.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)
        logger.debug(f"Saved metadata for file {file_id}")
    
    def load_metadata(self, file_id: str) -> Optional[dict]:
        """Load file metadata from disk"""
        metadata_file = self.metadata_path / f"{file_id}.json"
        if not metadata_file.exists():
            return None
        with open(metadata_file, "r") as f:
            return json.load(f)
    
    def list_all_metadata(self) -> list[dict]:
        """List all file metadata"""
        metadata_list = []
        for metadata_file in self.metadata_path.glob("*.json"):
            with open(metadata_file, "r") as f:
                metadata_list.append(json.load(f))
        return metadata_list
    
    def find_file_by_checksum(self, checksum: str) -> Optional[dict]:
        """Find a file by its checksum"""
        for metadata_file in self.metadata_path.glob("*.json"):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
                if metadata.get("checksum") == checksum:
                    return metadata
        return None
    
    def delete_file_metadata(self, file_id: str) -> bool:
        """
        Delete file metadata from disk
        
        Returns:
            True if deleted, False if not found
        """
        metadata_file = self.metadata_path / f"{file_id}.json"
        if metadata_file.exists():
            metadata_file.unlink()
            logger.info(f"Deleted metadata for file {file_id}")
            return True
        return False
    
    # ============================================================================
    # Batch Management
    # ============================================================================
    
    def save_batch(self, batch: BatchObject) -> None:
        """Save batch metadata to disk"""
        batch_file = self.batches_path / f"{batch.id}.json"
        with open(batch_file, "w") as f:
            json.dump(batch.model_dump(), f)
        logger.debug(f"Saved batch {batch.id}")
    
    def load_batch(self, batch_id: str) -> Optional[BatchObject]:
        """Load batch metadata from disk"""
        batch_file = self.batches_path / f"{batch_id}.json"
        if not batch_file.exists():
            return None
        with open(batch_file, "r") as f:
            data = json.load(f)
        return BatchObject(**data)
    
    def list_all_batches(self) -> list[BatchObject]:
        """List all batches sorted by creation time (newest first)"""
        batches = []
        for batch_file in self.batches_path.glob("*.json"):
            batch = self.load_batch(batch_file.stem)
            if batch:
                batches.append(batch)
        return sorted(batches, key=lambda x: x.created_at, reverse=True)
    
    def get_active_batch_for_file(self, input_file_id: str) -> Optional[BatchObject]:
        """
        Check if a file already has an active (non-completed) batch.
        
        Active statuses are: validating, in_progress, finalizing, cancelling
        Inactive statuses are: completed, failed, expired, cancelled
        """
        active_statuses = [
            BatchStatus.VALIDATING.value,
            BatchStatus.IN_PROGRESS.value,
            BatchStatus.FINALIZING.value,
            BatchStatus.CANCELLING.value,
        ]
        
        for batch_file in self.batches_path.glob("*.json"):
            batch = self.load_batch(batch_file.stem)
            if batch and batch.input_file_id == input_file_id and batch.status in active_statuses:
                return batch
        
        return None
    
    def has_pending_batch(self) -> bool:
        """
        Check if there are any batches waiting to be processed.
        
        A batch is pending if it has status 'validating'.
        
        Returns:
            True if there are pending batches, False otherwise
        """
        for batch_file in self.batches_path.glob("*.json"):
            batch = self.load_batch(batch_file.stem)
            if batch and batch.status == BatchStatus.VALIDATING.value:
                return True
        return False
    
    def get_next_pending_batch(self) -> tuple[Optional[BatchObject], Optional[Path]]:
        """
        Get the next pending batch to process.
        
        Returns the oldest batch with status 'validating' and its input file path.
        
        Returns:
            tuple: (batch_object, input_file_path) or (None, None) if no pending batches
        """
        pending_batches = []
        
        for batch_file in self.batches_path.glob("*.json"):
            batch = self.load_batch(batch_file.stem)
            if batch and batch.status == BatchStatus.VALIDATING.value:
                pending_batches.append(batch)
        
        if not pending_batches:
            return None, None
        
        # Sort by created_at to get the oldest first (FIFO)
        pending_batches.sort(key=lambda x: x.created_at)
        next_batch = pending_batches[0]
        
        # Get the input file path
        input_file_path = self.storage_path / next_batch.input_file_id
        
        if not input_file_path.exists():
            logger.error(f"Input file not found for batch {next_batch.id}: {input_file_path}")
            return None, None
        
        logger.info(f"Found pending batch {next_batch.id} created at {next_batch.created_at}")
        return next_batch, input_file_path
    
    def save_output_file_metadata(self, file_id: str, batch_id: str, bytes: int) -> None:
        """Save output file metadata to disk"""
        created_at = int(datetime.now().timestamp())
        metadata = {
            "id": file_id,
            "object": "file",
            "bytes": bytes,
            "created_at": created_at,
            "filename": f"{batch_id}_output.jsonl",
            "purpose": "batch_output",
            "status": "processed",
            "status_details": None,
        }
        
        metadata_file = self.metadata_path / f"{file_id}.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)
        
        return metadata
    
    def write_batch_output(self, batch_id: str, input_file_path: Path, queries: list[str]) -> str:
        """
        Write batch output file in OpenAI format using ChatCompletion schema.
        
        Returns:
            file_id: The ID of the created output file
        """
        # Generate file ID using the same pattern (extract UUID from batch_id)
        # batch_id format: "batch_<uuid>", we want "file-<uuid>"
        batch_uuid = batch_id.replace("batch_", "")
        file_id = f"file-{batch_uuid}"
        
        # Create output file path
        output_file_path = self.outputs_path / file_id
        
        # Read input file to get custom_ids and requests
        input_requests = []
        with open(input_file_path, 'r') as f:
            for line in f:
                if line.strip():
                    input_requests.append(json.loads(line.strip()))
        
        # Write output file in OpenAI batch output format
        with open(output_file_path, 'w') as f:
            for i, (input_request, query) in enumerate(zip(input_requests, queries)):
                # Get model from request body
                model = input_request["body"].get("model", "gpt-3.5-turbo")
                created_timestamp = int(datetime.now().timestamp())
                
                # Create ChatCompletion object using Pydantic schema
                chat_completion = ChatCompletion(
                    id=f"chatcmpl-{batch_uuid}-{i}",
                    object="chat.completion",
                    created=created_timestamp,
                    model=model,
                    choices=[
                        ChatCompletionChoice(
                            index=0,
                            message=ChatCompletionMessage(
                                role=ChatCompletionRole.ASSISTANT,
                                content=f"Mock response for request {i+1}",
                                refusal=None
                            ),
                            logprobs=None,
                            finish_reason=FinishReason.STOP
                        )
                    ],
                    usage=CompletionUsage(
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30
                    ),
                    system_fingerprint=None,
                    service_tier=None
                )
                
                # Create batch response wrapper
                output_line = {
                    "id": f"batch_req_{batch_uuid}_{i}",
                    "custom_id": input_request.get("custom_id", f"request-{i+1}"),
                    "response": {
                        "status_code": 200,
                        "request_id": f"req_{batch_uuid}_{i}",
                        "body": chat_completion.model_dump()
                    },
                    "error": None
                }
                f.write(json.dumps(output_line) + "\n")
        
        # Get file size
        file_size = output_file_path.stat().st_size
        
        # Save metadata
        self.save_output_file_metadata(file_id, batch_id, file_size)
        
        logger.info(f"Output file {file_id} created with {len(queries)} responses ({file_size} bytes)")
        
        return file_id

        