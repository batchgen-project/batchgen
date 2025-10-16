"""
Unit tests for StorageManager

Tests all functionality of the StorageManager class including:
- File metadata management (save, load, list, find, delete)
- Batch management (save, load, list, get active batches)
- Output file generation
- Pending batch processing
"""
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from batchgen.managers.storage import StorageManager
from batchgen.managers.batch_schema import (
    BatchObject,
    BatchStatus,
    BatchEndpoint,
    CompletionWindow,
    RequestCounts,
)
from batchgen.server_args import ServerArgs


class TestStorageManager(unittest.TestCase):
    """Test suite for StorageManager class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock ServerArgs with temporary storage path
        self.server_args = MagicMock(spec=ServerArgs)
        self.server_args.file_path = self.temp_dir
        
        # Initialize StorageManager
        self.storage = StorageManager(self.server_args)
        
        # Verify directories were created
        self.assertTrue(self.storage.storage_path.exists())
        self.assertTrue(self.storage.metadata_path.exists())
        self.assertTrue(self.storage.batches_path.exists())
        self.assertTrue(self.storage.outputs_path.exists())

    def tearDown(self):
        """Clean up test fixtures"""
        # Remove temporary directory and all contents
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # =========================================================================
    # File Metadata Management Tests
    # =========================================================================

    def test_save_and_load_metadata(self):
        """Test saving and loading file metadata"""
        file_id = "file-test123"
        metadata = {
            "id": file_id,
            "object": "file",
            "bytes": 1024,
            "created_at": int(datetime.now().timestamp()),
            "filename": "test.jsonl",
            "purpose": "batch",
            "status": "uploaded",
            "checksum": "abc123def456"
        }
        
        # Save metadata
        self.storage.save_metadata(file_id, metadata)
        
        # Verify file was created
        metadata_file = self.storage.metadata_path / f"{file_id}.json"
        self.assertTrue(metadata_file.exists())
        
        # Load metadata
        loaded_metadata = self.storage.load_metadata(file_id)
        
        # Verify loaded metadata matches
        self.assertEqual(loaded_metadata, metadata)

    def test_load_nonexistent_metadata(self):
        """Test loading metadata for non-existent file"""
        result = self.storage.load_metadata("file-nonexistent")
        self.assertIsNone(result)

    def test_list_all_metadata(self):
        """Test listing all file metadata"""
        # Create multiple metadata files
        metadata_list = []
        for i in range(3):
            file_id = f"file-test{i}"
            metadata = {
                "id": file_id,
                "object": "file",
                "bytes": 1024 * (i + 1),
                "created_at": int(datetime.now().timestamp()),
                "filename": f"test{i}.jsonl",
                "purpose": "batch"
            }
            self.storage.save_metadata(file_id, metadata)
            metadata_list.append(metadata)
        
        # List all metadata
        all_metadata = self.storage.list_all_metadata()
        
        # Verify count
        self.assertEqual(len(all_metadata), 3)
        
        # Verify all metadata is present (order may vary)
        for metadata in metadata_list:
            self.assertIn(metadata, all_metadata)

    def test_find_file_by_checksum(self):
        """Test finding a file by its checksum"""
        # Create files with different checksums
        file1_id = "file-test1"
        file1_checksum = "checksum123"
        metadata1 = {
            "id": file1_id,
            "checksum": file1_checksum,
            "filename": "file1.jsonl"
        }
        
        file2_id = "file-test2"
        file2_checksum = "checksum456"
        metadata2 = {
            "id": file2_id,
            "checksum": file2_checksum,
            "filename": "file2.jsonl"
        }
        
        self.storage.save_metadata(file1_id, metadata1)
        self.storage.save_metadata(file2_id, metadata2)
        
        # Find file by checksum
        found = self.storage.find_file_by_checksum(file1_checksum)
        
        # Verify correct file was found
        self.assertIsNotNone(found)
        self.assertEqual(found["id"], file1_id)
        self.assertEqual(found["checksum"], file1_checksum)
        
        # Test non-existent checksum
        not_found = self.storage.find_file_by_checksum("nonexistent")
        self.assertIsNone(not_found)

    def test_delete_file_metadata(self):
        """Test deleting file metadata"""
        file_id = "file-delete-test"
        metadata = {
            "id": file_id,
            "filename": "delete_test.jsonl"
        }
        
        # Save metadata
        self.storage.save_metadata(file_id, metadata)
        
        # Verify it exists
        self.assertIsNotNone(self.storage.load_metadata(file_id))
        
        # Delete metadata
        result = self.storage.delete_file_metadata(file_id)
        
        # Verify deletion was successful
        self.assertTrue(result)
        self.assertIsNone(self.storage.load_metadata(file_id))
        
        # Try to delete again (should return False)
        result = self.storage.delete_file_metadata(file_id)
        self.assertFalse(result)

    # =========================================================================
    # Batch Management Tests
    # =========================================================================

    def test_save_and_load_batch(self):
        """Test saving and loading batch objects"""
        batch = BatchObject(
            id="batch_test123",
            object="batch",
            endpoint=BatchEndpoint.CHAT_COMPLETIONS.value,
            input_file_id="file-input123",
            completion_window=CompletionWindow.HOURS_24.value,
            status=BatchStatus.VALIDATING,
            created_at=int(datetime.now().timestamp())
        )
        
        # Save batch
        self.storage.save_batch(batch)
        
        # Verify file was created
        batch_file = self.storage.batches_path / f"{batch.id}.json"
        self.assertTrue(batch_file.exists())
        
        # Load batch
        loaded_batch = self.storage.load_batch(batch.id)
        
        # Verify loaded batch matches
        self.assertIsNotNone(loaded_batch)
        self.assertEqual(loaded_batch.id, batch.id)
        self.assertEqual(loaded_batch.status, batch.status)
        self.assertEqual(loaded_batch.input_file_id, batch.input_file_id)

    def test_load_nonexistent_batch(self):
        """Test loading non-existent batch"""
        result = self.storage.load_batch("batch_nonexistent")
        self.assertIsNone(result)

    def test_list_all_batches_sorted(self):
        """Test listing all batches sorted by creation time"""
        import time
        
        # Create batches with different timestamps
        batches = []
        for i in range(3):
            timestamp = int(datetime.now().timestamp()) + i
            batch = BatchObject(
                id=f"batch_test{i}",
                object="batch",
                endpoint=BatchEndpoint.CHAT_COMPLETIONS.value,
                input_file_id=f"file-input{i}",
                completion_window=CompletionWindow.HOURS_24.value,
                status=BatchStatus.VALIDATING,
                created_at=timestamp
            )
            self.storage.save_batch(batch)
            batches.append(batch)
            time.sleep(0.01)  # Small delay to ensure different timestamps
        
        # List all batches
        all_batches = self.storage.list_all_batches()
        
        # Verify count
        self.assertEqual(len(all_batches), 3)
        
        # Verify sorting (newest first)
        for i in range(len(all_batches) - 1):
            self.assertGreaterEqual(
                all_batches[i].created_at,
                all_batches[i + 1].created_at
            )

    def test_get_active_batch_for_file(self):
        """Test finding active batch for a file"""
        input_file_id = "file-input123"
        
        # Create an active batch
        active_batch = BatchObject(
            id="batch_active",
            object="batch",
            endpoint=BatchEndpoint.CHAT_COMPLETIONS.value,
            input_file_id=input_file_id,
            completion_window=CompletionWindow.HOURS_24.value,
            status=BatchStatus.IN_PROGRESS,
            created_at=int(datetime.now().timestamp())
        )
        self.storage.save_batch(active_batch)
        
        # Create a completed batch for the same file
        completed_batch = BatchObject(
            id="batch_completed",
            object="batch",
            endpoint=BatchEndpoint.CHAT_COMPLETIONS.value,
            input_file_id=input_file_id,
            completion_window=CompletionWindow.HOURS_24.value,
            status=BatchStatus.COMPLETED,
            created_at=int(datetime.now().timestamp()) - 100
        )
        self.storage.save_batch(completed_batch)
        
        # Get active batch
        result = self.storage.get_active_batch_for_file(input_file_id)
        
        # Verify we got the active batch, not the completed one
        self.assertIsNotNone(result)
        self.assertEqual(result.id, active_batch.id)
        self.assertEqual(result.status, BatchStatus.IN_PROGRESS.value)

    def test_get_active_batch_for_file_no_active(self):
        """Test finding active batch when only completed batches exist"""
        input_file_id = "file-input456"
        
        # Create only completed batches
        completed_batch = BatchObject(
            id="batch_completed",
            object="batch",
            endpoint=BatchEndpoint.CHAT_COMPLETIONS.value,
            input_file_id=input_file_id,
            completion_window=CompletionWindow.HOURS_24.value,
            status=BatchStatus.COMPLETED,
            created_at=int(datetime.now().timestamp())
        )
        self.storage.save_batch(completed_batch)
        
        # Get active batch
        result = self.storage.get_active_batch_for_file(input_file_id)
        
        # Verify no active batch found
        self.assertIsNone(result)

    def test_has_pending_batch(self):
        """Test checking for pending batches"""
        # Initially no pending batches
        self.assertFalse(self.storage.has_pending_batch())
        
        # Create a validating batch
        pending_batch = BatchObject(
            id="batch_pending",
            object="batch",
            endpoint=BatchEndpoint.CHAT_COMPLETIONS.value,
            input_file_id="file-input123",
            completion_window=CompletionWindow.HOURS_24.value,
            status=BatchStatus.VALIDATING,
            created_at=int(datetime.now().timestamp())
        )
        self.storage.save_batch(pending_batch)
        
        # Now should have pending batch
        self.assertTrue(self.storage.has_pending_batch())
        
        # Create an in-progress batch (not pending)
        active_batch = BatchObject(
            id="batch_active",
            object="batch",
            endpoint=BatchEndpoint.CHAT_COMPLETIONS.value,
            input_file_id="file-input456",
            completion_window=CompletionWindow.HOURS_24.value,
            status=BatchStatus.IN_PROGRESS,
            created_at=int(datetime.now().timestamp())
        )
        self.storage.save_batch(active_batch)
        
        # Still should have pending batch
        self.assertTrue(self.storage.has_pending_batch())

    def test_get_next_pending_batch(self):
        """Test getting the next pending batch (FIFO)"""
        # Create multiple pending batches with different timestamps
        # Use explicit timestamps to ensure proper ordering
        base_timestamp = int(datetime.now().timestamp())
        
        batch1 = BatchObject(
            id="batch_pending1",
            object="batch",
            endpoint=BatchEndpoint.CHAT_COMPLETIONS.value,
            input_file_id="file-input1",
            completion_window=CompletionWindow.HOURS_24.value,
            status=BatchStatus.VALIDATING,
            created_at=base_timestamp
        )
        self.storage.save_batch(batch1)
        
        batch2 = BatchObject(
            id="batch_pending2",
            object="batch",
            endpoint=BatchEndpoint.CHAT_COMPLETIONS.value,
            input_file_id="file-input2",
            completion_window=CompletionWindow.HOURS_24.value,
            status=BatchStatus.VALIDATING,
            created_at=base_timestamp + 1
        )
        self.storage.save_batch(batch2)
        
        # Create the input files
        input_file1 = self.storage.storage_path / batch1.input_file_id
        input_file2 = self.storage.storage_path / batch2.input_file_id
        input_file1.write_text("test content 1")
        input_file2.write_text("test content 2")
        
        # Get next pending batch (should be the oldest)
        next_batch, input_path = self.storage.get_next_pending_batch()
        
        # Verify we got the oldest batch
        self.assertIsNotNone(next_batch)
        self.assertEqual(next_batch.id, batch1.id)
        self.assertIsNotNone(input_path)
        self.assertTrue(input_path.exists())

    def test_get_next_pending_batch_no_pending(self):
        """Test getting next pending batch when none exist"""
        batch, path = self.storage.get_next_pending_batch()
        self.assertIsNone(batch)
        self.assertIsNone(path)

    def test_get_next_pending_batch_missing_input_file(self):
        """Test getting next pending batch when input file is missing"""
        batch = BatchObject(
            id="batch_pending",
            object="batch",
            endpoint=BatchEndpoint.CHAT_COMPLETIONS.value,
            input_file_id="file-nonexistent",
            completion_window=CompletionWindow.HOURS_24.value,
            status=BatchStatus.VALIDATING,
            created_at=int(datetime.now().timestamp())
        )
        self.storage.save_batch(batch)
        
        # Get next pending batch (should fail due to missing file)
        result_batch, result_path = self.storage.get_next_pending_batch()
        
        # Should return None because input file doesn't exist
        self.assertIsNone(result_batch)
        self.assertIsNone(result_path)

    # =========================================================================
    # Output File Generation Tests
    # =========================================================================

    def test_save_output_file_metadata(self):
        """Test saving output file metadata"""
        file_id = "file-output123"
        batch_id = "batch_test123"
        file_bytes = 2048
        
        # Save output file metadata
        metadata = self.storage.save_output_file_metadata(file_id, batch_id, file_bytes)
        
        # Verify metadata was created
        self.assertEqual(metadata["id"], file_id)
        self.assertEqual(metadata["bytes"], file_bytes)
        self.assertEqual(metadata["purpose"], "batch_output")
        self.assertEqual(metadata["status"], "processed")
        
        # Verify it was saved to disk
        loaded = self.storage.load_metadata(file_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["id"], file_id)

    def test_write_batch_output(self):
        """Test writing batch output file in OpenAI format"""
        batch_id = "batch_abc123"
        
        # Create input file with test requests
        input_file_path = self.storage.storage_path / "file-input123"
        input_requests = [
            {
                "custom_id": "request-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Hello"}]
                }
            },
            {
                "custom_id": "request-2",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "Hi"}]
                }
            }
        ]
        
        with open(input_file_path, 'w') as f:
            for req in input_requests:
                f.write(json.dumps(req) + "\n")
        
        # Mock queries (responses)
        queries = ["Response 1", "Response 2"]
        
        # Write batch output
        file_id = self.storage.write_batch_output(batch_id, input_file_path, queries)
        
        # Verify file_id format
        self.assertTrue(file_id.startswith("file-"))
        
        # Verify output file was created
        output_file_path = self.storage.outputs_path / file_id
        self.assertTrue(output_file_path.exists())
        
        # Verify output file content
        with open(output_file_path, 'r') as f:
            lines = f.readlines()
        
        self.assertEqual(len(lines), 2)
        
        # Parse and verify first line
        output1 = json.loads(lines[0])
        self.assertEqual(output1["custom_id"], "request-1")
        self.assertEqual(output1["response"]["status_code"], 200)
        self.assertIsNotNone(output1["response"]["body"])
        self.assertIsNone(output1["error"])
        
        # Verify ChatCompletion structure
        body1 = output1["response"]["body"]
        self.assertEqual(body1["object"], "chat.completion")
        self.assertEqual(body1["model"], "gpt-3.5-turbo")
        self.assertIn("choices", body1)
        self.assertIn("usage", body1)
        
        # Verify metadata was created
        metadata = self.storage.load_metadata(file_id)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata["purpose"], "batch_output")

    def test_write_batch_output_empty_requests(self):
        """Test writing batch output with no requests"""
        batch_id = "batch_empty"
        
        # Create empty input file
        input_file_path = self.storage.storage_path / "file-empty"
        input_file_path.write_text("")
        
        # Write batch output
        file_id = self.storage.write_batch_output(batch_id, input_file_path, [])
        
        # Verify file was created but is empty
        output_file_path = self.storage.outputs_path / file_id
        self.assertTrue(output_file_path.exists())
        
        content = output_file_path.read_text()
        self.assertEqual(content, "")

    # =========================================================================
    # Integration Tests
    # =========================================================================

    def test_batch_lifecycle(self):
        """Test complete batch lifecycle"""
        # 1. Create and save batch
        batch_id = "batch_lifecycle"
        input_file_id = "file-lifecycle-input"
        
        batch = BatchObject(
            id=batch_id,
            object="batch",
            endpoint=BatchEndpoint.CHAT_COMPLETIONS.value,
            input_file_id=input_file_id,
            completion_window=CompletionWindow.HOURS_24.value,
            status=BatchStatus.VALIDATING,
            created_at=int(datetime.now().timestamp())
        )
        self.storage.save_batch(batch)
        
        # 2. Verify it's pending
        self.assertTrue(self.storage.has_pending_batch())
        
        # 3. Get pending batch
        input_file_path = self.storage.storage_path / input_file_id
        input_file_path.write_text('{"custom_id": "test"}\n')
        
        next_batch, path = self.storage.get_next_pending_batch()
        self.assertEqual(next_batch.id, batch_id)
        
        # 4. Update to in_progress
        batch.status = BatchStatus.IN_PROGRESS
        batch.in_progress_at = int(datetime.now().timestamp())
        self.storage.save_batch(batch)
        
        # 5. Complete batch
        batch.status = BatchStatus.COMPLETED
        batch.completed_at = int(datetime.now().timestamp())
        self.storage.save_batch(batch)
        
        # 6. Verify no longer active
        active = self.storage.get_active_batch_for_file(input_file_id)
        self.assertIsNone(active)
        
        # 7. Load and verify final state
        final_batch = self.storage.load_batch(batch_id)
        self.assertEqual(final_batch.status, BatchStatus.COMPLETED.value)
        self.assertIsNotNone(final_batch.completed_at)


class TestStorageManagerEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.server_args = MagicMock(spec=ServerArgs)
        self.server_args.file_path = self.temp_dir
        self.storage = StorageManager(self.server_args)

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_empty_storage(self):
        """Test operations on empty storage"""
        # List operations should return empty lists
        self.assertEqual(len(self.storage.list_all_metadata()), 0)
        self.assertEqual(len(self.storage.list_all_batches()), 0)
        
        # Find operations should return None
        self.assertIsNone(self.storage.find_file_by_checksum("any"))
        self.assertIsNone(self.storage.load_metadata("file-any"))
        self.assertIsNone(self.storage.load_batch("batch_any"))
        
        # Status checks should return False/None
        self.assertFalse(self.storage.has_pending_batch())
        batch, path = self.storage.get_next_pending_batch()
        self.assertIsNone(batch)
        self.assertIsNone(path)

    def test_multiple_active_statuses(self):
        """Test handling multiple active statuses"""
        input_file_id = "file-multi"
        
        # Create batches with different active statuses
        statuses = [
            BatchStatus.VALIDATING,
            BatchStatus.IN_PROGRESS,
            BatchStatus.FINALIZING,
            BatchStatus.CANCELLING,
        ]
        
        for i, status in enumerate(statuses):
            batch = BatchObject(
                id=f"batch_{status.value}",
                object="batch",
                endpoint=BatchEndpoint.CHAT_COMPLETIONS.value,
                input_file_id=input_file_id,
                completion_window=CompletionWindow.HOURS_24.value,
                status=status,
                created_at=int(datetime.now().timestamp()) + i
            )
            self.storage.save_batch(batch)
        
        # Should find an active batch (any one of them)
        active = self.storage.get_active_batch_for_file(input_file_id)
        self.assertIsNotNone(active)
        self.assertIn(active.status, [s.value for s in statuses])


if __name__ == "__main__":
    unittest.main()
