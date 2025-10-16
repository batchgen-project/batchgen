"""
Unit tests for ServerScheduler

Tests all functionality of the ServerScheduler class including:
- Initialization (single and multi-node)
- Batch processing workflow
- Storage integration
- Query creation from batch files
- Broadcast functionality (for multi-node setup)
"""
import json
import tempfile
import unittest
import time
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from batchgen.managers.scheduler import ServerScheduler
from batchgen.managers.storage import StorageManager
from batchgen.managers.batch_schema import (
    BatchObject,
    BatchStatus,
    BatchEndpoint,
    CompletionWindow,
    RequestCounts,
)
from batchgen.server_args import ServerArgs


class TestServerSchedulerInitialization(unittest.TestCase):
    """Test scheduler initialization"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.server_args = MagicMock(spec=ServerArgs)
        self.server_args.file_path = self.temp_dir
        self.server_args.nnodes = 1
        self.server_args.node_rank = 0

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_single_node_initialization(self):
        """Test scheduler initialization for single node"""
        self.server_args.nnodes = 1
        scheduler = ServerScheduler(self.server_args)
        
        # Verify scheduler was initialized
        self.assertIsNotNone(scheduler)
        self.assertIsNotNone(scheduler.storage)
        self.assertIsInstance(scheduler.storage, StorageManager)
        
        # Single node should not have broadcaster
        self.assertIsNone(scheduler.broadcaster)
        
        # Clean up
        scheduler.close()

    def test_multi_node_initialization(self):
        """Test scheduler initialization for multi-node setup"""
        # Mock broadcaster
        mock_broadcaster = MagicMock()
        
        self.server_args.nnodes = 2
        self.server_args.node_rank = 0
        self.server_args.create_broadcaster = MagicMock(return_value=mock_broadcaster)
        
        scheduler = ServerScheduler(self.server_args)
        
        # Verify broadcaster was created for multi-node
        self.assertIsNotNone(scheduler.broadcaster)
        self.server_args.create_broadcaster.assert_called_once()
        
        # Clean up
        scheduler.close()

    def test_close_cleanup(self):
        """Test scheduler cleanup on close"""
        scheduler = ServerScheduler(self.server_args)
        
        # Mock broadcaster
        mock_broadcaster = MagicMock()
        scheduler.broadcaster = mock_broadcaster
        
        # Close scheduler
        scheduler.close()
        
        # Verify broadcaster was closed
        mock_broadcaster.close.assert_called_once()
        self.assertIsNone(scheduler.broadcaster)


class TestServerSchedulerBroadcast(unittest.TestCase):
    """Test broadcast functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.server_args = MagicMock(spec=ServerArgs)
        self.server_args.file_path = self.temp_dir
        self.server_args.nnodes = 1
        self.server_args.node_rank = 0

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_broadcast_single_node(self):
        """Test broadcast in single node setup (no-op)"""
        scheduler = ServerScheduler(self.server_args)
        
        test_obj = {"key": "value"}
        result = scheduler.broadcast(test_obj)
        
        # In single node, broadcast should return the same object
        self.assertEqual(result, test_obj)
        
        scheduler.close()

    def test_broadcast_multi_node_rank_zero(self):
        """Test broadcast from rank 0 in multi-node setup"""
        self.server_args.nnodes = 2
        self.server_args.node_rank = 0
        
        # Mock broadcaster
        mock_broadcaster = MagicMock()
        mock_broadcaster.broadcast.return_value = {"broadcasted": "data"}
        mock_broadcaster.__len__.return_value = 2
        
        scheduler = ServerScheduler(self.server_args)
        scheduler.broadcaster = mock_broadcaster
        
        test_obj = {"key": "value"}
        result = scheduler.broadcast(test_obj)
        
        # Rank 0 should call broadcaster.broadcast with the object
        mock_broadcaster.broadcast.assert_called_once_with(test_obj)
        self.assertEqual(result, {"broadcasted": "data"})
        
        scheduler.close()

    def test_broadcast_multi_node_rank_nonzero(self):
        """Test broadcast receive on non-zero rank"""
        self.server_args.nnodes = 2
        self.server_args.node_rank = 1
        
        # Mock broadcaster
        mock_broadcaster = MagicMock()
        mock_broadcaster.broadcast.return_value = {"received": "data"}
        
        scheduler = ServerScheduler(self.server_args)
        scheduler.broadcaster = mock_broadcaster
        
        result = scheduler.broadcast(None)
        
        # Non-zero rank should call broadcast with None to receive
        mock_broadcaster.broadcast.assert_called_once_with(None)
        self.assertEqual(result, {"received": "data"})
        
        scheduler.close()


class TestServerSchedulerQueryCreation(unittest.TestCase):
    """Test query creation from batch files"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('batchgen.managers.scheduler.AutoTokenizer')
    def test_create_batch_queries(self, mock_tokenizer_class):
        """Test creating batch queries from file"""
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.side_effect = [
            "Formatted query 1",
            "Formatted query 2",
            "Formatted query 3"
        ]
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Create test batch file using real format
        batch_file = Path(self.temp_dir) / "test_batch.jsonl"
        requests = []
        for i in range(3):
            request = {
                "custom_id": f"request-{i+1}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"This is test request number {i+1}"}
                    ],
                    "max_tokens": 50
                }
            }
            requests.append(request)
        
        with open(batch_file, 'w') as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")
        
        # Create queries
        queries = ServerScheduler.create_batch_queries(str(batch_file))
        
        # Verify queries were created
        self.assertEqual(len(queries), 3)
        self.assertEqual(queries[0], "Formatted query 1")
        self.assertEqual(queries[1], "Formatted query 2")
        self.assertEqual(queries[2], "Formatted query 3")
        
        # Verify tokenizer was loaded and called
        mock_tokenizer_class.from_pretrained.assert_called_once_with(
            "gpt-3.5-turbo",
            trust_remote_code=True
        )
        self.assertEqual(mock_tokenizer.apply_chat_template.call_count, 3)

    @patch('batchgen.managers.scheduler.AutoTokenizer')
    def test_create_batch_queries_empty_file(self, mock_tokenizer_class):
        """Test creating queries from empty file"""
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Create empty batch file
        batch_file = Path(self.temp_dir) / "empty_batch.jsonl"
        batch_file.write_text("")
        
        # This should raise an error or return empty list
        # Depending on implementation, adjust assertion
        try:
            queries = ServerScheduler.create_batch_queries(str(batch_file))
            # If it doesn't raise, should be empty
            self.assertEqual(len(queries), 0)
        except (json.JSONDecodeError, ValueError, IndexError):
            # Expected if file is empty
            pass


class TestServerSchedulerBatchProcessing(unittest.TestCase):
    """Test batch processing functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.server_args = MagicMock(spec=ServerArgs)
        self.server_args.file_path = self.temp_dir
        self.server_args.nnodes = 1
        self.server_args.node_rank = 0

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('batchgen.managers.scheduler.time.sleep')
    def test_run_batch_processing(self, mock_sleep):
        """Test the run method for batch processing"""
        scheduler = ServerScheduler(self.server_args)
        
        # Create a test batch
        batch = BatchObject(
            id="batch_test123",
            object="batch",
            endpoint=BatchEndpoint.CHAT_COMPLETIONS.value,
            input_file_id="file-input123",
            completion_window=CompletionWindow.HOURS_24.value,
            status=BatchStatus.VALIDATING,
            created_at=int(datetime.now().timestamp())
        )
        
        # Create input file using real format
        input_file_path = Path(self.temp_dir) / batch.input_file_id
        requests = [
            {
                "custom_id": "request-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Test message"}
                    ],
                    "max_tokens": 50
                }
            }
        ]
        with open(input_file_path, 'w') as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")
        
        # Mock queries
        queries = ["Query 1", "Query 2", "Query 3"]
        
        # Run batch processing
        result_batch = scheduler.run(batch, input_file_path, queries)
        
        # Verify batch was processed
        self.assertEqual(result_batch.status, BatchStatus.COMPLETED.value)
        self.assertIsNotNone(result_batch.in_progress_at)
        self.assertIsNotNone(result_batch.finalizing_at)
        self.assertIsNotNone(result_batch.completed_at)
        self.assertEqual(result_batch.request_counts.total, 3)
        self.assertEqual(result_batch.request_counts.completed, 3)
        self.assertIsNotNone(result_batch.output_file_id)
        
        # Verify sleep was called (mocked processing)
        mock_sleep.assert_called()
        
        scheduler.close()

    @patch('batchgen.managers.scheduler.time.sleep')
    def test_run_updates_batch_status_progression(self, mock_sleep):
        """Test that batch status progresses correctly during processing"""
        scheduler = ServerScheduler(self.server_args)
        
        # Create a test batch
        batch = BatchObject(
            id="batch_status_test",
            object="batch",
            endpoint=BatchEndpoint.CHAT_COMPLETIONS.value,
            input_file_id="file-status-test",
            completion_window=CompletionWindow.HOURS_24.value,
            status=BatchStatus.VALIDATING,
            created_at=int(datetime.now().timestamp())
        )
        
        # Create input file using real format
        input_file_path = Path(self.temp_dir) / batch.input_file_id
        requests = [
            {
                "custom_id": "request-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Test message"}
                    ],
                    "max_tokens": 50
                }
            }
        ]
        with open(input_file_path, 'w') as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")
        
        queries = ["Query 1"]
        
        # Track status changes by mocking save_batch
        status_history = []
        original_save = scheduler.storage.save_batch
        
        def track_status(batch_obj):
            status_history.append(batch_obj.status)
            return original_save(batch_obj)
        
        scheduler.storage.save_batch = track_status
        
        # Run batch processing
        result_batch = scheduler.run(batch, input_file_path, queries)
        
        # Verify status progression: validating -> in_progress -> finalizing -> completed
        self.assertIn(BatchStatus.IN_PROGRESS.value, status_history)
        self.assertIn(BatchStatus.FINALIZING.value, status_history)
        self.assertIn(BatchStatus.COMPLETED.value, status_history)
        
        # Verify final status
        self.assertEqual(result_batch.status, BatchStatus.COMPLETED.value)
        
        scheduler.close()

    @patch('batchgen.managers.scheduler.time.sleep')
    def test_run_creates_output_file(self, mock_sleep):
        """Test that run creates an output file"""
        scheduler = ServerScheduler(self.server_args)
        
        # Create a test batch
        batch = BatchObject(
            id="batch_output_test",
            object="batch",
            endpoint=BatchEndpoint.CHAT_COMPLETIONS.value,
            input_file_id="file-output-test",
            completion_window=CompletionWindow.HOURS_24.value,
            status=BatchStatus.VALIDATING,
            created_at=int(datetime.now().timestamp())
        )
        
        # Create input file using real format
        input_file_path = Path(self.temp_dir) / batch.input_file_id
        requests = [
            {
                "custom_id": "request-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Test message"}
                    ],
                    "max_tokens": 50
                }
            }
        ]
        with open(input_file_path, 'w') as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")
        
        queries = ["Query 1"]
        
        # Run batch processing
        result_batch = scheduler.run(batch, input_file_path, queries)
        
        # Verify output file was created
        self.assertIsNotNone(result_batch.output_file_id)
        self.assertTrue(result_batch.output_file_id.startswith("file-"))
        
        # Verify output file exists
        output_path = scheduler.storage.outputs_path / result_batch.output_file_id
        self.assertTrue(output_path.exists())
        
        # Verify output file has content
        with open(output_path, 'r') as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)
        
        scheduler.close()


class TestServerSchedulerMainLoop(unittest.TestCase):
    """Test scheduler main loop"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.server_args = MagicMock(spec=ServerArgs)
        self.server_args.file_path = self.temp_dir
        self.server_args.nnodes = 1
        self.server_args.node_rank = 0

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('batchgen.managers.scheduler.time.sleep')
    @patch('batchgen.managers.scheduler.ServerScheduler.create_batch_queries')
    def test_main_loop_processes_pending_batch(self, mock_create_queries, mock_sleep):
        """Test that main loop processes pending batches"""
        scheduler = ServerScheduler(self.server_args)
        
        # Mock queries
        mock_create_queries.return_value = ["Query 1", "Query 2"]
        
        # Create a pending batch
        batch = BatchObject(
            id="batch_loop_test",
            object="batch",
            endpoint=BatchEndpoint.CHAT_COMPLETIONS.value,
            input_file_id="file-loop-test",
            completion_window=CompletionWindow.HOURS_24.value,
            status=BatchStatus.VALIDATING,
            created_at=int(datetime.now().timestamp())
        )
        scheduler.storage.save_batch(batch)
        
        # Create input file
        input_file_path = Path(self.temp_dir) / batch.input_file_id
        input_file_path.write_text(json.dumps({
            "custom_id": "request-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "test",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Test message"}
                ],
                "max_tokens": 50
            }
        }) + "\n")
        
        # Create stop event that triggers after first iteration
        stop_event = mp.Event()
        
        # Mock sleep to set stop event after first call
        def sleep_and_stop(duration):
            stop_event.set()
        
        mock_sleep.side_effect = sleep_and_stop
        
        # Run main loop (should process one batch and exit)
        scheduler(stop_event)
        
        # Verify batch was processed
        processed_batch = scheduler.storage.load_batch(batch.id)
        self.assertEqual(processed_batch.status, BatchStatus.COMPLETED.value)
        
        scheduler.close()

    @patch('batchgen.managers.scheduler.time.sleep')
    def test_main_loop_waits_when_no_pending_batches(self, mock_sleep):
        """Test that main loop waits when no batches are pending"""
        scheduler = ServerScheduler(self.server_args)
        
        # Create stop event
        stop_event = mp.Event()
        
        # Mock sleep to set stop event after a few iterations
        call_count = [0]
        
        def sleep_and_stop(duration):
            call_count[0] += 1
            if call_count[0] >= 3:
                stop_event.set()
        
        mock_sleep.side_effect = sleep_and_stop
        
        # Run main loop with no pending batches
        scheduler(stop_event)
        
        # Verify sleep was called multiple times
        self.assertGreaterEqual(call_count[0], 3)
        
        scheduler.close()


class TestServerSchedulerIntegration(unittest.TestCase):
    """Integration tests for scheduler"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.server_args = MagicMock(spec=ServerArgs)
        self.server_args.file_path = self.temp_dir
        self.server_args.nnodes = 1
        self.server_args.node_rank = 0

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('batchgen.managers.scheduler.AutoTokenizer')
    @patch('batchgen.managers.scheduler.time.sleep')
    def test_full_batch_lifecycle(self, mock_sleep, mock_tokenizer_class):
        """Test complete batch processing lifecycle"""
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "Formatted query"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        scheduler = ServerScheduler(self.server_args)
        
        # Create a batch
        batch = BatchObject(
            id="batch_integration",
            object="batch",
            endpoint=BatchEndpoint.CHAT_COMPLETIONS.value,
            input_file_id="file-integration",
            completion_window=CompletionWindow.HOURS_24.value,
            status=BatchStatus.VALIDATING,
            created_at=int(datetime.now().timestamp())
        )
        scheduler.storage.save_batch(batch)
        
        # Create input file using real format
        input_file_path = Path(self.temp_dir) / batch.input_file_id
        request = {
            "custom_id": "request-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Test message"}
                ],
                "max_tokens": 50
            }
        }
        with open(input_file_path, 'w') as f:
            f.write(json.dumps(request) + "\n")
        
        # Get pending batch
        pending_batch, path = scheduler.storage.get_next_pending_batch()
        self.assertIsNotNone(pending_batch)
        
        # Create queries
        queries = ServerScheduler.create_batch_queries(str(path))
        self.assertGreater(len(queries), 0)
        
        # Process batch
        result = scheduler.run(pending_batch, path, queries)
        
        # Verify complete lifecycle
        self.assertEqual(result.status, BatchStatus.COMPLETED.value)
        self.assertIsNotNone(result.output_file_id)
        
        # Verify output file exists and has content
        output_path = scheduler.storage.outputs_path / result.output_file_id
        self.assertTrue(output_path.exists())
        
        scheduler.close()


if __name__ == "__main__":
    unittest.main()
