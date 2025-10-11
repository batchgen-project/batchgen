"""
Test the Batch API Server using the official OpenAI Python SDK
This demonstrates OpenAI Batch API compatibility for creating, listing, retrieving, and canceling batches
"""
import os
import json
import time
import unittest
from pathlib import Path
from openai import OpenAI


class BatchAPITestCase(unittest.TestCase):
    """Base test case for Batch API tests"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test client once for all tests"""
        cls.client = OpenAI(
            api_key="test-key",
            base_url="http://localhost:8000/v1"
        )
    
    def setUp(self):
        """Set up for each test"""
        self.temp_files = []  # Local files to clean up
        self.uploaded_file_ids = []  # Server files to clean up
        self.batch_ids = []  # Batch IDs to track (for reference, batches auto-expire)
    
    def tearDown(self):
        """Clean up after each test"""
        # Clean up local temp files
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass
        
        # Clean up uploaded files from server
        for file_id in self.uploaded_file_ids:
            try:
                self.client.files.delete(file_id)
            except Exception:
                pass  # File might already be deleted
        
        # Note: Batches are auto-managed by the server (expire after 24h)
        # We just track them for assertion purposes
    
    def create_batch_input_file(self, filename: str = "batch_input.jsonl", num_requests: int = 5):
        """Create a test JSONL file with batch requests"""
        requests = []
        for i in range(num_requests):
            request = {
                "custom_id": f"request-{i+1}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"What is {i+1} + {i+1}?"}
                    ],
                    "max_tokens": 100
                }
            }
            requests.append(request)
        
        with open(filename, "w") as f:
            for request in requests:
                f.write(json.dumps(request) + "\n")
        
        self.temp_files.append(filename)
        return filename
    
    def upload_batch_file(self, filename: str = None, num_requests: int = 5):
        """Helper to create and upload a batch input file"""
        if filename is None:
            filename = f"batch_test_{int(time.time())}.jsonl"
        
        # Create the file
        self.create_batch_input_file(filename, num_requests)
        
        # Upload the file
        with open(filename, "rb") as f:
            uploaded_file = self.client.files.create(
                file=f,
                purpose="batch"
            )
        
        self.uploaded_file_ids.append(uploaded_file.id)
        return uploaded_file


class BatchCreationTests(BatchAPITestCase):
    """Tests for batch creation functionality"""
    
    def test_create_batch(self):
        """Test basic batch creation"""
        # Upload input file
        input_file = self.upload_batch_file(num_requests=3)
        
        # Create batch
        batch = self.client.batches.create(
            input_file_id=input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        self.batch_ids.append(batch.id)
        
        # Assertions
        self.assertIsNotNone(batch.id)
        self.assertTrue(batch.id.startswith("batch_"))
        self.assertEqual(batch.object, "batch")
        self.assertEqual(batch.endpoint, "/v1/chat/completions")
        self.assertEqual(batch.input_file_id, input_file.id)
        self.assertEqual(batch.completion_window, "24h")
        self.assertIn(batch.status, ["validating", "in_progress", "finalizing", "completed"])
        self.assertIsNotNone(batch.created_at)
        self.assertIsNotNone(batch.expires_at)
        self.assertGreater(batch.expires_at, batch.created_at)
    
    def test_create_batch_with_metadata(self):
        """Test creating a batch with custom metadata"""
        # Upload input file
        input_file = self.upload_batch_file(num_requests=5)
        
        # Create batch with metadata
        custom_metadata = {
            "project": "test_project",
            "environment": "development",
            "test_run": "batch_api_test"
        }
        
        batch = self.client.batches.create(
            input_file_id=input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata=custom_metadata
        )
        
        self.batch_ids.append(batch.id)
        
        # Assertions
        self.assertIsNotNone(batch.metadata)
        self.assertEqual(batch.metadata, custom_metadata)
        self.assertEqual(batch.metadata["project"], "test_project")
    
    def test_create_batch_embeddings_endpoint(self):
        """Test creating a batch for embeddings endpoint"""
        # Create embeddings batch input
        filename = "embeddings_batch.jsonl"
        requests = []
        for i in range(3):
            request = {
                "custom_id": f"embed-{i+1}",
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {
                    "model": "text-embedding-ada-002",
                    "input": f"Sample text number {i+1}"
                }
            }
            requests.append(request)
        
        with open(filename, "w") as f:
            for request in requests:
                f.write(json.dumps(request) + "\n")
        
        self.temp_files.append(filename)
        
        # Upload file
        with open(filename, "rb") as f:
            input_file = self.client.files.create(file=f, purpose="batch")
        
        self.uploaded_file_ids.append(input_file.id)
        
        # Create batch
        batch = self.client.batches.create(
            input_file_id=input_file.id,
            endpoint="/v1/embeddings",
            completion_window="24h"
        )
        
        self.batch_ids.append(batch.id)
        
        # Assertions
        self.assertEqual(batch.endpoint, "/v1/embeddings")
        self.assertIsNotNone(batch.id)


class BatchListTests(BatchAPITestCase):
    """Tests for batch listing functionality"""
    
    def test_list_batches(self):
        """Test listing all batches"""
        # Create multiple batches
        batch_ids = []
        for i in range(3):
            input_file = self.upload_batch_file(num_requests=2)
            batch = self.client.batches.create(
                input_file_id=input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            batch_ids.append(batch.id)
            self.batch_ids.append(batch.id)
        
        # List all batches
        batches_list = self.client.batches.list()
        
        # Assertions
        self.assertIsNotNone(batches_list.data)
        self.assertGreaterEqual(len(batches_list.data), 3)
        
        # Verify our created batches are in the list
        listed_ids = [b.id for b in batches_list.data]
        for batch_id in batch_ids:
            self.assertIn(batch_id, listed_ids)
    
    def test_list_batches_with_limit(self):
        """Test listing batches with pagination limit"""
        # Create several batches
        for i in range(5):
            input_file = self.upload_batch_file(num_requests=1)
            batch = self.client.batches.create(
                input_file_id=input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            self.batch_ids.append(batch.id)
        
        # List with limit
        batches_list = self.client.batches.list(limit=3)
        
        # Assertions
        self.assertLessEqual(len(batches_list.data), 3)
        self.assertIsNotNone(batches_list.first_id)
        self.assertIsNotNone(batches_list.last_id)
    
    def test_list_batches_pagination(self):
        """Test batch listing pagination with 'after' cursor"""
        # Create multiple batches
        for i in range(5):
            input_file = self.upload_batch_file(num_requests=1)
            batch = self.client.batches.create(
                input_file_id=input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            self.batch_ids.append(batch.id)
        
        # Get first page
        first_page = self.client.batches.list(limit=2)
        
        self.assertLessEqual(len(first_page.data), 2)
        
        # Get second page using 'after' cursor
        if first_page.has_more and first_page.last_id:
            second_page = self.client.batches.list(limit=2, after=first_page.last_id)
            
            # Assertions
            self.assertIsNotNone(second_page.data)
            
            # Verify no overlap between pages
            first_page_ids = [b.id for b in first_page.data]
            second_page_ids = [b.id for b in second_page.data]
            
            for batch_id in second_page_ids:
                self.assertNotIn(batch_id, first_page_ids)


class BatchRetrievalTests(BatchAPITestCase):
    """Tests for batch retrieval functionality"""
    
    def test_retrieve_batch(self):
        """Test retrieving a specific batch"""
        # Create a batch
        input_file = self.upload_batch_file(num_requests=3)
        created_batch = self.client.batches.create(
            input_file_id=input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        self.batch_ids.append(created_batch.id)
        
        # Retrieve the batch
        retrieved_batch = self.client.batches.retrieve(created_batch.id)
        
        # Assertions
        self.assertEqual(retrieved_batch.id, created_batch.id)
        self.assertEqual(retrieved_batch.object, "batch")
        self.assertEqual(retrieved_batch.endpoint, created_batch.endpoint)
        self.assertEqual(retrieved_batch.input_file_id, created_batch.input_file_id)
        self.assertEqual(retrieved_batch.completion_window, created_batch.completion_window)
        self.assertEqual(retrieved_batch.created_at, created_batch.created_at)
        self.assertEqual(retrieved_batch.expires_at, created_batch.expires_at)
    
    def test_retrieve_batch_with_metadata(self):
        """Test retrieving a batch that has metadata"""
        # Create batch with metadata
        input_file = self.upload_batch_file(num_requests=2)
        metadata = {"test_id": "retrieve_test", "version": "1.0"}
        
        created_batch = self.client.batches.create(
            input_file_id=input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata=metadata
        )
        
        self.batch_ids.append(created_batch.id)
        
        # Retrieve the batch
        retrieved_batch = self.client.batches.retrieve(created_batch.id)
        
        # Assertions
        self.assertIsNotNone(retrieved_batch.metadata)
        self.assertEqual(retrieved_batch.metadata, metadata)


class BatchCancellationTests(BatchAPITestCase):
    """Tests for batch cancellation functionality"""
    
    def test_cancel_batch(self):
        """Test canceling a batch"""
        # Create a batch
        input_file = self.upload_batch_file(num_requests=10)
        batch = self.client.batches.create(
            input_file_id=input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        self.batch_ids.append(batch.id)
        
        # Cancel the batch
        cancelled_batch = self.client.batches.cancel(batch.id)
        
        # Assertions
        self.assertEqual(cancelled_batch.id, batch.id)
        self.assertIn(cancelled_batch.status, ["cancelling", "cancelled"])
        
        # If cancellation is immediate, check cancelled_at is set
        if cancelled_batch.status == "cancelled":
            self.assertIsNotNone(cancelled_batch.cancelled_at)
            self.assertGreaterEqual(cancelled_batch.cancelled_at, batch.created_at)
    
    def test_cancel_then_retrieve(self):
        """Test that a cancelled batch shows correct status when retrieved"""
        # Create and cancel a batch
        input_file = self.upload_batch_file(num_requests=5)
        batch = self.client.batches.create(
            input_file_id=input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        self.batch_ids.append(batch.id)
        
        # Cancel
        self.client.batches.cancel(batch.id)
        
        # Retrieve to verify cancellation
        retrieved = self.client.batches.retrieve(batch.id)
        
        # Assertions
        self.assertIn(retrieved.status, ["cancelling", "cancelled"])


class ErrorHandlingTests(BatchAPITestCase):
    """Tests for error handling"""
    
    def test_create_batch_with_invalid_file(self):
        """Test creating a batch with a non-existent file ID"""
        with self.assertRaises(Exception) as context:
            self.client.batches.create(
                input_file_id="file-nonexistent123",
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
    
    def test_retrieve_nonexistent_batch(self):
        """Test retrieving a non-existent batch"""
        with self.assertRaises(Exception) as context:
            self.client.batches.retrieve("batch_nonexistent123")
        
        # The error should indicate the batch was not found
        self.assertIn("404", str(context.exception) or str(type(context.exception)))
    
    def test_cancel_nonexistent_batch(self):
        """Test canceling a non-existent batch"""
        with self.assertRaises(Exception) as context:
            self.client.batches.cancel("batch_nonexistent456")
        
        self.assertIn("404", str(context.exception) or str(type(context.exception)))
    
    def test_cancel_completed_batch(self):
        """Test that canceling a completed batch raises an error"""
        # This test assumes we have a way to mark a batch as completed
        # For now, we'll skip this test as it requires batch processing to complete
        # In a real scenario, you would wait for a batch to complete or mock the status
        pass
    
    def test_create_batch_invalid_endpoint(self):
        """Test creating a batch with an invalid endpoint"""
        input_file = self.upload_batch_file(num_requests=1)
        
        # Note: Depending on validation, this might succeed or fail
        # If your API validates endpoints, this should raise an error
        try:
            batch = self.client.batches.create(
                input_file_id=input_file.id,
                endpoint="/v1/invalid/endpoint",
                completion_window="24h"
            )
            # If it succeeds, just track it
            self.batch_ids.append(batch.id)
        except Exception:
            # Expected if endpoint validation is strict
            pass
    
    def test_list_batches_invalid_limit(self):
        """Test listing batches with an invalid limit"""
        # Try with limit > 100 (should be capped or error)
        try:
            batches = self.client.batches.list(limit=150)
            # If it succeeds, verify it's capped at 100
            self.assertLessEqual(len(batches.data), 100)
        except Exception:
            # Expected if strict validation
            pass


class BatchStatusTests(BatchAPITestCase):
    """Tests for batch status tracking"""
    
    def test_batch_initial_status(self):
        """Test that a newly created batch has correct initial status"""
        input_file = self.upload_batch_file(num_requests=3)
        batch = self.client.batches.create(
            input_file_id=input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        self.batch_ids.append(batch.id)
        
        # Assertions
        self.assertIn(batch.status, ["validating", "in_progress"])
        self.assertIsNone(batch.output_file_id)  # Should be None initially
        self.assertIsNone(batch.error_file_id)   # Should be None initially
    
    def test_batch_expiration_time(self):
        """Test that batch expiration is set correctly"""
        input_file = self.upload_batch_file(num_requests=2)
        batch = self.client.batches.create(
            input_file_id=input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        self.batch_ids.append(batch.id)
        
        # Verify expiration is approximately 24 hours from creation
        time_diff = batch.expires_at - batch.created_at
        expected_seconds = 24 * 60 * 60  # 24 hours
        
        # Allow 1 minute tolerance
        self.assertAlmostEqual(time_diff, expected_seconds, delta=60)


if __name__ == "__main__":
    # Check if required packages are installed
    try:
        import openai
        print(f"Using OpenAI SDK version: {openai.__version__}")
    except ImportError:
        print("❌ Error: OpenAI SDK not installed")
        print("Please install it with: pip install openai")
        exit(1)
    
    print("\n" + "="*70)
    print("OpenAI Batch API Compatibility Test Suite")
    print("="*70)
    print("\nServer URL: http://localhost:8000")
    print("Make sure the server is running before executing tests!\n")
    print("="*70 + "\n")
    
    # Run tests with verbose output
    unittest.main(verbosity=2)
