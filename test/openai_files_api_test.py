"""
Test the Files API Server using the official OpenAI Python SDK
This demonstrates OpenAI Files API compatibility for file upload, retrieval, listing, and deletion
"""
import os
import json
import time
import unittest
from pathlib import Path
from openai import OpenAI


class FilesAPITestCase(unittest.TestCase):
    """Base test case for Files API tests"""
    
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
    
    def create_test_file(self, filename: str = "test_batch.jsonl", num_requests: int = 5):
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
                        {"role": "user", "content": f"This is test request number {i+1}"}
                    ],
                    "max_tokens": 50
                }
            }
            requests.append(request)
        
        with open(filename, "w") as f:
            for request in requests:
                f.write(json.dumps(request) + "\n")
        
        self.temp_files.append(filename)
        return filename
    
    def create_sample_text_file(self, filename: str = "sample.txt", content: str = None):
        """Create a simple text file for testing"""
        if content is None:
            content = "This is a sample text file for testing the Files API.\n" * 10
        
        with open(filename, "w") as f:
            f.write(content)
        
        self.temp_files.append(filename)
        return filename


class FileUploadTests(FilesAPITestCase):
    """Tests for file upload functionality"""
    
    def test_upload_file(self):
        """Test basic file upload functionality"""
        # Create a test file
        test_file = self.create_test_file("upload_test.jsonl", num_requests=3)
        
        # Upload the file
        with open(test_file, "rb") as f:
            uploaded_file = self.client.files.create(
                file=f,
                purpose="batch"
            )
        
        # Track for cleanup
        self.uploaded_file_ids.append(uploaded_file.id)
        
        # Assertions
        self.assertIsNotNone(uploaded_file.id)
        self.assertEqual(uploaded_file.object, "file")
        self.assertEqual(uploaded_file.purpose, "batch")
        self.assertGreater(uploaded_file.bytes, 0)
        self.assertIsNotNone(uploaded_file.created_at)
    
    def test_large_file_upload(self):
        """Test uploading a larger file"""
        # Create a larger batch file
        test_file = self.create_test_file("large_batch.jsonl", num_requests=100)
        
        file_size = os.path.getsize(test_file)
        self.assertGreater(file_size, 10000)  # Should be > 10KB
        
        # Upload the large file
        start_time = time.time()
        
        with open(test_file, "rb") as f:
            uploaded_file = self.client.files.create(file=f, purpose="batch")
        
        upload_time = time.time() - start_time
        
        # Track for cleanup
        self.uploaded_file_ids.append(uploaded_file.id)
        
        # Assertions
        self.assertIsNotNone(uploaded_file.id)
        self.assertEqual(uploaded_file.bytes, file_size)
        self.assertLess(upload_time, 10)  # Should upload in less than 10 seconds


class FileListTests(FilesAPITestCase):
    """Tests for file listing functionality"""
    
    def test_list_files(self):
        """Test listing all files"""
        # Upload multiple files for this test
        file_ids = []
        for i in range(3):
            filename = f"list_test_{i+1}.jsonl"
            self.create_test_file(filename, num_requests=2)
            
            with open(filename, "rb") as f:
                uploaded = self.client.files.create(file=f, purpose="batch")
                file_ids.append(uploaded.id)
                self.uploaded_file_ids.append(uploaded.id)
        
        # List all files
        files_list = self.client.files.list()
        
        # Assertions
        self.assertIsNotNone(files_list.data)
        self.assertGreaterEqual(len(files_list.data), 3)
        
        # Verify our uploaded files are in the list
        listed_ids = [f.id for f in files_list.data]
        for file_id in file_ids:
            self.assertIn(file_id, listed_ids)
    
    def test_list_files_by_purpose(self):
        """Test listing files filtered by purpose"""
        # Upload a file with batch purpose for this test
        test_file = self.create_test_file("batch_purpose_test.jsonl", num_requests=2)
        
        with open(test_file, "rb") as f:
            uploaded = self.client.files.create(file=f, purpose="batch")
        
        self.uploaded_file_ids.append(uploaded.id)
        
        # List files by purpose
        batch_files = self.client.files.list(purpose="batch")
        
        # Assertions
        self.assertIsNotNone(batch_files.data)
        self.assertGreater(len(batch_files.data), 0)
        
        # All returned files should have batch purpose
        for file in batch_files.data:
            self.assertEqual(file.purpose, "batch")
        
        # Our uploaded file should be in the list
        listed_ids = [f.id for f in batch_files.data]
        self.assertIn(uploaded.id, listed_ids)


class FileRetrievalTests(FilesAPITestCase):
    """Tests for file retrieval functionality"""
    
    def test_retrieve_file_metadata(self):
        """Test retrieving file metadata"""
        # Upload a file for this test
        test_file = self.create_test_file("retrieve_test.jsonl", num_requests=4)
        
        with open(test_file, "rb") as f:
            uploaded_file = self.client.files.create(file=f, purpose="batch")
        
        self.uploaded_file_ids.append(uploaded_file.id)
        
        # Retrieve file metadata
        retrieved_file = self.client.files.retrieve(uploaded_file.id)
        
        # Assertions
        self.assertEqual(retrieved_file.id, uploaded_file.id)
        self.assertEqual(retrieved_file.object, "file")
        self.assertEqual(retrieved_file.filename, uploaded_file.filename)
        self.assertEqual(retrieved_file.purpose, uploaded_file.purpose)
        self.assertEqual(retrieved_file.bytes, uploaded_file.bytes)
        self.assertEqual(retrieved_file.created_at, uploaded_file.created_at)
    
    def test_retrieve_file_content(self):
        """Test retrieving file content"""
        # Upload a file with known content
        original_content = "Test content for file retrieval.\nLine 2.\nLine 3."
        test_file = self.create_sample_text_file("content_test.txt", original_content)
        
        with open(test_file, "rb") as f:
            uploaded_file = self.client.files.create(file=f, purpose="batch")
        
        self.uploaded_file_ids.append(uploaded_file.id)
        
        # Retrieve file content
        content = self.client.files.content(uploaded_file.id)
        retrieved_content = content.read().decode('utf-8')
        
        # Assertions
        self.assertEqual(retrieved_content, original_content)
        self.assertEqual(len(retrieved_content), len(original_content))


class FileDeletionTests(FilesAPITestCase):
    """Tests for file deletion functionality"""
    
    def test_delete_file(self):
        """Test file deletion"""
        # Upload a file to delete
        test_file = self.create_test_file("delete_test.jsonl", num_requests=2)
        
        with open(test_file, "rb") as f:
            uploaded_file = self.client.files.create(file=f, purpose="batch")
        
        # Note: We don't add to uploaded_file_ids since we're testing deletion
        
        # Delete the file
        deletion_status = self.client.files.delete(uploaded_file.id)
        
        # Assertions
        self.assertEqual(deletion_status.id, uploaded_file.id)
        self.assertTrue(deletion_status.deleted)
        
        # Verify deletion by trying to retrieve
        with self.assertRaises(Exception):
            self.client.files.retrieve(uploaded_file.id)


class ErrorHandlingTests(FilesAPITestCase):
    """Tests for error handling"""
    
    def test_retrieve_nonexistent_file(self):
        """Test retrieving a non-existent file raises an error"""
        with self.assertRaises(Exception):
            self.client.files.retrieve("file-nonexistent123")
    
    def test_delete_nonexistent_file(self):
        """Test deleting a non-existent file raises an error"""
        with self.assertRaises(Exception):
            self.client.files.delete("file-nonexistent456")
    
    def test_upload_with_invalid_purpose(self):
        """Test uploading with an invalid purpose raises an error"""
        test_file = self.create_sample_text_file("error_test.txt")
        
        with open(test_file, "rb") as f:
            with self.assertRaises(Exception):
                self.client.files.create(file=f, purpose="invalid-purpose")
    
    def test_retrieve_content_nonexistent_file(self):
        """Test retrieving content of non-existent file raises an error"""
        with self.assertRaises(Exception):
            self.client.files.content("file-nonexistent789")


if __name__ == "__main__":
    # Check if required packages are installed
    try:
        import openai
        print(f"Using OpenAI SDK version: {openai.__version__}")
    except ImportError:
        print("❌ Error: OpenAI SDK not installed")
        print("Please install it with: pip install openai")
        exit(1)
    
    # Run tests with verbose output
    unittest.main(verbosity=2)
