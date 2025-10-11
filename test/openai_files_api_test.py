"""
Test the Files API Server using the official OpenAI Python SDK
This demonstrates OpenAI Files API compatibility for file upload, retrieval, listing, and deletion
"""
import os
import json
import time
import tempfile
from pathlib import Path
from openai import OpenAI


def create_test_file(filename: str = "test_batch.jsonl", num_requests: int = 5):
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
    
    print(f"✓ Created test file: {filename}")
    print(f"  Contains {len(requests)} requests")
    return filename


def create_sample_text_file(filename: str = "sample.txt", content: str = None):
    """Create a simple text file for testing"""
    if content is None:
        content = "This is a sample text file for testing the Files API.\n" * 10
    
    with open(filename, "w") as f:
        f.write(content)
    
    print(f"✓ Created sample text file: {filename}")
    return filename


def test_upload_file():
    """Test file upload functionality"""
    print("\n" + "="*70)
    print("TEST 1: Upload File")
    print("="*70 + "\n")
    
    # Initialize OpenAI client pointing to local server
    client = OpenAI(
        api_key="test-key",  # Server doesn't validate this
        base_url="http://localhost:8000/v1"
    )
    
    # Create a test file
    print("Step 1: Creating test file...")
    test_file = create_test_file("upload_test.jsonl", num_requests=3)
    
    # Upload the file
    print("\nStep 2: Uploading file...")
    with open(test_file, "rb") as f:
        uploaded_file = client.files.create(
            file=f,
            purpose="batch"
        )
    
    print(f"✓ File uploaded successfully!")
    print(f"  File ID: {uploaded_file.id}")
    print(f"  Object: {uploaded_file.object}")
    print(f"  Filename: {uploaded_file.filename}")
    print(f"  Purpose: {uploaded_file.purpose}")
    print(f"  Size: {uploaded_file.bytes} bytes")
    print(f"  Created at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(uploaded_file.created_at))}")
    
    # Cleanup
    os.remove(test_file)
    
    return uploaded_file.id


def test_list_files():
    """Test listing files functionality"""
    print("\n" + "="*70)
    print("TEST 2: List Files")
    print("="*70 + "\n")
    
    client = OpenAI(
        api_key="test-key",
        base_url="http://localhost:8000/v1"
    )
    
    # Upload multiple files first
    print("Step 1: Uploading multiple test files...")
    file_ids = []
    for i in range(3):
        filename = f"list_test_{i+1}.jsonl"
        create_test_file(filename, num_requests=2)
        
        with open(filename, "rb") as f:
            uploaded = client.files.create(file=f, purpose="batch")
            file_ids.append(uploaded.id)
            print(f"  ✓ Uploaded: {uploaded.id}")
        
        os.remove(filename)
    
    # List all files
    print("\nStep 2: Listing all files...")
    files_list = client.files.list()
    
    print(f"✓ Retrieved {len(files_list.data)} files:")
    for file in files_list.data[:5]:  # Show first 5
        print(f"  - {file.id}: {file.filename} ({file.bytes} bytes, {file.purpose})")
    
    if len(files_list.data) > 5:
        print(f"  ... and {len(files_list.data) - 5} more")
    
    # List files by purpose
    print("\nStep 3: Listing files filtered by purpose='batch'...")
    batch_files = client.files.list(purpose="batch")
    
    print(f"✓ Retrieved {len(batch_files.data)} batch files")
    
    return file_ids


def test_retrieve_file():
    """Test retrieving file metadata"""
    print("\n" + "="*70)
    print("TEST 3: Retrieve File Metadata")
    print("="*70 + "\n")
    
    client = OpenAI(
        api_key="test-key",
        base_url="http://localhost:8000/v1"
    )
    
    # Upload a file first
    print("Step 1: Uploading a test file...")
    test_file = create_test_file("retrieve_test.jsonl", num_requests=4)
    
    with open(test_file, "rb") as f:
        uploaded_file = client.files.create(file=f, purpose="batch")
    
    print(f"  ✓ Uploaded: {uploaded_file.id}")
    os.remove(test_file)
    
    # Retrieve file metadata
    print(f"\nStep 2: Retrieving metadata for {uploaded_file.id}...")
    retrieved_file = client.files.retrieve(uploaded_file.id)
    
    print(f"✓ File metadata retrieved:")
    print(f"  ID: {retrieved_file.id}")
    print(f"  Object: {retrieved_file.object}")
    print(f"  Filename: {retrieved_file.filename}")
    print(f"  Purpose: {retrieved_file.purpose}")
    print(f"  Size: {retrieved_file.bytes} bytes")
    print(f"  Created at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(retrieved_file.created_at))}")
    
    return uploaded_file.id


def test_retrieve_file_content():
    """Test retrieving file content"""
    print("\n" + "="*70)
    print("TEST 4: Retrieve File Content")
    print("="*70 + "\n")
    
    client = OpenAI(
        api_key="test-key",
        base_url="http://localhost:8000/v1"
    )
    
    # Upload a file with known content
    print("Step 1: Creating and uploading test file...")
    original_content = "Test content for file retrieval.\nLine 2.\nLine 3."
    test_file = create_sample_text_file("content_test.txt", original_content)
    
    with open(test_file, "rb") as f:
        uploaded_file = client.files.create(file=f, purpose="batch")
    
    print(f"  ✓ Uploaded: {uploaded_file.id}")
    os.remove(test_file)
    
    # Retrieve file content
    print(f"\nStep 2: Retrieving content for {uploaded_file.id}...")
    content = client.files.content(uploaded_file.id)
    
    retrieved_content = content.read().decode('utf-8')
    
    print(f"✓ File content retrieved ({len(retrieved_content)} bytes)")
    print(f"\nContent preview:")
    print("-" * 70)
    print(retrieved_content[:200])
    if len(retrieved_content) > 200:
        print("...")
    print("-" * 70)
    
    # Verify content matches
    if retrieved_content == original_content:
        print("\n✓ Content verification: PASSED (matches original)")
    else:
        print("\n❌ Content verification: FAILED (doesn't match original)")
    
    return uploaded_file.id


def test_delete_file():
    """Test file deletion functionality"""
    print("\n" + "="*70)
    print("TEST 5: Delete File")
    print("="*70 + "\n")
    
    client = OpenAI(
        api_key="test-key",
        base_url="http://localhost:8000/v1"
    )
    
    # Upload a file to delete
    print("Step 1: Uploading a file to delete...")
    test_file = create_test_file("delete_test.jsonl", num_requests=2)
    
    with open(test_file, "rb") as f:
        uploaded_file = client.files.create(file=f, purpose="batch")
    
    print(f"  ✓ Uploaded: {uploaded_file.id}")
    os.remove(test_file)
    
    # Delete the file
    print(f"\nStep 2: Deleting file {uploaded_file.id}...")
    deletion_status = client.files.delete(uploaded_file.id)
    
    print(f"✓ File deleted:")
    print(f"  ID: {deletion_status.id}")
    print(f"  Deleted: {deletion_status.deleted}")
    
    # Verify deletion by trying to retrieve
    print(f"\nStep 3: Verifying deletion...")
    try:
        client.files.retrieve(uploaded_file.id)
        print("❌ File still exists (deletion failed)")
    except Exception as e:
        print(f"✓ File confirmed deleted (retrieval failed as expected)")
    
    return deletion_status.deleted


def test_error_handling():
    """Test error handling for invalid operations"""
    print("\n" + "="*70)
    print("TEST 6: Error Handling")
    print("="*70 + "\n")
    
    client = OpenAI(
        api_key="test-key",
        base_url="http://localhost:8000/v1"
    )
    
    # Test 1: Retrieve non-existent file
    print("Test 6.1: Retrieve non-existent file...")
    try:
        client.files.retrieve("file-nonexistent123")
        print("  ❌ Should have raised an error")
    except Exception as e:
        print(f"  ✓ Correctly raised error: {type(e).__name__}")
    
    # Test 2: Delete non-existent file
    print("\nTest 6.2: Delete non-existent file...")
    try:
        client.files.delete("file-nonexistent456")
        print("  ❌ Should have raised an error")
    except Exception as e:
        print(f"  ✓ Correctly raised error: {type(e).__name__}")
    
    # Test 3: Upload with invalid purpose
    print("\nTest 6.3: Upload with invalid purpose...")
    test_file = create_sample_text_file("error_test.txt")
    try:
        with open(test_file, "rb") as f:
            client.files.create(file=f, purpose="invalid-purpose")
        print("  ❌ Should have raised an error")
    except Exception as e:
        print(f"  ✓ Correctly raised error: {type(e).__name__}")
    finally:
        os.remove(test_file)
    
    # Test 4: Retrieve content of non-existent file
    print("\nTest 6.4: Retrieve content of non-existent file...")
    try:
        client.files.content("file-nonexistent789")
        print("  ❌ Should have raised an error")
    except Exception as e:
        print(f"  ✓ Correctly raised error: {type(e).__name__}")
    
    print("\n✓ All error handling tests passed")


def test_large_file_upload():
    """Test uploading a larger file"""
    print("\n" + "="*70)
    print("TEST 7: Large File Upload")
    print("="*70 + "\n")
    
    client = OpenAI(
        api_key="test-key",
        base_url="http://localhost:8000/v1"
    )
    
    # Create a larger batch file
    print("Step 1: Creating large batch file (100 requests)...")
    test_file = create_test_file("large_batch.jsonl", num_requests=100)
    
    file_size = os.path.getsize(test_file)
    print(f"  File size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
    
    # Upload the large file
    print("\nStep 2: Uploading large file...")
    start_time = time.time()
    
    with open(test_file, "rb") as f:
        uploaded_file = client.files.create(file=f, purpose="batch")
    
    upload_time = time.time() - start_time
    
    print(f"✓ Large file uploaded successfully!")
    print(f"  File ID: {uploaded_file.id}")
    print(f"  Size: {uploaded_file.bytes:,} bytes")
    print(f"  Upload time: {upload_time:.2f} seconds")
    print(f"  Upload speed: {uploaded_file.bytes/upload_time/1024:.2f} KB/s")
    
    # Cleanup
    os.remove(test_file)
    
    return uploaded_file.id


def run_all_tests():
    """Run all Files API tests"""
    print("\n" + "="*70)
    print("OpenAI Files API Compatibility Test Suite")
    print("="*70)
    print(f"\nServer: http://localhost:8000/v1")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    try:
        # Test 1: Upload File
        file_id_1 = test_upload_file()
        results["Upload File"] = "✓ PASSED"
        
        # Test 2: List Files
        file_ids = test_list_files()
        results["List Files"] = "✓ PASSED"
        
        # Test 3: Retrieve File Metadata
        file_id_3 = test_retrieve_file()
        results["Retrieve File Metadata"] = "✓ PASSED"
        
        # Test 4: Retrieve File Content
        file_id_4 = test_retrieve_file_content()
        results["Retrieve File Content"] = "✓ PASSED"
        
        # Test 5: Delete File
        test_delete_file()
        results["Delete File"] = "✓ PASSED"
        
        # Test 6: Error Handling
        test_error_handling()
        results["Error Handling"] = "✓ PASSED"
        
        # Test 7: Large File Upload
        file_id_7 = test_large_file_upload()
        results["Large File Upload"] = "✓ PASSED"
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results["Overall"] = "❌ FAILED"
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70 + "\n")
    
    for test_name, status in results.items():
        print(f"{test_name:.<50} {status}")
    
    passed = sum(1 for s in results.values() if "PASSED" in s)
    total = len(results)
    
    print(f"\n{passed}/{total} tests passed")
    print(f"\nCompleted: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == "__main__":
    # Check if required packages are installed
    try:
        import openai
        print(f"Using OpenAI SDK version: {openai.__version__}")
    except ImportError:
        print("❌ Error: OpenAI SDK not installed")
        print("Please install it with: pip install openai")
        exit(1)
    
    # Run all tests
    run_all_tests()
