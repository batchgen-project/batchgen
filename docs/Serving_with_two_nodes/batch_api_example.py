"""
	Show case of how to submit a batch job to BatchGen with python API.
	Align with OpenAI Batch API. https://platform.openai.com/docs/guides/batch
	API sementic difference:
	1. BatchGen batch job is created with an uploaded file, which contains all the sequences.
	2. Current does not support creating a list of batches. User can only create a batch after the server is in idle.
	3. Does not have expiration semantic.
"""

# 0: Prepare .jsonl file as input.
# The format of each sequence is as follows:
"""
{
	"custom_id": "request-1", 
	"method": "POST", 
	"url": "/v1/chat/completions", 
	"body": {"model": "deepseek-r1", 
	"messages": [{"role": "system", "content": "You are a helpful assistant."},
				{"role": "user", "content": "Hello world!"}],
	"max_tokens": 1000}
}
"""


# 1: upload file to the server.
# from batchgen import BatchGen
from openai import OpenAI
client = OpenAI(
    base_url="http://<addr>:<port>/v1",  # Change to batchgen http server addr and port
)

batch_input_file = client.files.create(
    file=open("batchinput.jsonl", "rb"),
    purpose="batch"
)

# 2: create a batch job with the uploaded file.
# Note that batchgen will create a batch with all the sequences in the uploaded file.
batch_input_file_id = batch_input_file.id
client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    metadata={
        "description": "nightly eval job"
    }
)

# 3: check the status of the batch job.
batch = client.batches.retrieve("batch_abc123")
"""
	{
		"id": "batch_abc123",
		"object": "batch",
		"endpoint": "/v1/completions",
		"model": "gpt-5-2025-08-07",
		"errors": null,
		"input_file_id": "file-abc123",
		"completion_window": "24h",
		"status": "completed",
		"output_file_id": "file-cvaTdG",
		...
	
		"error_file_id": "file-HOWS94",
		"created_at": 1711471533,
		"in_progress_at": 1711471538,
		"expires_at": 1711557933,
		"finalizing_at": 1711493133,
		"completed_at": 1711493163,
		"failed_at": null,
		"expired_at": null,
		"cancelling_at": null,
		"cancelled_at": null,
		"request_counts": {
			"total": 100,
			"completed": 95,
			"failed": 5
		},
		"usage": {
			"input_tokens": 1500,
			"input_tokens_details": {
			"cached_tokens": 1024
			},
			"output_tokens": 500,
			"output_tokens_details": {
			"reasoning_tokens": 300
			},
			"total_tokens": 2000
		},
		"metadata": {
			"customer_id": "user_123456789",
			"batch_description": "Nightly eval job",
		}
	}`

"""

# 4. get the results of the batch job.
# 4.1. Retrieve the completed batch object to get the file ID
batch_job = client.batches.retrieve("YOUR_BATCH_ID")
output_file_id = batch_job.output_file_id

if output_file_id:
    # 4.2. Retrieve the content of the output file
    file_content_response = client.files.content(output_file_id)
    
    # The content is in bytes, so decode it to a string
    file_content_str = file_content_response.content.decode('utf-8')

    # 4.3. Save the content to a .jsonl file
    with open("results.jsonl", "w") as f:
        f.write(file_content_str)
    
    print("Successfully downloaded results to results.jsonl")
else:
    print("Batch job is not completed or has no output file.")


"""
	{
		"id": "batch_req_123", 
		"custom_id": "request-2", 
		"response": {
						"status_code": 200, 
						"request_id": "req_123", 
						"body": {
									"id": "chatcmpl-123", 
									"object": "chat.completion", 
									"created": 1711652795, 
									"model": "gpt-3.5-turbo-0125", 
									"choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello."}, "logprobs": null, "finish_reason": "stop"}], 
									"usage": {"prompt_tokens": 22, "completion_tokens": 2, "total_tokens": 24}, "system_fingerprint": "fp_123"}
                                }, 
		"error": null
	}	
"""
