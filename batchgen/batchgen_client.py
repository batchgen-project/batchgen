import socket
import struct
import pickle
import time
import logging
import argparse
from typing import List, Optional, Dict, Any

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class BatchGenClient:
    """TCP client for BatchGen inference API calls."""

    def __init__(self, host: str = 'localhost', port: int = 10900):
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None

    def connect(self) -> None:
        """Establishes connection to the server."""
        try:
            logger.info(f"Connecting to BatchGen Server at {self.host}:{self.port}...")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            logger.info("Successfully connected.")
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.sock = None
            raise

    def send_request(self, data: Dict[str, Any]) -> Any:
        """
        Sends any Python object (List, Dict, etc.) to the server.
        Returns the response from the server.
        """
        if not self.sock:
            raise ConnectionError("Socket not connected.")

        try:
            # 1. Serialize
            payload = pickle.dumps(data)
            
            # 2. Send Length (4 bytes, Big Endian)
            self.sock.sendall(struct.pack('!I', len(payload)))
            
            # 3. Send Payload
            self.sock.sendall(payload)

            # 4. Receive Response Length
            size_bytes = self.sock.recv(4)
            if not size_bytes:
                raise ConnectionResetError(
                    "Server closed connection before sending a response."
                )
            resp_size = struct.unpack('!I', size_bytes)[0]

            # 5. Receive Response Payload (using bytearray for efficiency)
            resp_data = bytearray()
            while len(resp_data) < resp_size:
                chunk = self.sock.recv(min(4096, resp_size - len(resp_data)))
                if not chunk:
                    break
                resp_data.extend(chunk)

            # 6. Deserialize
            return pickle.loads(resp_data)

        except Exception as e:
            logger.error(f"Communication error: {e}")
            self.close()
            return []

    def submit_inference(
        self,
        queries: List[str],
        max_input_len: Optional[int] = None,
        max_output_len: int = 128,
        ignore_eos: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Optional[Dict]:
        """
        Submit inference request with generation parameters.

        Args:
            queries: List of prompt strings
            max_input_len: Maximum input sequence length. If None, determined
                          dynamically from the longest prompt in the batch.
            max_output_len: Maximum output/decoding length
            ignore_eos: If True, ignore EOS tokens and decode to max_output_len
                       (useful for benchmarking)
            temperature: Sampling temperature (None = greedy decoding)
            top_p: Nucleus sampling threshold (None = disabled)

        Returns:
            Server response dictionary
        """
        payload: Dict[str, Any] = {
            "command": "submit_inference",
            "queries": queries,
            "max_input_len": max_input_len,  # Can be None for dynamic detection
            "max_output_len": max_output_len,
            "ignore_eos": ignore_eos,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        return self.send_request(payload)

    def ping(self) -> Optional[Dict]:
        """Send ping command to server."""
        return self.send_request({"command": "ping"})

    def close(self) -> None:
        """Close the connection."""
        if self.sock:
            self.sock.close()
            self.sock = None


class BatchGenHttpClient:
    """HTTP client for BatchGen OpenAI-compatible API."""

    def __init__(self, base_url: str, timeout_s: Optional[float] = None) -> None:
        """Initialize the HTTP client.

        Args:
            base_url: Server base URL (e.g., http://localhost:10900)
            timeout_s: Request timeout in seconds (None = wait forever)
        """
        if not _REQUESTS_AVAILABLE:
            raise ImportError(
                "BatchGenHttpClient requires the 'requests' package. "
                "Install it with: pip install requests"
            )
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._session = requests.Session()

    def post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send a POST request with JSON payload.

        Args:
            path: API endpoint path (e.g., /v1/inference)
            payload: JSON payload dictionary

        Returns:
            Response JSON as dictionary
        """
        url = f"{self._base_url}{path}"
        response = self._session.post(url, json=payload, timeout=self._timeout_s)
        self._raise_for_status(response, "POST", url)
        if not response.content:
            return {}
        return response.json()

    def health_check(self) -> bool:
        """Check if the server is healthy.

        Returns:
            True if server responds with 200, False otherwise
        """
        try:
            url = f"{self._base_url}/health"
            response = self._session.get(url, timeout=10.0)
            return response.status_code == 200
        except Exception:
            return False

    def submit_inference(
        self,
        prompts: List[str],
        max_input_len: Optional[int] = None,
        max_output_len: int = 128,
        ignore_eos: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> List[str]:
        """Submit inference request and get decoded string results.

        Args:
            prompts: List of prompt strings
            max_input_len: Maximum input sequence length. If None, determined
                          dynamically from the longest prompt in the batch.
            max_output_len: Maximum output/decoding length
            ignore_eos: If True, ignore EOS tokens and decode to max_output_len
            temperature: Sampling temperature (None = greedy decoding)
            top_p: Nucleus sampling threshold (None = disabled)

        Returns:
            List of decoded output strings

        Raises:
            RuntimeError: If inference fails or returns unexpected format
        """
        payload: Dict[str, Any] = {
            "prompts": prompts,
            "max_input_len": max_input_len,
            "max_output_len": max_output_len,
            "ignore_eos": ignore_eos,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p

        response = self.post_json("/v1/inference", payload)

        if response.get("status") != "success":
            raise RuntimeError(f"Inference failed: {response}")

        results = response.get("results")
        if not results:
            raise RuntimeError("Server returned empty results.")

        if not isinstance(results, list):
            raise RuntimeError(f"Unexpected result format: {type(results)}")

        return results

    # ==================== Batch API Methods ====================

    def upload_file(
        self,
        file_path: str,
        purpose: str = "batch",
    ) -> Dict[str, Any]:
        """Upload a file to the server.

        Args:
            file_path: Path to the file to upload
            purpose: File purpose ('batch' for input files)

        Returns:
            File object with id, filename, etc.
        """
        url = f"{self._base_url}/v1/files"
        with open(file_path, "rb") as f:
            files = {"file": (file_path.split("/")[-1], f)}
            data = {"purpose": purpose}
            response = self._session.post(
                url, files=files, data=data, timeout=self._timeout_s
            )
        self._raise_for_status(response, "POST", url)
        return response.json()

    def create_batch(
        self,
        input_file_id: str,
        endpoint: str = "/v1/chat/completions",
        completion_window: str = "24h",
        metadata: Optional[Dict[str, Any]] = None,
        max_decoding_length: Optional[int] = None,
        max_context_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create a batch job.

        Args:
            input_file_id: ID of the uploaded input file
            endpoint: Target endpoint ('/v1/chat/completions' or '/v1/completions')
            completion_window: Time window for completion ('24h')
            metadata: Optional metadata dictionary
            max_decoding_length: Batch-level fallback max output tokens (None = require per-request)
            max_context_length: Max total context (prompt + decode). None = use model maximum.
            temperature: Default sampling temperature (None = greedy). Per-request values override.
            top_p: Default nucleus sampling threshold (None = disabled). Per-request values override.
            top_k: Default top-k filtering (None or 0 = disabled). Per-request values override.

        Returns:
            Batch object with id, status, etc.
        """
        payload: Dict[str, Any] = {
            "input_file_id": input_file_id,
            "endpoint": endpoint,
            "completion_window": completion_window,
        }
        if max_context_length is not None:
            payload["max_context_length"] = max_context_length
        if metadata:
            payload["metadata"] = metadata
        if max_decoding_length is not None:
            payload["max_decoding_length"] = max_decoding_length
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if top_k is not None:
            payload["top_k"] = top_k
        return self.post_json("/v1/batches", payload)

    def get_batch(self, batch_id: str) -> Dict[str, Any]:
        """Get batch status.

        Args:
            batch_id: ID of the batch

        Returns:
            Batch object with current status
        """
        url = f"{self._base_url}/v1/batches/{batch_id}"
        response = self._session.get(url, timeout=self._timeout_s)
        self._raise_for_status(response, "GET", url)
        return response.json()

    def wait_for_batch(
        self,
        batch_id: str,
        poll_interval: float = 5.0,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Wait for a batch to complete.

        Args:
            batch_id: ID of the batch
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait (None = unlimited)

        Returns:
            Final batch object

        Raises:
            TimeoutError: If timeout exceeded
            RuntimeError: If batch failed or was cancelled
        """
        start_time = time.time()
        terminal_statuses = {"completed", "failed", "cancelled"}

        while True:
            batch = self.get_batch(batch_id)
            status = batch.get("status")

            if status in terminal_statuses:
                if status == "failed":
                    raise RuntimeError(f"Batch failed: {batch.get('error')}")
                if status == "cancelled":
                    raise RuntimeError("Batch was cancelled")
                return batch

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(
                    f"Batch {batch_id} did not complete within {timeout}s"
                )

            logger.info(f"Batch {batch_id} status: {status}, waiting...")
            time.sleep(poll_interval)

    def download_file_content(self, file_id: str) -> bytes:
        """Download file content.

        Args:
            file_id: ID of the file to download

        Returns:
            File content as bytes
        """
        url = f"{self._base_url}/v1/files/{file_id}/content"
        response = self._session.get(url, timeout=self._timeout_s)
        self._raise_for_status(response, "GET", url)
        return response.content

    def get_file(self, file_id: str) -> Dict[str, Any]:
        """Get file metadata.

        Args:
            file_id: ID of the file

        Returns:
            File object with metadata
        """
        url = f"{self._base_url}/v1/files/{file_id}"
        response = self._session.get(url, timeout=self._timeout_s)
        self._raise_for_status(response, "GET", url)
        return response.json()

    def submit_batch(
        self,
        input_file_path: str,
        output_file_path: Optional[str] = None,
        endpoint: str = "/v1/chat/completions",
        poll_interval: float = 5.0,
        timeout: Optional[float] = None,
        max_decoding_length: Optional[int] = None,
        max_context_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Submit a batch job and wait for completion.

        This is a convenience method that:
        1. Uploads the input file
        2. Creates the batch
        3. Waits for completion
        4. Downloads results to output file (if specified)

        Args:
            input_file_path: Path to input JSONL file
            output_file_path: Path to save output JSONL (optional)
            endpoint: Target endpoint
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait
            max_decoding_length: Batch-level fallback max output tokens (None = require per-request)
            max_context_length: Max total context (prompt + decode). None = use model maximum.
            temperature: Default sampling temperature (None = greedy). Per-request values override.
            top_p: Default nucleus sampling threshold (None = disabled). Per-request values override.
            top_k: Default top-k filtering (None or 0 = disabled). Per-request values override.

        Returns:
            Final batch object with output_file_id
        """
        # 1. Upload input file
        logger.info(f"Uploading {input_file_path}...")
        file_obj = self.upload_file(input_file_path, purpose="batch")
        file_id = file_obj["id"]
        logger.info(f"Uploaded file: {file_id}")

        # 2. Create batch
        logger.info("Creating batch...")
        batch = self.create_batch(
            file_id,
            endpoint=endpoint,
            max_decoding_length=max_decoding_length,
            max_context_length=max_context_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        batch_id = batch["id"]
        logger.info(f"Created batch: {batch_id}")

        # 3. Wait for completion
        logger.info("Waiting for batch to complete...")
        batch = self.wait_for_batch(batch_id, poll_interval, timeout)
        logger.info(f"Batch completed: {batch['status']}")
        if batch.get("output_file_id"):
            output_fid = batch["output_file_id"]
            logger.info(f"Result file: {output_fid}.jsonl (in server storage_path/outputs/)")

        # 4. Download results if output path specified
        if output_file_path and batch.get("output_file_id"):
            output_file_id = batch["output_file_id"]
            logger.info(f"Downloading results from {output_file_id}...")
            content = self.download_file_content(output_file_id)
            with open(output_file_path, "wb") as f:
                f.write(content)
            logger.info(f"Saved results to {output_file_path}")

        return batch

    # ==================== Internal Methods ====================

    def _raise_for_status(
        self, response: "requests.Response", method: str, url: str
    ) -> None:
        """Raise an exception if the response status indicates an error."""
        if response.status_code < 400:
            return
        detail = None
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        raise RuntimeError(
            f"{method} {url} failed ({response.status_code}): {detail}"
        )

    def close(self) -> None:
        """Close the HTTP session."""
        self._session.close()


def main():
    parser = argparse.ArgumentParser(description="BatchGen Client")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=10900, help="Server port")
    parser.add_argument("--max-input-len", type=int, default=1024, help="Max input length")
    parser.add_argument("--max-output-len", type=int, default=128, help="Max output/decoding length")
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Ignore EOS tokens, decode all sequences to max_output_len (for benchmarking)"
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=["Tell me a joke about AI.", "What is the capital of France?"],
        help="List of query strings"
    )
    
    args = parser.parse_args()

    client = BatchGenClient(args.host, args.port)
    client.connect()

    if client.sock:
        print(f"\n--- Submitting Inference Request ---")
        print(f"  ignore_eos: {args.ignore_eos}")
        print(f"  max_input_len: {args.max_input_len}")
        print(f"  max_output_len: {args.max_output_len}")
        print(f"  num_queries: {len(args.queries)}")
        
        start_t = time.perf_counter()
        response = client.submit_inference(
            queries=args.queries,
            max_input_len=args.max_input_len,
            max_output_len=args.max_output_len,
            ignore_eos=args.ignore_eos,
        )
        dur = time.perf_counter() - start_t
        
        print(f"\nTime taken: {dur:.4f}s")
        print(f"Result status: {response.get('status') if response else 'None'}")
        
        if response and response.get('status') == 'success':
            results = response.get('results', [])
            print(f"Number of result tensors: {len(results)}")

        client.close()


if __name__ == "__main__":
    main()