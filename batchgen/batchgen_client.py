import time
import logging
from typing import List, Optional, Dict, Any

from batchgen.deprecation import LegacyInferenceDeprecated

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


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

    def _request_with_retry(
        self, method: str, url: str, max_retries: int = 5, **kwargs
    ) -> "requests.Response":
        """Send HTTP request with retry on 429 (server at capacity).

        Uses exponential backoff: 1s, 2s, 4s, 8s, 16s (capped at 60s).
        """
        import time as _time
        for attempt in range(max_retries):
            response = self._session.request(method, url, timeout=self._timeout_s, **kwargs)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait = int(retry_after) if retry_after else min(2 ** attempt, 60)
                logger.warning(
                    f"Server at capacity (429). Retrying in {wait}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                _time.sleep(wait)
                continue
            return response
        raise RuntimeError(
            f"{method} {url} failed: server at capacity after {max_retries} retries"
        )

    def post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send a POST request with JSON payload.

        Args:
            path: API endpoint path (e.g., /v1/inference)
            payload: JSON payload dictionary

        Returns:
            Response JSON as dictionary
        """
        url = f"{self._base_url}{path}"
        response = self._request_with_retry("POST", url, json=payload)
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
        """DEPRECATED and disabled. Use submit_batch() instead.

        The method is kept, rather than deleted, so a caller gets the
        explanation above instead of an AttributeError telling it only that
        something is gone. It raises without any network call: the server
        answers /v1/inference with 410 anyway, and failing here keeps the
        deprecated request off the wire entirely.

        Raises:
            LegacyInferenceDeprecated: always.
        """
        raise LegacyInferenceDeprecated()

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
            response = self._request_with_retry(
                "POST", url, files=files, data=data,
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