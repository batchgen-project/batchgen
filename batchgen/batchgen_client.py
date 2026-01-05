import socket
import struct
import pickle
import time
import logging
import argparse
from typing import List, Optional, Dict, Any

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
        return self.send_request(payload)

    def ping(self) -> Optional[Dict]:
        """Send ping command to server."""
        return self.send_request({"command": "ping"})

    def close(self) -> None:
        """Close the connection."""
        if self.sock:
            self.sock.close()
            self.sock = None


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