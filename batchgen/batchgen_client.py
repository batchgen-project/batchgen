import socket
import struct
import pickle
import time
import argparse
from typing import List, Optional, Dict, Any


class BatchGenClient:
    def __init__(self, host: str = 'localhost', port: int = 10900):
        self.host = host
        self.port = port
        self.sock = None

    def connect(self):
        """Establishes connection to the server."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            print(f"Connected to {self.host}:{self.port}")
        except Exception as e:
            print(f"Connection failed: {e}")
            self.sock = None

    def send_request(self, data: Dict[str, Any]) -> Optional[Dict]:
        """
        Sends any Python object (List, Dict, etc.) to the server.
        Returns the response from the server.
        """
        if not self.sock:
            print("Socket not connected.")
            return None

        try:
            # 1. Serialize
            serialized_data = pickle.dumps(data)
            
            # 2. Send Length (4 bytes, Big Endian)
            length_prefix = struct.pack('!I', len(serialized_data))
            self.sock.sendall(length_prefix)
            
            # 3. Send Payload
            self.sock.sendall(serialized_data)

            # 4. Receive Response Length
            size_data = self.sock.recv(4)
            if not size_data:
                return None
            resp_size = struct.unpack('!I', size_data)[0]

            # 5. Receive Response Payload
            resp_data = b''
            while len(resp_data) < resp_size:
                chunk = self.sock.recv(min(4096, resp_size - len(resp_data)))
                if not chunk:
                    break
                resp_data += chunk

            # 6. Deserialize
            return pickle.loads(resp_data)

        except Exception as e:
            print(f"Communication error: {e}")
            self.close()
            return None

    def submit_inference(
        self,
        queries: List[str],
        max_input_len: int = 1024,
        max_output_len: int = 128,
        ignore_eos: bool = False,
    ) -> Optional[Dict]:
        """
        Submit inference request with generation parameters.
        
        Args:
            queries: List of prompt strings
            max_input_len: Maximum input sequence length
            max_output_len: Maximum output/decoding length
            ignore_eos: If True, ignore EOS tokens and decode to max_output_len
                       (useful for benchmarking)
        
        Returns:
            Server response dictionary
        """
        payload = {
            "command": "submit_inference",
            "queries": queries,
            "max_input_len": max_input_len,
            "max_output_len": max_output_len,
            "ignore_eos": ignore_eos,
        }
        return self.send_request(payload)

    def ping(self) -> Optional[Dict]:
        """Send ping command to server."""
        return self.send_request({"command": "ping"})

    def close(self):
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