import socket
import struct
import pickle
import time

class BatchGenClient:
    def __init__(self, host='localhost', port=10900):
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

    def send_request(self, data):
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
                if not chunk: break
                resp_data += chunk

            # 6. Deserialize
            return pickle.loads(resp_data)

        except Exception as e:
            print(f"Communication error: {e}")
            self.close()
            return None

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None

if __name__ == "__main__":
    # Configuration
    HOST = 'localhost'
    PORT = 10900  # Matches your Server default

    client = BatchGenClient(HOST, PORT)
    client.connect()

    if client.sock:
        # --- TEST 1: Standard Inference Request ---
        print("\n--- Test 1: Sending Structured Inference Request ---")
        
        # New Payload Structure
        payload = {
            "command": "submit_inference",
            "queries": [
                "Tell me a joke about AI.",
                "What is the capital of France?",
                "Explain quantum physics in 5 words."
            ],
            # Params required by the new worker logic
            "max_input_len": 512, 
            "max_output_len": 64
        }
        
        start_t = time.perf_counter()
        response = client.send_request(payload)
        dur = time.perf_counter() - start_t
        
        print(f"Time taken: {dur:.4f}s")
        print(f"Result: {response}")

        # --- TEST 2: Long Context Request (Different Params) ---
        print("\n--- Test 2: Sending Long Context Request ---")
        
        payload_long = {
            "command": "submit_inference",
            "queries": [
                "Write a short poem about rust.",
            ],
            "max_input_len": 2048, 
            "max_output_len": 256 # Requesting longer output
        }
        
        response = client.send_request(payload_long)
        print(f"Result: {response}")

        # --- TEST 3: Server Command (Ping) ---
        print("\n--- Test 3: Control Command (Ping) ---")
        cmd_payload = {"command": "ping"}
        response = client.send_request(cmd_payload)
        print(f"Result: {response}")

        client.close()