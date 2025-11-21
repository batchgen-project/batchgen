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
    PORT = 32000

    client = BatchGenClient(HOST, PORT)
    client.connect()

    if client.sock:
        # --- TEST 1: List of Strings ---
        print("\n--- Test 1: Sending List of Strings ---")
        str_payload = [
            "Tell me a joke about AI.",
            "What is the capital of France?",
            "Explain quantum physics in 5 words."
        ]
        response = client.send_request(str_payload)
        print(f"Response type: {type(response)}")
        print(f"Result: {response}")

        # --- TEST 2: List of Dictionaries (Advanced) ---
        print("\n--- Test 2: Sending List of Dicts ---")
        dict_payload = [
            {"role": "user", "content": "Hello!"},
            {"role": "user", "content": "Translate this to Spanish."}
        ]
        response = client.send_request(dict_payload)
        print(f"Result: {response}")

        # --- TEST 3: Server Command (Ping) ---
        print("\n--- Test 3: Control Command (Ping) ---")
        cmd_payload = {"command": "ping"}
        response = client.send_request(cmd_payload)
        print(f"Result: {response}")

        client.close()