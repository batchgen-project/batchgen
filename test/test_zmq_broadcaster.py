"""
Test suite for ZMQBroadcaster using unittest and multiprocessing
"""
import unittest
import multiprocessing as mp
import time
import zmq
import logging
from typing import Optional

from pydantic import BaseModel, Field

# Import the broadcaster (adjust path as needed)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from batchgen.utils import ZMQBroadcaster

# Make this module importable by name for dynamic class loading
if __name__ == '__main__':
    # When run as main, register the module
    import __main__
    sys.modules['test_zmq_broadcaster'] = __main__


# Test Pydantic models
class SimpleConfig(BaseModel):
    """Simple configuration for testing"""
    name: str = Field(..., description="Configuration name")
    value: int = Field(..., description="Configuration value")


class ComplexConfig(BaseModel):
    """Complex configuration with nested data"""
    model_name: str
    batch_size: int
    learning_rate: float
    layers: list
    metadata: dict


def worker_process_simple(rank: int, world_size: int, endpoint: str, result_queue: mp.Queue):
    """Worker process for simple broadcast test"""
    # Configure logging for this worker process
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Worker {rank}] %(asctime)s - %(levelname)s - %(message)s',
        force=True
    )
    
    try:
        broadcaster = ZMQBroadcaster(rank=rank, world_size=world_size, endpoint=endpoint)
        
        print(f"Rank {rank} starting broadcast test")
        if rank == 0:
            # Rank 0 sends the config
            config = SimpleConfig(name="test_config", value=42)
            result = broadcaster.broadcast(config)
        else:
            # Other ranks receive
            result = broadcaster.broadcast(None)
        
        broadcaster.close()
        
        # Verify result
        assert isinstance(result, SimpleConfig), f"Rank {rank}: Expected SimpleConfig, got {type(result)}"
        assert result.name == "test_config", f"Rank {rank}: Expected name='test_config', got '{result.name}'"
        assert result.value == 42, f"Rank {rank}: Expected value=42, got {result.value}"
        
        result_queue.put((rank, True, None))
    except Exception as e:
        result_queue.put((rank, False, str(e)))


def worker_process_complex(rank: int, world_size: int, endpoint: str, result_queue: mp.Queue):
    """Worker process for complex broadcast test"""
    # Configure logging for this worker process
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Worker {rank}] %(asctime)s - %(levelname)s - %(message)s',
        force=True
    )
    
    try:
        broadcaster = ZMQBroadcaster(rank=rank, world_size=world_size, endpoint=endpoint)
        
        if rank == 0:
            # Rank 0 sends complex config
            config = ComplexConfig(
                model_name="gpt-3",
                batch_size=32,
                learning_rate=0.001,
                layers=[128, 256, 512],
                metadata={"version": "1.0", "author": "test"}
            )
            result = broadcaster.broadcast(config)
        else:
            # Other ranks receive
            result = broadcaster.broadcast(None)
        
        broadcaster.close()
        
        # Verify result
        assert isinstance(result, ComplexConfig), f"Rank {rank}: Expected ComplexConfig, got {type(result)}"
        assert result.model_name == "gpt-3"
        assert result.batch_size == 32
        assert result.learning_rate == 0.001
        assert result.layers == [128, 256, 512]
        assert result.metadata == {"version": "1.0", "author": "test"}
        
        result_queue.put((rank, True, None))
    except Exception as e:
        result_queue.put((rank, False, str(e)))


def worker_process_dict(rank: int, world_size: int, endpoint: str, result_queue: mp.Queue):
    """Worker process for dict broadcast test"""
    # Configure logging for this worker process
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Worker {rank}] %(asctime)s - %(levelname)s - %(message)s',
        force=True
    )
    
    try:
        broadcaster = ZMQBroadcaster(rank=rank, world_size=world_size, endpoint=endpoint)
        
        if rank == 0:
            # Rank 0 sends a dict
            data = {"key1": "value1", "key2": 123, "key3": [1, 2, 3]}
            result = broadcaster.broadcast(data)
        else:
            # Other ranks receive
            result = broadcaster.broadcast(None)
        
        broadcaster.close()
        
        # Verify result
        assert isinstance(result, dict), f"Rank {rank}: Expected dict, got {type(result)}"
        assert result["key1"] == "value1"
        assert result["key2"] == 123
        assert result["key3"] == [1, 2, 3]
        
        result_queue.put((rank, True, None))
    except Exception as e:
        result_queue.put((rank, False, str(e)))


def worker_process_multiple_broadcasts(rank: int, world_size: int, endpoint: str, result_queue: mp.Queue):
    """Worker process for multiple sequential broadcasts"""
    # Configure logging for this worker process
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Worker {rank}] %(asctime)s - %(levelname)s - %(message)s',
        force=True
    )
    
    try:
        broadcaster = ZMQBroadcaster(rank=rank, world_size=world_size, endpoint=endpoint)
        
        # First broadcast
        if rank == 0:
            config1 = SimpleConfig(name="first", value=1)
            result1 = broadcaster.broadcast(config1)
        else:
            result1 = broadcaster.broadcast(None)
        
        assert result1.name == "first"
        assert result1.value == 1
        
        # Second broadcast (no sleep needed - barrier handles synchronization)
        if rank == 0:
            config2 = SimpleConfig(name="second", value=2)
            result2 = broadcaster.broadcast(config2)
        else:
            result2 = broadcaster.broadcast(None)
        
        assert result2.name == "second"
        assert result2.value == 2
        
        # Third broadcast
        if rank == 0:
            config3 = SimpleConfig(name="third", value=3)
            result3 = broadcaster.broadcast(config3)
        else:
            result3 = broadcaster.broadcast(None)
        
        assert result3.name == "third"
        assert result3.value == 3
        
        broadcaster.close()
        result_queue.put((rank, True, None))
    except Exception as e:
        result_queue.put((rank, False, str(e)))


def worker_process_context_manager(rank: int, world_size: int, endpoint: str, result_queue: mp.Queue):
    """Worker process testing context manager usage"""
    # Configure logging for this worker process
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Worker {rank}] %(asctime)s - %(levelname)s - %(message)s',
        force=True
    )
    
    try:
        with ZMQBroadcaster(rank=rank, world_size=world_size, endpoint=endpoint) as broadcaster:
            if rank == 0:
                config = SimpleConfig(name="context_test", value=99)
                result = broadcaster.broadcast(config)
            else:
                result = broadcaster.broadcast(None)
            
            assert result.name == "context_test"
            assert result.value == 99
        
        # Broadcaster should be closed after context manager
        result_queue.put((rank, True, None))
    except Exception as e:
        result_queue.put((rank, False, str(e)))


def worker_process_large(rank: int, world_size: int, endpoint: str, result_queue: mp.Queue):
    """Worker process for large object broadcast test"""
    # Configure logging for this worker process
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Worker {rank}] %(asctime)s - %(levelname)s - %(message)s',
        force=True
    )
    
    try:
        broadcaster = ZMQBroadcaster(rank=rank, world_size=world_size, endpoint=endpoint)
        
        if rank == 0:
            # Create large config with lots of data
            large_data = {
                "data": [i for i in range(10000)],
                "text": "x" * 10000,
                "nested": {"key" + str(i): i for i in range(1000)}
            }
            result = broadcaster.broadcast(large_data)
        else:
            result = broadcaster.broadcast(None)
        
        broadcaster.close()
        
        # Verify
        assert len(result["data"]) == 10000
        assert len(result["text"]) == 10000
        assert len(result["nested"]) == 1000
        
        result_queue.put((rank, True, None))
    except Exception as e:
        result_queue.put((rank, False, str(e)))


def worker_process_none(rank: int, world_size: int, endpoint: str, result_queue: mp.Queue):
    """Worker process for None broadcast test"""
    # Configure logging for this worker process
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Worker {rank}] %(asctime)s - %(levelname)s - %(message)s',
        force=True
    )
    
    try:
        broadcaster = ZMQBroadcaster(rank=rank, world_size=world_size, endpoint=endpoint)
        
        if rank == 0:
            result = broadcaster.broadcast(None)
        else:
            result = broadcaster.broadcast(None)
        
        broadcaster.close()
        assert result is None
        result_queue.put((rank, True, None))
    except Exception as e:
        result_queue.put((rank, False, str(e)))


def worker_process_empty_dict(rank: int, world_size: int, endpoint: str, result_queue: mp.Queue):
    """Worker process for empty dict broadcast test"""
    # Configure logging for this worker process
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Worker {rank}] %(asctime)s - %(levelname)s - %(message)s',
        force=True
    )
    
    try:
        broadcaster = ZMQBroadcaster(rank=rank, world_size=world_size, endpoint=endpoint)
        
        if rank == 0:
            result = broadcaster.broadcast({})
        else:
            result = broadcaster.broadcast(None)
        
        broadcaster.close()
        assert result == {}
        result_queue.put((rank, True, None))
    except Exception as e:
        result_queue.put((rank, False, str(e)))


def worker_process_len_test(rank: int, world_size: int, endpoint: str, result_queue: mp.Queue):
    """Worker process for testing __len__ function"""
    # Configure logging for this worker process
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Worker {rank}] %(asctime)s - %(levelname)s - %(message)s',
        force=True
    )
    
    try:
        broadcaster = ZMQBroadcaster(rank=rank, world_size=world_size, endpoint=endpoint)
        
        # Test __len__ method
        assert len(broadcaster) == world_size, f"Rank {rank}: Expected len={world_size}, got {len(broadcaster)}"
        
        # Also test a simple broadcast
        if rank == 0:
            config = SimpleConfig(name="len_test", value=len(broadcaster))
            result = broadcaster.broadcast(config)
        else:
            result = broadcaster.broadcast(None)
        
        assert result.value == world_size
        
        broadcaster.close()
        result_queue.put((rank, True, None))
    except Exception as e:
        result_queue.put((rank, False, str(e)))


class TestZMQBroadcaster(unittest.TestCase):
    """Test suite for ZMQBroadcaster"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.base_port = 5555
        self.endpoint_counter = 0
    
    def get_endpoint(self):
        """Get a unique endpoint for each test"""
        endpoint = f"tcp://127.0.0.1:{self.base_port + self.endpoint_counter}"
        self.endpoint_counter += 1
        return endpoint
    
    def run_multi_process_test(self, worker_func, world_size=4, timeout=10):
        """
        Helper to run multi-process tests
        
        Args:
            worker_func: Worker function to run in each process
            world_size: Number of processes
            timeout: Timeout in seconds
        
        Returns:
            List of (rank, success, error_msg) tuples
        """
        endpoint = self.get_endpoint()
        result_queue = mp.Queue()
        processes = []
        
        # Start all worker processes
        for rank in range(world_size):
            p = mp.Process(target=worker_func, args=(rank, world_size, endpoint, result_queue))
            p.start()
            processes.append(p)
        
        # Collect results
        results = []
        for _ in range(world_size):
            try:
                result = result_queue.get(timeout=timeout)
                results.append(result)
            except mp.queues.Empty:
                self.fail(f"Timeout waiting for worker results after {timeout}s")
        
        # Wait for all processes to complete
        for p in processes:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
                p.join()
        
        return results
    
    def test_simple_broadcast(self):
        """Test broadcasting a simple Pydantic model"""
        results = self.run_multi_process_test(worker_process_simple, world_size=4)
        
        # Check all workers succeeded
        for rank, success, error_msg in results:
            self.assertTrue(success, f"Rank {rank} failed: {error_msg}")
    
    def test_complex_broadcast(self):
        """Test broadcasting a complex Pydantic model with nested data"""
        results = self.run_multi_process_test(worker_process_complex, world_size=4)
        
        for rank, success, error_msg in results:
            self.assertTrue(success, f"Rank {rank} failed: {error_msg}")
    
    def test_dict_broadcast(self):
        """Test broadcasting a plain dictionary"""
        results = self.run_multi_process_test(worker_process_dict, world_size=3)
        
        for rank, success, error_msg in results:
            self.assertTrue(success, f"Rank {rank} failed: {error_msg}")
    
    def test_multiple_broadcasts(self):
        """Test multiple sequential broadcasts"""
        results = self.run_multi_process_test(worker_process_multiple_broadcasts, world_size=3, timeout=15)
        
        for rank, success, error_msg in results:
            self.assertTrue(success, f"Rank {rank} failed: {error_msg}")
    
    def test_context_manager(self):
        """Test using ZMQBroadcaster as a context manager"""
        results = self.run_multi_process_test(worker_process_context_manager, world_size=3)
        
        for rank, success, error_msg in results:
            self.assertTrue(success, f"Rank {rank} failed: {error_msg}")
    
    def test_two_ranks(self):
        """Test with minimum number of ranks (2)"""
        results = self.run_multi_process_test(worker_process_simple, world_size=2)
        
        for rank, success, error_msg in results:
            self.assertTrue(success, f"Rank {rank} failed: {error_msg}")
    
    def test_many_ranks(self):
        """Test with many ranks (8)"""
        results = self.run_multi_process_test(worker_process_simple, world_size=8, timeout=15)
        
        for rank, success, error_msg in results:
            self.assertTrue(success, f"Rank {rank} failed: {error_msg}")
    
    def test_large_object(self):
        """Test broadcasting a large object"""
        results = self.run_multi_process_test(worker_process_large, world_size=3, timeout=15)
        
        for rank, success, error_msg in results:
            self.assertTrue(success, f"Rank {rank} failed: {error_msg}")

    """Test edge cases and error handling"""
    def test_none_broadcast(self):
        """Test broadcasting None value"""
        results = self.run_multi_process_test(worker_process_none, world_size=2, timeout=10)
        
        for rank, success, error_msg in results:
            self.assertTrue(success, f"Rank {rank} failed: {error_msg}")
    
    def test_empty_dict(self):
        """Test broadcasting an empty dictionary"""
        results = self.run_multi_process_test(worker_process_empty_dict, world_size=2, timeout=10)
        
        for rank, success, error_msg in results:
            self.assertTrue(success, f"Rank {rank} failed: {error_msg}")
    
    def test_broadcaster_len(self):
        """Test __len__ method returns world_size"""
        results = self.run_multi_process_test(worker_process_len_test, world_size=4, timeout=10)
        
        for rank, success, error_msg in results:
            self.assertTrue(success, f"Rank {rank} failed: {error_msg}")


if __name__ == "__main__":
    # Set multiprocessing start method (important for compatibility)
    mp.set_start_method('spawn', force=True)
    
    # Configure logging for the main process
    logging.basicConfig(
        level=logging.INFO,
        format='[Main] %(asctime)s - %(levelname)s - %(message)s',
        force=True
    )
    
    print("\n" + "="*70)
    print("ZMQBroadcaster Test Suite")
    print("="*70)
    print("\nTesting ZMQ-based broadcasting with Pydantic models")
    print("Using multiprocessing to simulate distributed environment\n")
    print("="*70 + "\n")
    
    # Run tests with verbose output
    unittest.main(verbosity=2)
