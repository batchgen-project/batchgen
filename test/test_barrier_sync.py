#!/usr/bin/env python3
"""
Test suite for ZMQ Barrier Synchronization using unittest
"""
import unittest
import multiprocessing as mp
import logging
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pydantic import BaseModel, Field
from batchgen.utils import ZMQBroadcaster


class TestConfig(BaseModel):
    """Test configuration model"""
    name: str
    value: int


def worker_rapid_broadcast(rank: int, world_size: int, endpoint: str, result_queue: mp.Queue):
    """Worker process for rapid successive broadcasts without sleep"""
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Rank {rank}] %(asctime)s - %(message)s',
        force=True
    )
    
    try:
        broadcaster = ZMQBroadcaster(rank=rank, world_size=world_size, endpoint=endpoint)
        logging.info(f"Broadcaster created, len={len(broadcaster)}")
        
        # Test multiple broadcasts without any sleep - barrier should handle synchronization
        num_broadcasts = 5
        for i in range(num_broadcasts):
            if rank == 0:
                config = TestConfig(name=f"test_{i}", value=i * 10)
                logging.info(f"Broadcasting #{i}: {config}")
                result = broadcaster.broadcast(config)
            else:
                result = broadcaster.broadcast(None)
                logging.info(f"Received #{i}: {result}")
            
            assert result.name == f"test_{i}", f"Expected name=test_{i}, got {result.name}"
            assert result.value == i * 10, f"Expected value={i*10}, got {result.value}"
        
        broadcaster.close()
        logging.info("Test passed!")
        result_queue.put((rank, True, None))
    except Exception as e:
        logging.error(f"Test failed: {e}")
        result_queue.put((rank, False, str(e)))


def worker_timing_test(rank: int, world_size: int, endpoint: str, result_queue: mp.Queue):
    """Worker process to verify no sleep is needed between broadcasts"""
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Rank {rank}] %(asctime)s - %(message)s',
        force=True
    )
    
    try:
        start_time = time.time()
        broadcaster = ZMQBroadcaster(rank=rank, world_size=world_size, endpoint=endpoint)
        
        # Perform 10 broadcasts back-to-back with no sleep
        for i in range(10):
            if rank == 0:
                config = TestConfig(name=f"timing_{i}", value=i)
                result = broadcaster.broadcast(config)
            else:
                result = broadcaster.broadcast(None)
            
            assert result.name == f"timing_{i}"
            assert result.value == i
        
        broadcaster.close()
        elapsed = time.time() - start_time
        
        logging.info(f"Completed 10 broadcasts in {elapsed:.3f}s")
        result_queue.put((rank, True, elapsed))
    except Exception as e:
        result_queue.put((rank, False, str(e)))


def worker_barrier_verification(rank: int, world_size: int, endpoint: str, result_queue: mp.Queue):
    """Worker process to verify barrier ensures all ranks are synchronized"""
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Rank {rank}] %(asctime)s - %(message)s',
        force=True
    )
    
    try:
        broadcaster = ZMQBroadcaster(rank=rank, world_size=world_size, endpoint=endpoint)
        
        # Record when each rank receives each broadcast
        receive_times = []
        
        for i in range(3):
            if rank == 0:
                config = TestConfig(name=f"barrier_{i}", value=i)
                result = broadcaster.broadcast(config)
                receive_time = time.time()
            else:
                result = broadcaster.broadcast(None)
                receive_time = time.time()
            
            receive_times.append(receive_time)
            assert result.name == f"barrier_{i}"
            
            # After broadcast returns, all ranks should be synchronized
            # So we can immediately proceed without race conditions
            logging.info(f"Synchronized after broadcast #{i}")
        
        broadcaster.close()
        result_queue.put((rank, True, receive_times))
    except Exception as e:
        result_queue.put((rank, False, str(e)))


def worker_stress_test(rank: int, world_size: int, endpoint: str, result_queue: mp.Queue):
    """Worker process for stress test with many rapid broadcasts"""
    logging.basicConfig(
        level=logging.WARNING,  # Less verbose for stress test
        format=f'[Rank {rank}] %(message)s',
        force=True
    )
    try:
        broadcaster = ZMQBroadcaster(rank=rank, world_size=world_size, endpoint=endpoint)
        
        # 20 rapid broadcasts
        for i in range(20):
            if rank == 0:
                config = TestConfig(name=f"stress_{i}", value=i)
                result = broadcaster.broadcast(config)
            else:
                result = broadcaster.broadcast(None)
            
            assert result.name == f"stress_{i}"
        
        broadcaster.close()
        result_queue.put((rank, True, None))
    except Exception as e:
        result_queue.put((rank, False, str(e)))


class TestZMQBarrierSync(unittest.TestCase):
    """Test suite for ZMQ barrier synchronization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.base_port = 6000
        self.endpoint_counter = 0
    
    def get_endpoint(self):
        """Get a unique endpoint for each test"""
        endpoint = f"tcp://127.0.0.1:{self.base_port + self.endpoint_counter}"
        self.endpoint_counter += 1
        return endpoint
    
    def run_multi_process_test(self, worker_func, world_size=4, timeout=15):
        """
        Helper to run multi-process tests
        
        Args:
            worker_func: Worker function to run in each process
            world_size: Number of processes
            timeout: Timeout in seconds
        
        Returns:
            List of (rank, success, data) tuples
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
    
    def test_rapid_successive_broadcasts(self):
        """Test that multiple broadcasts work without sleep delays"""
        results = self.run_multi_process_test(worker_rapid_broadcast, world_size=4)
        
        # Check all workers succeeded
        for rank, success, error_msg in results:
            self.assertTrue(success, f"Rank {rank} failed: {error_msg}")
    
    def test_no_sleep_needed(self):
        """Test that broadcasts complete quickly without artificial delays"""
        results = self.run_multi_process_test(worker_timing_test, world_size=4, timeout=20)
        
        # Check all workers succeeded
        for rank, success, data in results:
            self.assertTrue(success, f"Rank {rank} failed: {data}")
            if success:
                elapsed = data
                # 10 broadcasts should complete quickly (well under 5 seconds)
                # If we needed sleep(0.1) between each, it would take at least 1 second
                self.assertLess(elapsed, 5.0, 
                    f"Rank {rank} took {elapsed}s for 10 broadcasts - barrier may not be working")
    
    def test_barrier_synchronization(self):
        """Test that barrier ensures all ranks are synchronized after each broadcast"""
        results = self.run_multi_process_test(worker_barrier_verification, world_size=4)
        
        # Check all workers succeeded
        for rank, success, data in results:
            self.assertTrue(success, f"Rank {rank} failed: {data}")
    
    def test_two_ranks_rapid(self):
        """Test rapid broadcasts with minimum world_size"""
        results = self.run_multi_process_test(worker_rapid_broadcast, world_size=2)
        
        for rank, success, error_msg in results:
            self.assertTrue(success, f"Rank {rank} failed: {error_msg}")
    
    def test_many_ranks_rapid(self):
        """Test rapid broadcasts with many ranks"""
        results = self.run_multi_process_test(worker_rapid_broadcast, world_size=6, timeout=20)
        
        for rank, success, error_msg in results:
            self.assertTrue(success, f"Rank {rank} failed: {error_msg}")
    
    def test_stress_test(self):
        """Stress test with many rapid broadcasts"""
        results = self.run_multi_process_test(worker_stress_test, world_size=4, timeout=30)
        
        for rank, success, error_msg in results:
            self.assertTrue(success, f"Rank {rank} failed: {error_msg}")


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Configure logging for main process
    logging.basicConfig(
        level=logging.INFO,
        format='[Main] %(asctime)s - %(message)s',
        force=True
    )
    
    print("\n" + "="*70)
    print("ZMQ Barrier Synchronization Test Suite")
    print("="*70)
    print("\nTesting that barrier synchronization eliminates need for sleep delays")
    print("="*70 + "\n")
    
    # Run tests with verbose output
    unittest.main(verbosity=2)
