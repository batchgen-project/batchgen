import torch
import math
import logging
import tqdm
from typing import Optional, List, Dict, Union
import torch.distributed as dist


"""
	Data structure to manage queries including:
	1. The streaming query pool keep receiving incoming queries.
	2. For each query, we keep records of its status, e.g., prefill done or not, decode done or not, decoding step, etc.
"""

class QueryStatus:
    """
		TBD
	"""
    def __init__(self, query_id: int, string: str):
        self.query_id = query_id
        self.string = string
        self.prefill_done = False
        self.decode_done = False
        self.decode_step = 0
        self.max_decode_step = 0

class QueryManager:
    def __init__(self):
        self.query_pool = {}
        self._next_id = 1
        self._free_ids = set()
    
    def add_query(self, query):
        """Add query with ID recycling."""
        if isinstance(query, str):
            # Get ID from free pool or generate new
            if self._free_ids:
                query_id = min(self._free_ids)
                self._free_ids.remove(query_id)
            else:
                query_id = self._next_id
                self._next_id += 1
            
            self.query_pool[query_id] = QueryStatus(query_id, query)
            return query_id
            
        elif isinstance(query, list):
            query_ids = []
            for q in query:
                if self._free_ids:
                    query_id = min(self._free_ids)
                    self._free_ids.remove(query_id)
                else:
                    query_id = self._next_id
                    self._next_id += 1
                
                self.query_pool[query_id] = QueryStatus(query_id, q)
                query_ids.append(query_id)
            return query_ids
        
        return None
    
    def remove_query(self, query_id: int) -> bool:
        """Remove query and free its ID."""
        if query_id in self.query_pool:
            del self.query_pool[query_id]
            self._free_ids.add(query_id)
            return True
        return False
	

