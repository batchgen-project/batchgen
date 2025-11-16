import torch
import math
import logging
import tqdm
from typing import Optional, List, Dict
import torch.distributed as dist
from dataclasses import dataclass

"""
	This class handle the schedule of the decode and the prefill phase of BatchGen.
	It keeps track of status of all the sequences and the status of the nodes.
	It decides when to prefill, decode and what would be the batch to send to each node.
"""

@dataclass
class NodeStatus:
	node_id: int
	host_kv_pool_size: int,
	host_kv_pool_size_used: int

# state machine: idle, prefill and decode
class SchedulerState:
	IDLE = "idle"
	PREFILL = "prefill"
	DECODE = "decode"

class Scheduler:
	def __init__(self):
		self.state = SchedulerState.IDLE
		self.node_statuses = {}

	def update_node_status(self, node_id: int, kv_pool_size: int, kv_pool_size_used: int):
		self.node_statuses[node_id] = NodeStatus(
			node_id=node_id,
			host_kv_pool_size=kv_pool_size,
			host_kv_pool_size_used=kv_pool_size_used
		)
	
	def schedule(self) -> SchedulerState:
		"""
		Based on the current state and node statuses, decide the next action:
		- If all nodes are idle, decide to prefill or decode based on the workload.
		- If some nodes are busy, wait or adjust the workload distribution.
		"""
		pass

	
