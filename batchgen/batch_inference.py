"""
	This is the entry-point of BatchGen. 
	All the components(inference_runtime, core_engine, scheduler etc.) are initialized and orchestrated here.
	Each node would instantiate one BatchInference object to run the batch inference.
	All the node would accept batches from http server which by default runs on node 0.
"""

import torch
import math
import logging
import tqdm
from typing import Optional, List, Dict
import torch.distributed as dist
from dataclasses import dataclass


class BatchInference:
	def __init__(self):
		pass

	def init(self, **kwargs):
		"""
		Initialize all the components needed for batch inference.
		"""
		pass

	def run(self):
		"""
		Main loop to run the batch inference.
		1. Get the next batch from the scheduler.
		2. Depending on the phase (prefill or decode), call the appropriate method in InferenceRuntime.
		3. Update the scheduler with the results.
		"""
		pass

	def shutdown(self):
		"""
		Clean up resources and shutdown the inference runtime.
		"""
		pass


