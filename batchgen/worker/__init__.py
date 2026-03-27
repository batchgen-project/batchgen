# batchgen/worker/ — Composable scheduler for BatchGen inference engine
#
# Architecture:
#   BatchGenWorker (thin orchestrator) delegates to:
#   - WorkerState: central shared state
#   - IndexManager: local↔uuid↔global index mappings
#   - SyncCoordinator: cross-rank collective operations
#   - KVCacheManager: GPU + host KV lifecycle (step 5)
#   - BatchFormation: tokenization + assignment (step 6)
#   - PrefillScheduler: prefill batch selection + execution (step 7)
#   - BoundaryHandler: page boundary decisions (step 8)
#   - DecodeScheduler: decode loop + strategy pattern (step 9)
#   - HostKVRebalancer: migration + eviction (step 10)
#   - CompletionHandler: EOS detection (step 11)
#
# Migration status: Steps 1-10 (+ host_rebalancer)
# The main BatchGenWorker class remains in batchgen/batchgen_worker.py
# and delegates to these sub-managers.

from batchgen.worker.state import WorkerState
from batchgen.worker.indexing import IndexManager
from batchgen.worker.sync import SyncCoordinator
from batchgen.worker.kv_manager import KVCacheManager
from batchgen.worker.batch_formation import BatchFormation
from batchgen.worker.prefill import PrefillScheduler
from batchgen.worker.boundary import BoundaryHandler
from batchgen.worker.decode import DecodeScheduler
from batchgen.worker.host_rebalancer import HostKVRebalancer
