"""Worker handler package — incremental decouple of `batchgen.batchgen_worker`.

Each module here owns one cohesive subsystem extracted from the worker
monolith. Handlers follow the Phase A/B/C cuda-graph contract pattern:

  - Inputs arrive as frozen ``@dataclass`` snapshots; the worker is the
    single source of truth for canonical state and the only mutator.
  - Handler methods return plain values or frozen ``*Plan`` dataclasses
    describing decisions; the worker applies them.
  - No shared mutable ``WorkerState`` container — each handler call
    receives exactly the snapshot it needs.

Slice tracking: milestone "Worker decouple"
(https://github.com/batchgen-project/batchgen/milestone/2).
"""
