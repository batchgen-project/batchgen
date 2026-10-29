"""Native extension load gate for the BatchGen runtime.

Every component that needs ``batchgen.core_engine`` (server, worker, KV cache
and model packages) imports it through this module, so the native runtime is
validated once, before any caller touches it:

* the pip ``libucx-cu12`` UCX libraries are preloaded so core_engine's
  distributed-weight daemon resolves libucp/libuct/libucs/libucm at load time.
  A missing or broken UCX runtime is a deployment error; do not silently
  continue and let a worker discover it after startup.
* ``core_engine`` must be an ahead-of-time built native module; implicit JIT
  compilation during server startup is refused.
"""

try:
    import libucx

    libucx.load_library()
except Exception as exc:
    raise RuntimeError(
        "UCX runtime preflight failed; install and load libucx-cu12 before "
        "starting BatchGen"
    ) from exc

try:
    from batchgen import core_engine
except ImportError as exc:
    raise RuntimeError(
        "AOT batchgen.core_engine is unavailable; refusing implicit JIT "
        "compilation during server startup"
    ) from exc

if not str(getattr(core_engine, "__file__", "")).endswith((".so", ".pyd", ".dylib")):
    raise RuntimeError(
        "batchgen.core_engine is not an AOT native module; refusing implicit JIT "
        "compilation during server startup"
    )
