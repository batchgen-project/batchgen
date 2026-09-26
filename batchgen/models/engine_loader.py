# Preload the pip libucx-cu12 UCX libraries so core_engine's distributed-weight
# daemon resolves libucp/libuct/libucs/libucm at load time.  A missing or broken
# UCX runtime is a deployment error; do not silently continue and let a worker
# discover it after startup.
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
