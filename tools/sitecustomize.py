import os

if os.getenv("V4_COLL_TRACE", "0") == "1":
    try:
        import v4_collective_tracer  # noqa: F401
    except Exception:
        try:
            from tools import v4_collective_tracer  # noqa: F401
        except Exception:
            pass
