"""Entry point for the BatchGen HTTP server with OpenAI-compatible API.

Usage:
    python -m batchgen.launch_http_server --model deepseek-ai/DeepSeek-V3 [options]

This launches the FastAPI server with:
- OpenAI-compatible batch API endpoints
- Direct inference endpoint
- Worker health monitoring via watchdog
"""

import logging
import sys

from batchgen.eventlog.process import (
    configure_process,
    new_run_identity,
    recording_exit,
)
from batchgen.server.server_args import prepare_server_args
from batchgen.runtime_preflight import RuntimePreflightError, run_runtime_preflight


def main():
    # Nothing is configured yet: a parse failure reaches stderr through
    # logging.lastResort, and the event log starts once the args are known.
    try:
        server_args = prepare_server_args()
    except ValueError as exc:
        logging.error(f"Invalid arguments: {exc}")
        sys.exit(1)
    except Exception as exc:
        logging.error(f"Failed to parse arguments: {exc}")
        sys.exit(1)

    # Node 0 names the run directory; every other node buffers its records
    # until it learns that name from node 0 over the restart control store.
    node_rank = server_args.node_rank
    run = (
        new_run_identity(server_args.runtime_identity.run_id)
        if node_rank == 0
        else None
    )
    configure_process(
        "server",
        node=node_rank,
        run=run,
        log_dir=server_args.log_dir,
        level=server_args.log_level,
        fallback_run_id=server_args.runtime_identity.run_id,
    )

    # A failed startup (uvicorn exits 3) leaves a process.exit record, and a
    # node other than 0 that ends before node 0 names the run still writes its
    # buffered records, under a run of its own.
    with recording_exit():
        _serve(server_args)


def _serve(server_args) -> None:
    logging.info(f"Starting BatchGen HTTP server for model: {server_args.model}")
    logging.info(f"Listen address: {server_args.listen_ip}:{server_args.listen_port}")
    logging.info(f"Watchdog timeout: {server_args.watchdog_timeout}s")

    try:
        # Run before importing worker_manager/http_server so missing native
        # dependencies fail before workers, SHM, ports, or model weights exist.
        run_runtime_preflight(server_args)
        from batchgen.server.http_server import launch_server

        launch_server(server_args)
    except RuntimePreflightError as exc:
        logging.error("Runtime preflight failed: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logging.info("Server shutdown requested")
    except RuntimeError as exc:
        logging.error(f"Server error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
