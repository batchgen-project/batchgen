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

from batchgen.server.http_server import launch_server
from batchgen.server.server_args import prepare_server_args


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [BatchGenServer] - %(levelname)s - %(message)s",
    )

    try:
        server_args = prepare_server_args()
    except ValueError as exc:
        logging.error(f"Invalid arguments: {exc}")
        sys.exit(1)
    except Exception as exc:
        logging.error(f"Failed to parse arguments: {exc}")
        sys.exit(1)

    logging.info(f"Starting BatchGen HTTP server for model: {server_args.model}")
    logging.info(f"Listen address: {server_args.listen_ip}:{server_args.listen_port}")
    logging.info(f"Watchdog timeout: {server_args.watchdog_timeout}s")

    try:
        launch_server(server_args)
    except KeyboardInterrupt:
        logging.info("Server shutdown requested")
    except RuntimeError as exc:
        logging.error(f"Server error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
