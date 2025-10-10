import os
import sys

from batchgen.entrypoints.http_server import launch_server
from batchgen.server_args import prepare_server_args
from batchgen.utils import kill_process_tree

if __name__ == "__main__":
    # Add the parent directory to sys.path to allow imports from batchgen
    server_args = prepare_server_args(sys.argv[1:])

    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=True)
