import argparse
import dataclasses
import os
import tempfile
from typing import Optional
from pathlib import Path

from batchgen.utils import is_port_available


@dataclasses.dataclass
class ServerArgs:
    model_path: str
    file_path: Optional[str] = None

    # http server
    host: str = "127.0.0.1"
    port: int = 40000

    # Multi-node distributed serving
    dist_init_addr: Optional[str] = None
    nnodes: int = 1
    node_rank: int = 0

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model_path",
            type=str,
            required=True,
            help="Path to the model checkpoint or model name",
        )
        parser.add_argument(
            "--file_path",
            type=str,
            default=None,
            help="Path to the file storage directory",
        )
        parser.add_argument(
            "--host",
            type=str,
            default=ServerArgs.host,
            help="Host for the HTTP server",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=ServerArgs.port,
            help="Port for the HTTP server",
        )

        # Multi-node distributed serving
        parser.add_argument(
            "--dist-init-addr",
            type=str,
            help="The host address for initializing distributed backend (e.g., `192.168.0.2:25000`).",
        )
        parser.add_argument(
            "--nnodes",
            type=int,
            default=ServerArgs.nnodes,
            help="The number of nodes.",
        )
        parser.add_argument(
            "--node-rank",
            type=int,
            default=ServerArgs.node_rank,
            help="The node rank.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "ServerArgs":
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})
    
    def __post_init__(self):
        if not is_port_available(self.port):
            raise ValueError(f"Port {self.port} is not available. Please choose another port.")
        
        # if file_path is not provided, use a model path
        if not self.file_path:
            self.file_path = os.path.join(os.path.dirname(self.model_path), "files")

        self.file_path = Path(os.path.abspath(self.file_path))

def prepare_server_args(argv: list[str]) -> ServerArgs:
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    raw_args = parser.parse_args(argv)
    server_args = ServerArgs.from_cli_args(raw_args)
    return server_args


ZMQ_TCP_PORT_DELTA = 233


@dataclasses.dataclass
class PortArgs:
    # The ipc filename for scheduler (rank 0) (zmq)
    scheduler_input_ipc_name: str
    scheduler_output_ipc_name: str

    @staticmethod
    def init_new(server_args: ServerArgs) -> "PortArgs":
        if server_args.nnodes == 1 and server_args.dist_init_addr is None:
            return PortArgs(
                scheduler_input_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
                scheduler_output_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            )

        dist_init_addr = server_args.dist_init_addr.split(":")

        assert len(dist_init_addr) == 2, (
            "please provide --dist-init-addr as host:port of head node"
        )

        dist_init_host, dist_init_port = dist_init_addr
        port_base = int(dist_init_port) + 1

        scheduler_input_port = port_base + 4
        scheduler_output_port = port_base + 5
        return PortArgs(
            scheduler_input_ipc_name=f"tcp://{dist_init_host}:{scheduler_input_port}",
            scheduler_output_ipc_name=f"tcp://{dist_init_host}:{scheduler_output_port}",
        )
