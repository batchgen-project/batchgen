"""BatchGen HTTP server module with OpenAI-compatible API and watchdog."""

# Lazy imports to avoid circular dependency:
# server_worker_main_loop imports server.process_utils
# server.__init__ imports http_server -> batch_scheduler -> worker_manager -> server_worker_main_loop


def create_app(*args, **kwargs):
    from .http_server import create_app as _create_app
    return _create_app(*args, **kwargs)


def launch_server(*args, **kwargs):
    from .http_server import launch_server as _launch_server
    return _launch_server(*args, **kwargs)


from .watchdog import Watchdog
from .server_args import ServerArgs

__all__ = [
    "create_app",
    "launch_server",
    "Watchdog",
    "ServerArgs",
]
