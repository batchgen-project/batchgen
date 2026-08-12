"""The single place the /v1/inference deprecation is spelled out.

Three modules must agree on this text and cannot import one another:
`batchgen_client` (imported by `batchgen/__init__`, so it must carry no heavy
dependencies), `server/http_server` (FastAPI + pydantic) and `batchgen_worker`
(torch). A module that imports nothing at all is the only seam the three of
them can share.
"""

LEGACY_INFERENCE_ERROR_CODE = "legacy_inference_deprecated"

LEGACY_INFERENCE_MESSAGE = (
    "/v1/inference is deprecated and disabled. Submit inference through the "
    "batch API instead: POST /v1/files (purpose=batch), then POST /v1/batches. "
    "The legacy path carried no request-id routing: on a pool-mode server it "
    "parked in the worker admission loop and then took the next completion off "
    "the shared response queue, corrupting a concurrent batch."
)


class LegacyInferenceDeprecated(RuntimeError):
    """Raised wherever a /v1/inference-shaped request is refused."""

    def __init__(self, message: str = LEGACY_INFERENCE_MESSAGE) -> None:
        super().__init__(message)
        self.code = LEGACY_INFERENCE_ERROR_CODE
