import asyncio
from contextlib import asynccontextmanager

import uvicorn
import uvloop
from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse

from batchgen.managers.io_struct import (
    BatchGenerateRequest,
    BatchGenerateResponse,
)
from batchgen.server_args import ServerArgs

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


@asynccontextmanager
async def lifespan(app: FastAPI):
    # initialize state
    yield
    # cleanup state


app = FastAPI(lifespan=lifespan)


@app.post(
    "/batch_generate",
    response_model=BatchGenerateResponse,
    response_class=ORJSONResponse,
)
async def batch_generate(obj: BatchGenerateRequest, request: Request):
    """
    Handle batch generation requests.
    """
    input_texts: list[str] = obj.input_texts
    max_new_tokens: int = obj.max_new_tokens
    # pass
    response = BatchGenerateResponse(
        generated_texts=[f"Generated text for: {text}" for text in input_texts],
        status="success",
        message=None,
    )
    return response


@app.get("/health")
async def health_check():
    return {"status": "ok"}


def launch_server(server_args: ServerArgs):
    app.server_args = server_args
    uvicorn.run(
        app,
        host=server_args.host,
        port=server_args.port,
        log_level="info",
        timeout_keep_alive=300,
        loop="uvloop",
    )




