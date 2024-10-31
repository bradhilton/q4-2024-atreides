import asyncio
from openai import AsyncOpenAI
import os
import random
import socket
from typing import Any, Callable
from uvicorn.config import LOGGING_CONFIG

os.environ["VLLM_LOGGING_CONFIG_PATH"] = __file__.replace(
    "vllm.py", "vllm-logging-config.json"
)
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils import FlexibleArgumentParser


async def start_vllm_server(
    uvicorn_log_file="./logs/vllm.log",
    kill_competing_processes: bool = True,
    **kwargs: Any,
) -> tuple[Callable[[], bool], AsyncOpenAI]:
    if "api_key" not in kwargs:
        kwargs["api_key"] = "vllm-" + "".join(random.choices("0123456789abcdef", k=32))
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args(args=[])
    for key, value in kwargs.items():
        setattr(args, key, value)
    validate_parsed_serve_args(args)
    with open(uvicorn_log_file, "w"):
        pass
    LOGGING_CONFIG["handlers"]["default"] = {
        "class": "logging.FileHandler",
        "formatter": "default",
        "filename": uvicorn_log_file,
        "mode": "a",
    }
    LOGGING_CONFIG["handlers"]["access"] = {
        "class": "logging.FileHandler",
        "formatter": "access",
        "filename": uvicorn_log_file,
        "mode": "a",
    }
    if kill_competing_processes:
        os.system(
            f"lsof -ti :{args.port} | grep -v {os.getpid()} | xargs kill -9 2>/dev/null || true"
        )
    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((args.host or "0.0.0.0", args.port))
            break
        except socket.error:
            if "port" in kwargs:
                raise RuntimeError(f"Port {args.port} is already in use")
            args.port += 1
        finally:
            sock.close()
    server_task = asyncio.create_task(run_server(args, log_config=LOGGING_CONFIG))
    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=f"http://{args.host or '0.0.0.0'}:{args.port}/v1",
    )
    while True:
        try:
            await client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}],
                model=args.model,
                max_tokens=1,
            )
            break
        except Exception:
            continue
    return server_task.cancel, client
