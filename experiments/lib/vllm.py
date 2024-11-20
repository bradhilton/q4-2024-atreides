import asyncio
import copy
from dataclasses import dataclass
import httpx
import json
from openai import AsyncOpenAI
from openai import DefaultAsyncHttpxClient
import os
import random
import re
import socket
from typing import Any, Callable, Optional
from uvicorn.config import LOGGING_CONFIG as UVICORN_LOGGING_CONFIG

os.environ["VLLM_LOGGING_CONFIG_PATH"] = __file__.replace(
    "vllm.py", "vllm-logging-config.json"
)
VLLM_LOGGING_CONFIG = json.load(open(os.environ["VLLM_LOGGING_CONFIG_PATH"]))
LOG_FILENAME = VLLM_LOGGING_CONFIG["handlers"]["vllm"]["filename"]
os.makedirs(os.path.dirname(LOG_FILENAME), exist_ok=True)
with open(LOG_FILENAME, "w"):
    pass

from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils import FlexibleArgumentParser

from .utils import read_last_n_lines


async def start_vllm_server(
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
    uvicorn_logging_config = copy.deepcopy(UVICORN_LOGGING_CONFIG)
    uvicorn_logging_config["handlers"]["default"] = {
        "class": "logging.FileHandler",
        "formatter": "default",
        "filename": LOG_FILENAME,
        "mode": "a",
    }
    uvicorn_logging_config["handlers"]["access"] = {
        "class": "logging.FileHandler",
        "formatter": "access",
        "filename": LOG_FILENAME,
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
    server_task = asyncio.create_task(
        run_server(args, log_config=uvicorn_logging_config)
    )
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


@dataclass
class vLLM:
    client: AsyncOpenAI
    process: asyncio.subprocess.Process

    def __del__(self) -> None:
        self.process.terminate()


async def start_vllm(
    model: str,
    timeout: float = 120.0,
    env: Optional[dict[str, str]] = None,
    max_concurrent_requests: int = 128,
    **kwargs: Any,
) -> vLLM:
    stdouterr = open("./logs/vllm.log", "a")
    args = [
        "vllm",
        "serve",
        model,
        *[
            f"--{key.replace('_', '-')}{f'={value}' if value != True else ''}"
            for key, value in kwargs.items()
        ],
        "--api-key=default",
    ]
    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=stdouterr,
        stderr=stdouterr,
        env={
            **os.environ,
            **(env or {}),
        },
    )
    print(f"$ {' '.join(args)}")
    client = AsyncOpenAI(
        api_key="default",
        base_url=f"http://{kwargs.get('host', '0.0.0.0')}:{kwargs.get('port', 8000)}/v1",
        http_client=DefaultAsyncHttpxClient(
            limits=httpx.Limits(
                max_connections=max_concurrent_requests * 4,
                max_keepalive_connections=max_concurrent_requests,
            )
        ),
    )
    start = asyncio.get_event_loop().time()
    while True:
        try:
            await client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}],
                model=model,
                max_tokens=1,
            )
            break
        except Exception:
            if asyncio.get_event_loop().time() - start > timeout:
                process.terminate()
                raise TimeoutError("vLLM server did not start in time")
            continue
    return vLLM(client, process)


def vllm_server_metrics(last_n_lines: int = 5) -> tuple[int, int]:
    log_str = read_last_n_lines(LOG_FILENAME, last_n_lines)
    pattern = r"Running: (\d+) reqs, Swapped: \d+ reqs, Pending: (\d+) reqs"
    matches = list(re.finditer(pattern, log_str))
    if not matches:
        if len(log_str.splitlines()) == last_n_lines:
            return vllm_server_metrics(last_n_lines * 2)
        else:
            return (0, 0)
    last_match = matches[-1]
    return (int(last_match.group(1)), int(last_match.group(2)))
