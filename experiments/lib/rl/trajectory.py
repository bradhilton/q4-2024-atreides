import asyncio
import httpx
from fastapi import FastAPI
import functools
import itertools
import nest_asyncio
import os
from pathlib import Path
from pydantic import BaseModel
import torch
from torch.utils.data import Dataset, get_worker_info, IterableDataset
from typing import Iterable, Optional, TypedDict
import uvicorn

from .completion import Completion
from .episode import Episode
from .episode_buffer import EpisodeBuffer

nest_asyncio.apply()


class Trajectory:
    buffer: EpisodeBuffer
    episode: Episode
    terminus: Completion
    abs_advantage: float
    token_count: int
    decayed: bool

    def __init__(self, buffer: EpisodeBuffer, episode: Episode) -> None:
        self.buffer = buffer
        self.episode = episode
        self.terminus = episode.best_leaf(
            buffer.tokenizer,
            split_method=buffer.split_method,
            split_separators=buffer.split_separators,
        )
        self.abs_advantage = self.terminus.all_abs_advantage()
        self.token_count = self.terminus.all_token_count(buffer.tokenizer)
        self.decayed = False

    def score(self) -> float:
        return self.episode.weight * self.abs_advantage / self.token_count

    def decay(self) -> None:
        if self.decayed:
            return
        self.episode.weight *= self.buffer.episode_decay
        for completion in self.terminus.ancestors(including_self=True):
            completion.weight *= self.buffer.completion_decay
        self.decayed = True


async def get_trajectory_batch(
    buffer: EpisodeBuffer, batch_size: int, seqlen: int
) -> list[list[Trajectory]]:
    batch: list[list[Trajectory]] = []
    trajectories: list[Trajectory] = []
    for _ in range(batch_size):
        sequence: list[Trajectory] = []
        while sum(trajectory.token_count for trajectory in sequence) < seqlen:
            if not trajectories:
                _ = [t.decay() for s in batch for t in s]
                trajectories = sorted(
                    (
                        Trajectory(buffer=buffer, episode=episode)
                        for episode in buffer.episodes
                        if episode.completion.children
                    ),
                    key=lambda trajectory: trajectory.score(),
                )
                if not trajectories:
                    if pending_tasks := [
                        task for task in buffer.tasks.values() if not task.done()
                    ]:
                        await asyncio.wait(
                            pending_tasks,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    else:
                        await asyncio.sleep(buffer.sleep_time)
                    continue
            sequence.append(trajectories.pop())
        trajectory = sequence.pop()
        if trajectory.token_count <= seqlen:
            trajectories.append(trajectory)
        batch.append(sequence)
    _ = [t.decay() for s in batch for t in s]
    return batch


class TrajectoryTensors(TypedDict):
    tokens: torch.Tensor
    advantages: torch.Tensor
    logprobs: torch.Tensor


def trajectory_tensors(batch: list[list[Trajectory]], seqlen: int) -> TrajectoryTensors:
    tokenizer = batch[0][0].buffer.tokenizer
    tokens = torch.full(
        (len(batch), seqlen),
        tokenizer.get_pad_token_id() or 0,
        dtype=torch.int64,
    )
    tensor_dict: TrajectoryTensors = {
        "tokens": tokens,
        "advantages": torch.full_like(
            tokens, fill_value=torch.nan, dtype=torch.float32
        ),
        "logprobs": torch.full_like(tokens, fill_value=torch.nan, dtype=torch.float32),
    }
    write_trajectory_batch(batch, tensor_dict, start=0)
    return tensor_dict


def write_trajectory_batch(
    batch: list[list[Trajectory]], tensors: TrajectoryTensors, start: int
) -> None:
    tokenizer = batch[0][0].buffer.tokenizer
    rows, seqlen = tensors["tokens"].shape
    for row, selected_trajectories in enumerate(batch, start):
        # Wrap row index back to start if it exceeds number of rows, enabling circular buffer behavior
        row %= rows
        tensors["tokens"][row] = tokenizer.encode(
            [
                trajectory.terminus.all_message_params()
                for trajectory in selected_trajectories
            ],  # type: ignore
            concatenate=True,
            seqlen=seqlen,
        )
        replacement_token = "<|reserved_special_token_250|>"
        mask = tokenizer.encode(
            [trajectory.terminus.all_message_params(replacement_token=replacement_token) for trajectory in selected_trajectories],  # type: ignore
            concatenate=True,
            seqlen=seqlen,
        ) == tokenizer.get_token_id(replacement_token)
        mask_size = mask.sum()
        tensors["advantages"][row] = torch.full_like(
            mask, fill_value=torch.nan, dtype=torch.float32
        )
        tensors["advantages"][row][mask] = torch.tensor(
            list(
                advantage
                for trajectory in selected_trajectories
                for advantage in trajectory.terminus.all_token_advantages()
            )[:mask_size]
        )
        tensors["logprobs"][row] = torch.full_like(
            mask, fill_value=torch.nan, dtype=torch.float32
        )
        tensors["logprobs"][row][mask] = torch.tensor(
            list(
                advantage
                for trajectory in selected_trajectories
                for advantage in trajectory.terminus.all_logprobs()
            )[:mask_size]
        )


class Trajectories(BaseModel, Dataset[TrajectoryTensors], frozen=True):
    dir: str
    rows: int
    seqlen: int

    @functools.cached_property
    def tensors(self) -> TrajectoryTensors:
        os.makedirs(self.dir, exist_ok=True)
        return {
            "tokens": torch.from_file(
                str(Path(self.dir) / "tokens.pt"),
                shared=True,
                size=self.rows * self.seqlen,
                dtype=torch.int64,
            ).view(-1, self.seqlen),
            "advantages": torch.from_file(
                str(Path(self.dir) / "advantages.pt"),
                shared=True,
                size=self.rows * self.seqlen,
                dtype=torch.float32,
            ).view(-1, self.seqlen),
            "logprobs": torch.from_file(
                str(Path(self.dir) / "logprobs.pt"),
                shared=True,
                size=self.rows * self.seqlen,
                dtype=torch.float32,
            ).view(-1, self.seqlen),
        }

    def __len__(self) -> int:
        return self.rows

    def __getitem__(self, index: int) -> TrajectoryTensors:
        return {
            "tokens": self.tensors["tokens"][index],
            "advantages": self.tensors["advantages"][index],
            "logprobs": self.tensors["logprobs"][index],
        }


class TrajectoryBatchRequest(Trajectories, frozen=True):
    start: int
    stop: int


def serve_trajectory_batch_api(
    buffer: EpisodeBuffer,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> uvicorn.Server:
    app = FastAPI()

    @app.get("/")
    async def _() -> int:
        return os.getpid()

    @app.post("/trajectory-batch")
    async def _(
        request: TrajectoryBatchRequest,
    ) -> TrajectoryBatchRequest:
        write_trajectory_batch(
            batch=await get_trajectory_batch(
                buffer, request.stop - request.start, request.seqlen
            ),
            tensors=request.tensors,
            start=request.start,
        )
        return request

    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, workers=2))
    asyncio.create_task(server.serve())
    return server


class IterableTrajectories(
    Trajectories, IterableDataset[TrajectoryTensors], frozen=True
):
    batch_api_address: tuple[str, int] = ("127.0.0.1", 8000)
    buffer: int = 1

    def __iter__(self) -> Iterable[TrajectoryTensors]:
        worker_info = get_worker_info()
        if worker_info:  # in a worker process
            # split workload
            per_worker = int(self.rows / worker_info.num_workers)
            start = worker_info.id * per_worker
            stop = min(start + per_worker, self.rows)
        else:  # single-process data loading
            start = 0
            stop = self.rows
        client = httpx.AsyncClient()
        task: Optional[asyncio.Task[httpx.Response]] = None
        yield_indices = lambda: itertools.cycle(range(start, stop))
        buffer_indices = itertools.cycle(
            itertools.islice(
                yield_indices(),
                self.buffer,
                stop - start + self.buffer,
            )
        )
        for yield_index, buffer_index in zip(yield_indices(), buffer_indices):
            if not task or (buffer_index - start) % self.buffer == 0:
                response = task and (
                    getattr(task, "_result") or asyncio.run(get_response(task))
                )
                task = asyncio.create_task(
                    client.post(
                        f"http://{self.batch_api_address[0]}:{self.batch_api_address[1]}/trajectory-batch",
                        json=TrajectoryBatchRequest(
                            dir=self.dir,
                            rows=self.rows,
                            seqlen=self.seqlen,
                            start=buffer_index if response else start,
                            stop=min(buffer_index + self.buffer, stop),
                        ).model_dump(mode="json"),
                    )
                )
                if not response:
                    response = asyncio.run(get_response(task))
                response.raise_for_status()
            yield self[yield_index]


async def get_response(task: asyncio.Task[httpx.Response]) -> httpx.Response:
    return await task
