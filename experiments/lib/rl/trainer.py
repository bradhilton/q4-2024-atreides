from aioitertools.builtins import iter
from aioitertools import itertools as ait
from aioitertools.helpers import maybe_await
import asyncio
import math
from tqdm import tqdm
from typing import (
    Any,
    AsyncIterable,
    Iterable,
    Literal,
    Optional,
    Union,
)

from .completion_sampler import CompletionSampler
from .episode import Episode
from .pack import packed_tensors
from ..tokenizer import Tokenizer
from ..vllm import start_vllm, vLLM


Episodes = Union[Iterable[Episode], AsyncIterable[Episode]]


class Trainer:
    def __init__(
        self,
        base_model: str,
        samples_per_episode: int,
        branch_factor: int,
        train_episodes: Episodes,
        episodes_per_iteration: Optional[int] = None,
        val_episodes: Optional[Episodes] = None,
        val_samples_per_episode: int = 1,
        test_episodes: Optional[Episodes] = None,
        test_samples_per_episode: int = 1,
        tune_episode_sample_fraction: float = 1.0,
        tune_sequence_length: int = 8192,
        tqdm: Optional[type[tqdm]] = None,
        vllm_env: Optional[dict[str, str]] = None,
        vllm_kwargs: Optional[dict[str, Any]] = None,
        vllm_max_concurrent_requests: int = 128,
        vllm_timeout: float = 120.0,
    ) -> None:
        self.models = [base_model]
        self.samples_per_episode = samples_per_episode
        self.branch_factor = branch_factor
        self._train_iterator = ait.cycle(train_episodes)
        self._first_train_episode: Optional[Episode] = None
        self.episodes_per_iteration: Optional[int] = (
            episodes_per_iteration or getattr(train_episodes, "__len__", lambda: None)()
        )
        self.eval_episodes = {
            "val": val_episodes,
            "test": test_episodes,
        }
        self.eval_samples_per_episode = {
            "val": val_samples_per_episode,
            "test": test_samples_per_episode,
        }
        self.eval_scores: dict[str, dict[str, float]] = {"val": {}, "test": {}}
        self.tune_episode_sample_fraction = tune_episode_sample_fraction
        self.tune_sequence_length = tune_sequence_length
        self.tqdm = tqdm
        self.vllm_kwargs = vllm_kwargs or {}
        self.vllm_kwargs["env"] = vllm_env
        self.vllm_kwargs["timeout"] = vllm_timeout
        self.vllm_max_concurrent_requests = vllm_max_concurrent_requests
        self._vllm_task: Optional[asyncio.Task[vLLM]] = None
        self._completion_sampler: Optional[CompletionSampler] = None
        self.tokenizer = Tokenizer(base_model)
        self._substitute_episodes: dict[Episode, Episode] = {}

    @property
    def model(self) -> str:
        return self.models[-1]

    async def train(self, iterations: int) -> None:
        for _ in range(iterations):
            _, episodes = await asyncio.gather(self.eval("val", 0), self.explore(1))
            await self.tune(episodes)
        await asyncio.gather(self.eval("val", 0), self.eval("test", 1))

    async def eval(
        self, split: Literal["val", "test"], pbar_position: Optional[int] = None
    ) -> Optional[float]:
        if self.eval_episodes[split] is None:
            return
        completion_sampler = await self.completion_sampler()
        pbar = (
            self.tqdm(desc=split, total=1, unit="episode", position=pbar_position)
            if self.tqdm
            else None
        )
        tasks: list[asyncio.Task] = []
        episodes: list[Episode] = []
        async for episode in iter(self.eval_episodes[split] or ()):
            episodes.append(episode)
            task = asyncio.create_task(
                episode.sample_completions_v2(
                    completion_sampler,
                    branch_factor=self.eval_samples_per_episode[split],
                )
            )
            if pbar is not None:
                pbar.total += 1
            task.add_done_callback(lambda _: pbar.update(1) if pbar else None)
            tasks.append(task)
        if pbar is not None:
            pbar.total = len(episodes)
        self.eval_episodes[split] = episodes
        await asyncio.gather(*tasks)
        score = sum(
            episode.completion.value(model=self.model) for episode in episodes
        ) / max(
            sum(
                1
                for episode in episodes
                if any(
                    child.model == self.model for child in episode.completion.children
                )
            ),
            1,
        )
        if pbar is not None:
            pbar.set_postfix(avg=score)
        self.eval_scores[split][self.model] = score
        return score

    async def explore(self, pbar_position: Optional[int] = None) -> list[Episode]:
        pbar = (
            self.tqdm(
                desc="explore",
                total=self.episodes_per_iteration,
                unit="episode",
                position=pbar_position,
            )
            if self.tqdm
            else None
        )
        tasks: list[asyncio.Task] = []
        async for episode in ait.islice(
            self._train_iterator, self.episodes_per_iteration
        ):
            if not self._first_train_episode:
                self._first_train_episode = episode
            elif (
                not self.episodes_per_iteration and episode is self._first_train_episode
            ):
                self._train_iterator = ait.chain([episode], self._train_iterator)
                break
            tasks.append(asyncio.create_task(self._explore_episode(episode, pbar)))
        return await asyncio.gather(*tasks)

    async def _explore_episode(self, episode: Episode, pbar: Optional[tqdm]) -> Episode:
        if episode in self._substitute_episodes:
            return await self._explore_episode(self._substitute_episodes[episode], pbar)
        completion_sampler = await self.completion_sampler()
        frac = None
        while remaining_samples := max(
            self.samples_per_episode - episode.num_samples(model=self.model), 0
        ):
            _frac = remaining_samples / self.samples_per_episode
            if pbar and frac:
                pbar.update(frac - _frac)
            frac = _frac
            if not await episode.sample_completions_v2(
                completion_sampler=completion_sampler,
                branch_factor=self.branch_factor,
                max_parallel_splits=int(
                    math.ceil(remaining_samples / (self.branch_factor - 1))
                ),
            ):
                break
            if frac == 1.0 and all(
                c.advantage(cache=True) == 0
                for c in episode.completion.children
                if c.model == self.model
            ):
                if (
                    episode.get_easier_episode
                    and episode.min_value is not None
                    and episode.completion.value() <= episode.min_value
                ):
                    substitute = await maybe_await(episode.get_easier_episode())
                elif (
                    episode.get_harder_episode
                    and episode.max_value is not None
                    and episode.completion.value() >= episode.max_value
                ):
                    substitute = await maybe_await(episode.get_harder_episode())
                elif episode.get_similar_episode:
                    substitute = await maybe_await(episode.get_similar_episode())
                else:
                    continue
                self._substitute_episodes[episode] = substitute
                return await self._explore_episode(substitute, pbar)
        if pbar:
            pbar.update(frac or 0)
        return episode

    async def tune(self, episodes: list[Episode]) -> None:
        vllm = await self.vllm()
        vllm.process.terminate()
        self._vllm_task, self._completion_sampler = None, None
        tensors = packed_tensors(
            episodes,
            model=self.model,
            sequence_length=self.tune_sequence_length,
            trajectories_per_episode=(
                int(self.samples_per_episode * self.tune_episode_sample_fraction)
                if self.tune_episode_sample_fraction < 1.0
                else None
            ),
            tokenizer=self.tokenizer,
        )

    async def completion_sampler(self) -> CompletionSampler:
        if self._completion_sampler:
            return self._completion_sampler
        vllm = await self.vllm()
        self._completion_sampler = CompletionSampler(
            vllm.client,
            max_concurrent_requests=self.vllm_max_concurrent_requests,
            model=self.model,
        )
        return self._completion_sampler

    async def vllm(self) -> vLLM:
        if not self._vllm_task:
            self._vllm_task = asyncio.create_task(
                start_vllm(
                    self.model,
                    max_concurrent_requests=self.vllm_max_concurrent_requests,
                    **self.vllm_kwargs,
                )
            )
        return await self._vllm_task
