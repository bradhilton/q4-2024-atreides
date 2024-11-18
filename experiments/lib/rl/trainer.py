from aioitertools import itertools as ait
import asyncio
from typing import (
    Any,
    AsyncIterable,
    Coroutine,
    Iterable,
    Optional,
    Union,
)

from .completion_sampler import CompletionSampler
from .episode import Episode
from ..tokenizer import Tokenizer
from ..vllm import start_vllm, vLLM


Episodes = Union[Iterable[Episode], AsyncIterable[Episode]]


class Trainer:
    def __init__(
        self,
        base_model: str,
        train_episodes: Episodes,
        episodes_per_iteration: Optional[int],
        val_episodes: Optional[Episodes] = None,
        test_episodes: Optional[Episodes] = None,
        vllm_env: Optional[dict[str, str]] = None,
        vllm_kwargs: Optional[dict[str, Any]] = None,
        vllm_timeout: float = 120.0,
        parallel: bool = False,
    ) -> None:
        self.models = [base_model]
        self._train_iterator = ait.cycle(train_episodes)
        self._first_train_episode: Optional[Episode] = None
        self.episodes_per_iteration: Optional[int] = (
            episodes_per_iteration or getattr(train_episodes, "__len__", lambda: None)()
        )
        self._val_iterator = ait.iter(val_episodes) if val_episodes else None
        self._val_episodes: list[Episode] = []
        self.val_scores: dict[str, float] = {}
        self._test_iterator = ait.iter(test_episodes) if test_episodes else None
        self.vllm_kwargs = vllm_kwargs or {}
        self.vllm_kwargs["env"] = vllm_env
        self.vllm_kwargs["timeout"] = vllm_timeout
        self._vllm_task: Optional[asyncio.Task[vLLM]] = None
        self._completion_sampler: Optional[CompletionSampler] = None
        self.parallel = parallel
        self.tokenizer = Tokenizer(base_model)
        self._substitute_episodes: dict[Episode, Episode] = {}

    async def train(self, iterations: int) -> None:
        for _ in range(iterations):
            await asyncio.gather(self.validate(), self.sample_completions())
            vllm = await self.vllm()
            vllm.process.terminate()
            await self.tune()
        await asyncio.gather(self.validate(), self.test())

    async def validate(self) -> None:
        if not self._val_iterator:
            return
        completion_sampler = await self.completion_sampler()
        tasks: list[asyncio.Task] = []
        save = True
        if self._val_episodes:
            self._val_iterator = ait.iter(self._val_episodes)
            save = False
        async for episode in self._val_iterator:
            if save:
                self._val_episodes.append(episode)
            tasks.append(
                asyncio.create_task(
                    episode.sample_completions(
                        completion_sampler, self.tokenizer, branch_factor=1
                    )
                )
            )
        await asyncio.gather(*tasks)
        self.val_scores[self.models[-1]] = sum(
            episode.completion.value(model=self.models[-1])
            for episode in self._val_episodes
        ) / len(self._val_episodes)

    async def sample_completions(self) -> None:
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
            tasks.append(asyncio.create_task(self.sample_episode_completions(episode)))
        await asyncio.gather(*tasks)

    async def sample_episode_completions(self, episode: Episode) -> None:
        if episode in self._substitute_episodes:
            return await self.sample_episode_completions(
                self._substitute_episodes[episode]
            )
        completion_sampler = await self.completion_sampler()
        while episode.num_samples() < 100:
            if not await episode.sample_completions(
                completion_sampler,
                self.tokenizer,
                branch_factor=1,
            ):
                break
            if all(c.advantage() == 0 for c in episode.completion.children):
                if (
                    episode.get_easier_episode
                    and episode.min_value is not None
                    and episode.completion.value() <= episode.min_value
                ):
                    substitute = episode.get_easier_episode()
                elif (
                    episode.get_harder_episode
                    and episode.max_value is not None
                    and episode.completion.value() >= episode.max_value
                ):
                    substitute = episode.get_harder_episode()
                elif episode.get_similar_episode:
                    substitute = episode.get_similar_episode()
                else:
                    return await self.sample_episode_completions(episode)
                if isinstance(substitute, Coroutine):
                    substitute = await substitute
                self._substitute_episodes[episode] = substitute
                return await self.sample_episode_completions(substitute)

    async def tune(self) -> None: ...

    async def test(self) -> None: ...

    async def completion_sampler(self) -> CompletionSampler:
        if self._completion_sampler:
            return self._completion_sampler
        vllm = await self.vllm()
        self._completion_sampler = CompletionSampler(vllm.client, model=self.models[-1])
        return self._completion_sampler

    async def vllm(self) -> vLLM:
        if not self._vllm_task:
            self._vllm_task = asyncio.create_task(
                start_vllm(self.models[-1], **self.vllm_kwargs)
            )
        return await self._vllm_task
