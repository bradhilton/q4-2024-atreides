from aioitertools.builtins import iter, enumerate as aenumerate
from aioitertools import itertools as ait
from aioitertools.helpers import maybe_await
import asyncio
import math
import os
import subprocess
from torchtune.models.llama3_1 import llama3_1_8b
from torchtune.training.metric_logging import DiskLogger
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook
from typing import (
    Any,
    AsyncIterable,
    Iterable,
    Literal,
    Optional,
    overload,
    Union,
)

from .completion_sampler import CompletionSampler
from .episode import Episode
from .pack import PackedDataset, packed_tensors
from .ppo import PPOLoss
from ..recipes.rl import ComponentConfig, RLConfig, recipe_main
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
        self.vllm_kwargs = vllm_kwargs or {}
        self.vllm_kwargs["env"] = vllm_env
        self.vllm_kwargs["timeout"] = vllm_timeout
        self.vllm_max_concurrent_requests = vllm_max_concurrent_requests
        self._vllm_task: Optional[asyncio.Task[vLLM]] = None
        self._completion_sampler: Optional[CompletionSampler] = None
        self.tokenizer = Tokenizer(base_model)
        self._substitute_episodes: dict[Episode, Episode] = {}
        try:
            get_ipython  # type: ignore
            self.tqdm = tqdm_notebook
        except NameError:
            self.tqdm = tqdm

    @property
    def model(self) -> str:
        return self.models[-1]

    async def train(self, iterations: int) -> None:
        for _ in range(iterations):
            val_score, episodes = await asyncio.gather(
                self.eval("val", 0), self.explore(1)
            )
            await self.tune(episodes)
        val_score, test_score = await asyncio.gather(
            self.eval("val", 0), self.eval("test", 1)
        )

    @overload
    async def eval(
        self,
        split: Literal["val", "test"],
        pbar_position: Optional[int] = None,
        *,
        return_exceptions: Literal[False] = False,
    ) -> Optional[float]: ...

    @overload
    async def eval(
        self,
        split: Literal["val", "test"],
        pbar_position: Optional[int] = None,
        *,
        return_exceptions: Literal[True] = True,
    ) -> tuple[Optional[float], list[Exception]]: ...

    async def eval(
        self,
        split: Literal["val", "test"],
        pbar_position: Optional[int] = None,
        *,
        return_exceptions: bool = False,
    ) -> Union[Optional[float], tuple[Optional[float], list[Exception]]]:
        if self.eval_episodes[split] is None:
            if return_exceptions:
                return None, []
            return None
        completion_sampler = await self.completion_sampler()
        pbar = self.tqdm(
            desc=split,
            total=getattr(self.eval_episodes[split], "__len__", lambda: None)(),
            unit="episode",
            dynamic_ncols=True,
            position=pbar_position,
        )
        tasks: list[asyncio.Task] = []
        episodes: list[Episode] = []
        exceptions: list[Exception] = []

        def get_score() -> float:
            return sum(
                episode.completion.value(cache=True, model=self.model)
                for episode in episodes
            ) / max(
                sum(
                    1
                    for episode in episodes
                    if any(
                        child.model == self.model
                        for child in episode.completion.children
                    )
                ),
                1,
            )

        def done_callback(task: asyncio.Task[bool]) -> None:
            pbar.update(1)
            try:
                task.result()
            except Exception as exception:
                exceptions.append(exception)
            pbar.set_postfix(avg=get_score(), exceptions=len(exceptions))

        async for episode in iter(self.eval_episodes[split] or ()):
            episodes.append(episode)
            task = asyncio.create_task(
                episode.sample_completions_v2(
                    completion_sampler,
                    branch_factor=self.eval_samples_per_episode[split],
                )
            )
            task.add_done_callback(done_callback)
            tasks.append(task)
        pbar.total = len(episodes)
        pbar.refresh()
        self.eval_episodes[split] = episodes
        await asyncio.gather(*tasks)
        pbar.close()
        score = get_score()
        self.eval_scores[split][self.model] = get_score()
        if return_exceptions:
            return score, exceptions
        return score

    @overload
    async def explore(
        self, pbar_position: Optional[int] = None, *, return_exceptions: Literal[True]
    ) -> tuple[list[Episode], list[Exception]]: ...

    @overload
    async def explore(
        self,
        pbar_position: Optional[int] = None,
        *,
        return_exceptions: Literal[False] = False,
    ) -> list[Episode]: ...

    async def explore(
        self, pbar_position: Optional[int] = None, *, return_exceptions: bool = False
    ) -> Union[list[Episode], tuple[list[Episode], list[Exception]]]:
        await self.completion_sampler()
        pbar = self.tqdm(
            desc="explore",
            total=self.episodes_per_iteration,
            unit="episode",
            dynamic_ncols=True,
            position=pbar_position,
        )
        pbar.set_postfix(completed=0, exceptions=0)
        tasks: list[asyncio.Task[Episode]] = []
        num_episodes: int = 0
        async for priority, episode in aenumerate(
            ait.islice(self._train_iterator, self.episodes_per_iteration)
        ):
            if not self._first_train_episode:
                self._first_train_episode = episode
            elif (
                not self.episodes_per_iteration and episode is self._first_train_episode
            ):
                self._train_iterator = ait.chain([episode], self._train_iterator)
                break
            num_episodes += 1
            task = asyncio.create_task(self._explore_episode(episode, pbar, priority))
            tasks.append(task)
        if self.episodes_per_iteration is None:
            self.episodes_per_iteration = num_episodes
            pbar.total = num_episodes
            pbar.refresh()
        episodes = []
        exceptions: list[Exception] = []
        for future in asyncio.as_completed(tasks):
            try:
                episode = await future
                episodes.append(episode)
            except Exception as exception:
                exceptions.append(exception)
            pbar.set_postfix(completed=len(episodes), exceptions=len(exceptions))
        pbar.n = len(episodes)
        pbar.close()
        if return_exceptions:
            return episodes, exceptions
        return episodes

    async def _explore_episode(
        self, episode: Episode, pbar: tqdm, priority: int
    ) -> Episode:
        if episode in self._substitute_episodes:
            return await self._explore_episode(
                self._substitute_episodes[episode], pbar, priority
            )
        completion_sampler = await self.completion_sampler()
        frac = None
        while remaining_samples := max(
            self.samples_per_episode - episode.num_samples(model=self.model), 0
        ):
            _frac = remaining_samples / self.samples_per_episode
            if frac:
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
                return await self._explore_episode(substitute, pbar, priority)
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

        checkpoint_dir = subprocess.run(
            f"HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download {self.model[0]}",
            shell=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        checkpoint_output_dir = "/home/ubuntu/atreides/experiments/models/rl"
        os.makedirs(checkpoint_output_dir, exist_ok=True)

        PLACEHOLDER: Any = None

        config = RLConfig(
            # Dataset
            dataset=ComponentConfig(PackedDataset, tensors=tensors),
            seed=42,
            shuffle=False,
            # Model
            model=ComponentConfig(llama3_1_8b),
            num_output_chunks=4,
            # Checkpointer
            checkpointer=ComponentConfig(
                "torchtune.training.FullModelHFCheckpointer",
                checkpoint_dir=checkpoint_dir,
                checkpoint_files=[
                    "model-00001-of-00004.safetensors",
                    "model-00002-of-00004.safetensors",
                    "model-00003-of-00004.safetensors",
                    "model-00004-of-00004.safetensors",
                ],
                recipe_checkpoint=None,
                output_dir=checkpoint_output_dir,
                model_type="LLAMA3",
            ),
            resume_from_checkpoint=False,
            # Fine-tuning arguments
            batch_size=2,
            epochs=1,
            optimizer=ComponentConfig(
                # AdamW,
                "bitsandbytes.optim.PagedAdamW8bit",
                params=PLACEHOLDER,
                lr=5e-6,
                # fused=True,
            ),
            loss=ComponentConfig(
                PPOLoss,
                # clip_epsilon=0.3,
                # entropy_coef=0.0,
                # kl_coef=0.0,
                clip_epsilon=0.3,
                entropy_coef=0.025,
                kl_coef=0.025,
            ),
            max_steps_per_epoch=None,
            compile=False,
            optimizer_in_bwd=False,
            gradient_accumulation_steps=1,
            # Training env
            device="cuda",
            # Memory management
            enable_activation_checkpointing=True,
            enable_activation_offloading=False,
            custom_sharded_layers=["tok_embeddings", "output"],
            # Reduced precision
            dtype="bf16",
            # Logging
            metric_logger=ComponentConfig(
                DiskLogger, log_dir="/home/ubuntu/atreides/experiments/logs"
            ),
            output_dir="/home/ubuntu/atreides/experiments/logs",
            log_every_n_steps=1,
            log_peak_memory_stats=True,
        )

        recipe_main(config)

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
