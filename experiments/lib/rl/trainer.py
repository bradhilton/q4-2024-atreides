from aioitertools.builtins import iter, enumerate as aenumerate, next as anext
from aioitertools import itertools as ait
from aioitertools.helpers import maybe_await
from aioitertools.types import AnyIterable
import asyncio
from dataclasses import dataclass
import glob
import itertools as it
from omegaconf import OmegaConf
from openai import AsyncOpenAI
import os
import re
import shutil
import sys
import torch
from torchtune.modules import TransformerDecoder
from torchtune.training import cleanup_before_training, FullModelHFCheckpointer
from torchtune.training.metric_logging import DiskLogger, WandBLogger
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook
from typing import (
    Any,
    AsyncIterable,
    Callable,
    IO,
    Iterable,
    Literal,
    Optional,
    overload,
    Union,
)
import wandb

from .completion import Completion, SplitMethod
from .completion_sampler import SamplingKwargs, CompletionSampler, CompletionSamplerPool
from .episode import Episode
from .explore_result import ExploreResult
from .pack import packed_tensors, PackedDataset, PackedTensors
from .recipe import ComponentConfig, recipe_main, TuneRecipeConfig
from ..tokenizer import Tokenizer
from ..utils import get_semaphore
from ..vllm import start_vllms, vLLM


Episodes = Union[
    Iterable[Union[Episode, BaseException]],
    AsyncIterable[Union[Episode, BaseException]],
]


@dataclass
class ExploreOptions:
    """
    During the exploration step, the trainer will sample completions for each episode `iterations` times. The
    total number of sampled completions per episode will be close to `iterations * num_parents * (branch_factor - 1)`.

    Attributes:
    iterations (int): The number of iterations to explore each episode.

    num_parents (int): The number of parent nodes to sample completions from for each episode each iteration.

    branch_factor (int): The number of completions to sample from each parent node.

    max_split_points (Optional[int]): The maximum number of points that a completion can be split per
    iteration. If None, completions may be split up to `num_parents` times.

    patience (float): The maximum number of seconds to wait per episode for exploration progress.

    recovery_pattern (Optional[Union[str, Callable[[], str]]]): An optional prefix pattern to use when sampling
    from completions with negative advantage. May provide a string pattern or a callable that returns a pattern to
    introduce some randomness (suggested). Introduces additional overhead to the sampling process.

    sample_probability_power (Optional[float]): The power to raise the sample probabilities to. If None,
    defaults to 1 / branch_factor.

    sampling_kwargs (Optional[SamplingKwargs]): Optional keyword arguments when sampling completions. See the
    documentation for `openai.resources.chat.completions.AsyncCompletions.create` for the standard keyword
    arguments. Additional vLLM-specific keyword arguments can be found at:
    https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters-for-chat-completions-api

    split_method (SplitMethod): The method to weight previously sampled completions' token logprobs before
    splitting. Options include "count", "prob", and "logprob". For example, splitting a completion in two with
    the "logprob" method would divide the completion into two roughly equally likely sequences. "count" just splits
    by the number of tokens and "prob" strikes a balance between the two.

    split_point_std_deviation (float): The standard deviation of the normal distribution used to sample split points
    at equally spaced intervals. If 0 (the default value), split points are chosen deterministically. The larger the
    standard deviation, the more random the split points will be and larger values will tend to a uniform distribution.

    split_separator (Optional[set[str]]): An optional set of strings to divide tokens into unsplittable atomic units.
    If None, completions may be split at any token boundary.
    """

    iterations: int
    num_parents: int
    branch_factor: int
    max_split_points: Optional[int] = None
    patience: float = 1.0
    recovery_pattern: Optional[Union[str, Callable[[Completion], str]]] = None
    sample_probability_power: Optional[float] = None
    sampling_kwargs: Optional[SamplingKwargs] = None
    split_method: SplitMethod = "count"
    split_point_std_deviation: float = 0.0
    split_separators: Optional[set[str]] = None

    @property
    def get_recovery_pattern(self) -> Optional[Callable[[Completion], str]]:
        recovery_pattern = self.recovery_pattern
        if isinstance(recovery_pattern, str):
            return lambda _: recovery_pattern
        else:
            return recovery_pattern

    def get_sample_probability_power(self) -> float:
        return self.sample_probability_power or 1 / self.branch_factor

    @property
    def num_samples(self) -> int:
        return self.iterations * self.num_parents * (self.branch_factor - 1)


Verbosity = Literal[0, 1, 2]


@dataclass
class vLLMConfig:
    env: Optional[dict[str, str]] = None
    kwargs: Optional[dict[str, Any]] = None
    max_concurrent_samples: int = 128
    min_time_between_requests: float = 0.0
    num: int = torch.cuda.device_count()
    timeout: float = 120.0


class Trainer:
    def __init__(
        self,
        *,
        base_model: str,
        base_model_checkpoint_files: Optional[list[str]] = None,
        output_dir: str,
        explore_options: ExploreOptions,
        reference_clients: Optional[AnyIterable[AsyncOpenAI]] = None,
        reference_model: Optional[str] = None,
        reference_model_checkpoint_files: Optional[list[str]] = None,
        reference_roundtrips_per_episode: int = 1,
        reference_vllm_config: Optional[vLLMConfig] = None,
        train_episodes: Episodes,
        episodes_per_iteration: Optional[int] = None,
        max_mask_sequence_batch_size: Optional[int] = None,
        test_episodes: Optional[Episodes] = None,
        test_patience: float = 5.0,
        test_samples_per_episode: int = 1,
        test_sampling_kwargs: Optional[SamplingKwargs] = None,
        torchrun_kwargs: Optional[dict[str, Any]] = None,
        tune_episode_sample_fraction: float = 1.0,
        tune_model: Callable[[], TransformerDecoder],
        tune_model_type: str,
        tune_recipe_config: Optional[TuneRecipeConfig] = None,
        tune_run: bool = True,
        tune_run_env: Optional[dict[str, str]] = None,
        tune_sequence_length: int = 8192,
        val_episodes: Optional[Episodes] = None,
        val_patience: float = 5.0,
        val_samples_per_episode: int = 1,
        val_sampling_kwargs: Optional[SamplingKwargs] = None,
        vllm_config: Optional[vLLMConfig] = None,
        wandb_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self.base_model = base_model
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        last_iteration = max(
            (
                int(subdir)
                for subdir in os.listdir(self.output_dir)
                if os.path.isdir(os.path.join(self.output_dir, subdir))
                and subdir.isdigit()
            ),
            default=None,
        )
        self.models = [
            (
                f"{self.output_dir}/{last_iteration:04d}"
                if last_iteration is not None
                else base_model
            )
        ]
        if self.model != base_model:
            print(f"Resuming from {self.model}")
        self.base_model_checkpoint_files = base_model_checkpoint_files
        self.explore_options = explore_options
        self.reference_clients = (
            ait.cycle(reference_clients) if reference_clients else None
        )
        self.reference_model = reference_model or self.base_model
        self.reference_model_checkpoint_files = reference_model_checkpoint_files or (
            self.base_model_checkpoint_files
            if reference_model == self.base_model
            else None
        )
        self.reference_roundtrips_per_episode = reference_roundtrips_per_episode
        self.reference_vllm_config = reference_vllm_config
        self._train_iterator = ait.cycle(train_episodes)
        self._first_train_episode: Optional[Episode] = None
        self.episodes_per_iteration: Optional[int] = (
            episodes_per_iteration or getattr(train_episodes, "__len__", lambda: None)()
        )
        self.max_mask_sequence_batch_size = max_mask_sequence_batch_size or max(
            32768 // tune_sequence_length, 1
        )
        self.eval_entropies: dict[str, dict[str, float]] = {"val": {}, "test": {}}
        self.eval_episodes = {
            "val": val_episodes,
            "test": test_episodes,
        }
        self.eval_exceptions = {
            "val": [],
            "test": [],
        }
        self.eval_patience = {
            "val": val_patience,
            "test": test_patience,
        }
        self.eval_samples_per_episode = {
            "val": val_samples_per_episode,
            "test": test_samples_per_episode,
        }
        self.eval_sampling_kwargs = {
            "val": val_sampling_kwargs,
            "test": test_sampling_kwargs,
        }
        self.eval_scores: dict[str, dict[str, float]] = {"val": {}, "test": {}}
        self.explore_results: list[ExploreResult] = []
        self.torchrun_kwargs = torchrun_kwargs or (
            {
                "nnodes": 1,
                "nproc_per_node": torch.cuda.device_count(),
            }
            if torch.cuda.device_count() > 1
            else {}
        )
        self.tune_episode_sample_fraction = tune_episode_sample_fraction
        self.tune_model = tune_model
        self.tune_model_type = tune_model_type
        self.tune_recipe_config = tune_recipe_config or TuneRecipeConfig()
        self.tune_run = tune_run
        self.tune_run_env = tune_run_env or {}
        self.tune_sequence_length = tune_sequence_length
        self.vllm_config = vllm_config or vLLMConfig()
        self._vllm_task: Optional[asyncio.Task[list[vLLM]]] = None
        self._completion_sampler: Optional[CompletionSampler] = None
        self.tokenizer = Tokenizer(base_model)
        self._substitute_episodes: dict[Episode, Episode] = {}
        try:
            get_ipython  # type: ignore
            self.tqdm = tqdm_notebook
        except NameError:
            self.tqdm = tqdm
        self._wandb_kwargs = wandb_kwargs.copy() if wandb_kwargs else {}
        if self._wandb_kwargs:
            self._wandb_kwargs["resume"] = "allow"
            self._wandb_kwargs["reinit"] = True
        self._wandb_run = (
            wandb.init(**self._wandb_kwargs) if self._wandb_kwargs else None
        )

    @property
    def model(self) -> str:
        return self.models[-1]

    async def train(
        self, iterations: int, test: bool = False, verbosity: Verbosity = 2
    ) -> None:
        for _ in range(iterations):
            _, result = await asyncio.gather(
                self.eval("val", 0, return_exceptions=True, verbosity=verbosity),
                self.explore(1, return_exceptions=True, verbosity=verbosity),
            )
            await self.tune(result, verbosity=verbosity)
        _, _ = await asyncio.gather(
            self.eval("val", 0, return_exceptions=True, verbosity=verbosity),
            self.eval("test", 1, verbosity=verbosity) if test else asyncio.sleep(0),
        )

    @overload
    async def eval(
        self,
        split: Literal["val", "test"],
        pbar_position: Optional[int] = None,
        *,
        return_exceptions: Literal[False] = False,
        verbosity: Verbosity = 2,
    ) -> Optional[float]: ...

    @overload
    async def eval(
        self,
        split: Literal["val", "test"],
        pbar_position: Optional[int] = None,
        *,
        return_exceptions: Literal[True] = True,
        verbosity: Verbosity = 2,
    ) -> tuple[Optional[float], list[BaseException]]: ...

    async def eval(
        self,
        split: Literal["val", "test"],
        pbar_position: Optional[int] = None,
        *,
        return_exceptions: bool = True,
        verbosity: Verbosity = 2,
    ) -> Union[Optional[float], tuple[Optional[float], list[BaseException]]]:
        if self.eval_episodes[split] is None:
            if return_exceptions:
                return None, []
            return None
        completion_sampler = await self.get_completion_sampler(verbosity=verbosity)
        pbar = self.tqdm(
            desc=split,
            total=getattr(self.eval_episodes[split], "__len__", lambda: None)(),
            unit="episode",
            dynamic_ncols=True,
            position=pbar_position,
            disable=not verbosity,
        )
        tasks: list[asyncio.Task] = []
        episodes: list[Episode] = []
        exceptions: list[BaseException] = []

        def num_episodes_with_model_completions() -> int:
            return max(
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

        def get_score() -> float:
            return (
                sum(
                    episode.completion.value(cache=True, model=self.model)
                    for episode in episodes
                )
                / num_episodes_with_model_completions()
            )

        def get_entropy() -> float:
            return (
                sum(
                    leaf.all_entropy()
                    for episode in episodes
                    for leaf in episode.completion.leaves(model=self.model)
                )
                / num_episodes_with_model_completions()
            )

        def done_callback(task: asyncio.Task[bool]) -> None:
            pbar.update(1)
            try:
                task.result()
            except Exception as exception:
                exceptions.append(exception)
            pbar.set_postfix(
                avg=get_score(), entropy=get_entropy(), exceptions=len(exceptions)
            )

        sampling_kwargs = (self.eval_sampling_kwargs[split] or {}).copy()
        sampling_kwargs["top_logprobs"] = 20

        async for episode in iter(
            self.eval_episodes[split] or (),
        ):
            if isinstance(episode, BaseException):
                if return_exceptions:
                    exceptions.append(episode)
                    continue
                else:
                    raise episode
            episodes.append(episode)
            samples = self.eval_samples_per_episode[split]
            task = asyncio.create_task(
                episode.sample_completions_v2(
                    completion_sampler,
                    num_parents=1,
                    branch_factor=samples,
                    sampling_kwargs=sampling_kwargs,
                )
                if episode.num_samples(model=self.model) < samples
                else maybe_await(True)
            )
            task.add_done_callback(done_callback)
            tasks.append(task)
            await asyncio.sleep(1e-6)
        pbar.total = len(episodes)
        pbar.refresh()
        self.eval_episodes[split] = episodes
        for future in asyncio.as_completed(tasks):
            remaining_episodes = pbar.total - pbar.n
            patience = self.eval_patience[split] * remaining_episodes
            try:
                await asyncio.wait_for(future, timeout=patience)
            except asyncio.TimeoutError:
                if verbosity > 0:
                    print(
                        f"Early stopping {split} evaluation due to expired patience ({remaining_episodes} remaining episodes x {self.eval_patience[split]} patience per episode = {patience} seconds)"
                    )
                for task in tasks:
                    task.cancel()
                break
            except BaseException as exception:
                if return_exceptions:
                    exceptions.append(exception)
                else:
                    raise exception
        pbar.close()
        score = get_score()
        entropy = get_entropy()
        self.eval_scores[split][self.model] = score
        self.eval_entropies[split][self.model] = entropy
        self.eval_exceptions[split].extend(exceptions)
        if self._wandb_run:
            wandb.log({f"{split}": score, f"{split}_entropy": entropy})
        if return_exceptions:
            return score, exceptions
        return score

    async def explore(
        self,
        pbar_position: Optional[int] = None,
        *,
        return_exceptions: bool = True,
        verbosity: Verbosity = 2,
    ) -> ExploreResult:
        await self.get_completion_sampler(verbosity=verbosity)
        pbar = self.tqdm(
            desc="explore",
            total=self.episodes_per_iteration,
            unit="episode",
            dynamic_ncols=True,
            position=pbar_position,
            disable=not verbosity,
        )
        pbar.set_postfix(completed=0, exceptions=0)
        tasks: list[asyncio.Task[Episode]] = []
        result = ExploreResult(
            pbar=pbar,
            max_mask_sequence_batch_size=self.max_mask_sequence_batch_size,
            model=self.model,
            sample_probability_power=self.explore_options.get_sample_probability_power(),
            sequence_length=self.tune_sequence_length,
            tensor_dir=self.output_dir + "/tensors",
            tokenizer=self.tokenizer,
            trajectories_per_episode=(
                int(
                    self.explore_options.num_samples * self.tune_episode_sample_fraction
                )
                if self.tune_episode_sample_fraction < 1.0
                else None
            ),
        )
        async for priority, episode in iter(
            aenumerate(ait.islice(self._train_iterator, self.episodes_per_iteration))
        ):
            if isinstance(episode, BaseException):
                if return_exceptions:
                    result.add_exception(episode)
                    continue
                else:
                    raise episode
            if not self._first_train_episode:
                self._first_train_episode = episode
            elif (
                not self.episodes_per_iteration and episode is self._first_train_episode
            ):
                self._train_iterator = ait.chain([episode], self._train_iterator)
                break
            task = asyncio.create_task(self._explore_episode(episode, pbar, priority))
            task.add_done_callback(result.done_callback)
            tasks.append(task)
            if self.episodes_per_iteration is None:
                self.episodes_per_iteration = len(tasks)
                pbar.total = len(tasks)
                pbar.refresh()
            await asyncio.sleep(1e-6)  # yield to other tasks
        for future in asyncio.as_completed(tasks):
            remaining_episodes = self.explore_options.num_samples * (
                len(tasks) - len(result.episodes)
            )
            patience = self.explore_options.patience * remaining_episodes
            try:
                await asyncio.wait_for(future, timeout=patience)
            except asyncio.TimeoutError:
                if verbosity > 0:
                    print(
                        f"Early stopping exploration due to expired patience ({remaining_episodes} remaining episodes x {self.explore_options.patience} patience per episode = {patience} seconds)"
                    )
                for task in tasks:
                    task.cancel()
                break
            except BaseException as exception:
                if return_exceptions:
                    result.add_exception(exception)
                else:
                    raise exception
        self.explore_results.append(result)
        return result.completed()

    async def _explore_episode(
        self, episode: Episode, pbar: tqdm, priority: int
    ) -> Episode:
        if episode in self._substitute_episodes:
            return await self._explore_episode(
                self._substitute_episodes[episode], pbar, priority
            )
        completion_sampler = await self.get_completion_sampler()
        for iteration in range(self.explore_options.iterations):
            await episode.sample_completions_v2(
                completion_sampler=completion_sampler,
                num_parents=self.explore_options.num_parents,
                branch_factor=self.explore_options.branch_factor,
                get_recovery_pattern=self.explore_options.get_recovery_pattern,
                max_splits_per_completion=self.explore_options.max_split_points
                or self.explore_options.num_parents,
                priority=priority,
                sample_probability_power=self.explore_options.get_sample_probability_power(),
                sampling_kwargs=self.explore_options.sampling_kwargs,
                split_by=self.explore_options.split_method,
                split_separators=self.explore_options.split_separators,
            )
            if iteration == 0 and all(
                c.advantage(cache=True, model=self.model) == 0
                for c in episode.completion.children
                if c.model == self.model
            ):
                if (
                    episode.get_easier_episode
                    and episode.min_value is not None
                    and episode.completion.value(cache=True, model=self.model)
                    <= episode.min_value
                ):
                    substitute = await maybe_await(episode.get_easier_episode())
                elif (
                    episode.get_harder_episode
                    and episode.max_value is not None
                    and episode.completion.value(cache=True, model=self.model)
                    >= episode.max_value
                ):
                    substitute = await maybe_await(episode.get_harder_episode())
                elif episode.get_similar_episode:
                    substitute = await maybe_await(episode.get_similar_episode())
                else:
                    continue
                self._substitute_episodes[episode] = substitute
                return await self._explore_episode(substitute, pbar, priority)
            pbar.update(
                1
                / self.explore_options.iterations
                / (
                    2
                    if self.reference_clients and self.reference_model != self.model
                    else 1
                )
            )
        if self.reference_clients and self.reference_model != self.model:
            await self._get_episode_reference_logprobs(
                episode, client=await anext(self.reference_clients)
            )
            pbar.update(0.5)
        return episode

    async def _get_episode_reference_logprobs(
        self,
        episode: Episode,
        client: AsyncOpenAI,
    ) -> Episode:
        leaves = list(episode.completion.leaves(model=self.model))
        for offset in range(self.reference_roundtrips_per_episode):
            await asyncio.gather(
                *(
                    self._get_completion_reference_logprobs(leaf, client)
                    for leaf in leaves[offset :: self.reference_roundtrips_per_episode]
                )
            )
        return episode

    async def _get_completion_reference_logprobs(
        self,
        completion: Completion,
        client: AsyncOpenAI,
    ) -> None:
        tokens = completion.all_tokens(self.tokenizer, cache=True).tolist()
        async with get_semaphore(client):
            plain_completion = await client.completions.create(
                model=self.reference_model,
                prompt=tokens,
                max_tokens=1,
                extra_body={
                    "prompt_logprobs": True,
                },
            )
        prompt_logprobs: list[dict[str, dict[str, Any]]] = plain_completion.choices[0].prompt_logprobs  # type: ignore
        reference_logprobs = [
            (prompt_logprob[str(token)]["logprob"] if prompt_logprob else torch.nan)
            for token, prompt_logprob in zip(tokens, prompt_logprobs)
        ]
        for completion in completion.ancestors(including_self=True, reverse=True):
            count = completion.token_count(self.tokenizer, cache=True)
            completion.reference_logprobs, reference_logprobs = (
                torch.tensor(reference_logprobs[:count]),
                reference_logprobs[count:],
            )

    def tensors(self, episodes: list[Episode]) -> PackedTensors:
        return packed_tensors(
            episodes,
            model=self.model,
            sample_probability_power=self.explore_options.get_sample_probability_power(),
            sequence_length=self.tune_sequence_length,
            trajectories_per_episode=(
                int(
                    self.explore_options.num_samples * self.tune_episode_sample_fraction
                )
                if self.tune_episode_sample_fraction < 1.0
                else None
            ),
            tokenizer=self.tokenizer,
        )

    async def tune(self, result: ExploreResult, *, verbosity: Verbosity = 2) -> None:
        await self.stop_vllms()
        if (
            not self.reference_clients
            and self.reference_model != self.model
            and self.reference_vllm_config
        ):
            await self._get_reference_logprobs(
                result, verbosity, self.reference_vllm_config
            )
            await self.stop_vllms()
        checkpoint_dir = await self._get_checkpoint_dir(self.model)
        self.tune_recipe_config.checkpointer = self._get_checkpointer_config(
            checkpoint_dir,
            checkpoint_files=(
                self.base_model_checkpoint_files
                if self.model != self.base_model
                else None
            ),
        )
        if not self.tune_recipe_config.metric_logger:
            self.tune_recipe_config.metric_logger = (
                ComponentConfig(WandBLogger, **self._wandb_kwargs)
                if self._wandb_kwargs
                else ComponentConfig(DiskLogger, log_dir=self.output_dir + "/logs")
            )
        self.tune_recipe_config.model = ComponentConfig(self.tune_model)
        self.tune_recipe_config.dataset = ComponentConfig(
            PackedDataset, **result.disk_packed_tensors()
        )
        if (
            not self.reference_clients
            and self.reference_model != self.model
            and not self.reference_vllm_config
        ):
            self.tune_recipe_config.reference_checkpointer = (
                self._get_checkpointer_config(
                    await self._get_checkpoint_dir(self.reference_model),
                    checkpoint_files=self.reference_model_checkpoint_files,
                )
            )
        if self.tune_run:
            await self._tune_run(verbosity)
        else:
            cleanup_before_training()
            recipe_main(self.tune_recipe_config)
        self._save(checkpoint_dir)

    async def _tune_run(self, verbosity: Verbosity) -> None:
        dict_config = self.tune_recipe_config.dict_config()
        OmegaConf.save(dict_config, self.output_dir + "/config.yaml")
        args = [
            "tune",
            "run",
            *[
                f"--{key.replace('_', '-')}{f'={value}' if value is not True else ''}"
                for key, value in self.torchrun_kwargs.items()
            ],
            "lib.rl.recipe.TuneRecipe",
            "--config",
            self.output_dir + "/config.yaml",
        ]
        if verbosity > 0:
            print(f"$ {' '.join(args)}")
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={
                **os.environ,
                **self.tune_run_env,
            },
        )

        async def log_output(stream: asyncio.StreamReader, io: IO[str]) -> None:
            output = ""
            while True:
                try:
                    chunk = await stream.read(4096)
                    if not chunk:
                        break
                    output += chunk.decode()
                    if verbosity > 1:
                        io.write(output)
                        io.flush()
                        output = ""
                    elif verbosity == 1:
                        output = output.split("\n")[-1]
                        pbar_regex = re.compile(
                            r"\[(?:\d+:)?\d+:\d+<(?:\d+:)?\d+:\d+.*\]"
                        )
                        if pbar_regex.search(output):
                            io.write(output)
                            io.flush()
                            output = ""
                except Exception:
                    break

        tasks = []
        if process.stdout:
            tasks.append(asyncio.create_task(log_output(process.stdout, sys.stdout)))
        if process.stderr:
            tasks.append(asyncio.create_task(log_output(process.stderr, sys.stderr)))
        _ = await asyncio.gather(*tasks)

    async def _get_reference_logprobs(
        self, result: ExploreResult, verbosity: Verbosity, vllm_config: vLLMConfig
    ) -> None:
        reference_clients = it.cycle(
            vllm.client
            for vllm in await self.get_or_start_vllms(
                model=self.reference_model,
                config=vllm_config,
                verbosity=verbosity,
            )
        )
        exceptions: list[BaseException] = []
        pbar = self.tqdm(
            desc="reference logprobs",
            total=len(result.episodes),
            unit="episode",
            dynamic_ncols=True,
            disable=not verbosity,
        )
        for future in asyncio.as_completed(
            self._get_episode_reference_logprobs(episode, client)
            for episode, client in zip(result.episodes, reference_clients)
        ):
            try:
                episode = await future
                for completion in episode.completion.descendants(including_self=True):
                    result.write_reference_logprobs(completion)
            except BaseException as exception:
                exceptions.append(exception)
            pbar.update(1)
            pbar.set_postfix(exceptions=len(exceptions))
        result.exceptions.extend(exceptions)
        pbar.close()

    async def _get_checkpoint_dir(self, model: str) -> str:
        if os.path.exists(os.path.abspath(model)):
            return os.path.abspath(model)
        process = await asyncio.create_subprocess_shell(
            f"HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download {model}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()
        return stdout.decode().strip()

    def _get_checkpointer_config(
        self, checkpoint_dir: str, checkpoint_files: Optional[list[str]]
    ) -> ComponentConfig[FullModelHFCheckpointer]:
        return ComponentConfig(
            FullModelHFCheckpointer,
            checkpoint_dir=checkpoint_dir,
            checkpoint_files=checkpoint_files
            or [
                file
                for ext in ["safetensors", "pt", "ckpt", "bin", "pth"]
                for file in glob.glob(f"{checkpoint_dir}/*.{ext}")
            ],
            recipe_checkpoint=None,
            output_dir=self.output_dir,
            model_type=self.tune_model_type,
        )

    def _save(self, base_checkpoint_dir: str) -> None:
        # Find the latest epoch number from model checkpoint files
        epoch = max(
            (
                int(result.group(1))
                for result in (
                    re.search(r"hf_model_\d+_(\d+)\.pt", file)
                    for file in glob.glob(f"{self.output_dir}/hf_model_*_*.pt")
                )
                if result
            ),
            default=None,
        )

        assert (
            epoch is not None
        ), f"No model checkpoint files found to save in output directory {self.output_dir}"

        # Find the next iteration number by looking at existing subdirectories
        iteration = (
            max(
                (
                    int(subdir)
                    for subdir in os.listdir(self.output_dir)
                    if os.path.isdir(os.path.join(self.output_dir, subdir))
                    and subdir.isdigit()
                ),
                default=0,
            )
            + 1
        )
        model_name = f"{self.output_dir}/{iteration:04d}"

        # Create a new directory for this iteration
        iteration_dir = f"{self.output_dir}/{iteration:04d}"
        os.makedirs(iteration_dir, exist_ok=True)

        # Copy configuration files (non-model files) to the iteration directory
        for file in os.listdir(base_checkpoint_dir):
            if not any(
                file.endswith(suffix)
                for suffix in (".safetensors", ".pt", ".ckpt", ".bin", ".pth", ".h5")
            ):
                src = os.path.join(base_checkpoint_dir, file)
                dst = os.path.join(iteration_dir, file)
                shutil.copy2(src, dst)

        # Move model checkpoint files to the iteration directory
        for src in glob.glob(f"{self.output_dir}/hf_model_*_{epoch}.pt"):
            dst = f"{iteration_dir}/{os.path.basename(src)}"
            shutil.move(src, dst)

        # Delete remaining model checkpoint files in checkpoint_output_dir root
        for file in glob.glob(f"{self.output_dir}/hf_model_*_*.pt"):
            if os.path.isfile(file):
                os.remove(file)

        print(f"Saved iteration {iteration} model files to {model_name}")
        self.models.append(model_name)

    async def get_completion_sampler(
        self, *, verbosity: Verbosity = 2
    ) -> CompletionSampler:
        if not self._vllm_task:
            self._completion_sampler = None
        if self._completion_sampler:
            return self._completion_sampler
        vllms = await self.get_or_start_vllms(
            model=self.model, config=self.vllm_config, verbosity=verbosity
        )
        self._completion_sampler = CompletionSamplerPool(
            [
                CompletionSampler(
                    vllm.client,
                    max_concurrent_samples=self.vllm_config.max_concurrent_samples,
                    min_time_between_requests=self.vllm_config.min_time_between_requests,
                    model=self.model,
                )
                for vllm in vllms
            ]
        )
        return self._completion_sampler

    async def get_or_start_vllms(
        self, *, model: str, config: vLLMConfig, verbosity: Verbosity = 2
    ) -> list[vLLM]:
        if not self._vllm_task:
            self._vllm_task = asyncio.create_task(
                start_vllms(
                    model=model,
                    n=config.num,
                    timeout=config.timeout,
                    env=config.env,
                    max_concurrent_requests=config.max_concurrent_samples,
                    verbosity=verbosity,
                    **config.kwargs or {},
                )
            )
        try:
            return await self._vllm_task
        except BaseException as exception:
            self._vllm_task = None
            raise exception

    async def stop_vllms(self) -> None:
        if self._vllm_task:
            try:
                vllms = await self._vllm_task
                for vllm in vllms:
                    vllm.process.terminate()
            except BaseException as exception:
                print(type(exception), exception)
            finally:
                self._vllm_task = None
