from aioitertools.builtins import iter
from aioitertools import itertools as ait
from aioitertools.helpers import maybe_await
import asyncio
from dataclasses import dataclass
import glob
import json
import nest_asyncio
from omegaconf import OmegaConf
import os
from pydantic import BaseModel, field_serializer, field_validator
import random
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
    Protocol,
    Union,
)
import wandb

from .completion import Completion, SplitMethod
from .completion_sampler import SamplingKwargs, CompletionSampler, CompletionSamplerPool
from .episode import Episode
from .explore_result import ExploreResult
from .mlp_head_checkpointer import MLPHeadCheckpointer
from .pack import packed_tensors, PackedDataset, PackedTensors
from .recipe import ComponentConfig, recipe_main, TuneRecipeConfig
from ..tokenizer import Tokenizer
from ..utils import get_semaphore
from ..vllm import start_vllms, vLLM

nest_asyncio.apply()
Episodes = Union[
    Iterable[Union[Episode, BaseException]],
    AsyncIterable[Union[Episode, BaseException]],
]


@dataclass
class Eval:
    name: str
    episodes: Episodes
    patience: float = 60.0
    samples_per_episode: int = 1
    sampling_kwargs: Optional[SamplingKwargs] = None


class EvalResult(BaseModel):
    name: str
    model: str
    avg: float
    max: float
    entropy: float
    tokens: int
    exceptions: list[BaseException]

    @property
    def wandb_data(self) -> dict[str, Any]:
        return {
            f"{self.name}/avg": self.avg,
            f"{self.name}/max": self.max,
            f"{self.name}/entropy": self.entropy,
            f"{self.name}/tokens": self.tokens,
        }

    @field_serializer("exceptions")
    def serialize_exceptions(self, exceptions: list[BaseException]) -> list[str]:
        return [str(e) for e in exceptions]

    @field_validator("exceptions")
    @classmethod
    def validate_exceptions(cls, exceptions: list[str]) -> list[BaseException]:
        return [Exception(e) for e in exceptions]


class ExploreImpl(Protocol):
    async def __call__(
        self,
        completion_sampler_pool: CompletionSamplerPool,
        tokenizer: Tokenizer,
        ready_episodes: asyncio.Queue[Episode],
        done_episodes: asyncio.Queue[Episode | BaseException],
        update_progress: Callable[[float], None],
    ) -> None: ...


@dataclass
class ExploreOptions:
    """
    During the exploration step, the trainer will sample completions for each episode `iterations` times. The
    total number of sampled completions per episode will be close to `iterations * num_parents * (branch_factor - 1)`.

    Attributes:
    iterations (int): The number of iterations to explore each episode.

    num_parents (int): The number of parent nodes to sample completions from for each episode each iteration.

    branch_factor (int): The number of completions to sample from each parent node.

    advantage_max_weight (float): How much weight to give to the maximum reward-to-go (instead of the Q-value) when
    computing the advantage.

    normalize_values (bool): Whether to normalize the values discovered in the exploration step.

    normalize_advantages (bool): Whether to normalize the advantages discovered in the exploration step.

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
    advantage_max_weight: float = 0.0
    max_split_points: Optional[int] = None
    normalize_values: bool = True
    normalize_advantages: bool = True
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
        explore_impl: ExploreImpl,
        force_terminate_vllms: bool = False,
        train_episodes: Episodes,
        episodes_per_iteration: Optional[int] = None,
        max_mask_sequence_batch_size: Optional[int] = None,
        evals: Optional[Iterable[Eval]] = None,
        torchrun_kwargs: Optional[dict[str, Any]] = None,
        tune_episode_sample_fraction: float = 1.0,
        tune_model: Callable[[], TransformerDecoder],
        tune_model_type: str,
        tune_recipe_configs: list[TuneRecipeConfig] = [],
        tune_run: bool = True,
        tune_run_env: Optional[dict[str, str]] = None,
        tune_sequence_length: int = 8192,
        vllm_config: Optional[vLLMConfig] = None,
        wandb_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self.base_model = base_model
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.models = sorted(
            [
                os.path.join(self.output_dir, subdir)
                for subdir in os.listdir(self.output_dir)
                if os.path.isdir(os.path.join(self.output_dir, subdir))
                and subdir.isdigit()
            ]
        )
        if not self.models:
            checkpoint_dir = asyncio.run(self._get_checkpoint_dir(self.base_model))
            self.models = [
                self._create_iteration_dir(checkpoint_dir, copy_model_files=True)[1]
                for _ in range(torch.cuda.device_count())
            ]
        print(f"Resuming from {self.latest_models}")
        self.base_model_checkpoint_files = base_model_checkpoint_files
        self.explore_options = explore_options
        self.explore_impl = explore_impl
        self.force_terminate_vllms = force_terminate_vllms
        self._train_iterator = ait.cycle(train_episodes)
        self._first_train_episode: Optional[Episode] = None
        self.episodes_per_iteration: Optional[int] = (
            episodes_per_iteration or getattr(train_episodes, "__len__", lambda: None)()
        )
        self.max_mask_sequence_batch_size = max_mask_sequence_batch_size or max(
            32768 // tune_sequence_length, 1
        )
        self.evals = {eval.name: eval for eval in evals or []}
        eval_results_path = os.path.join(self.output_dir, "eval-results.json")
        if os.path.exists(eval_results_path):
            with open(eval_results_path) as f:
                self.eval_results = {k: [EvalResult(**r) for r in v] for k, v in json.load(f).items()}
        else:
            self.eval_results: dict[str, list[EvalResult]] = {}
        self.eval_episodes = {eval.name: eval.episodes for eval in self.evals.values()}
        self.explore_results: list[ExploreResult] = []
        self.torchrun_kwargs = torchrun_kwargs or {}
        self.tune_episode_sample_fraction = tune_episode_sample_fraction
        self.tune_model = tune_model
        self.tune_model_type = tune_model_type
        self.tune_recipe_configs = tune_recipe_configs or [TuneRecipeConfig()]
        self.tune_run = tune_run
        self.tune_run_env = tune_run_env or {}
        self.tune_sequence_length = tune_sequence_length
        self.vllm_config = vllm_config or vLLMConfig()
        self._vllm_task: Optional[asyncio.Task[list[vLLM]]] = None
        self._completion_sampler_pool: Optional[CompletionSamplerPool] = None
        self.tokenizer = Tokenizer(base_model)
        try:
            get_ipython  # type: ignore
            self.tqdm = tqdm_notebook
        except NameError:
            self.tqdm = tqdm
        self._wandb_kwargs = wandb_kwargs.copy() if wandb_kwargs else {}
        if self._wandb_kwargs:
            self._wandb_kwargs["resume"] = "allow"
        self._wandb_run = (
            wandb.run or wandb.init(**self._wandb_kwargs)
            if self._wandb_kwargs
            else None
        )

    @property
    def latest_models(self) -> list[str]:
        return self.models[-torch.cuda.device_count() :]

    async def train(self, iterations: int, verbosity: Verbosity = 2) -> None:
        for _ in range(iterations):
            results = await asyncio.gather(
                *(
                    self.eval(
                        eval_name,
                        pbar_position,
                        verbosity=verbosity,
                    )
                    for pbar_position, eval_name in enumerate(self.evals)
                ),
                self.explore(
                    pbar_position=len(self.evals) * torch.cuda.device_count(),
                    return_exceptions=True,
                    verbosity=verbosity,
                ),
            )
            explore_result = results[-1]
            assert isinstance(explore_result, ExploreResult)
            await self.tune(explore_result, verbosity=verbosity)
        _ = await asyncio.gather(
            *(
                self.eval(
                    eval_name,
                    pbar_position,
                    verbosity=verbosity,
                )
                for pbar_position, eval_name in enumerate(self.evals)
            )
        )

    async def eval(
        self,
        eval_name: str,
        pbar_position: Optional[int] = None,
        *,
        verbosity: Verbosity = 2,
    ) -> EvalResult:
        pool = await self.get_completion_sampler_pool(verbosity=verbosity)
        results = await asyncio.gather(
            *(
                self._eval(
                    eval_name,
                    (pbar_position or 0) * len(pool.samplers) + id,
                    id,
                    completion_sampler,
                    verbosity,
                )
                for id, completion_sampler in enumerate(pool.samplers)
            )
        )
        combined_result = EvalResult(
            name=eval_name,
            model=",".join(result.model for result in results),
            avg=sum(result.avg for result in results) / len(results),
            max=sum(result.max for result in results) / len(results),
            entropy=sum(result.entropy for result in results) / len(results),
            tokens=round(sum(result.tokens for result in results) / len(results)),
            exceptions=[
                exception for result in results for exception in result.exceptions
            ],
        )
        results.append(combined_result)
        for result in results:
            self.eval_results.setdefault(result.name, []).append(result)
        if self._wandb_run:
            wandb.log(
                {
                    key: value
                    for result in results
                    for key, value in result.wandb_data.items()
                }
            )
        with open(f"{self.output_dir}/eval-results.json", "w") as f:
            json.dump(
                {k: [r.model_dump() for r in v] for k, v in self.eval_results.items()},
                f,
            )
        return combined_result

    async def _eval(
        self,
        eval_name: str,
        pbar_position: int,
        id: int,
        completion_sampler: CompletionSampler,
        verbosity: Verbosity,
    ) -> EvalResult:
        pbar = self.tqdm(
            desc=f"{eval_name}/{id}",
            total=getattr(self.eval_episodes[eval_name], "__len__", lambda: None)(),
            unit="episode",
            dynamic_ncols=True,
            position=pbar_position,
            disable=not verbosity,
        )
        tasks: list[asyncio.Task] = []
        episodes: list[Episode] = []
        exceptions: list[BaseException] = []
        model = await completion_sampler.get_model()

        def num_episodes_with_model_completions() -> int:
            return (
                sum(
                    1
                    for episode in episodes
                    if any(
                        child.model == model for child in episode.completion.children
                    )
                )
                or 1
            )

        def num_model_completions() -> int:
            return (
                sum(
                    sum(1 for _ in episode.completion.leaves(models={model}))
                    for episode in episodes
                )
                or 1
            )

        def get_avg_score() -> float:
            return (
                sum(
                    episode.completion.value(cache=True, models={model})
                    for episode in episodes
                )
                / num_episodes_with_model_completions()
            )

        def get_max_score() -> float:
            return (
                sum(
                    max(
                        (
                            sum(
                                completion.reward
                                for completion in leaf.ancestors(including_self=True)
                            )
                            for leaf in episode.completion.leaves(models={model})
                        ),
                        default=0,
                    )
                    for episode in episodes
                )
                / num_episodes_with_model_completions()
            )

        def get_entropy() -> float:
            return (
                sum(
                    leaf.all_entropy(cache=True)
                    for episode in episodes
                    for leaf in episode.completion.leaves(models={model})
                )
                / num_model_completions()
            )

        def get_avg_tokens() -> int:
            return round(
                sum(
                    leaf.all_num_token_logprobs()
                    for episode in episodes
                    for leaf in episode.completion.leaves(models={model})
                )
                / num_model_completions()
            )

        def done_callback(task: asyncio.Task[bool]) -> None:
            pbar.update(1)
            try:
                task.result()
            except Exception as exception:
                exceptions.append(exception)
            pbar.set_postfix(
                avg=get_avg_score(),
                max=get_max_score(),
                entropy=get_entropy(),
                tokens=get_avg_tokens(),
                exceptions=len(exceptions),
            )

        sampling_kwargs = (self.evals[eval_name].sampling_kwargs or {}).copy()
        sampling_kwargs["top_logprobs"] = 20

        async for episode in iter(
            self.eval_episodes[eval_name] or (),
        ):
            if isinstance(episode, BaseException):
                exceptions.append(episode)
                continue
            episodes.append(episode)
            samples = self.evals[eval_name].samples_per_episode
            task = asyncio.create_task(
                episode.sample_completions(
                    completion_sampler,
                    tokenizer=self.tokenizer,
                    num_parents=1,
                    branch_factor=samples,
                    sampling_kwargs=sampling_kwargs,
                )
                if episode.num_samples(models={model}) < samples
                else maybe_await(True)
            )
            task.add_done_callback(done_callback)
            tasks.append(task)
            await asyncio.sleep(1e-6)
        pbar.total = len(episodes)
        pbar.refresh()
        self.eval_episodes[eval_name] = episodes
        for future in asyncio.as_completed(tasks):
            remaining_episodes = pbar.total - pbar.n
            patience = self.evals[eval_name].patience * remaining_episodes
            try:
                await asyncio.wait_for(future, timeout=patience)
            except asyncio.TimeoutError:
                if verbosity > 0:
                    print(
                        f"Early stopping {eval_name} evaluation due to expired patience ({remaining_episodes} remaining episodes x {self.evals[eval_name].patience} patience per episode = {patience} seconds)"
                    )
                for task in tasks:
                    task.cancel()
                break
            except BaseException as exception:
                exceptions.append(exception)
        pbar.close()
        avg_score = get_avg_score()
        return EvalResult(
            name=f"{eval_name}/{id}",
            model=model,
            avg=avg_score,
            max=get_max_score(),
            entropy=get_entropy(),
            tokens=get_avg_tokens(),
            exceptions=exceptions,
        )

    async def explore(
        self,
        pbar_position: Optional[int] = None,
        *,
        return_exceptions: bool = True,
        verbosity: Verbosity = 2,
    ) -> ExploreResult:
        pool = await self.get_completion_sampler_pool(verbosity=verbosity)
        pbar = self.tqdm(
            desc="explore",
            total=self.episodes_per_iteration,
            unit="episode",
            dynamic_ncols=True,
            position=pbar_position,
            disable=not verbosity,
        )
        pbar.set_postfix(completed=0, exceptions=0, max=0.0)
        tasks: list[asyncio.Task[Episode]] = []
        result = ExploreResult(
            pbar=pbar,
            max_mask_sequence_batch_size=self.max_mask_sequence_batch_size,
            models=set(self.latest_models),
            abs_weighted_sum=200_000.0,
            advantage_max_weight=self.explore_options.advantage_max_weight,
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
            normalize_values=self.explore_options.normalize_values,
            normalize_advantages=self.explore_options.normalize_advantages,
        )
        ready_episodes = asyncio.Queue[Episode]()
        done_episodes = asyncio.Queue[Episode | BaseException]()
        explore_task = asyncio.create_task(
            self.explore_impl(
                pool,
                self.tokenizer,
                ready_episodes,
                done_episodes,
                lambda progress: pbar.update(progress),  # type: ignore
            )
        )

        async def put_ready_episode(episode: Episode) -> Episode:
            await ready_episodes.put(episode)
            episode_or_exception = await done_episodes.get()
            if isinstance(episode_or_exception, BaseException):
                raise episode_or_exception
            return episode_or_exception

        async for episode in ait.islice(
            self._train_iterator, self.episodes_per_iteration
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
            task = asyncio.create_task(put_ready_episode(episode))
            task.add_done_callback(result.done_callback)
            tasks.append(task)
            await asyncio.sleep(1e-6)  # yield to other tasks
        if self.episodes_per_iteration is None:
            self.episodes_per_iteration = len(tasks)
            pbar.total = len(tasks)
            pbar.refresh()
        for future in asyncio.as_completed(tasks):
            remaining_episodes = len(tasks) - len(result.episodes)
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
                if not return_exceptions:
                    explore_task.cancel()
                    raise exception
        explore_task.cancel()
        self.explore_results.append(result)
        with open(f"{self.output_dir}/explore-results.jsonl", "a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")
        return result.completed()

    async def tune(self, result: ExploreResult, *, verbosity: Verbosity = 2) -> None:
        print(f"Tuning model on {len(result.sequences)} sequences")
        await self.stop_vllms()
        model_names = await asyncio.gather(
            *[
                self._tune_model(
                    result,
                    model,
                    id,
                    self.tune_recipe_configs[id % len(self.tune_recipe_configs)],
                    verbosity,
                )
                for id, model in enumerate(self.latest_models)
            ]
        )
        self.models.extend(model_names)

    async def _tune_model(
        self,
        result: ExploreResult,
        model: str,
        id: int,
        config: TuneRecipeConfig,
        verbosity: Verbosity,
    ) -> str:
        checkpoint_dir = await self._get_checkpoint_dir(model)
        base_checkpoint_dir = await self._get_checkpoint_dir(self.base_model)
        output_subdir = f"/cuda:{id}"
        os.makedirs(self.output_dir + output_subdir, exist_ok=True)
        config.checkpointer = self._get_checkpointer_config(
            checkpoint_dir,
            checkpoint_files=(
                self.base_model_checkpoint_files if model != self.base_model else None
            ),
            mlp_head_checkpointer=True,
            output_subdir=output_subdir,
        )
        config.reference_checkpointer = self._get_checkpointer_config(
            base_checkpoint_dir,
            checkpoint_files=self.base_model_checkpoint_files,
        )
        if not config.metric_logger:
            config.metric_logger = (
                ComponentConfig(WandBLogger, **self._wandb_kwargs)
                if self._wandb_kwargs
                else ComponentConfig(DiskLogger, log_dir=self.output_dir + "/logs")
            )
        config.model = ComponentConfig(self.tune_model)
        config.dataset = ComponentConfig(
            PackedDataset, **result.disk_packed_tensors(drop_last=True)
        )
        config.loss.model_id = min(id, len(set(self.latest_models)) - 1)
        config.seed = random.randint(0, 2**32 - 1)
        if self.tune_run:
            dict_config = config.dict_config()
            config_path = f"{self.output_dir}{output_subdir}/config.yaml"
            OmegaConf.save(dict_config, config_path)
            await self._tune_run(
                config_path=config_path,
                id=id,
                total=min(
                    len(result.sequences),
                    config.max_steps_per_epoch or 1024,
                ),
                verbosity=verbosity,
            )
        else:
            cleanup_before_training()
            recipe_main(config)
        return self._save(checkpoint_dir, output_subdir)

    async def _tune_run(
        self, config_path: str, id: int, total: int, verbosity: Verbosity
    ) -> None:
        args = [
            "tune",
            "run",
            *[
                f"--{key.replace('_', '-')}{f'={value}' if value is not True else ''}"
                for key, value in self.torchrun_kwargs.items()
            ],
            "lib.rl.recipe.TuneRecipe",
            "--config",
            config_path,
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
                "CUDA_VISIBLE_DEVICES": str(id),
            },
        )
        if verbosity == 1:
            pbar = self.tqdm(
                total=total,
                position=id,
            )
        else:
            pbar = None

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
                        if pbar:
                            pbar_start = re.compile(r"(\d+)\|(\d+)\|Loss: ([\d.]+):")
                            if match := pbar_start.search(output):
                                epoch, step, loss = match.groups()
                                pbar.update(int(step) - pbar.n)
                                pbar.set_description(f"{epoch}|{step}|Loss: {loss}")
                            metrics = {
                                key: value
                                for key, value in re.findall(r"(\w+)=([\d.-]+)", output)
                            }
                            if metrics:
                                pbar.set_postfix(**metrics)
                                output = ""
                        else:
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
        if pbar:
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
        self,
        checkpoint_dir: str,
        checkpoint_files: Optional[list[str]],
        mlp_head_checkpointer: bool = False,
        output_subdir: str = "",
    ) -> ComponentConfig[FullModelHFCheckpointer]:
        return ComponentConfig(
            MLPHeadCheckpointer if mlp_head_checkpointer else FullModelHFCheckpointer,
            checkpoint_dir=checkpoint_dir,
            checkpoint_files=checkpoint_files
            or [
                file
                for ext in ["safetensors", "pt", "ckpt", "bin", "pth"]
                for file in glob.glob(f"{checkpoint_dir}/*.{ext}")
                if not file.endswith("mlp_head.pt")
            ],
            recipe_checkpoint=None,
            output_dir=self.output_dir + output_subdir,
            model_type=self.tune_model_type,
        )

    def _save(self, base_checkpoint_dir: str, output_subdir: str) -> str:
        """
        Saves and returns the directory of the latest checkpoint.
        """
        # Find the latest epoch number from model checkpoint files
        epoch = max(
            (
                int(result.group(1))
                for result in (
                    re.search(r"hf_model_\d+_(\d+)\.pt", file)
                    for file in glob.glob(
                        f"{self.output_dir}{output_subdir}/hf_model_*_*.pt"
                    )
                )
                if result
            ),
            default=None,
        )

        assert (
            epoch is not None
        ), f"No model checkpoint files found to save in output directory {self.output_dir}{output_subdir}"

        iteration, iteration_dir = self._create_iteration_dir(base_checkpoint_dir)

        # Move model checkpoint files to the iteration directory
        for src in [
            path
            for extension in ("pt", "pt.ignore")
            for path in glob.glob(
                f"{self.output_dir}{output_subdir}/*_{epoch}.{extension}"
            )
        ]:
            dst = f"{iteration_dir}/{os.path.basename(src).replace(f'_{epoch}.pt', '.pt')}"
            shutil.move(src, dst)

        # Delete entire output subdirectory
        output_subdir_path = f"{self.output_dir}{output_subdir}"
        if os.path.exists(output_subdir_path):
            shutil.rmtree(output_subdir_path)

        print(f"Saved iteration {iteration} model files to {iteration_dir}")
        return iteration_dir

    def _create_iteration_dir(
        self, base_checkpoint_dir: str, copy_model_files: bool = False
    ) -> tuple[int, str]:
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

        # Create a new directory for this iteration
        iteration_dir = f"{self.output_dir}/{iteration:04d}"
        os.makedirs(iteration_dir, exist_ok=True)

        # Copy configuration files (non-model files) to the iteration directory
        for file in os.listdir(base_checkpoint_dir):
            if not any(
                file.endswith(suffix)
                for suffix in (
                    ()
                    if copy_model_files
                    else (".safetensors", ".pt", ".ckpt", ".bin", ".pth", ".h5")
                )
            ):
                src = os.path.join(base_checkpoint_dir, file)
                dst = os.path.join(iteration_dir, file)
                shutil.copy2(src, dst)

        return iteration, iteration_dir

    async def get_completion_sampler_pool(
        self, *, verbosity: Verbosity = 2
    ) -> CompletionSamplerPool:
        if not self._vllm_task:
            self._completion_sampler_pool = None
        if self._completion_sampler_pool:
            return self._completion_sampler_pool
        vllms = await self.get_or_start_vllms(
            models=self.latest_models,
            config=self.vllm_config,
            verbosity=verbosity,
        )
        if self._completion_sampler_pool:
            return self._completion_sampler_pool
        self._completion_sampler_pool = CompletionSamplerPool(
            [
                CompletionSampler(
                    vllm.client,
                    max_concurrent_tokens=vllm.max_concurrent_tokens,
                    min_time_between_requests=self.vllm_config.min_time_between_requests,
                    model=model,
                )
                for vllm, model in zip(vllms, self.latest_models)
            ]
        )
        return self._completion_sampler_pool

    async def get_or_start_vllms(
        self, *, models: list[str], config: vLLMConfig, verbosity: Verbosity = 2
    ) -> list[vLLM]:
        if not self._vllm_task:
            self._vllm_task = asyncio.create_task(
                start_vllms(
                    models=models,
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

    async def stop_vllms(self, *, timeout: float = 5.0) -> None:
        if self._vllm_task:
            try:
                vllms = await self._vllm_task
                for vllm in vllms:
                    vllm.process.terminate()
                    # Forcefully terminate any remaining GPU processes
                    if self.force_terminate_vllms:
                        os.system(
                            "nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9"
                        )
                    await asyncio.wait_for(vllm.process.wait(), timeout=timeout)
            except BaseException as exception:
                print(
                    "Experienced the following exception while stopping vLLM servers:",
                    type(exception),
                    exception,
                )
            finally:
                self._vllm_task = None
