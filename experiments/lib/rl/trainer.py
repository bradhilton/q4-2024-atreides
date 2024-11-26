from aioitertools.builtins import iter, enumerate as aenumerate
from aioitertools import itertools as ait
from aioitertools.helpers import maybe_await
import asyncio
import glob
import math
from omegaconf import OmegaConf
import os
import re
import shutil
import sys
from torchtune.modules import TransformerDecoder
from torchtune.training import FullModelHFCheckpointer
from torchtune.training.metric_logging import DiskLogger
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

from .completion import SplitMethod
from .completion_sampler import CompletionSampler, CompletionSamplerPool
from .episode import Episode
from .explore_result import ExploreResult
from .pack import packed_tensors, PackedDataset, PackedTensors, packed_tensors_to_dir
from .recipe import ComponentConfig, recipe_main, TuneRecipeConfig
from ..tokenizer import Tokenizer
from ..vllm import start_vllms, vLLM


Episodes = Union[
    Iterable[Union[Episode, BaseException]],
    AsyncIterable[Union[Episode, BaseException]],
]


class Trainer:
    def __init__(
        self,
        *,
        base_model: str,
        base_model_checkpoint_files: Optional[list[str]] = None,
        output_dir: str,
        samples_per_episode: Optional[int] = None,
        branch_factor: int = 2,
        sample_probability_power: float = 1.0,
        split_method: SplitMethod = "count",
        split_separators: Optional[set[str]] = None,
        train_episodes: Episodes,
        episodes_per_iteration: Optional[int] = None,
        patience_per_sample: float = 1.0,
        max_mask_sequence_batch_size: Optional[int] = None,
        val_episodes: Optional[Episodes] = None,
        val_samples_per_episode: int = 1,
        test_episodes: Optional[Episodes] = None,
        test_samples_per_episode: int = 1,
        torchrun_kwargs: Optional[dict[str, Any]] = None,
        tune_episode_sample_fraction: float = 1.0,
        tune_model: Callable[[], TransformerDecoder],
        tune_model_type: str,
        tune_recipe_config: Optional[TuneRecipeConfig] = None,
        tune_run: bool = True,
        tune_run_env: Optional[dict[str, str]] = None,
        tune_sequence_length: int = 8192,
        vllm_env: Optional[dict[str, str]] = None,
        vllm_kwargs: Optional[dict[str, Any]] = None,
        vllm_max_concurrent_samples: int = 128,
        vllm_min_time_between_requests: float = 0.0,
        vllm_num: int = 1,
        vllm_timeout: float = 120.0,
    ) -> None:
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
        self.samples_per_episode = samples_per_episode or branch_factor
        self.branch_factor = branch_factor
        self.completion_sample_probability_power = sample_probability_power
        self.split_by: SplitMethod = split_method
        self.split_separators = split_separators
        self._train_iterator = ait.cycle(train_episodes)
        self._first_train_episode: Optional[Episode] = None
        self.episodes_per_iteration: Optional[int] = (
            episodes_per_iteration or getattr(train_episodes, "__len__", lambda: None)()
        )
        self.patience_per_sample = patience_per_sample
        self.max_mask_sequence_batch_size = max_mask_sequence_batch_size or max(
            32768 // tune_sequence_length, 1
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
        self.torchrun_kwargs = torchrun_kwargs or {}
        self.tune_episode_sample_fraction = tune_episode_sample_fraction
        self.tune_model = tune_model
        self.tune_model_type = tune_model_type
        self.tune_recipe_config = tune_recipe_config or TuneRecipeConfig()
        self.tune_run = tune_run
        self.tune_run_env = tune_run_env or {}
        self.tune_sequence_length = tune_sequence_length
        self.vllm_kwargs = vllm_kwargs or {}
        self.vllm_kwargs["env"] = vllm_env
        self.vllm_kwargs["timeout"] = vllm_timeout
        self.vllm_max_concurrent_requests = vllm_max_concurrent_samples
        self.vllm_min_time_between_requests = vllm_min_time_between_requests
        self.vllm_num = vllm_num
        self._vllm_task: Optional[asyncio.Task[list[vLLM]]] = None
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

    async def train(self, iterations: int, test: bool = False) -> None:
        for _ in range(iterations):
            _, result = await asyncio.gather(
                self.eval("val", 0, return_exceptions=True),
                self.explore(1, return_exceptions=True),
            )
            await self.tune(result.episodes)
        _, _ = await asyncio.gather(
            self.eval("val", 0, return_exceptions=True),
            self.eval("test", 1) if test else asyncio.sleep(0),
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
    ) -> tuple[Optional[float], list[BaseException]]: ...

    async def eval(
        self,
        split: Literal["val", "test"],
        pbar_position: Optional[int] = None,
        *,
        return_exceptions: bool = True,
    ) -> Union[Optional[float], tuple[Optional[float], list[BaseException]]]:
        if self.eval_episodes[split] is None:
            if return_exceptions:
                return None, []
            return None
        completion_sampler = await self.get_completion_sampler()
        pbar = self.tqdm(
            desc=split,
            total=getattr(self.eval_episodes[split], "__len__", lambda: None)(),
            unit="episode",
            dynamic_ncols=True,
            position=pbar_position,
        )
        tasks: list[asyncio.Task] = []
        episodes: list[Episode] = []
        exceptions: list[BaseException] = []

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
            task = asyncio.create_task(
                episode.sample_completions_v2(
                    completion_sampler,
                    branch_factor=self.eval_samples_per_episode[split],
                )
            )
            task.add_done_callback(done_callback)
            tasks.append(task)
            await asyncio.sleep(1e-6)
        pbar.total = len(episodes)
        pbar.refresh()
        self.eval_episodes[split] = episodes
        await asyncio.gather(*tasks, return_exceptions=return_exceptions)
        pbar.close()
        score = get_score()
        self.eval_scores[split][self.model] = get_score()
        if return_exceptions:
            return score, exceptions
        return score

    async def explore(
        self, pbar_position: Optional[int] = None, *, return_exceptions: bool = True
    ) -> ExploreResult:
        await self.get_completion_sampler()
        pbar = self.tqdm(
            desc="explore",
            total=self.episodes_per_iteration,
            unit="episode",
            dynamic_ncols=True,
            position=pbar_position,
        )
        pbar.set_postfix(completed=0, exceptions=0)
        tasks: list[asyncio.Task[Episode]] = []
        result = ExploreResult(
            pbar=pbar,
            max_mask_sequence_batch_size=self.max_mask_sequence_batch_size,
            model=self.model,
            sample_probability_power=self.completion_sample_probability_power,
            sequence_length=self.tune_sequence_length,
            tensor_dir=self.output_dir + "/tensors",
            tokenizer=self.tokenizer,
            trajectories_per_episode=(
                int(self.samples_per_episode * self.tune_episode_sample_fraction)
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
            remaining_samples = self.samples_per_episode * (pbar.total - pbar.n)
            patience = self.patience_per_sample * remaining_samples
            try:
                await asyncio.wait_for(future, timeout=patience)
            except asyncio.TimeoutError:
                print(
                    f"Early stopping exploration due to expired patience ({remaining_samples} remaining samples x {self.patience_per_sample} patience per sample = {patience} seconds)"
                )
                for task in tasks:
                    task.cancel()
                break
            except BaseException as exception:
                if return_exceptions:
                    result.add_exception(exception)
                else:
                    raise exception
        return result.completed()

    async def _explore_episode(
        self, episode: Episode, pbar: tqdm, priority: int
    ) -> Episode:
        if episode in self._substitute_episodes:
            return await self._explore_episode(
                self._substitute_episodes[episode], pbar, priority
            )
        completion_sampler = await self.get_completion_sampler()
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
                priority=priority,
                sample_probability_power=self.completion_sample_probability_power,
                split_by=self.split_by,
                split_separators=self.split_separators,
            ):
                break
            await asyncio.sleep(1e-6)
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

    def tensors(self, episodes: list[Episode]) -> PackedTensors:
        return packed_tensors(
            episodes,
            model=self.model,
            sample_probability_power=self.completion_sample_probability_power,
            sequence_length=self.tune_sequence_length,
            trajectories_per_episode=(
                int(self.samples_per_episode * self.tune_episode_sample_fraction)
                if self.tune_episode_sample_fraction < 1.0
                else None
            ),
            tokenizer=self.tokenizer,
        )

    async def tune(self, episodes: list[Episode]) -> None:
        await self.stop_vllms()
        tensors = packed_tensors(
            episodes,
            model=self.model,
            sample_probability_power=self.completion_sample_probability_power,
            sequence_length=self.tune_sequence_length,
            trajectories_per_episode=(
                int(self.samples_per_episode * self.tune_episode_sample_fraction)
                if self.tune_episode_sample_fraction < 1.0
                else None
            ),
            tokenizer=self.tokenizer,
        )
        if os.path.exists(os.path.abspath(self.model)):
            checkpoint_dir = os.path.abspath(self.model)
            checkpoint_files = None
        else:
            process = await asyncio.create_subprocess_shell(
                f"HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download {self.model}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await process.communicate()
            checkpoint_dir = stdout.decode().strip()
            checkpoint_files = self.base_model_checkpoint_files
        checkpoint_files = checkpoint_files or [
            file
            for ext in ["safetensors", "pt", "ckpt", "bin", "pth"]
            for file in glob.glob(f"{checkpoint_dir}/*.{ext}")
        ]
        self.tune_recipe_config.checkpointer = ComponentConfig(
            FullModelHFCheckpointer,
            checkpoint_dir=checkpoint_dir,
            checkpoint_files=checkpoint_files,
            recipe_checkpoint=None,
            output_dir=self.output_dir,
            model_type=self.tune_model_type,
        )
        if not self.tune_recipe_config.metric_logger:
            self.tune_recipe_config.metric_logger = ComponentConfig(
                DiskLogger, log_dir=self.output_dir + "/logs"
            )
        self.tune_recipe_config.model = ComponentConfig(self.tune_model)
        self.tune_recipe_config.dataset = ComponentConfig(
            PackedDataset,
            **packed_tensors_to_dir(tensors, self.output_dir + "/tensors"),
        )
        if self.tune_run:
            dict_config = self.tune_recipe_config.dict_config()
            OmegaConf.save(dict_config, self.output_dir + "/config.yaml")
            args = [
                "tune",
                "run",
                *[
                    f"--{key.replace('_', '-')}{f'={value}' if value is not True else ''}"
                    for key, value in self.torchrun_kwargs.items()
                ],
                "lib.recipes.rl.RLRecipe",
                "--config",
                self.output_dir + "/config.yaml",
            ]
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
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    decoded_line = line.decode()
                    io.write(decoded_line)
                    io.flush()

            tasks = []
            if process.stdout:
                tasks.append(
                    asyncio.create_task(log_output(process.stdout, sys.stdout))
                )
            if process.stderr:
                tasks.append(
                    asyncio.create_task(log_output(process.stderr, sys.stderr))
                )
            _ = await asyncio.gather(*tasks)
        else:
            recipe_main(self.tune_recipe_config)
        self.save(checkpoint_dir)

    def save(self, base_checkpoint_dir: str) -> None:
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

        if epoch is None:
            print(
                f"No model checkpoint files found to save in output directory {self.output_dir}"
            )
            return

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

    async def get_completion_sampler(self) -> CompletionSampler:
        if not self._vllm_task:
            self._completion_sampler = None
        if self._completion_sampler:
            return self._completion_sampler
        vllms = await self.get_or_start_vllms()
        self._completion_sampler = CompletionSamplerPool(
            [
                CompletionSampler(
                    vllm.client,
                    max_concurrent_samples=self.vllm_max_concurrent_requests,
                    min_time_between_requests=self.vllm_min_time_between_requests,
                    model=self.model,
                )
                for vllm in vllms
            ]
        )
        return self._completion_sampler

    async def get_or_start_vllms(self) -> list[vLLM]:
        if not self._vllm_task:
            self._vllm_task = asyncio.create_task(
                start_vllms(
                    model=self.model,
                    n=self.vllm_num,
                    max_concurrent_requests=self.vllm_max_concurrent_requests,
                    **self.vllm_kwargs,
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
