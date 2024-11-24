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

from .completion_sampler import CompletionSampler
from .episode import Episode
from .pack import PackedDataset, packed_tensors, packed_tensors_to_dir, PackedTensors
from .recipe import ComponentConfig, recipe_main, TuneRecipeConfig
from ..tokenizer import Tokenizer
from ..vllm import start_vllm, vLLM


Episodes = Union[Iterable[Episode], AsyncIterable[Episode]]


class Trainer:
    def __init__(
        self,
        *,
        base_model: str,
        base_model_checkpoint_files: Optional[list[str]] = None,
        output_dir: str,
        samples_per_episode: Optional[int] = None,
        branch_factor: int = 2,
        train_episodes: Episodes,
        episodes_per_iteration: Optional[int] = None,
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
        vllm_max_concurrent_requests: int = 128,
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

    async def train(self, iterations: int, test: bool = False) -> None:
        for _ in range(iterations):
            _, (episodes, _) = await asyncio.gather(
                self.eval("val", 0, return_exceptions=True),
                self.explore(1, return_exceptions=True),
            )
            await self.tune(episodes)
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
        await asyncio.gather(*tasks, return_exceptions=return_exceptions)
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
        await self.stop_vllm()
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

    async def tune_resources(
        self, episodes: list[Episode]
    ) -> tuple[PackedTensors, str, list[str]]:
        await self.stop_vllm()
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
        return tensors, checkpoint_dir, checkpoint_files

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
        vllm = await self.get_or_start_vllm()
        self._completion_sampler = CompletionSampler(
            vllm.client,
            max_concurrent_requests=self.vllm_max_concurrent_requests,
            model=self.model,
        )
        return self._completion_sampler

    async def get_or_start_vllm(self) -> vLLM:
        if not self._vllm_task:
            self._vllm_task = asyncio.create_task(
                start_vllm(
                    self.model,
                    max_concurrent_requests=self.vllm_max_concurrent_requests,
                    **self.vllm_kwargs,
                )
            )
        try:
            return await self._vllm_task
        except BaseException as exception:
            self._vllm_task = None
            raise exception

    async def stop_vllm(self) -> None:
        if self._vllm_task:
            try:
                vllm = await self._vllm_task
                vllm.process.terminate()
            except BaseException as exception:
                print(type(exception), exception)
            finally:
                self._vllm_task = None
