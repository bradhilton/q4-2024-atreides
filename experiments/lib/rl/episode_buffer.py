import asyncio
from typing import Any, Coroutine, Literal, Optional, Union

from .completion import SplitMethod
from .completion_sampler import CompletionSampler
from .episode import Episode
from .episode_sampler import EpisodeSampler, EpisodeSamplerRouter

# from .trajectory import Trajectory
from ..tokenizer import Tokenizer
from ..vllm import vllm_server_metrics


class EpisodeBuffer:
    def __init__(
        self,
        episode_sampler_router: EpisodeSamplerRouter,
        completion_sampler: CompletionSampler,
        tokenizer: Tokenizer,
        branch_factor: int,
        split_method: SplitMethod,
        split_separators: Optional[set[str]],
        size: int,
        episode_decay: float,
        completion_decay: float,
        fork_decay: float,
        sleep_time: float = 5.0,
        start_buffering: bool = False,
    ) -> None:
        self.episode_sampler_router = episode_sampler_router
        self.completion_sampler = completion_sampler
        self.tokenizer = tokenizer
        self.branch_factor = branch_factor
        self.split_method: SplitMethod = split_method
        self.split_separators = split_separators
        self.size = size
        self.episode_decay = episode_decay
        self.completion_decay = completion_decay
        self.fork_decay = fork_decay
        self.sleep_time = sleep_time
        self.pending_episodes: list[tuple[asyncio.Task[Episode], EpisodeSampler]] = []
        self.episodes: list[Episode] = []
        self.tasks: dict[Episode, asyncio.Task[bool]] = {}
        self.episode_async_sample_exceptions: list[Exception] = []
        self.max_running_requests = 0
        self.buffer_task = (
            asyncio.create_task(self.buffer_indefinitely()) if start_buffering else None
        )

    def start_buffering(self) -> bool:
        if self.buffer_task and not self.buffer_task.done():
            return False
        self.buffer_task = asyncio.create_task(self.buffer_indefinitely())
        return True

    def stop_buffering(self) -> bool:
        return self.buffer_task.cancel() if self.buffer_task else False

    def weighted_size(self) -> float:
        return len(self.pending_episodes) + sum(
            episode.weight
            for episode in self.episodes
            if result(self.tasks[episode]) != False
        )

    async def buffer_indefinitely(self) -> None:
        while True:
            for task, episode_sampler in self.pending_episodes:
                if task.done():
                    try:
                        episode = task.result()
                        self.tasks[episode] = asyncio.create_task(
                            self.process_new_episode(episode, episode_sampler)
                        )
                    except Exception as exception:
                        self.episode_async_sample_exceptions.append(exception)
            self.pending_episodes = [
                (task, episode_sampler)
                for task, episode_sampler in self.pending_episodes
                if not task.done()
            ]
            for _ in range(self.size - int(self.weighted_size())):
                episode_sampler = self.episode_sampler_router.get_sampler()
                episode = episode_sampler.sample()
                episode_sampler.num_samples += 1
                if isinstance(episode, Coroutine):
                    self.pending_episodes.append(
                        (asyncio.create_task(episode), episode_sampler)
                    )
                else:
                    self.tasks[episode] = asyncio.create_task(
                        self.process_new_episode(episode, episode_sampler)
                    )
            running_requests, pending_requests = vllm_server_metrics()
            self.max_running_requests = max(self.max_running_requests, running_requests)
            pending_requests = max(
                sum(1 for episode in self.episodes if not self.tasks[episode].done()),
                pending_requests,
            )
            for episode in sorted(
                (
                    episode
                    for episode in self.episodes
                    if self.tasks[episode].done()
                    and result(self.tasks[episode]) != False
                ),
                key=lambda episode: episode.num_samples(),
            )[: self.max_running_requests - pending_requests]:
                self.tasks[episode] = asyncio.create_task(
                    episode.sample_completions(
                        self.completion_sampler,
                        self.tokenizer,
                        self.branch_factor,
                        self.fork_decay,
                        self.split_method,
                        self.split_separators,
                    )
                )
            await asyncio.sleep(self.sleep_time)

    async def process_new_episode(
        self, episode: Episode, episode_sampler: EpisodeSampler
    ) -> bool:
        self.episodes.append(episode)
        if not await episode.sample_completions(
            self.completion_sampler,
            self.tokenizer,
            self.branch_factor,
            self.fork_decay,
            self.split_method,
            self.split_separators,
        ):
            return self.episodes.remove(episode) or False
        goldilocks = True
        if (
            episode.get_easier_episode
            and episode.min_value is not None
            and episode.completion.value() <= episode.min_value
        ):
            self.episode_sampler_router.other_samplers.append(
                EpisodeSampler(episode.get_easier_episode)
            )
            goldilocks = False
        elif (
            episode.get_harder_episode
            and episode.max_value is not None
            and episode.completion.value() >= episode.max_value
        ):
            self.episode_sampler_router.other_samplers.append(
                EpisodeSampler(episode.get_harder_episode)
            )
            goldilocks = False
        elif episode.get_similar_episode:
            self.episode_sampler_router.other_samplers.append(
                EpisodeSampler(episode.get_similar_episode)
            )
        if all(c.advantage() == 0 for c in episode.completion.children):
            return self.episodes.remove(episode) or False
        if goldilocks:
            episode_sampler.num_goldilocks += 1
        return True

    # def trajectories(self) -> list[Trajectory]:
    #     return sorted(
    #         (
    #             episode.best_trajectory(
    #                 self.tokenizer,
    #                 episode_decay=self.episode_decay,
    #                 completion_decay=self.completion_decay,
    #             )
    #             for episode in self.episodes
    #         ),
    #         key=lambda trajectory: trajectory.score(),
    #     )

    def __del__(self) -> None:
        if self.buffer_task:
            self.buffer_task.cancel()


def result(task: asyncio.Task) -> Any:
    try:
        return task.result()
    except:
        return None
