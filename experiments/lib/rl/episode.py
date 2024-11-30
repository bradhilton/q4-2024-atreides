import asyncio
from dataclasses import dataclass
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from typing import (
    Any,
    Callable,
    Coroutine,
    Literal,
    Optional,
    overload,
    Protocol,
    Sequence,
)

from .completion import Completion, SplitMethod
from .completion_sampler import SamplingKwargs, CompletionSampler

from ..tokenizer import Tokenizer


@dataclass
class EpisodeCompletion:
    _completion: Completion
    _sampler: CompletionSampler
    _sampling_kwargs: Optional[SamplingKwargs] = None
    _priority: Optional[int] = None

    @property
    def last_assistant_message(self) -> ChatCompletionAssistantMessageParam:
        return next(
            message
            for message in reversed(self._completion.all_message_params())
            if message["role"] == "assistant"
        )

    @property
    def messages(self) -> list[ChatCompletionMessageParam]:
        return self._completion.all_message_params()

    @property
    def reward(self) -> float:
        return self._completion.reward

    @reward.setter
    def reward(self, value: float) -> None:
        self._completion.reward = value

    def commit(self, reward: Optional[float] = None) -> None:
        self._completion.commit(reward)

    async def follow_up(
        self, messages: list[ChatCompletionMessageParam]
    ) -> "EpisodeCompletion":
        completions = await self._sampler.sample_completions(
            Completion(parent=self._completion, messages=messages),
            priority=self._priority or 0,
            **self._sampling_kwargs or {},
        )
        return EpisodeCompletion(_completion=completions[0], _sampler=self._sampler)


class SampleEpisode(Protocol):
    def __call__(self) -> "Episode" | Coroutine[Any, Any, "Episode"]: ...


class Episode:
    def __init__(
        self,
        messages: list[ChatCompletionMessageParam],
        on_sample: Callable[
            [list[EpisodeCompletion]], None | Coroutine[None, None, None]
        ],
        get_easier_episode: Optional[tuple[float, SampleEpisode]] = None,
        get_similar_episode: Optional[SampleEpisode] = None,
        get_harder_episode: Optional[tuple[float, SampleEpisode]] = None,
    ) -> None:
        self.completion = Completion(messages=messages)  # type: ignore
        self.on_sample = on_sample
        self.min_value = (get_easier_episode or [None])[0]
        self.max_value = (get_harder_episode or [None])[0]
        self.get_easier_episode = (get_easier_episode or [None, None])[1]
        self.get_similar_episode = get_similar_episode
        self.get_harder_episode = (get_harder_episode or [None, None])[1]
        self.weight = 1.0

    def __repr__(self) -> str:
        parts = [f"samples={self.num_samples()}"]
        if self.min_value is not None:
            parts.append(f"min_value={self.min_value}")
        if self.max_value is not None:
            parts.append(f"max_value={self.max_value}")
        parts.append(f"weight={self.weight}")
        return f"Episode({', '.join(parts)})"

    def __str__(self) -> str:
        return self.__repr__()

    def __hash__(self) -> int:
        return id(self)

    def num_samples(self, model: Optional[str] = None) -> int:
        if len(self.completion.children) == 0:
            return 0
        return sum(1 for _ in self.completion.leaves(model=model))

    async def sample_completions(
        self,
        completion_sampler: CompletionSampler,
        tokenizer: Tokenizer,
        branch_factor: int,
        fork_decay: float = 1.0,
        split_by: SplitMethod = "count",
        split_separators: Optional[set[str]] = None,
    ) -> bool:
        model = await completion_sampler.get_model()
        parent = self.get_sampleable_parent(
            model, tokenizer, split_by, split_separators
        )
        if parent is None:
            return False
        return await self._sample_completions(
            parent,
            model,
            completion_sampler,
            branch_factor,
            fork_decay,
            None,
            split_separators or set(),
            {},
        )

    def get_sampleable_parent(
        self,
        model: str,
        tokenizer: Tokenizer,
        split_method: SplitMethod,
        split_separators: Optional[set[str]],
    ) -> Optional[Completion]:
        if not any(child.model == model for child in self.completion.children):
            return self.completion
        try:
            leaf = self.best_leaf(
                tokenizer,
                model=model,
                split_method=split_method,
                split_separators=split_separators,
                where_leaf_or_ancestor_is_splittable=True,
            )
        except ValueError:
            return None
        try:
            parent = max(
                (
                    c
                    for c in leaf.ancestors(including_self=True)
                    if c.can_split(by=split_method, separators=split_separators)
                ),
                key=lambda c: abs(c.advantage()) * c.split_weight(by=split_method),
            )
            assert parent.split(
                by=split_method, separators=split_separators
            ), "Unable to split completion"
            return parent
        except BaseException as e:
            print(type(e), e)

    def best_leaf(
        self,
        tokenizer: Tokenizer,
        *,
        model: Optional[str] = None,
        split_method: SplitMethod,
        split_separators: Optional[set[str]] = None,
        where_leaf_or_ancestor_is_splittable: Optional[Literal[True]] = None,
        cache: bool = True,
    ) -> Completion:
        return max(
            (
                completion
                for completion in self.completion.leaves(model=model)
                if not where_leaf_or_ancestor_is_splittable
                or any(
                    c.can_split(split_method, separators=split_separators)
                    for c in completion.ancestors(including_self=True)
                )
            ),
            key=lambda c: c.all_abs_advantage_per_token(tokenizer, cache=cache),
        )

    async def sample_completions_v2(
        self,
        completion_sampler: CompletionSampler,
        branch_factor: int,
        get_recovery_pattern: Optional[Callable[[], Optional[str]]] = None,
        max_parallel_splits: int = 1,
        priority: Optional[int] = None,
        sample_probability_power: float = 1.0,
        sampling_kwargs: Optional[SamplingKwargs] = None,
        split_by: SplitMethod = "count",
        split_separators: Optional[set[str]] = None,
    ) -> bool:
        model = await completion_sampler.get_model()
        parents = self._get_sampleable_parents(
            max_parallel_splits,
            model,
            sample_probability_power,
            branch_factor > 1,
            split_by,
            split_separators,
        )
        if not parents:
            return False
        return any(
            await asyncio.gather(
                *(
                    self._sample_completions(
                        parent,
                        model,
                        completion_sampler,
                        branch_factor,
                        fork_decay=1.0,
                        get_recovery_pattern=get_recovery_pattern,
                        split_separators=split_separators or set(),
                        priority=priority,
                        sampling_kwargs=sampling_kwargs or {},
                    )
                    for parent in parents
                )
            )
        )

    def _get_sampleable_parents(
        self,
        max_parallel_splits: int,
        model: str,
        sample_probability_power: float,
        split: bool,
        split_method: SplitMethod,
        split_separators: Optional[set[str]],
    ) -> list[Completion]:
        if not any(child.model == model for child in self.completion.children):
            return [self.completion]
        if not split:
            return []
        parents = sorted(
            (
                c
                for c in self.completion.descendants(model=model)
                if c.can_split(by=split_method, separators=split_separators)
            ),
            key=lambda c: abs(c.advantage(cache=True, model=model))
            * (c.split_weight(by=split_method) / c.num_token_logprobs())
            * c.sample_weight(cache=True, model=model, power=sample_probability_power),
            reverse=True,
        )[:max_parallel_splits]
        for parent in parents:
            assert parent.split(
                by=split_method, separators=split_separators
            ), "Unable to split completion"
        return parents

    async def _sample_completions(
        self,
        parent: Completion,
        model: str,
        completion_sampler: CompletionSampler,
        branch_factor: int,
        fork_decay: float,
        get_recovery_pattern: Optional[Callable[[], Optional[str]]],
        split_separators: set[str],
        sampling_kwargs: SamplingKwargs,
        priority: Optional[int] = None,
    ) -> bool:
        num_children = sum(1 for child in parent.children if child.model == model)
        n = branch_factor - num_children
        if n <= 0:
            return False
        completions = await completion_sampler.sample_completions(
            parent,
            strip=split_separators,
            priority=priority or 0,
            get_recovery_pattern=get_recovery_pattern,
            **{**sampling_kwargs, "n": n},
        )
        if num_children:
            for completion in completions:
                completion.weight *= fork_decay
                completion.fork = True
        on_sample = self.on_sample(
            [
                EpisodeCompletion(
                    completion,
                    completion_sampler,
                    sampling_kwargs,
                    priority,
                )
                for completion in completions
            ]
        )
        if isinstance(on_sample, Coroutine):
            await on_sample
        return True
