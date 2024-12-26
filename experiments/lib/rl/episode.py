import asyncio
from collections import Counter
from dataclasses import dataclass
import numpy as np
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from typing import (
    Any,
    Callable,
    Coroutine,
    Iterable,
    Literal,
    Optional,
    Protocol,
    Union,
    Unpack,
)

from .completion import Completion, SplitMethod
from .completion_sampler import SamplingKwargs, CompletionSampler

from ..tokenizer import Tokenizer


@dataclass
class EpisodeCompletion:
    _completion: Completion
    _sampler: CompletionSampler
    _sampling_kwargs: SamplingKwargs
    _tokenizer: Tokenizer
    _priority: Optional[int]

    @property
    def absent_stop_tokens(self) -> int:
        return self._completion.absent_stop_tokens()

    @property
    def all_absent_stop_tokens(self) -> int:
        return self._completion.all_absent_stop_tokens()

    @property
    def completion_tokens(self) -> int:
        return self._completion.num_token_logprobs()

    @property
    def all_completion_tokens(self) -> int:
        return self._completion.all_num_token_logprobs()

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
        self,
        messages: list[ChatCompletionMessageParam],
        **sampling_kwargs: Unpack[SamplingKwargs],  # type: ignore
    ) -> "EpisodeCompletion":
        for message in self._completion.messages:
            if not isinstance(message, dict):
                message.finish_reason
        completions = await self._sampler.sample_completions(
            Completion(parent=self._completion, messages=messages),
            tokenizer=self._tokenizer,
            priority=self._priority or 0,
            **{**self._sampling_kwargs, **sampling_kwargs, "n": 1},
        )
        return EpisodeCompletion(
            _completion=completions[0],
            _sampler=self._sampler,
            _sampling_kwargs=self._sampling_kwargs,
            _tokenizer=self._tokenizer,
            _priority=self._priority,
        )


class SampleEpisode(Protocol):
    def __call__(self) -> "Episode" | Coroutine[Any, Any, "Episode"]: ...


class Episode:
    def __init__(
        self,
        messages: list[ChatCompletionMessageParam],
        on_sample: Callable[
            [list[EpisodeCompletion]], None | Coroutine[None, None, None]
        ],
        examples: Union[
            list[ChatCompletionMessageParam],
            Callable[[], list[ChatCompletionMessageParam]],
        ] = [],
        get_easier_episode: Optional[tuple[float, SampleEpisode]] = None,
        get_similar_episode: Optional[SampleEpisode] = None,
        get_harder_episode: Optional[tuple[float, SampleEpisode]] = None,
    ) -> None:
        self.completion = Completion(messages=messages)  # type: ignore
        self.on_sample = on_sample
        self.examples = examples
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
        num_parents: int,
        branch_factor: int,
        get_recovery_pattern: Optional[Callable[[Completion], Optional[str]]] = None,
        max_splits_per_completion: int = 1,
        priority: Optional[int] = None,
        sample_probability_power: float = 1.0,
        sampling_kwargs: Optional[SamplingKwargs] = None,
        split_by: SplitMethod = "count",
        split_point_std_deviation: float = 0.0,
        split_separators: Optional[set[str]] = None,
    ) -> bool:
        """Sample completions for the episode.

        Args:
            completion_sampler: The completion sampler to use.
            tokenizer: The tokenizer to use.
            num_parents: The number of parents to sample completions from.
            branch_factor: The number of completions to sample per parent.
            get_recovery_pattern: A function that returns a recovery pattern to use.
            max_splits_per_completion: The maximum number of splits to perform per completion.
            priority: The priority of the completions.
            sample_probability_power: The power to use for the sample probability.
            sampling_kwargs: The sampling kwargs to use.
            split_by: The method to use for splitting.
            split_point_std_deviation: The standard deviation to use for sampling split points.
                A deviation of 0 is deterministic while larger values approach uniform sampling.
            split_separators: The separators to use for splitting.

        Returns:
            True if any completions were sampled, False otherwise.
        """
        model = await completion_sampler.get_model()
        if not any(child.model == model for child in self.completion.children):
            parents = [self.completion]
        elif branch_factor > 1:
            parents = self._get_sampleable_parents(
                branch_factor,
                num_parents,
                max_splits_per_completion,
                model,
                sample_probability_power,
                split_by,
                split_point_std_deviation,
                split_separators,
            )
        else:
            return False
        return any(
            await asyncio.gather(
                *(
                    self._sample_completions(
                        parent,
                        model,
                        completion_sampler,
                        tokenizer,
                        branch_factor=(
                            max(branch_factor, num_parents * (branch_factor - 1))
                            if len(parents) == 1
                            else branch_factor
                        ),
                        fork_decay=1.0,
                        recovery_pattern=get_recovery_pattern,
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
        branch_factor: int,
        num_parents: int,
        max_splits_per_completion: int,
        model: str,
        sample_probability_power: float,
        split_method: SplitMethod,
        split_point_std_deviation: float,
        split_separators: Optional[set[str]],
    ) -> list[Completion]:
        """Get the parents to sample completions from.

        This function selects a set of completions to split and sample from,
        based on a scoring system that considers the advantage, split weight,
        number of tokens, and sample weight of each completion.

        Args:
            branch_factor: The number of completions to sample per parent.
            num_parents: The number of parents to sample completions from.
            max_splits_per_completion: The maximum number of splits to perform on a single completion.
            model: The model to use.
            sample_probability_power: The power to use for the sample probability.
            split: Whether to split the completions.
            split_method: The method to use for splitting.
            split_point_std_deviation: The standard deviation to use for the split point.
            split_separators: The separators to use for splitting.

        Returns:
            A list of completions to sample from.
        """
        get_split_value: Callable[[Completion], float] = lambda c: (
            abs(c.advantage(cache=True, model=model))
            * (c.split_weight(by=split_method, cache=True) / c.num_token_logprobs())
            * c.sample_weight(cache=True, model=model, power=sample_probability_power)
        )

        # Create a Counter to track the number of times each completion is selected
        # for splitting.
        completions = Counter(
            c
            for c, _ in sorted(
                (
                    (
                        c,
                        split_value
                        / (split + 1)
                        * (1 / branch_factor) ** (sample_probability_power * split),
                    )
                    for c, max_splits, split_value in (
                        (
                            c,
                            max_splits,
                            get_split_value(c),
                        )
                        for c, max_splits in (
                            (
                                c,
                                min(
                                    num_parents,
                                    max_splits_per_completion,
                                    c.max_splits(
                                        by=split_method,
                                        separators=split_separators,
                                        cache=True,
                                    ),
                                ),
                            )
                            for c in self.completion.descendants(model=model)
                        )
                        if max_splits > 0
                    )
                    for split in range(max_splits)
                ),
                key=lambda x: x[1],
            )[:num_parents]
        )

        # Split the selected completions and return the resulting parents.
        return [
            parent
            for completion, num_splits in completions.items()
            for parent in list(
                completion.split(
                    by=split_method,
                    at=self._split_points(num_splits, split_point_std_deviation),
                    separators=split_separators,
                    cache=True,
                )
            )[:-1]
        ]

    def _split_points(self, num_splits: int, std_deviation: float) -> Iterable[float]:
        """Sample a sequence of split points.

        Args:
            num_splits: The number of split points to sample.
            std_deviation: The standard deviation of the split points.

        Yields:
            The split points.
        """
        for i in range(num_splits):
            split_point = (i + 1) / (num_splits + 1)
            if std_deviation:
                split_point = np.random.normal(split_point, std_deviation / num_splits)
                if split_point < 0 or split_point >= 1:
                    # If the split point is out of bounds, sample a new one using a
                    # uniform distribution.
                    split_point = np.random.uniform()
                yield split_point
            else:
                yield split_point

    async def _sample_completions(
        self,
        parent: Completion,
        model: str,
        completion_sampler: CompletionSampler,
        tokenizer: Tokenizer,
        branch_factor: int,
        fork_decay: float,
        recovery_pattern: Optional[Callable[[Completion], Optional[str]]],
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
            tokenizer=tokenizer,
            examples=self.examples,
            priority=priority or 0,
            recovery_pattern=recovery_pattern,
            strip=split_separators,
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
                    tokenizer,
                    priority,
                )
                for completion in completions
            ]
        )
        if isinstance(on_sample, Coroutine):
            await on_sample
        return True
