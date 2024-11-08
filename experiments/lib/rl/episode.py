import asyncio
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from typing import Any, Callable, Coroutine, Literal, Optional, Protocol

from .completion import Completion, SplitMethod
from .completion_sampler import CompletionSampler

# from .trajectory import Trajectory
from ..tokenizer import Tokenizer


class SampleEpisode(Protocol):
    def __call__(self) -> "Episode" | Coroutine[Any, Any, "Episode"]: ...


class Episode:
    def __init__(
        self,
        messages: list[ChatCompletionMessageParam],
        on_sample: Callable[[list[Completion]], None | Coroutine[None, None, None]],
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

    def num_samples(self) -> int:
        if len(self.completion.children) == 0:
            return 0
        return sum(1 for _ in self.completion.leaves())

    async def sample_completions(
        self,
        completion_sampler: CompletionSampler,
        tokenizer: Tokenizer,
        branch_factor: int,
        split_by: SplitMethod = "count",
    ) -> bool:
        parent = self.get_sampleable_parent(tokenizer, split_by)
        if parent is None:
            return False
        completions = await completion_sampler.sample_completions(
            parent,
            n=branch_factor - len(parent.children),
        )
        on_sample = self.on_sample(completions)
        if isinstance(on_sample, Coroutine):
            await on_sample
        return True

    def get_sampleable_parent(
        self, tokenizer: Tokenizer, split_by: SplitMethod
    ) -> Optional[Completion]:
        if len(self.completion.children) == 0:
            return self.completion
        try:
            leaf = self.best_leaf(tokenizer, where_leaf_or_ancestor_is_splittable=True)
            parent = max(
                (c for c in leaf.ancestors(including_self=True) if c.can_split()),
                key=lambda c: abs(c.advantage()) * c.token_count(tokenizer),
            )
            assert parent.split(by=split_by), "Unable to split completion"
            return parent
        except BaseException as e:
            print(type(e), e)

    def best_leaf(
        self,
        tokenizer: Tokenizer,
        *,
        where_leaf_or_ancestor_is_splittable: Optional[Literal[True]] = None,
        cache: bool = True,
    ) -> Completion:
        return max(
            (
                completion
                for completion in self.completion.leaves()
                if not where_leaf_or_ancestor_is_splittable
                or any(c.can_split() for c in completion.ancestors(including_self=True))
            ),
            key=lambda c: c.all_abs_advantage_per_token(tokenizer, cache=cache),
        )

    # def best_trajectory(
    #     self, tokenizer: Tokenizer, *, episode_decay: float, completion_decay: float
    # ) -> Trajectory:
    #     terminus = self.best_leaf(tokenizer)
    #     return Trajectory(
    #         episode=self,
    #         terminus=terminus,
    #         abs_advantage=terminus.all_abs_advantage(cache=True),
    #         token_count=terminus.all_token_count(tokenizer),
    #         episode_decay=episode_decay,
    #         completion_decay=completion_decay,
    #     )
