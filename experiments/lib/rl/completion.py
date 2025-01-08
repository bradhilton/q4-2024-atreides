from copy import deepcopy
import numpy as np
from openai.types.chat.chat_completion import Choice, ChoiceLogprobs
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_content_part_param import (
    ChatCompletionContentPartParam,
)
from openai.types.chat.chat_completion_function_message_param import (
    ChatCompletionFunctionMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
from pydantic import (
    BaseModel,
    Field,
    model_validator,
    SkipValidation,
)
import torch
from typing import (
    Any,
    Iterable,
    Literal,
    Optional,
    Required,
    Sequence,
    TypeAlias,
    Union,
)

from ..tokenizer import Tokenizer
from ..utils import get_token, get_token_id


class ChatCompletionUserMessageParamOverride(
    ChatCompletionUserMessageParam, total=False
):
    # Override the `content` field to skip validation for the Iterable type to work around Pydantic bug.
    content: Required[
        Union[str, SkipValidation[Iterable[ChatCompletionContentPartParam]]]
    ]
    """The contents of the user message."""


ChatCompletionMessageParam: TypeAlias = Union[
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParamOverride,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionFunctionMessageParam,
]

SplitMethod = Literal["count", "prob", "logprob"]


class Completion:
    def __init__(
        self,
        parent: Optional["Completion"] = None,
        messages: Optional[Sequence[Union[ChatCompletionMessageParam, Choice]]] = None,
        reward: float = 0.0,
        children: Optional[set["Completion"]] = None,
        weight: float = 1.0,
        model: Optional[str] = None,
        fork: bool = False,
        logprobs_mask: Optional[set[str]] = None,
    ):
        self.parent = parent
        self.messages = list(messages or [])
        self.reward = reward
        self.children = children or set()
        self.weight = weight
        self.model = model
        self.fork = fork
        self.logprobs_mask = logprobs_mask or set()
        self.reference_logprobs: Optional[torch.Tensor] = None
        self._cached_values: dict[
            tuple[Optional[frozenset[str]], Optional[bool]], float
        ] = {}
        self._cached_max_values: dict[Optional[frozenset[str]], float] = {}
        self._cached_sample_weights: dict[
            tuple[Optional[frozenset[str]], float], float
        ] = {}
        self._cached_tokens_and_mask: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._cached_entropy_sum: Optional[float] = None
        self._cached_split_weights: dict[
            tuple[SplitMethod, Optional[frozenset[str]]], list[float]
        ] = {}
        for child in self.children:
            if child.parent is not self:
                raise ValueError("Child completion's parent must be this completion.")

    def commit(self, reward: Optional[float] = None) -> None:
        """
        Recursively commits this and uncommitted ancestor completions to the tree.

        Before committing, be sure:
        1) To set any non-zero rewards for this completion and its ancestors if they have not already been set.
        2) That this completion is the terminal step in the trajectory.

        May optionally set the reward for this completion at the time of commit.
        """
        if reward is not None:
            self.reward = reward
            self._cached_values = {}
            self._cached_max_values = {}
            self._cached_sample_weights = {}
        if not self.parent:
            return
        self.parent.commit()
        self.parent.children.add(self)
        self.parent._cached_values = {}
        self.parent._cached_max_values = {}
        self.parent._cached_sample_weights = {}

    def value(
        self,
        cache: bool,
        models: Optional[set[str]],
        fork: Optional[bool] = None,
    ) -> float:
        """
        Estimates the undiscounted Q-value for the current state-action (`parent`, `messages`) pair.

        Computes the Q-value by summing the immediate reward (`reward`) and the
        average of the values of sampled successor state-action pairs (`children`).

        Args:
            cache (bool): Whether to cache Q-value estimates for faster retrieval.
            models (Optional[set[str]]): Optionally limit child Q-value estimates to specific models.
            fork (Optional[bool]): Optionally limit child Q-value estimates to only forked or unforked completions.

        Returns:
            float: Estimated Q-value for the state-action pair.
        """
        cache_key = (frozenset(models) if models else None, fork)
        if cache and cache_key in self._cached_values:
            return self._cached_values[cache_key]
        child_values = [
            completion.value(cache=cache, models=models, fork=fork)
            for completion in self.children
            if completion.matches_models(models)
            and (fork is None or completion.fork == fork)
        ]
        value = self.reward + (sum(child_values) / max(len(child_values), 1))
        if cache:
            self._cached_values[cache_key] = value
        return value

    def max_value(self, cache: bool, models: Optional[set[str]]) -> float:
        """
        Returns the maximum reward-to-go for this completion.
        """
        cache_key = frozenset(models) if models else None
        if cache and cache_key in self._cached_max_values:
            return self._cached_max_values[cache_key]
        max_value = self.reward + (
            max(child.max_value(cache=cache, models=models) for child in self.children)
            if self.children
            else 0.0
        )
        if cache:
            self._cached_max_values[cache_key] = max_value
        return max_value

    def weighted_value(
        self,
        cache: bool,
        models: Optional[set[str]],
        max_weight: float,
    ) -> float:
        return (1 - max_weight) * self.value(
            cache=cache, models=models
        ) + max_weight * self.max_value(cache=cache, models=models)

    def advantage(
        self,
        cache: bool,
        models: Optional[set[str]],
        max_weight: float = 0.0,
    ) -> float:
        """
        Calculates the advantage value for this completion relative to its parent.

        The advantage represents how much better this completion's value is compared to
        the baseline value of its parent's other children. It is calculated as the
        difference between this completion's value and the parent's value (excluding
        the parent's immediate reward).

        For root nodes (no parent), the advantage is always 0.

        Returns:
            float: The advantage value. Positive values indicate this completion
                  performed better than its siblings on average.
        """
        if self.parent is None:
            return 0.0
        return (
            self.weighted_value(cache=cache, models=models, max_weight=max_weight)
            - (
                self.parent.weighted_value(
                    cache=cache, models=models, max_weight=max_weight
                )
                - self.parent.reward
            )
        ) * self.weight

    # def adjustment(self, lambda_: float = 1.0) -> float:
    #     return 1 - (
    #         lambda_
    #         / (
    #             lambda_
    #             + max(sum(c.adjustment(lambda_) for c in self.children), 0.5) * 2
    #         )
    #     )

    # def adjusted_value(self, lambda_: float = 1.0) -> float:
    #     if not self.parent:
    #         return self.value()
    #     return (self.value() - self.parent.value()) * self.adjustment(
    #         lambda_
    #     ) + self.parent.value()

    # def adjusted_advantage(self, lambda_: float = 1.0) -> float:
    #     return self.advantage() * self.adjustment(lambda_)

    # def all_abs_advantage(self, cache: bool, model: Optional[str]) -> float:
    #     return abs(self.advantage(cache=cache, model=model)) + (
    #         self.parent.all_abs_advantage(cache=cache, model=model)
    #         if self.parent
    #         else 0
    #     )

    # def all_abs_advantage_per_token(
    #     self, tokenizer: Tokenizer, cache: bool, model: Optional[str]
    # ) -> float:
    #     return self.all_abs_advantage(cache=cache, model=model) / self.all_token_count(
    #         tokenizer, cache=cache
    #     )

    def ancestors(
        self, including_self: bool = False, reverse: bool = False
    ) -> Iterable["Completion"]:
        if not reverse and including_self:
            yield self
        if self.parent:
            yield from self.parent.ancestors(including_self=True, reverse=reverse)
        if reverse and including_self:
            yield self

    def cumulative_reward(self) -> float:
        return self.reward + (self.parent.cumulative_reward() if self.parent else 0)

    def matches_models(self, models: Optional[set[str]]) -> bool:
        return not models or not self.model or self.model in models

    def descendants(
        self,
        models: Optional[set[str]],
        including_self: bool = False,
    ) -> Iterable["Completion"]:
        if including_self and self.matches_models(models):
            yield self
        for child in self.children:
            yield from child.descendants(
                including_self=True,
                models=models,
            )

    def leaves(self, models: Optional[set[str]]) -> Iterable["Completion"]:
        if not self.children and self.matches_models(models):
            yield self
        for child in self.children:
            yield from child.leaves(models=models)

    def depth(self) -> int:
        return 1 + self.parent.depth() if self.parent else 0

    def depths(
        self,
        models: Optional[set[str]],
        depth: int = 0,
    ) -> Iterable[int]:
        if self.matches_models(models):
            yield depth
        for child in self.children:
            yield from child.depths(models=models, depth=depth + 1)

    def max_depth(self, models: Optional[set[str]]) -> int:
        return max(self.depths(models=models), default=0)

    def absent_stop_tokens(self) -> int:
        return sum(
            1
            for message in self.messages
            if not isinstance(message, dict) and message.finish_reason == "length"
        )

    def all_absent_stop_tokens(self) -> int:
        return self.absent_stop_tokens() + (
            self.parent.all_absent_stop_tokens() if self.parent else 0
        )

    def entropy(self, cache: bool = False) -> float:
        """Calculate the average entropy per token.

        Args:
            cache (bool): Whether to cache entropy calculations for faster retrieval.

        Returns:
            float: The average entropy per token.
        """
        num_token_logprobs = self.num_token_logprobs()
        if not num_token_logprobs:
            return 0.0
        return self.entropy_sum(cache=cache) / num_token_logprobs

    def entropy_sum(self, cache: bool = False) -> float:
        """Calculate the sum of entropy across all token logprobs.

        Args:
            cache (bool): Whether to cache the entropy sum for faster retrieval.

        Returns:
            float: The sum of entropy values.
        """
        if cache and self._cached_entropy_sum is not None:
            return self._cached_entropy_sum

        top_logprobs = torch.tensor(
            [
                [top_logprob.logprob for top_logprob in token_logprob.top_logprobs]
                for token_logprobs in self._token_logprob_sequences()
                for token_logprob in token_logprobs
            ]
        )
        if not top_logprobs.size(0):
            return 0.0

        entropy_sum = (
            torch.distributions.Categorical(probs=torch.exp(top_logprobs))
            .entropy()
            .sum()
            .item()
        )

        if cache:
            self._cached_entropy_sum = entropy_sum

        return entropy_sum

    def all_entropy(self, cache: bool = False) -> float:
        """Calculate the average entropy per token across this completion and all ancestors.

        Args:
            cache (bool): Whether to cache entropy calculations for faster retrieval.

        Returns:
            float: The average entropy per token across the completion chain.
        """
        all_num_token_logprobs = self.all_num_token_logprobs()
        if not all_num_token_logprobs:
            return 0.0
        return self.all_entropy_sum(cache=cache) / all_num_token_logprobs

    def all_entropy_sum(self, cache: bool = False) -> float:
        """Calculate the sum of entropy across this completion and all its ancestors.

        Args:
            cache (bool): Whether to cache entropy sums for faster retrieval.

        Returns:
            float: The total sum of entropy values.
        """
        return self.entropy_sum(cache=cache) + (
            self.parent.all_entropy_sum(cache=cache) if self.parent else 0.0
        )

    def merge(self) -> "Completion":
        """
        Merges this completion with its parent and returns the merged (parent) completion.

        Preconditions:
        1) This completion must have a parent.
        2) This completion must be the only child of its parent.
        3) This completion must not have any uncommitted children.

        Returns:
            Completion: The merged completion.
        """
        assert self.parent, "Cannot merge a completion without a parent."
        assert self.parent.children == {
            self
        }, "Cannot merge a completion with siblings."
        assert (
            self.parent.model == self.model
        ), "Cannot merge completions from different models."
        assert (
            self.parent.fork == self.fork
        ), "Cannot merge forked and unforked completions."
        if (
            self.parent.messages
            and self.messages
            and isinstance(self.parent.messages[-1], Choice)
            and isinstance(self.messages[0], Choice)
            and role(self.parent.messages[-1]) == role(self.messages[0]) == "assistant"
        ):
            parent_message = self.parent.messages[-1]
            child_message = self.messages[0]
            assert bool(parent_message.logprobs) == bool(
                child_message.logprobs
            ), "Cannot merge choices with differing presence of logprobs."
            assert not (
                parent_message.message.audio and child_message.message.audio
            ), "Cannot merge audio messages."
            assert not (
                parent_message.message.function_call
                and child_message.message.function_call
            ), "Cannot merge function call messages."
            parent_message.finish_reason = child_message.finish_reason
            if parent_message.logprobs and child_message.logprobs:
                if parent_message.logprobs.content or child_message.logprobs.content:
                    parent_message.logprobs.content = (
                        parent_message.logprobs.content or []
                    ) + (child_message.logprobs.content or [])
                if parent_message.logprobs.refusal or child_message.logprobs.refusal:
                    parent_message.logprobs.refusal = (
                        parent_message.logprobs.refusal or []
                    ) + (child_message.logprobs.refusal or [])
            if parent_message.message.content or child_message.message.content:
                parent_message.message.content = (
                    parent_message.message.content or ""
                ) + (child_message.message.content or "")
            if parent_message.message.refusal or child_message.message.refusal:
                parent_message.message.refusal = (
                    parent_message.message.refusal or ""
                ) + (child_message.message.refusal or "")
            parent_message.message.audio = (
                parent_message.message.audio or child_message.message.audio
            )
            parent_message.message.function_call = (
                parent_message.message.function_call
                or child_message.message.function_call
            )
            if parent_message.message.tool_calls or child_message.message.tool_calls:
                parent_message.message.tool_calls = (
                    parent_message.message.tool_calls or []
                ) + (child_message.message.tool_calls or [])
            _ = self.messages.pop(0)
        self.parent.messages.extend(self.messages)
        self.parent.reward += self.reward
        self.parent.children = self.children
        for child in self.children:
            child.parent = self.parent
        self.parent.weight += self.weight
        self.parent.weight /= 2
        self.parent._cached_values = {}
        self.parent._cached_sample_weights = {}
        self.parent._cached_tokens_and_mask = None
        self.parent._cached_entropy_sum = None
        self.parent._cached_split_weights = {}
        return self.parent

    def message_params(
        self, replacement_token: Optional[str] = None
    ) -> list[ChatCompletionMessageParam]:
        return [
            (
                message_param(message_or_choice, replacement_token=replacement_token)
                if isinstance(message_or_choice, Choice)
                else message_or_choice
            )
            for message_or_choice in self.messages
        ]

    def all_message_params(
        self,
        join_consecutive_assistant_messages: Union[bool, str] = True,
        replacement_token: Optional[str] = None,
    ) -> list[ChatCompletionMessageParam]:
        message_params = self.message_params(replacement_token=replacement_token)
        if not self.parent:
            return message_params
        parent_message_params = self.parent.all_message_params(
            join_consecutive_assistant_messages=join_consecutive_assistant_messages,
            replacement_token=replacement_token,
        )
        if (
            join_consecutive_assistant_messages == False
            or not parent_message_params
            or not message_params
            or not (
                parent_message_params[-1]["role"]
                == message_params[0]["role"]
                == "assistant"
            )
        ):
            return parent_message_params + message_params
        return (
            parent_message_params[:-1]
            + joined_assistant_message_params(
                parent_message_params[-1],  # type: ignore
                message_params[0],  # type: ignore
                joiner=(
                    join_consecutive_assistant_messages
                    if isinstance(join_consecutive_assistant_messages, str)
                    else ""
                ),
            )
            + message_params[1:]
        )

    def logprobs(self) -> Iterable[float]:
        mask: Optional[str] = None
        pending_logprobs: list[float] = []
        for choice in self.messages:
            if isinstance(choice, Choice) and choice.logprobs:
                for token_logprob in (
                    choice.logprobs.content or choice.logprobs.refusal or []
                ):
                    token = get_token(token_logprob)
                    if mask is not None:
                        if mask == "":
                            for _ in range(len(pending_logprobs)):
                                yield float("nan")
                            mask = None
                            pending_logprobs = []
                        elif token.startswith(mask):
                            mask = mask.removeprefix(token)
                            pending_logprobs.append(token_logprob.logprob)
                        else:
                            mask = None
                            yield from pending_logprobs
                            pending_logprobs = []
                    else:
                        for mask in self.logprobs_mask:
                            if token.startswith(mask):
                                mask = mask.removeprefix(token)
                                pending_logprobs = [token_logprob.logprob]
                                break
                    if mask is None:
                        yield token_logprob.logprob
        if mask == "":
            for _ in range(len(pending_logprobs)):
                yield float("nan")
        else:
            yield from pending_logprobs

    def all_logprobs(self) -> Iterable[float]:
        if self.parent:
            yield from self.parent.all_logprobs()
        yield from self.logprobs()

    def estimated_completion_tokens(
        self, including_self: bool = False
    ) -> Optional[float]:
        estimate: Optional[float] = None
        if including_self:
            for message in self.messages:
                if isinstance(message, Choice):
                    if not message.logprobs:
                        return None
                    token_logprobs = (
                        message.logprobs.content or message.logprobs.refusal
                    )
                    if not token_logprobs:
                        return None
                    estimate = (estimate or 0.0) + len(token_logprobs)
                else:
                    return estimate
        if (estimate or not including_self or not self.messages) and self.children:
            estimate = (estimate or 0.0) + sum(
                child.estimated_completion_tokens(including_self=True) or 0
                for child in self.children
            ) / len(self.children)
            if estimate == 0.0:
                return None
        return estimate

    def num_prefix_tokens(self) -> int:
        num_prefix_tokens = 0
        for message in reversed(self.messages):
            if isinstance(message, Choice):
                assert message.logprobs, "Choice must have logprobs."
                token_logprobs = message.logprobs.content or message.logprobs.refusal
                assert token_logprobs, "Choice must have token logprobs."
                num_prefix_tokens += len(token_logprobs)
            else:
                return num_prefix_tokens
        if self.parent:
            num_prefix_tokens += self.parent.num_prefix_tokens()
        return num_prefix_tokens

    def recursive_copy(
        self, commit: bool = True, copy_root: bool = False, model: Optional[str] = None
    ) -> "Completion":
        if not copy_root and not self.parent:
            return self
        completion = Completion(
            parent=(
                self.parent.recursive_copy(
                    commit=False, copy_root=copy_root, model=model
                )
                if self.parent
                else None
            ),
            messages=deepcopy(self.messages),
            reward=self.reward,
            children=set(),
            weight=self.weight,
            model=model or self.model,
            fork=self.fork,
            logprobs_mask=self.logprobs_mask,
        )
        if commit:
            completion.commit()
        return completion

    def root(self) -> "Completion":
        return self.parent.root() if self.parent else self

    def to_dict(self, models: Optional[set[str]] = None) -> dict[str, Any]:
        return {
            "messages": [
                message if isinstance(message, dict) else message.model_dump()
                for message in self.messages
            ],
            "reward": self.reward,
            "model": self.model,
            "children": [
                child.to_dict(models=models)
                for child in self.children
                if child.matches_models(models)
            ],
        }

    def token_advantage(self, cache: bool, models: Optional[set[str]]) -> float:
        return self.advantage(cache=cache, models=models) / (
            self.num_token_logprobs() or 1
        )

    def token_advantages(
        self, cache: bool, models: Optional[set[str]], max_weight: float = 0.0
    ) -> list[float]:
        advantage = self.advantage(cache=cache, models=models, max_weight=max_weight)
        token_logprobs = [
            token_logprob
            for token_logprobs in self._token_logprob_sequences()
            for token_logprob in token_logprobs
        ]
        token_advantage = advantage / max(len(token_logprobs), 1)
        token_advantages = [token_advantage] * len(token_logprobs)
        if (
            token_advantages
            and get_token(token_logprobs[-1])
            == "<|im_end|>"  # TODO: Don't hardcode this token
        ):
            token_advantages[-1] = max(token_advantage, 0.0)
        return token_advantages

    def all_token_advantages(
        self, cache: bool, models: Optional[set[str]]
    ) -> Iterable[float]:
        if self.parent:
            yield from self.parent.all_token_advantages(cache=cache, models=models)
        yield from self.token_advantages(cache=cache, models=models)

    def tokens_and_mask(
        self, tokenizer: Tokenizer, *, cache: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cache and self._cached_tokens_and_mask is not None:
            return self._cached_tokens_and_mask
        tokens_and_masks: list[tuple[torch.Tensor, bool]] = []
        for i, message in enumerate(self.messages):
            tokens = tokenizer.encode(
                [
                    (
                        {
                            "role": role(message),
                            "content": "!",
                        }
                        if isinstance(message, Choice)
                        else message
                    )
                ],
                remove_bos=self.parent is not None or i > 0,
                first_message_is_continuation=(
                    i == 0
                    and self.parent is not None
                    and role(self.parent.messages[-1]) == role(message) == "assistant"
                ),
                continue_final_message=(
                    i + 1 == len(self.messages)
                    and role(message) == "assistant"
                    and any(
                        role(child.messages[0]) == "assistant"
                        for child in self.children
                    )
                ),
            )
            if isinstance(message, Choice):
                assert message.logprobs
                zero_idx: int = (tokens == 0).nonzero()[0]  # type: ignore
                tokens_and_masks.append((tokens[:zero_idx], False))
                tokens_and_masks.append(
                    (
                        torch.tensor(
                            [
                                get_token_id(token_logprob)
                                for token_logprob in (
                                    message.logprobs.content
                                    or message.logprobs.refusal
                                    or []
                                )
                            ]
                        ),
                        True,
                    )
                )
                tokens_and_masks.append((tokens[zero_idx + 1 :], False))
            else:
                tokens_and_masks.append((tokens, False))
        tokens_and_mask = (
            torch.cat([tokens for tokens, _ in tokens_and_masks]),
            torch.cat(
                [
                    torch.full_like(tokens, mask, dtype=torch.bool)
                    for tokens, mask in tokens_and_masks
                ]
            ),
        )
        if cache:
            self._cached_tokens_and_mask = tokens_and_mask
        return tokens_and_mask

    def token_count(self, tokenizer: Tokenizer, *, cache: bool) -> int:
        return self.tokens_and_mask(tokenizer, cache=cache)[0].size(0)

    def all_token_count(self, tokenizer: Tokenizer, *, cache: bool) -> int:
        return self.token_count(tokenizer, cache=cache) + (
            self.parent.all_token_count(tokenizer, cache=cache) if self.parent else 0
        )

    def tokens(self, tokenizer: Tokenizer, *, cache: bool) -> torch.Tensor:
        return self.tokens_and_mask(tokenizer, cache=cache)[0]

    def all_tokens(self, tokenizer: Tokenizer, *, cache: bool = False) -> torch.Tensor:
        if self.parent:
            return torch.cat(
                (
                    self.parent.all_tokens(tokenizer, cache=cache),
                    self.tokens(tokenizer, cache=cache),
                )
            )
        return self.tokens(tokenizer, cache=cache)

    def sample_weight(
        self, cache: bool, models: Optional[set[str]], power: float
    ) -> float:
        if not self.matches_models(models):
            return 0.0
        if not self.parent or not power:
            return 1.0
        cache_key = (frozenset(models) if models else None, power)
        if cache and cache_key in self._cached_sample_weights:
            return self._cached_sample_weights[cache_key]
        weight = (
            self.parent.sample_weight(cache=cache, models=models, power=power)
            / sum(child.matches_models(models) for child in self.parent.children)
        ) ** power
        if cache:
            self._cached_sample_weights[cache_key] = weight
        return weight

    def max_splits(
        self,
        by: SplitMethod,
        separators: Optional[set[str]] = None,
        *,
        cache: bool,
    ) -> int:
        return max(
            sum(
                1
                for weight in self._split_weights(
                    by, separators=separators, cache=cache
                )
                if weight > 0
            )
            - 1,
            0,
        )

    def split_weight(
        self,
        by: SplitMethod,
        separators: Optional[set[str]] = None,
        *,
        cache: bool,
    ) -> float:
        return sum(self._split_weights(by, separators=separators, cache=cache))

    def split(
        self,
        by: SplitMethod,
        at: Iterable[float] = (0.5,),
        *,
        separators: Optional[set[str]] = None,
        weights: Optional[Iterable[float]] = None,
        cache: bool,
    ) -> Iterable["Completion"]:
        """
        Splits this completion into two or more completions at the specified split points.

        NOTE: Currently only supports splitting `Choice`s with `logprobs.content`.

        Args:
            by (Literal["count", "prob", "logprob"]): The weighting method for splitting the completion.
            at (Iterable[float]): The split points as a fraction of the total weight in the range (0, 1).
            separators (Optional[set[str]]): An optional set of separators to group tokens when splitting.
            weights (Optional[list[float]]): An optional list of weights for each token logprob. Must be the same length as the number of tokens.
            cache (bool): Whether to cache split weights for faster retrieval.

        Yields:
            Completion: The original completion followed by split completions.
            May yield fewer completions than split points if some splits aren't possible.

        Raises:
            AssertionError: If weights length doesn't match `num_token_logprobs()`.
        """
        yield self
        split_points = sorted(at)
        if not split_points:
            return
        split_point = split_points[0]
        split_points = [
            (point - split_point) / (1 - split_point) for point in split_points[1:]
        ]
        weights = list(
            weights or self._split_weights(by, separators=separators, cache=cache)
        )
        assert (
            not separators or len(weights) == self.num_token_logprobs()
        ), "Number of weights does not match number of tokens."
        if len(weights) < 2:
            return
        cumsum = np.cumsum(weights)
        split = np.searchsorted(
            cumsum, max(cumsum[-1] * split_point, cumsum[0]), side="right"
        )
        if split == 0:
            # Cannot split at start of completion, so yield the remaining splits
            yield from self.split(
                by=by, at=split_points, separators=separators, cache=cache
            )
        elif split == len(weights):
            # Cannot split at the end of the completion, so return early
            return
        assert weights[split] != 0, "Split point has zero weight."
        i = 0
        for j, choice in enumerate(self.messages):
            if (
                not isinstance(choice, Choice)
                or not choice.logprobs
                or not choice.logprobs.content
                or not choice.message.content
            ):
                continue
            for k in range(len(choice.logprobs.content)):
                if i < split:
                    i += 1
                    continue
                l = sum(
                    len(get_token(token_logprob))
                    for token_logprob in choice.logprobs.content[:k]
                )
                for child in self.children:
                    child.parent = None
                child = Completion(
                    parent=self,
                    messages=[
                        Choice(
                            finish_reason=choice.finish_reason,
                            index=choice.index,
                            logprobs=ChoiceLogprobs(
                                content=choice.logprobs.content[k:],
                            ),
                            message=choice.message.model_copy(
                                update=dict(content=choice.message.content[l:])
                            ),
                        )
                    ]
                    + self.messages[j + 1 :],
                    reward=self.reward,
                    weight=self.weight,
                    model=self.model,
                )
                child.children = self.children
                for grandchild in child.children:
                    grandchild.parent = child
                choice.logprobs.content = choice.logprobs.content[:k]
                choice.message.content = choice.message.content[:l]
                choice.message.function_call = None
                choice.message.tool_calls = None
                self.messages = self.messages[: j + 1]
                self.reward = 0.0
                self.children = {child}
                self._cached_tokens_and_mask = None
                self._cached_entropy_sum = None
                self._cached_split_weights = {}
                yield from child.split(
                    by=by, at=split_points, weights=weights[split:], cache=False
                )
                return

    def _split_weights(
        self, by: SplitMethod, separators: Optional[set[str]], *, cache: bool
    ) -> list[float]:
        if cache:
            cache_key = (by, frozenset(separators) if separators else None)
            if cache_key in self._cached_split_weights:
                return self._cached_split_weights[cache_key]

        weights = []
        weight = 0
        n = 0
        for sequence in self._token_logprob_sequences():
            separator = True
            for token_logprob in sequence:
                if (
                    separator
                    and n
                    and (not separators or not get_token(token_logprob) in separators)
                ):
                    weights.append(weight)
                    weights.extend([0] * (n - 1))
                    weight = 0
                    n = 0
                weight += dict(
                    count=lambda _: 1,
                    prob=lambda x: 1 - np.exp(x),
                    logprob=lambda x: -x,
                )[by](token_logprob.logprob)
                n += 1
                separator = not separators or get_token(token_logprob) in separators
        if n:
            weights.append(weight)
            weights.extend([0] * (n - 1))

        if cache:
            self._cached_split_weights[cache_key] = weights
        return weights

    def _token_logprob_sequences(self) -> Iterable[list[ChatCompletionTokenLogprob]]:
        return (
            choice.logprobs.content
            for choice in self.messages
            if isinstance(choice, Choice)
            and choice.logprobs
            and choice.logprobs.content
        )

    def token_logprobs(self) -> Iterable[ChatCompletionTokenLogprob]:
        return (
            token_logprob
            for sequence in self._token_logprob_sequences()
            for token_logprob in sequence
        )

    def all_token_logprobs(self) -> Iterable[ChatCompletionTokenLogprob]:
        if self.parent:
            yield from self.parent.all_token_logprobs()
        yield from self.token_logprobs()

    def num_token_logprobs(self) -> int:
        return sum(len(sequence) for sequence in self._token_logprob_sequences())

    def all_num_token_logprobs(self) -> int:
        return self.num_token_logprobs() + (
            self.parent.all_num_token_logprobs() if self.parent else 0
        )

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    def html(
        self,
        scale_colors: float,
        cache: bool = False,
        models: Optional[set[str]] = None,
    ) -> str:
        if not self.messages:
            return ""
        token_advantage = (
            self.token_advantage(cache=cache, models=models) * scale_colors
        )
        color = f"rgba({255 if token_advantage < 0 else 0},0,{255 if token_advantage > 0 else 0}, {abs(token_advantage)})"
        html = (
            self.parent.html(scale_colors=scale_colors, cache=cache)
            if self.parent
            else ""
        )
        for i, message in enumerate(self.messages):
            if (
                i > 0
                or not self.parent
                or not self.parent.messages
                or not (role(self.parent.messages[-1]) == role(message) == "assistant")
            ):
                html += f"\n\n<b>{role(message).capitalize()}</b>:\n"
            if isinstance(message, Choice):
                html += f"<span style='background-color: {color};'>{message.message.content}</span>"
            else:
                content = message.get("content") or message.get("refusal") or ""
                if not isinstance(content, str):
                    content = "".join(
                        (
                            part["text"]  # type: ignore
                            if part["type"] == "text"
                            else (part["refusal"] if part["type"] == "refusal" else "")  # type: ignore
                        )
                        for part in content
                    )
                html += content
        return html.strip()


def message_param(
    choice: Choice,
    replacement_token: Optional[str] = None,
) -> ChatCompletionAssistantMessageParam:
    message = choice.message
    if choice.logprobs:
        message = message.model_copy()
        if choice.logprobs.content:
            message.content = "".join(
                replacement_token or get_token(token_logprob)
                for token_logprob in choice.logprobs.content
            )
        if choice.logprobs.refusal:
            message.refusal = "".join(
                replacement_token or get_token(token_logprob)
                for token_logprob in choice.logprobs.refusal
            )
    message_param: ChatCompletionAssistantMessageParam = {
        "role": message.role,
    }
    if message.audio:
        message_param["audio"] = {"id": message.audio.id}
    if message.content:
        message_param["content"] = message.content
    if message.function_call:
        message_param["function_call"] = {
            "arguments": message.function_call.arguments,
            "name": message.function_call.name,
        }
    if isinstance(getattr(message, "name", None), str):
        message_param["name"] = getattr(message, "name")
    if message.refusal:
        message_param["refusal"] = message.refusal
    if message.tool_calls:
        message_param["tool_calls"] = [
            {
                "id": tool_call.id,
                "function": {
                    "arguments": tool_call.function.arguments,
                    "name": tool_call.function.name,
                },
                "type": tool_call.type,
            }
            for tool_call in message.tool_calls
        ]
    return message_param


def joined_assistant_message_params(
    first: ChatCompletionAssistantMessageParam,
    second: ChatCompletionAssistantMessageParam,
    joiner: str,
) -> list[ChatCompletionAssistantMessageParam]:
    if (
        first.get("audio")
        or second.get("audio")
        or first.get("function_call")
        or first.get("tool_calls")
        or first.get("name") != second.get("name")
    ):
        return [first, second]
    message_param: ChatCompletionAssistantMessageParam = {
        "role": "assistant",
    }
    first_content = first.get("content", "")
    second_content = second.get("content", "")
    assert isinstance(first_content, str) and isinstance(
        second_content, str
    ), "Merging assistant message params with non-string content is not supported at this time."
    assert not isinstance(first.get("refusal"), str) and not isinstance(
        second.get("refusal"), str
    ), "Merging assistant message params with refusal(s) is not supported at this time."
    message_param["content"] = first_content + joiner + second_content
    if second.get("function_call"):
        message_param["function_call"] = second.get("function_call")
    if first.get("name"):
        message_param["name"] = first.get("name", "")
    if second.get("tool_calls"):
        message_param["tool_calls"] = second.get("tool_calls", [])
    return [message_param]


def role(message: Union[ChatCompletionMessageParam, Choice]) -> str:
    return message.message.role if isinstance(message, Choice) else message["role"]
