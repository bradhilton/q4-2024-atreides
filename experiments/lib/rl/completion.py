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
import random
import torch
from typing import (
    cast,
    Iterable,
    Literal,
    Optional,
    Self,
    Required,
    Sequence,
    TypeAlias,
    Union,
)

from ..tokenizer import Tokenizer


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
    ):
        self.parent = parent
        self.messages = list(messages or [])
        self.reward = reward
        self.children = children or set()
        self.weight = weight
        self.model = model
        self.fork = fork
        self._cached_values: dict[tuple[Optional[str], Optional[bool]], float] = {}
        self._cached_sample_weights: dict[tuple[Optional[str], float], float] = {}
        self._cached_tokens: Optional[torch.Tensor] = None
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
            self._cached_sample_weights = {}
        if not self.parent:
            return
        self.parent.commit()
        self.parent.children.add(self)
        self.parent._cached_values = {}
        self.parent._cached_sample_weights = {}

    def value(
        self,
        cache: bool = False,
        model: Optional[str] = None,
        fork: Optional[bool] = None,
    ) -> float:
        """
        Estimates the undiscounted Q-value for the current state-action (`parent`, `messages`) pair.

        Computes the Q-value by summing the immediate reward (`reward`) and the
        average of the values of sampled successor state-action pairs (`children`).

        Args:
            cache (bool): Whether to cache Q-value estimates for faster retrieval.
            model (Optional[str]): Optionally limit child Q-value estimates to a specific model.
            fork (Optional[bool]): Optionally limit child Q-value estimates to only forked or unforked completions.

        Returns:
            float: Estimated Q-value for the state-action pair.
        """
        if cache and (model, fork) in self._cached_values:
            return self._cached_values[(model, fork)]
        child_values = [
            completion.value(cache=cache, model=model, fork=fork)
            for completion in self.children
            if (model is None or completion.model is None or completion.model == model)
            and (fork is None or completion.fork == fork)
        ]
        value = self.reward + (sum(child_values) / max(len(child_values), 1))
        if cache:
            self._cached_values[(model, fork)] = value
        return value

    def advantage(self, cache: bool = False, model: Optional[str] = None) -> float:
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
            self.value(cache=cache, model=model)
            - (self.parent.value(cache=cache, model=model) - self.parent.reward)
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

    def all_abs_advantage(self, cache: bool = False) -> float:
        return abs(self.advantage(cache=cache)) + (
            self.parent.all_abs_advantage(cache=cache) if self.parent else 0
        )

    def all_abs_advantage_per_token(
        self, tokenizer: Tokenizer, cache: bool = False
    ) -> float:
        return self.all_abs_advantage(cache=cache) / self.all_token_count(
            tokenizer, cache=cache
        )

    def ancestors(
        self, including_self: bool = False, reverse: bool = False
    ) -> Iterable["Completion"]:
        if not reverse and including_self:
            yield self
        if self.parent:
            yield from self.parent.ancestors(including_self=True, reverse=reverse)
        if reverse and including_self:
            yield self

    def matches_model(self, model: Optional[str]) -> bool:
        return not model or not self.model or self.model == model

    def descendants(
        self, including_self: bool = False, model: Optional[str] = None
    ) -> Iterable["Completion"]:
        if including_self and self.matches_model(model):
            yield self
        for child in self.children:
            yield from child.descendants(including_self=True, model=model)

    def leaves(self, model: Optional[str] = None) -> Iterable["Completion"]:
        if not self.children and self.matches_model(model):
            yield self
        for child in self.children:
            yield from child.leaves(model=model)

    def depths(self, depth: int = 0, model: Optional[str] = None) -> Iterable[int]:
        if self.matches_model(model):
            yield depth
        for child in self.children:
            yield from child.depths(depth + 1, model=model)

    def max_depth(self, model: Optional[str] = None) -> int:
        return max(self.depths(model=model), default=0)

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
        self.parent._cached_tokens = None
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
        for choice in self.messages:
            if isinstance(choice, Choice) and choice.logprobs:
                for token_logprob in (
                    choice.logprobs.content or choice.logprobs.refusal or []
                ):
                    yield token_logprob.logprob

    def all_logprobs(self) -> Iterable[float]:
        if self.parent:
            yield from self.parent.all_logprobs()
        yield from self.logprobs()

    def root(self) -> "Completion":
        return self.parent.root() if self.parent else self

    def token_advantage(self, cache: bool = False) -> float:
        return self.advantage(cache=cache) / (self.num_token_logprobs() or 1)

    def token_advantages(self, cache: bool = False) -> Iterable[float]:
        advantage = self.advantage(cache=cache)
        num_token_logprobs = self.num_token_logprobs()
        token_advantage = advantage / max(num_token_logprobs, 1)
        return (token_advantage for _ in range(num_token_logprobs))

    def all_token_advantages(self, cache: bool = False) -> Iterable[float]:
        if self.parent:
            yield from self.parent.all_token_advantages(cache=cache)
        yield from self.token_advantages(cache=cache)

    def token_count(self, tokenizer: Tokenizer, *, cache: bool = False) -> int:
        return self.tokens(tokenizer, cache=cache).size(0)

    def all_token_count(self, tokenizer: Tokenizer, *, cache: bool = False) -> int:
        return self.token_count(tokenizer, cache=cache) + (
            self.parent.all_token_count(tokenizer, cache=cache) if self.parent else 0
        )

    def tokens(
        self,
        tokenizer: Tokenizer,
        *,
        cache: bool = False,
        replacement_token: Optional[str] = None,
    ) -> torch.Tensor:
        if cache and self._cached_tokens is not None and replacement_token is None:
            return self._cached_tokens
        tokens = tokenizer.encode(
            self.message_params(replacement_token=replacement_token),  # type: ignore
            remove_bos=self.parent is not None,
            first_message_is_continuation=(
                self.parent is not None
                and (
                    role(self.parent.messages[-1])
                    == role(self.messages[0])
                    == "assistant"
                )
            ),
            continue_final_message=role(self.messages[-1]) == "assistant"
            and any(role(child.messages[0]) == "assistant" for child in self.children),
            replace_suffix=(
                ("<|im_end|>\n", "\n")
                if isinstance(self.messages[-1], Choice)
                and self.messages[-1].logprobs
                and self.messages[-1].logprobs.content
                and self.messages[-1].logprobs.content[-1].token == "<|im_end|>"
                else None
            ),
        )
        if cache and replacement_token is None:
            self._cached_tokens = tokens
        return tokens

    def sample_weight(
        self, cache: bool = False, model: Optional[str] = None, power: float = 1.0
    ) -> float:
        if not self.matches_model(model):
            return 0.0
        if not self.parent or not power:
            return 1.0
        if cache and (model, power) in self._cached_sample_weights:
            return self._cached_sample_weights[(model, power)]
        weight = (
            self.parent.sample_weight(cache=cache, model=model, power=power)
            / sum(child.matches_model(model) for child in self.parent.children)
        ) ** power
        if cache:
            self._cached_sample_weights[(model, power)] = weight
        return weight

    def can_split(self, by: SplitMethod, separators: Optional[set[str]] = None) -> bool:
        positive_weights = 0
        for weight in self._split_weights(by, separators=separators):
            if weight > 0:
                positive_weights += 1
            if positive_weights > 1:
                return True
        return False

    def split_weight(
        self, by: SplitMethod, separators: Optional[set[str]] = None
    ) -> float:
        return sum(self._split_weights(by, separators=separators))

    def split(self, by: SplitMethod, separators: Optional[set[str]] = None) -> bool:
        """
        Creates a new child completion and splits the completable contents between the parent (this) and the child.

        NOTE: Currently only supports splitting `Choice`s with `logprobs.content`.

        Args:
            by (Literal["count", "prob", "logprob"]): The weighting method for splitting the completion.
            separators (Optional[set[str]]): An optional set of separators to group tokens when splitting.

        Returns:
            bool: Whether the completion was successfully split.
        """
        split_weights = list(self._split_weights(by, separators=separators))
        assert (
            separators is None or len(split_weights) == self.num_token_logprobs()
        ), "Number of weights does not match number of tokens."
        if len(split_weights) < 2:
            return False
        cumsum = np.cumsum(split_weights)
        np.isclose(cumsum[-1], self.split_weight(by))
        assert separators is None or np.isclose(
            cumsum[-1], self.split_weight(by)
        ), "Grouped weights do not sum to total weight."
        split = np.searchsorted(cumsum, max(cumsum[-1] / 2, cumsum[0]), side="right")
        assert split != 0, "Cannot split at start of completion."
        assert split != len(split_weights), "Cannot split at end of completion."
        assert split_weights[split] != 0, "Split point has zero weight."
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
                    len(token_logprob.token)
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
                self._cached_tokens = None
                return True
        return False

    def _split_weights(
        self, by: SplitMethod, separators: Optional[set[str]]
    ) -> Iterable[float]:
        weight = 0
        n = 0
        for sequence in self._token_logprob_sequences():
            separator = True
            for token_logprob in sequence:
                if (
                    separator
                    and n
                    and (separators is None or not token_logprob.token in separators)
                ):
                    yield weight
                    yield from (0 for _ in range(n - 1))
                    weight = 0
                    n = 0
                weight += dict(
                    count=lambda _: 1,
                    prob=lambda x: 1 - np.exp(x),
                    logprob=lambda x: -x,
                )[by](token_logprob.logprob)
                n += 1
                separator = separators is None or token_logprob.token in separators
        if n:
            yield weight
            yield from (0 for _ in range(n - 1))

    def _token_logprob_sequences(self) -> Iterable[list[ChatCompletionTokenLogprob]]:
        return (
            choice.logprobs.content
            for choice in self.messages
            if isinstance(choice, Choice)
            and choice.logprobs
            and choice.logprobs.content
        )

    def num_token_logprobs(self) -> int:
        return sum(len(sequence) for sequence in self._token_logprob_sequences())

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    def html(self, scale_colors: float, cache: bool = False) -> str:
        if not self.messages:
            return ""
        token_advantage = self.token_advantage(cache=cache) * scale_colors
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
                replacement_token or token_logprob.token
                for token_logprob in choice.logprobs.content
            )
        if choice.logprobs.refusal:
            message.refusal = "".join(
                replacement_token or token_logprob.token
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
