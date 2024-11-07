from itertools import chain
import numpy as np
from openai.types.chat import ChatCompletionMessage
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
from pydantic._internal._model_construction import ModelMetaclass
from typing import (
    cast,
    Iterable,
    Literal,
    Optional,
    Self,
    Required,
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


class Completion(BaseModel):
    parent: Optional["Completion"] = cast(None, Field(None, exclude=True))  # State
    messages: list[Union[ChatCompletionMessageParam, Choice]] = []  # Action
    reward: float = 0.0  # Reward
    # Next state, action, reward triples
    children: set["Completion"] = set()
    weight: float = 1.0

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
        if not self.parent:
            return
        self.parent.commit()
        self.parent.children.add(self)

    def value(self) -> float:
        """
        Estimates the undiscounted Q-value for the current state-action (`parent`, `messages`) pair.

        Computes the Q-value by summing the immediate reward (`reward`) and the
        average of the values of sampled successor state-action pairs (`children`).

        Returns:
            float: Estimated Q-value for the state-action pair.
        """
        return self.reward + (
            sum(c.value() for c in self.children) / max(len(self.children), 1)
        )

    def advantage(self) -> float:
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
        return (self.value() - (self.parent.value() - self.parent.reward)) * self.weight

    def adjustment(self, lambda_: float = 1.0) -> float:
        return 1 - (
            lambda_
            / (
                lambda_
                + max(sum(c.adjustment(lambda_) for c in self.children), 0.5) * 2
            )
        )

    def adjusted_value(self, lambda_: float = 1.0) -> float:
        if not self.parent:
            return self.value()
        return (self.value() - self.parent.value()) * self.adjustment(
            lambda_
        ) + self.parent.value()

    def adjusted_advantage(self, lambda_: float = 1.0) -> float:
        return self.advantage() * self.adjustment(lambda_)

    def all_abs_advantage(self) -> float:
        return abs(self.advantage()) + (
            self.parent.all_abs_advantage() if self.parent else 0
        )

    def ancestors(self, including_self: bool = False) -> Iterable["Completion"]:
        if including_self:
            yield self
        if not self.parent:
            return
        yield self.parent
        yield from self.parent.ancestors()

    def descendants(self) -> Iterable["Completion"]:
        for child in self.children:
            yield child
            yield from child.descendants()

    def leaves(self) -> Iterable["Completion"]:
        if not self.children:
            yield self
        for child in self.children:
            yield from child.leaves()

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

    def token_advantages(self) -> Iterable[float]:
        advantage = self.advantage()
        num_token_logprobs = self._num_token_logprobs()
        token_advantage = advantage / num_token_logprobs
        return (token_advantage for _ in range(num_token_logprobs))

    def all_token_advantages(self) -> Iterable[float]:
        if self.parent:
            yield from self.parent.all_token_advantages()
        yield from self.token_advantages()

    def token_count(self, tokenizer: Tokenizer) -> int:
        if not self.messages:
            return 0
        token_count = sum(
            message_token_count(message, tokenizer) for message in self.messages
        )
        join_parent = (
            self.parent
            and self.parent.messages
            and not (
                role(self.parent.messages[-1]) == role(self.messages[0]) == "assistant"
            )
        )
        token_count += tokenizer.join_token_count * max(
            0, len(self.messages) - (0 if join_parent else 1)
        )
        if not self.parent:
            token_count += tokenizer.prefix_token_count
        return token_count

    def all_token_count(self, tokenizer: Tokenizer) -> int:
        return self.token_count(tokenizer) + (
            self.parent.all_token_count(tokenizer) if self.parent else 0
        )

    def can_split(self) -> bool:
        num_logprobs = 0
        for choice in self.messages:
            if isinstance(choice, Choice) and choice.logprobs:
                num_logprobs += len(
                    choice.logprobs.content or choice.logprobs.refusal or []
                )
                if num_logprobs > 1:
                    return True
        return False

    def split(self, by: SplitMethod) -> bool:
        """
        Creates a new child completion and splits the completable contents between the parent (this) and the child.

        NOTE: Currently only supports splitting `Choice`s with `logprobs.content`.

        Args:
            by (Literal["count", "prob", "logprob"]): The weighting method for splitting the completion.

        Returns:
            bool: Whether the completion was successfully split.
        """
        token_logprobs = [
            token_logprob
            for sequence in self._token_logprob_sequences()
            for token_logprob in sequence
        ]
        if len(token_logprobs) < 2:
            return False
        cumsum = np.cumsum(
            [
                dict(
                    count=lambda _: 1,
                    prob=lambda x: 1 - np.exp(x),
                    logprob=lambda x: -x,
                )[by](token_logprob.logprob)
                for token_logprob in token_logprobs
            ]
        )
        split = np.searchsorted(cumsum, cumsum[-1] / 2)
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
                return True
        return False

    def _token_logprob_sequences(self) -> Iterable[list[ChatCompletionTokenLogprob]]:
        return (
            choice.logprobs.content
            for choice in self.messages
            if isinstance(choice, Choice)
            and choice.logprobs
            and choice.logprobs.content
        )

    def _num_token_logprobs(self) -> int:
        return sum(len(sequence) for sequence in self._token_logprob_sequences())

    def __hash__(self) -> int:
        return id(self)

    @model_validator(mode="after")
    def validate_children(self) -> Self:
        for child in self.children:
            if child.parent is None:
                child.parent = self
            elif child.parent is not self:
                raise ValueError("Child completion's parent must be this completion.")
        return self


def message_param(
    choice: Choice,
    replacement_token: Optional[str] = None,
) -> ChatCompletionAssistantMessageParam:
    message = choice.message
    if replacement_token and choice.logprobs:
        replacement = replacement_token
        if choice.logprobs.content:
            message.content = replacement * len(choice.logprobs.content)
        if choice.logprobs.refusal:
            message.refusal = replacement * len(choice.logprobs.refusal)
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


def message_token_count(
    message: Union[ChatCompletionMessageParam, Choice], tokenizer: Tokenizer
) -> int:
    token_count = 0
    if isinstance(message, Choice):
        if message.logprobs:
            token_count += len(
                message.logprobs.content or message.logprobs.refusal or []
            )
        else:
            token_count += tokenizer.get_token_count(
                message.message.content or message.message.refusal or ""
            )
    else:
        content = message.get("content")
        if isinstance(content, str):
            token_count += tokenizer.get_token_count(content)
        elif content is not None:
            token_count += sum(
                tokenizer.get_token_count(
                    part["text"]
                    if part["type"] == "text"
                    else part["refusal"] if part["type"] == "refusal" else ""
                )
                for part in content
            )
        refusal = message.get("refusal")
        if isinstance(refusal, str):
            token_count += tokenizer.get_token_count(refusal)
    return token_count


def role(message: Union[ChatCompletionMessageParam, Choice]) -> str:
    return message.message.role if isinstance(message, Choice) else message["role"]
