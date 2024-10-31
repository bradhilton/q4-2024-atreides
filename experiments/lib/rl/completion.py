import numpy as np
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice, ChoiceLogprobs
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
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
    Any,
    cast,
    Iterable,
    Literal,
    Optional,
    Self,
    Union,
)


class Completion(BaseModel):
    parent: Optional["Completion"] = cast(None, Field(None, exclude=True))  # State
    messages: list[Union[SkipValidation[ChatCompletionMessageParam], Choice]] = (
        []
    )  # Action
    reward: float = 0.0  # Reward
    # Next state, action, reward triples
    children: set["Completion"] = set()

    def commit(self, reward: Optional[float] = None) -> None:
        """
        Recursively commits this and uncommitted ancestor completions to the tree.

        Before committing, be sure:
        1) To set any non-zero rewards for this completion and its ancestors if they have not already been set.
        2) That this completion is the terminal step in the trajectory.

        May optionally set the reward for this completion at the time of commit.

        Example:
        ```python
        for completion in await sampler.sample_completions(...):
            completion.commit(reward=/* calculated reward... */)
        ```
        """
        if reward is not None:
            self.reward = reward
        if not self.parent:
            return
        self.parent.commit()
        self.parent.children.add(self)

    @property
    def committed(self) -> bool:
        """
        Whether this completion has been committed to the tree.

        Sampled completions are uncommitted until they and their ancestor completions are committed. (See `commit` method.)

        NOTE:
        Root nodes are always considered committed.
        """
        return self.parent is None or (
            self in self.parent.children and self.parent.committed
        )

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
        return self.value() - (self.parent.value() - self.parent.reward)

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

    def descendants(self) -> Iterable["Completion"]:
        for child in self.children:
            yield child
            yield from child.descendants()

    def leaves(self) -> Iterable["Completion"]:
        if not self.children:
            yield self
        for child in self.children:
            yield from child.leaves()

    def message_params(self) -> list[ChatCompletionMessageParam]:
        return [
            (
                message_param(message_or_choice.message)
                if isinstance(message_or_choice, Choice)
                else message_or_choice
            )
            for message_or_choice in self.messages
        ]

    def all_message_params(
        self, join_consecutive_assistant_messages: Union[bool, str] = True
    ) -> list[ChatCompletionMessageParam]:
        message_params = self.message_params()
        if not self.parent:
            return message_params
        parent_message_params = self.parent.all_message_params(
            join_consecutive_assistant_messages=join_consecutive_assistant_messages
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

    def can_split(self) -> bool:
        return sum(len(sequence) for sequence in self._token_logprob_sequences()) > 1

    def split(self, by: Literal["count", "prob", "logprob"]) -> bool:
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
                child = Completion(
                    parent=self,
                    messages=[
                        Choice(
                            _request_id=choice._request_id,
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
                    children=self.children,
                )
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
    message: ChatCompletionMessage,
) -> ChatCompletionAssistantMessageParam:
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
    first_content = first.get("content")
    second_content = second.get("content")
    assert isinstance(first_content, str) and isinstance(
        second_content, str
    ), "Merging assistant message params with absent or non-string content is not supported at this time."
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
