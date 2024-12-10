import asyncio
import bisect
from collections import Counter
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.completion_create_params import CompletionCreateParamsBase
from types import TracebackType
from typing import (
    Any,
    Callable,
    cast,
    Never,
    Optional,
    Protocol,
    Union,
    Unpack,
)

from .completion import Completion
from ..utils import get_token


class SamplingKwargs(CompletionCreateParamsBase, total=False):
    messages: Never
    model: Optional[str]
    extra_body: dict[str, Any]


class ThrottledPrioritySemaphore:
    def __init__(
        self,
        max_concurrent_actions: int = 2**31 - 1,
        min_time_between_actions: float = 0.0,
    ) -> None:
        self.max_concurrent_actions = max_concurrent_actions
        self.min_time_between_actions = min_time_between_actions
        self.concurrent_actions = 0
        self.last_action_time = asyncio.get_event_loop().time()
        self.queue: list[tuple[asyncio.Event, float]] = []
        self.task: asyncio.Task = asyncio.create_task(self._wait())

    def __call__(
        self, n: int = 1, priority: float = 0.0
    ) -> "ThrottledPrioritySemaphoreContextManager":
        return ThrottledPrioritySemaphoreContextManager(self, n, priority)

    async def acquire(self, n: int = 1, priority: float = 0.0) -> None:
        event = asyncio.Event()
        bisect.insort(self.queue, (event, priority), key=lambda x: x[1])
        self._wait_if_needed()
        await event.wait()
        self.concurrent_actions += n
        self._wait_if_needed()

    def release(self, n: int = 1) -> None:
        self.concurrent_actions -= n
        self._wait_if_needed()

    def _wait_if_needed(self) -> None:
        if (
            self.queue
            and self.concurrent_actions < self.max_concurrent_actions
            and self.task.done()
        ):
            self.task = asyncio.create_task(self._wait())

    async def _wait(self) -> None:
        await asyncio.sleep(
            max(
                0,
                self.last_action_time
                + self.min_time_between_actions
                - asyncio.get_event_loop().time(),
            )
        )
        self.last_action_time = asyncio.get_event_loop().time()
        if self.queue and self.concurrent_actions < self.max_concurrent_actions:
            event, _ = self.queue.pop(0)
            event.set()


class ThrottledPrioritySemaphoreContextManager:
    def __init__(
        self,
        semaphore: ThrottledPrioritySemaphore,
        n: int,
        priority: float,
    ) -> None:
        self.semaphore = semaphore
        self.n = n
        self.priority = priority

    async def __aenter__(self) -> None:
        await self.semaphore.acquire(self.n, self.priority)

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.semaphore.release(self.n)


class CompletionSamplerProtocol(Protocol):
    async def sample_completions(
        self,
        parent: Completion,
        continue_last_message_if_assistant: bool = True,
        strip: set[str] = set(),
        priority: float = 0.0,
        **kwargs: Unpack[SamplingKwargs],
    ) -> list[Completion]: ...

    async def get_model(self) -> str: ...


class CompletionSampler:
    def __init__(
        self,
        client: AsyncOpenAI,
        max_concurrent_samples: int = 2**31 - 1,
        min_time_between_requests: float = 0.0,
        **kwargs: Unpack[SamplingKwargs],
    ) -> None:
        self.client = client
        self.semaphore = ThrottledPrioritySemaphore(
            max_concurrent_samples,
            min_time_between_requests,
        )
        self.kwargs = kwargs
        self.model = kwargs.get("model")
        self.queue: list[asyncio.Event] = []

    async def sample_completions(
        self,
        parent: Completion,
        *,
        continue_last_message_if_assistant: bool = True,
        examples: Union[
            list[ChatCompletionMessageParam],
            Callable[[], list[ChatCompletionMessageParam]],
        ] = [],
        get_recovery_pattern: Optional[Callable[[], Optional[str]]] = None,
        priority: float = 0.0,
        strip: set[str] = set(),
        **kwargs: Unpack[SamplingKwargs],
    ) -> list[Completion]:
        if callable(examples):
            example_sets: list[tuple[list[ChatCompletionMessageParam], int]] = []
            for _ in range(kwargs.pop("n", 1) or 1):
                some_examples = examples()
                for i, (other_examples, count) in enumerate(example_sets.copy()):
                    if some_examples == other_examples:
                        example_sets[i] = (other_examples, count + 1)
                        break
                else:
                    example_sets.append((some_examples, 1))
            return [
                completion
                for completions in await asyncio.gather(
                    *(
                        self.sample_completions(
                            parent,
                            continue_last_message_if_assistant=continue_last_message_if_assistant,
                            examples=examples,
                            get_recovery_pattern=get_recovery_pattern,
                            priority=priority,
                            strip=strip,
                            n=count,
                            **kwargs,  # type: ignore
                        )
                        for examples, count in example_sets
                    )
                )
                for completion in completions
            ]
        if get_recovery_pattern and parent.advantage(cache=True) < 0:
            patterns = Counter(
                get_recovery_pattern() for _ in range(kwargs.get("n", 1) or 1)
            )
            if None not in patterns or len(patterns) > 1:
                _ = kwargs.pop("n", None)
                extra_body = kwargs.pop("extra_body", {})
                return [
                    completion
                    for completions in await asyncio.gather(
                        *(
                            self.sample_completions(
                                parent,
                                continue_last_message_if_assistant=continue_last_message_if_assistant,
                                priority=priority,
                                strip=strip,
                                n=count,
                                extra_body={
                                    **extra_body,
                                    "guided_regex": pattern,
                                },
                                **kwargs,  # type: ignore
                            )
                            for pattern, count in patterns.items()
                        )
                    )
                    for completion in completions
                ]
        messages = examples + parent.all_message_params()
        untyped_kwargs: dict = {
            "messages": messages,
            "logprobs": True,
            **self.kwargs,
            **kwargs,
            "extra_headers": {
                **self.kwargs.get("extra_headers", {}),
                **kwargs.get("extra_headers", {}),
            },
            "extra_query": {
                **self.kwargs.get("extra_query", {}),
                **kwargs.get("extra_query", {}),
            },
            "extra_body": {
                **self.kwargs.get("extra_body", {}),
                **kwargs.get("extra_body", {}),
                "skip_special_tokens": "guided_regex" in kwargs.get("extra_body", {}),
            },
        }
        if continue_last_message_if_assistant and messages[-1]["role"] == "assistant":
            prefix = messages[-1].get("content") or ""
            if not isinstance(prefix, str):
                prefix = "".join(
                    part["text"] if part["type"] == "text" else part["refusal"]
                    for part in prefix
                )
            untyped_kwargs["extra_body"]["add_generation_prompt"] = False
            untyped_kwargs["extra_body"]["continue_final_message"] = True
        else:
            prefix = ""
        if not "model" in untyped_kwargs:
            untyped_kwargs["model"] = await self.get_model()
        async with self.semaphore(untyped_kwargs.get("n", 1), priority or 0):
            chat_completion = cast(
                ChatCompletion,
                await self.client.chat.completions.create(**untyped_kwargs),
            )
        completions = [
            Completion(
                parent=parent,
                messages=[
                    self._remove_prefix_and_unwanted_leading_tokens(
                        choice, prefix, strip
                    )
                ],
                weight=parent.weight,
                model=untyped_kwargs["model"],
            )
            for choice in chat_completion.choices
        ]
        if kwargs.get("extra_body", {}).pop("guided_regex", None):
            kwargs["n"] = 1
            completions = [
                completion.merge()
                for completions in await asyncio.gather(
                    *(
                        self.sample_completions(
                            completion,
                            continue_last_message_if_assistant=continue_last_message_if_assistant,
                            examples=examples,
                            strip=strip,
                            priority=priority,
                            **kwargs,
                        )
                        for completion in completions
                    )
                )
                for completion in completions
            ]
        return completions

    _get_model_task: Optional[asyncio.Task[str]] = None

    async def get_model(self) -> str:
        if self.model:
            return self.model
        if self._get_model_task is None:
            self._get_model_task = asyncio.create_task(self._get_model())
        return await self._get_model_task

    async def _get_model(self) -> str:
        async for model in self.client.models.list():
            print(f"Using model: {model.id}")
            self.model = model.id
            return model.id
        raise RuntimeError("No models available")

    def _remove_prefix_and_unwanted_leading_tokens(
        self, choice: Choice, prefix: str, strip: set[str]
    ) -> Choice:
        if strip and choice.logprobs:
            logprobs = choice.logprobs.content or choice.logprobs.refusal or []
            while logprobs:
                if get_token(logprobs[0]) in strip:
                    prefix += get_token(logprobs.pop(0))
                else:
                    break
        if choice.message.content:
            choice.message.content = choice.message.content.removeprefix(prefix)
        if choice.message.refusal:
            choice.message.refusal = choice.message.refusal.removeprefix(prefix)
        return choice

    def __eq__(self, other: Any) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


class CompletionSamplerPool(CompletionSampler):
    def __init__(
        self,
        samplers: list[CompletionSampler],
    ) -> None:
        self.samplers = samplers
        self.router: dict[Completion, CompletionSampler] = {}

    async def sample_completions(
        self,
        parent: Completion,
        continue_last_message_if_assistant: bool = True,
        strip: set[str] = set(),
        priority: float = 0.0,
        **kwargs: Unpack[SamplingKwargs],
    ) -> list[Completion]:
        root = parent.root()
        if root in self.router:
            completion_sampler = self.router[root]
        else:
            counter = Counter(self.router.values())
            completion_sampler = self.router[root] = min(
                self.samplers,
                key=lambda sampler: counter[sampler],
            )
        return await completion_sampler.sample_completions(
            parent,
            continue_last_message_if_assistant=continue_last_message_if_assistant,
            strip=strip,
            priority=priority,
            **kwargs,
        )

    async def get_model(self) -> str:
        return await self.samplers[0].get_model()
