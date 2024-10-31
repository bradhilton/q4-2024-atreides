import asyncio
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.completion_create_params import CompletionCreateParamsBase
from typing import (
    cast,
    Never,
    Optional,
    Unpack,
)

from .completion import Completion


class Kwargs(CompletionCreateParamsBase, total=False):
    messages: Never
    model: Optional[str]


class CompletionSampler:
    def __init__(
        self,
        client: AsyncOpenAI,
        max_parallel_requests: int = 2**31 - 1,
        **kwargs: Unpack[Kwargs],
    ) -> None:
        self.client = client
        self.semaphore = asyncio.Semaphore(max_parallel_requests)
        self.kwargs = kwargs
        self.model = kwargs.get("model")

    async def sample_completions(
        self, parent: Completion, **kwargs: Unpack[Kwargs]
    ) -> list[Completion]:
        prefix = ""
        kwargs = dict(
            messages=parent.all_message_params(),
            logprobs=True,
            **self.kwargs,  # type: ignore
            **kwargs,  # type: ignore
            extra_headers={
                **self.kwargs.get("extra_headers", {}),
                **kwargs.get("extra_headers", {}),
            },
            extra_query={
                **self.kwargs.get("extra_query", {}),
                **kwargs.get("extra_query", {}),
            },
            extra_body={
                **self.kwargs.get("extra_body", {}),
                **kwargs.get("extra_body", {}),
            },
        )
        if not "model" in kwargs:
            kwargs["model"] = await self._get_model()
        async with self.semaphore:
            chat_completion = cast(
                ChatCompletion,
                await self.client.chat.completions.create(
                    **kwargs,  # type: ignore
                ),
            )
        return [
            Completion(
                parent=parent,
                messages=[self._remove_prefix(choice, prefix)],
            )
            for choice in chat_completion.choices
        ]

    _get_model_task: Optional[asyncio.Task[str]] = None

    async def _get_model(self) -> str:
        if self.model:
            return self.model
        if self._get_model_task is None:
            self._get_model_task = asyncio.create_task(self.__get_model())
        return await self._get_model_task

    async def __get_model(self) -> str:
        async for model in self.client.models.list():
            print(f"Using model: {model.id}")
            self.model = model.id
            return model.id
        raise RuntimeError("No models available")

    def _remove_prefix(self, choice: Choice, prefix: str) -> Choice:
        if choice.message.content:
            choice.message.content = choice.message.content.removeprefix(prefix)
        return choice
