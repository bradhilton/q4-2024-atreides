import asyncio
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion, Choice
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
        self,
        parent: Completion,
        continue_last_message_if_assistant: bool = True,
        **kwargs: Unpack[Kwargs],
    ) -> list[Completion]:
        messages = parent.all_message_params()
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
            # import copy
            # import random

            # if prefix and random.random() < 0.25 and parent.advantage(cache=True) < 0:
            #     untyped_kwargs["messages"] = copy.deepcopy(untyped_kwargs["messages"])
            #     untyped_kwargs["messages"][-1]["content"] += random.choice(
            #         ["...", "—"]
            #     ) + random.choice(
            #         [
            #             "",
            #             "",
            #             "",
            #             "hmm",
            #             "wait a second",
            #             "sorry",
            #             "sorry, I made a mistake",
            #             "no",
            #             "no",
            #             "but",
            #             "however",
            #             "alternatively",
            #             "actually",
            #             "technically",
            #             "what I mean to say is",
            #             "hold on",
            #             "let's take a step back",
            #         ]
            #     )  # type: ignore
        else:
            prefix = ""
        if not "model" in untyped_kwargs:
            untyped_kwargs["model"] = await self._get_model()

        async with self.semaphore:
            chat_completion = cast(
                ChatCompletion,
                await self.client.chat.completions.create(**untyped_kwargs),
            )
        return [
            Completion(
                parent=parent,
                messages=[self._remove_prefix(choice, prefix)],
                weight=parent.weight,
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
        if choice.message.refusal:
            choice.message.refusal = choice.message.refusal.removeprefix(prefix)
        return choice
