from dataclasses import dataclass
import httpx
from openai import AsyncOpenAI
from openai._types import Body, Headers, Query
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.completion_create_params import CompletionCreateParamsBase
from typing import Never


class ChatCompletionCreateParams(CompletionCreateParamsBase, total=False):
    messages: Never
    model: Never
    extra_headers: Headers | None
    extra_query: Query | None
    extra_body: Body | None


@dataclass
class Model:
    name: str
    base_model: str


@dataclass
class Prompt:
    messages: list[ChatCompletionMessageParam]
    params: ChatCompletionCreateParams


@dataclass
class Request:
    api_key: str | None
    organization: str | None
    project: str | None
    base_url: str | httpx.URL | None
    params: ChatCompletionCreateParams


@dataclass
class Response:
    prompt: Prompt
    model: str
    messages: list[ChatCompletion | ChatCompletionChunk | Request]


async def test(request: Request) -> None:
    Request(None, None, None, None, params={})
    client = AsyncOpenAI()
    await client.chat.completions.create(
        **request.params, **{"messages": [], "model": ""}, stream=True
    )
