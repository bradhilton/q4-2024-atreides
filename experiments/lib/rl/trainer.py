from typing import AsyncIterable, Iterable, Optional, Union

from .episode import Episode


Episodes = Union[Iterable[Episode], AsyncIterable[Episode]]


class Trainer:
    def __init__(
        self,
        model: str,
        train: Episodes,
        val: Optional[Episodes] = None,
        test: Optional[Episodes] = None,
    ) -> None: ...

    async def train(self) -> None: ...
