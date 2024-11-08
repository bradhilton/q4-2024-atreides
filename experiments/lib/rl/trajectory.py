from dataclasses import dataclass

from .completion import Completion
from .episode import Episode


@dataclass
class Trajectory:
    episode: Episode
    terminus: Completion
    abs_advantage: float
    token_count: int
    episode_decay: float
    completion_decay: float

    def score(self) -> float:
        return self.episode.weight * self.abs_advantage / self.token_count
