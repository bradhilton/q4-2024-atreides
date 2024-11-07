from .episode import Episode
from .episode_sampler import EpisodeSamplerRouter


class EpisodeBuffer:
    episode_sampler_router: EpisodeSamplerRouter
    abs_buffer_size = 20
    weighted_buffer_size = 40
    buffer: list[Episode] = []
    max_running = 100
