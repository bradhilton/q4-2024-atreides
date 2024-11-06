from dataclasses import dataclass, field
import numpy as np
import random

from .episode import SampleEpisode


@dataclass
class EpisodeSampler:
    sample: SampleEpisode
    num_samples: int = 0
    num_goldilocks: int = 0

    def goldilocks_rate(self, prior: float, effective_sample_size: float) -> float:
        return (self.num_goldilocks + prior * effective_sample_size) / (
            self.num_samples + effective_sample_size
        )


@dataclass
class EpisodeSamplerRouter:
    random_sampler: EpisodeSampler
    exploitation_factor: float = 1.0
    min_random_episode_sample_probability_half_life: int = 100
    other_samplers: list[EpisodeSampler] = field(default_factory=list)

    def get_sampler(self) -> EpisodeSampler:
        if not self.other_samplers:
            return self.random_sampler
        prior, effective_sample_size = (
            self.goldilocks_rate_prior_and_effective_sample_size()
        )
        min_random_goldilocks_rate = 1.0 * np.exp(
            -np.log(2)
            / self.min_random_episode_sample_probability_half_life
            * self.random_sampler.num_samples
        )
        random_goldilocks_rate = max(
            self.random_sampler.goldilocks_rate(prior, effective_sample_size),
            min_random_goldilocks_rate,
        )
        other_goldilocks_rates = np.array(
            [
                sampler.goldilocks_rate(prior, effective_sample_size)
                for sampler in self.other_samplers
            ]
        )
        other_sampler_weights = other_goldilocks_rates**self.exploitation_factor
        other_sampler_weights /= other_sampler_weights.sum()
        other_expected_goldilocks_rate = other_goldilocks_rates @ other_sampler_weights
        hierachical_weights = (
            np.array([random_goldilocks_rate, other_expected_goldilocks_rate])
            ** self.exploitation_factor
        )
        hierachical_weights /= hierachical_weights.sum()
        if random.random() < hierachical_weights[0]:
            return self.random_sampler
        else:
            return random.choices(self.other_samplers, weights=other_sampler_weights)[0]

    def goldilocks_rate_prior_and_effective_sample_size(self) -> tuple[float, float]:
        num_goldilocks = self.random_sampler.num_goldilocks + sum(
            s.num_goldilocks for s in self.other_samplers
        )
        num_samples = self.random_sampler.num_samples + sum(
            s.num_samples for s in self.other_samplers
        )
        return (
            num_goldilocks / num_samples
            if num_goldilocks != 0 and num_samples != 0
            else 1.0
        ), max(num_samples / (len(self.other_samplers) + 1), 1)
