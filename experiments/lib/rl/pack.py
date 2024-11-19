from collections import Counter
import torch
from typing import Optional, TypedDict

from .completion import Completion
from .episode import Episode
from ..tokenizer import Tokenizer
from ..utils import truncate_pad


class PackedTensors(TypedDict):
    tokens: torch.Tensor
    advantages: torch.Tensor
    logprobs: torch.Tensor
    weights: torch.Tensor
    ids: torch.Tensor
    parent_ids: torch.Tensor


def packed_tensors(
    episodes: list[Episode],
    model: str,
    sequence_length: int,
    trajectories_per_episode: Optional[int],
    tokenizer: Tokenizer,
) -> PackedTensors:
    sequences, completion_weights = packed_sequences(
        episodes, model, sequence_length, trajectories_per_episode, tokenizer
    )
    completion_tensors = {
        completion: get_completion_tensors(completion, weight, tokenizer)
        for completion, weight in completion_weights.items()
    }
    return {
        key: torch.cat(
            [
                truncate_pad(
                    torch.cat(
                        [
                            completion_tensors[completion]["tokens"]
                            for completion in sequence
                        ]
                    ),
                    [sequence_length],
                    value=pad_value,
                )
                for sequence in sequences
            ]
        )
        for key, pad_value in {
            "tokens": tokenizer.get_pad_token_id() or 0,
            "advantages": torch.nan,
            "logprobs": torch.nan,
            "weights": 0.0,
            "ids": 0,
            "parent_ids": 0,
        }.items()
    }  # type: ignore


def packed_sequences(
    episodes: list[Episode],
    model: str,
    sequence_length: int,
    trajectories_per_episode: Optional[int],
    tokenizer: Tokenizer,
) -> tuple[list[list[Completion]], dict[Completion, float]]:
    sequences: list[Counter[Completion]] = []
    completions: Counter[Completion] = Counter()
    for episode in episodes:
        termini: list[Completion] = []
        for terminus in (
            (
                episode.completion.sample_terminus(model=model)
                for _ in range(trajectories_per_episode)
            )
            if trajectories_per_episode is not None
            else episode.completion.leaves(model=model)
        ):
            for terminus in terminus.ancestors(including_self=True):
                if (
                    terminus.advantage(cache=True, model=model) != 0
                    and terminus.all_token_count(tokenizer) <= sequence_length
                ):
                    break
            termini.append(terminus)
        for terminus in termini:
            while True:
                for completion in terminus.ancestors(including_self=True, reverse=True):
                    completions[completion] += 1
                    if (
                        sum(c.token_count(tokenizer) for c in completions)
                        > sequence_length
                    ):
                        for c in completion.ancestors(including_self=True):
                            completions[c] -= 1
                        sequences.append(completions)
                        completions = Counter()
                    else:
                        break
    sequences.append(completions)
    total_occurances = sum(sequences, Counter())
    sequence_occurences = Counter(
        completion for completions in sequences for completion in completions
    )
    weights: dict[Completion, float] = {
        completion: (
            (
                (
                    total_occurances[completion]
                    if trajectories_per_episode is not None
                    else completion.sample_probability(cache=True, model=model)
                )
                / sequence_occurences[completion]
            )
            if completion.advantage(cache=True, model=model) != 0
            else 0
        )
        for completion in total_occurances
    }
    average_weight = sum(weights.values()) / len(weights)
    weights = {
        completion: weight / average_weight for completion, weight in weights.items()
    }
    return [list(sequence) for sequence in sequences], weights


def get_completion_tensors(
    completion: Completion, weight: float, tokenizer: Tokenizer
) -> PackedTensors:
    tokens = completion.tokens(tokenizer)
    replacement_token = get_replacement_token(tokens, tokenizer)
    mask = (
        completion.tokens(tokenizer, replacement_token=replacement_token)
        == replacement_token
    )
    advantages = torch.full_like(mask, fill_value=torch.nan, dtype=torch.float32)
    advantages[mask] = torch.tensor(
        [advantage for advantage in completion.token_advantages(cache=True)]
    )
    logprobs = torch.full_like(mask, fill_value=torch.nan, dtype=torch.float32)
    logprobs[mask] = torch.tensor([logprob for logprob in completion.logprobs()])
    return {
        "tokens": tokens,
        "advantages": advantages,
        "logprobs": logprobs,
        "weights": torch.tensor([weight for _ in range(tokens.shape[0])]),
        "ids": torch.tensor([id(completion) for _ in range(tokens.shape[0])]),
        "parent_ids": torch.tensor(
            [id(completion.parent) for _ in range(tokens.shape[0])]
        ),
    }


def get_replacement_token(tokens: torch.Tensor, tokenizer: Tokenizer) -> str:
    max_token = int(tokens.max().item())
    try:
        return tokenizer.get_token(max_token + 1)
    except:
        for i in range(max_token - 1, 0, -1):
            if i in tokens:
                continue
            try:
                return tokenizer.get_token(i)
            except:
                continue
    raise ValueError("No replacement token found")
