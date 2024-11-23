from dataclasses import dataclass, field, fields
import torch
import torch.nn as nn
from typing import Iterable, Optional, Union


ignore_labels_cache: dict[
    tuple[torch.Size, Union[int, float], torch.dtype, torch.device], torch.Tensor
] = {}


def shift(
    labels: torch.Tensor, ignore_label: Optional[Union[int, float]] = None
) -> torch.Tensor:
    if ignore_label is None:
        ignore_label = (
            -100
            if labels.dtype in (torch.int32, torch.int64, torch.int16, torch.int8)
            else float("nan")
        )

    # Create a tensor of ignore labels every time if we are compiling, otherwise cache it
    if torch.compiler.is_compiling():
        ignore_labels = torch.full(
            (labels.shape[0], 1), ignore_label, dtype=labels.dtype, device=labels.device
        )
    else:
        key = (labels.shape[-1:], ignore_label, labels.dtype, labels.device)
        if key not in ignore_labels_cache:
            ignore_labels_cache[key] = torch.full(
                (labels.shape[0], 1),
                ignore_label,
                dtype=labels.dtype,
                device=labels.device,
            )
        ignore_labels = ignore_labels_cache[key]

    # Shift labels to compute loss
    return torch.cat((labels[..., 1:], ignore_labels), dim=1)


tensor_field = lambda: field(default_factory=lambda: torch.tensor(0.0))


@dataclass
class PPOResult:
    policy_weight: torch.Tensor = tensor_field()
    entropy_weight: torch.Tensor = tensor_field()
    kl_weight: torch.Tensor = tensor_field()
    policy_loss: torch.Tensor = tensor_field()
    entropy_bonus: torch.Tensor = tensor_field()
    kl_divergence: torch.Tensor = tensor_field()
    num_tokens: torch.Tensor = field(default_factory=lambda: torch.tensor(0))

    def named_tensors(self) -> Iterable[tuple[str, torch.Tensor]]:
        for field in fields(self):
            yield field.name, getattr(self, field.name)

    def tensors(self) -> Iterable[torch.Tensor]:
        return (tensor for _, tensor in self.named_tensors())

    def to(self, target: Union[torch.device, torch.dtype]) -> "PPOResult":
        return PPOResult(
            **{name: tensor.to(target) for name, tensor in self.named_tensors()}
        )

    def __iadd__(self, other: "PPOResult") -> "PPOResult":
        for tensor, other_tensor in zip(self.tensors(), other.tensors()):
            tensor += other_tensor.to(tensor.device)
        return self

    @property
    def total_loss(self) -> torch.Tensor:
        return (
            (self.policy_weight / self.num_tokens) * self.policy_loss
            - (self.entropy_weight / self.num_tokens) * self.entropy_bonus
            + (self.kl_weight / self.num_tokens) * self.kl_divergence
        )


class PPOLoss(nn.Module):
    def __init__(
        self,
        policy_coef: float = 1.0,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        kl_coef=0.0,
        normalize_advantages: bool = True,
    ):
        """
        Initializes the PPO Loss module.

        Args:
            policy_coef (float): Coefficient for the policy loss. Defaults to 1.0.
            clip_epsilon (float): Clipping parameter for PPO (typically between 0.1 and 0.3).
            entropy_coef (float): Coefficient for the entropy bonus to encourage exploration.
            kl_coef (float): Coefficient for KL divergence penalty (defaults to 0.0).
        """
        super().__init__()
        self.policy_coef = policy_coef
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.kl_coef = kl_coef
        self.normalize_advantages = normalize_advantages

    def forward(
        self,
        logits: Union[torch.Tensor, list[torch.Tensor]],
        tokens: torch.Tensor,
        advantages: torch.Tensor,
        logprobs: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        bos_id: Optional[int] = None,
    ) -> PPOResult:
        """
        Computes the PPO loss for sequence data, supporting both regular and chunked inputs.

        Args:
            logits (Union[Tensor, List[Tensor]]):
                Either a single tensor of shape (batch_size, sequence_length, vocab_size)
                or a list of chunked tensors, each of shape
                (batch_size, sequence_length/num_chunks, vocab_size).

            tokens (Tensor):
                Shape: (batch_size, sequence_length)
                Token indices sampled under the old policy.

            advantages (Tensor):
                Shape: (batch_size, sequence_length)
                Advantage estimates for each token.

            logprobs (Tensor):
                Shape: (batch_size, sequence_length)
                Log probabilities of the sampled tokens under the old policy.

            weights (Optional[Tensor]):
                Shape: (batch_size, sequence_length)
                Optional weights for each token in the sequence.

            bos_id (Optional[int] = None):
                Index of the beginning of sequence token in the vocabulary. If None, defaults
                to the first token in `tokens`.

        Returns:
            PPOResult: The combined loss results across all chunks.
        """
        if bos_id is None:
            bos_id = int(tokens.view(-1)[0].item())

        if isinstance(logits, list):
            result = PPOResult().to(logits[0].device)
            num_chunks = len(logits)
            for chunked_args in zip(
                logits,
                tokens.chunk(num_chunks, dim=1),
                advantages.chunk(num_chunks, dim=1),
                logprobs.chunk(num_chunks, dim=1),
                (
                    weights.chunk(num_chunks, dim=1)
                    if weights is not None
                    else [None] * num_chunks
                ),
            ):
                result += self._forward_chunk(*chunked_args, bos_id=bos_id)
            return result

        return self._forward_chunk(
            logits, tokens, advantages, logprobs, weights, bos_id=bos_id
        )

    def _forward_chunk(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        advantages: torch.Tensor,
        logprobs: torch.Tensor,
        weights: Optional[torch.Tensor],
        bos_id: int,
    ) -> PPOResult:
        """
        Processes a single chunk of the PPO loss computation.
        """
        # Flatten logits tensor to shape (batch_size * sequence_length, vocab_size)
        logits = logits.view(-1, logits.size(-1))
        # Shape: (batch_size * sequence_length,)
        tokens = shift(tokens, ignore_label=bos_id).view(-1)
        # Shape: (batch_size * sequence_length,)
        advantages = shift(advantages).view(-1)
        logprobs = shift(logprobs).view(-1)  # Shape: (batch_size * sequence_length,)
        if weights is not None:
            weights = shift(weights).view(-1)  # Shape: (batch_size * sequence_length,)

        # Create a Categorical distribution from logits
        # Shape: (batch_size * sequence_length,)
        dist = torch.distributions.Categorical(logits=logits)

        # Calculate new log probabilities of the taken actions
        new_logprobs = dist.log_prob(tokens)  # Shape: (batch_size * sequence_length,)

        # Debugging
        if False:
            # Shape: (batch_size * sequence_length,)
            kl_divergence = torch.nn.functional.kl_div(
                new_logprobs, logprobs, reduction="none", log_target=True
            )
            # Debugging KL divergence distribution
            import matplotlib.pyplot as plt
            import numpy as np

            # Convert to numpy and remove NaN values
            kl_np = kl_divergence.detach().cpu().numpy()

            # Create line plot
            plt.figure(figsize=(10, 6))
            plt.plot(kl_np, linewidth=2)
            plt.xlabel("Token Index")
            plt.ylabel("KL Divergence")
            plt.title("KL Divergence Across Tokens")
            plt.grid(True, alpha=0.3)
            plt.show()

            # Find first token with KL divergence > 10
            high_kl_idx = (kl_np > 10).nonzero()[0]
            if len(high_kl_idx) > 0:
                print(f"First token with KL divergence > 10: {high_kl_idx[0]}")
                print(f"Token: {tokens[high_kl_idx[0]]}")
                print(f"Logprob: {logprobs[high_kl_idx[0]]}")
                print(f"New Logprob: {new_logprobs[high_kl_idx[0]]}")
            else:
                print("No tokens found with KL divergence > 1")

            raise ValueError("KL Divergence Debugging")

        # Calculate entropy for each token
        entropy = dist.entropy()  # Shape: (batch_size * sequence_length,)

        # Create mask where advantages and logprobs are not NaN
        # Shape: (batch_size * sequence_length,)
        mask = ~torch.isnan(advantages) & ~torch.isnan(logprobs)
        num_tokens = mask.sum()

        # Apply mask
        new_logprobs = new_logprobs[mask]  # Shape: (num_tokens,)
        logprobs = logprobs[mask]  # Shape: (num_tokens,)
        advantages = advantages[mask]  # Shape: (num_tokens,)
        entropy = entropy[mask]  # Shape: (num_tokens,)
        if weights is not None:
            weights = weights[mask]
        else:
            weights = torch.ones_like(
                entropy, dtype=entropy.dtype, device=entropy.device
            )

        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Calculate the probability ratio (π_θ(a|s) / π_θ_old(a|s))
        ratio = torch.exp(new_logprobs - logprobs)  # Shape: (num_valid_tokens,)

        # Calculate the surrogate losses
        surrogate1 = ratio * advantages  # Shape: (num_valid_tokens,)
        surrogate2 = (
            torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
            * advantages
        )  # Shape: (num_valid_tokens,)

        # Take the minimum of the two surrogate losses
        policy_loss = -torch.min(surrogate1, surrogate2).mul(weights).sum()  # Scalar

        # Entropy bonus
        entropy_bonus = entropy.mul(weights).sum()  # Scalar

        # Calculate KL divergence between the old and new policies
        kl_divergence = (
            torch.nn.functional.kl_div(
                new_logprobs, logprobs, reduction="none", log_target=True
            )
            .mul(weights)
            .sum()
        )

        return PPOResult(
            policy_weight=self.policy_coef * num_tokens,
            entropy_weight=self.entropy_coef * num_tokens,
            kl_weight=self.kl_coef * num_tokens,
            policy_loss=policy_loss,
            entropy_bonus=entropy_bonus,
            kl_divergence=kl_divergence,
            num_tokens=num_tokens,
        )
