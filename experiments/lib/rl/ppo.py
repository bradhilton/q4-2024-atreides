from dataclasses import dataclass, field, fields
import torch
import torch.nn as nn
from typing import Iterable, Optional, Union


from .running_normalizer import DeviationType, RunningNormalizer


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
    key = (labels.shape[-1:], ignore_label, labels.dtype, labels.device)
    if key not in ignore_labels_cache:
        ignore_labels_cache[key] = torch.full(
            (labels.shape[0], 1), ignore_label, dtype=labels.dtype, device=labels.device
        )
    # Shift labels to compute loss
    # equivalent to doing labels[..., 1:] and logits[..., :-1, :]
    # But this way we dont need to slice the logits. We just add an ignore index to labels.
    return torch.cat((labels[..., 1:], ignore_labels_cache[key]), dim=1)


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

    def tensors(self) -> Iterable[torch.Tensor]:
        for field in fields(self):
            yield getattr(self, field.name)

    def to(self, target: Union[torch.device, torch.dtype]) -> "PPOResult":
        for tensor in self.tensors():
            tensor.to(target)
        return self

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
        advantage_normalization_alpha: Optional[float] = 1.0,
        advantage_normalization_deviation: DeviationType = "std",
    ):
        """
        Initializes the PPO Loss module.

        Args:
            policy_coef (float): Coefficient for the policy loss. Defaults to 1.0.
            clip_epsilon (float): Clipping parameter for PPO (typically between 0.1 and 0.3).
            entropy_coef (float): Coefficient for the entropy bonus to encourage exploration.
            kl_coef (float): Coefficient for KL divergence penalty (defaults to 0.0).
            advantage_normalization_alpha (Optional[float] = 1.0):
                Alpha parameter [0, 1] for the running mean and deviation used to normalize
                advantages (if None, no normalization is applied):
                - alpha == 0.0: Expanding window/equal-weight normalization.
                - 0.0 < alpha < 1.0: Exponentially weighted normalization.
                - alpha == 1.0: Batch normalization.
                - alpha is None: No normalization.
                Defaults to 1.0 (batch normalization).
            advantage_normalization_deviation (DeviationType = "std"):
                Type of deviation to use for advantage normalization (defaults to "std"):
                - "std": Standard deviation.
                - "mad": Mean absolute deviation.
        """
        super(PPOLoss, self).__init__()
        self.policy_coef = policy_coef
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.kl_coef = kl_coef
        self.advantage_normalizer = (
            RunningNormalizer(
                advantage_normalization_alpha, advantage_normalization_deviation
            )
            if advantage_normalization_alpha is not None
            else None
        )

    def forward(
        self,
        logits: Union[torch.Tensor, list[torch.Tensor]],
        tokens: torch.Tensor,
        advantages: torch.Tensor,
        logprobs: torch.Tensor,
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

            bos_id (Optional[int] = None):
                Index of the beginning of sequence token in the vocabulary. If None, defaults
                to the first token in `tokens`.

        Returns:
            PPOResult: The combined loss results across all chunks.
        """
        if isinstance(logits, list):
            result = PPOResult()
            num_chunks = len(logits)
            for logit_chunk, token_chunk, adv_chunk, logprob_chunk in zip(
                logits,
                tokens.chunk(num_chunks, dim=1),
                advantages.chunk(num_chunks, dim=1),
                logprobs.chunk(num_chunks, dim=1),
            ):
                result += self._forward_chunk(
                    logit_chunk, token_chunk, adv_chunk, logprob_chunk, bos_id
                )
            return result

        return self._forward_chunk(logits, tokens, advantages, logprobs, bos_id)

    def _forward_chunk(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        advantages: torch.Tensor,
        logprobs: torch.Tensor,
        bos_id: Optional[int],
    ) -> PPOResult:
        """
        Processes a single chunk of the PPO loss computation.
        """
        if bos_id is None:
            bos_id = int(tokens.view(-1)[0].item())
        # Flatten logits tensor to shape (batch_size * sequence_length, vocab_size)
        logits = logits.view(-1, logits.size(-1))
        # Shape: (batch_size * sequence_length,)
        tokens = shift(tokens, ignore_label=bos_id).view(-1)
        # Shape: (batch_size * sequence_length,)
        advantages = shift(advantages).view(-1)
        logprobs = shift(logprobs).view(-1)  # Shape: (batch_size * sequence_length,)

        # Create a Categorical distribution from logits
        # Shape: (batch_size * sequence_length,)
        dist = torch.distributions.Categorical(logits=logits)

        # Calculate new log probabilities of the taken actions
        new_logprobs = dist.log_prob(tokens)  # Shape: (batch_size * sequence_length,)

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

        # Normalize advantages
        if self.advantage_normalizer is not None:
            advantages = self.advantage_normalizer.normalize(advantages)

        # Calculate the probability ratio (π_θ(a|s) / π_θ_old(a|s))
        ratio = torch.exp(new_logprobs - logprobs)  # Shape: (num_valid_tokens,)

        # Calculate the surrogate losses
        surrogate1 = ratio * advantages  # Shape: (num_valid_tokens,)
        surrogate2 = (
            torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
            * advantages
        )  # Shape: (num_valid_tokens,)

        # Take the minimum of the two surrogate losses
        policy_loss = -torch.min(surrogate1, surrogate2).sum()  # Scalar

        # Entropy bonus (to encourage exploration)
        entropy_bonus = entropy.sum()  # Scalar

        # Debugging
        if False:
            new_probs = torch.exp(new_logprobs)
            old_probs = torch.exp(logprobs)
            print(
                torch.corrcoef(torch.stack([new_probs, old_probs], dim=1).T)[
                    0, 1
                ].item()
            )

        # Calculate KL divergence between the old and new policies
        kl_divergence = torch.nn.functional.kl_div(
            new_logprobs, logprobs, reduction="sum", log_target=True
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
