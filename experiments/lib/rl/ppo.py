import torch
import torch.nn as nn
from typing import Optional, Union


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


class PPOLoss(nn.Module):
    def __init__(
        self,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        advantage_normalization_alpha: Optional[float] = 1.0,
        advantage_normalization_deviation: DeviationType = "std",
    ):
        """
        Initializes the PPO Loss module.

        Args:
            clip_epsilon (float): Clipping parameter for PPO (typically between 0.1 and 0.3).
            entropy_coef (float): Coefficient for the entropy bonus to encourage exploration.
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
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.advantage_normalizer = (
            RunningNormalizer(
                advantage_normalization_alpha, advantage_normalization_deviation
            )
            if advantage_normalization_alpha is not None
            else None
        )

    def forward(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        advantages: torch.Tensor,
        logprobs: torch.Tensor,
        bos_id: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the PPO loss for sequence data.

        Args:
            logits (Tensor):
                Shape: (batch_size, sequence_length, vocab_size)
                Logits output by the new policy for next token preferences.

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
            total_loss (Tensor):
                Scalar tensor representing the combined PPO loss.

            num_tokens (Tensor):
                Scalar tensor representing the number of valid tokens used to compute the loss.
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

        # Total loss
        total_loss = policy_loss - self.entropy_coef * entropy_bonus  # Scalar

        return total_loss, num_tokens
