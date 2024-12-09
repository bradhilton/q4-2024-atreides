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
    unclipped_policy_weight: torch.Tensor = tensor_field()
    tanh_log_policy_weight: torch.Tensor = tensor_field()
    value_weight: torch.Tensor = tensor_field()
    entropy_weight: torch.Tensor = tensor_field()
    entropy_target_weight: torch.Tensor = tensor_field()
    kl_weight: torch.Tensor = tensor_field()
    reverse_kl_weight: torch.Tensor = tensor_field()
    weighted_entropy_weight: torch.Tensor = tensor_field()
    weighted_kl_weight: torch.Tensor = tensor_field()
    weighted_reverse_kl_weight: torch.Tensor = tensor_field()
    weighted_ce_weight: torch.Tensor = tensor_field()
    policy_loss: torch.Tensor = tensor_field()
    unclipped_policy_loss: torch.Tensor = tensor_field()
    tanh_log_policy_loss: torch.Tensor = tensor_field()
    value_loss: torch.Tensor = tensor_field()
    entropy_bonus: torch.Tensor = tensor_field()
    entropy_target: torch.Tensor = tensor_field()
    kl_divergence: torch.Tensor = tensor_field()
    reverse_kl_divergence: torch.Tensor = tensor_field()
    weighted_entropy_bonus: torch.Tensor = tensor_field()
    weighted_kl_divergence: torch.Tensor = tensor_field()
    weighted_reverse_kl_divergence: torch.Tensor = tensor_field()
    weighted_ce_loss: torch.Tensor = tensor_field()
    num_tokens: torch.Tensor = field(default_factory=lambda: torch.tensor(0))

    def named_tensors(self) -> Iterable[tuple[str, torch.Tensor]]:
        for field in fields(self):
            yield field.name, getattr(self, field.name)

    def per_token(self) -> "PPOResult":
        return PPOResult(
            **{name: tensor / self.num_tokens for name, tensor in self.named_tensors()}
        )

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
    def entropy_target_loss(self) -> torch.Tensor:
        return torch.abs(self.entropy_bonus - self.entropy_target)

    @property
    def total_loss(self) -> torch.Tensor:
        return (
            (self.policy_weight / self.num_tokens) * self.policy_loss
            + (self.unclipped_policy_weight / self.num_tokens)
            * self.unclipped_policy_loss
            + (self.tanh_log_policy_weight / self.num_tokens)
            * self.tanh_log_policy_loss
            + (self.value_weight / self.num_tokens) * self.value_loss
            - (self.entropy_weight / self.num_tokens) * self.entropy_bonus
            + (self.entropy_target_weight / self.num_tokens) * self.entropy_target_loss
            + (self.kl_weight / self.num_tokens) * self.kl_divergence
            + (self.reverse_kl_weight / self.num_tokens) * self.reverse_kl_divergence
            - (self.weighted_entropy_weight / self.num_tokens)
            * self.weighted_entropy_bonus
            + (self.weighted_kl_weight / self.num_tokens) * self.weighted_kl_divergence
            + (self.weighted_reverse_kl_weight / self.num_tokens)
            * self.weighted_reverse_kl_divergence
            + (self.weighted_ce_weight / self.num_tokens) * self.weighted_ce_loss
        )


class PPOLoss(nn.Module):
    def __init__(
        self,
        policy_coef: float = 1.0,
        unclipped_policy_coef: float = 0.0,
        tanh_log_policy_coef: float = 0.0,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.0,
        entropy_coef: float = 0.01,
        entropy_target: float = 0.5,
        entropy_target_coef: float = 0.0,
        kl_coef: float = 0.0,
        reverse_kl_coef: float = 0.0,
        weighted_entropy_coef: float = 0.0,
        weighted_kl_coef: float = 0.0,
        weighted_reverse_kl_coef: float = 0.0,
        weighted_ce_coef: float = 0.0,
        normalize_values: bool = True,
        normalize_value_predictions: bool = True,
        normalize_advantages: bool = True,
    ):
        """
        Initializes the PPO Loss module.

        Args:
            policy_coef (float): Coefficient for the clipped policy loss. Defaults to 1.0.
            unclipped_policy_coef (float): Coefficient for the unclipped policy loss. Defaults to 0.0.
            tanh_log_policy_coef (float): Coefficient for the tanh log policy loss. Defaults to 0.0.
            clip_epsilon (float): Clipping parameter for PPO (typically between 0.1 and 0.3).
            value_coef (float): Coefficient for the value loss (defaults to 0.0).
            entropy_coef (float): Coefficient for the entropy bonus to encourage exploration.
            entropy_target (float): Target entropy (defaults to 0.5).
            entropy_target_coef (float): Coefficient for the entropy target loss.
            kl_coef (float): Coefficient for KL divergence penalty (defaults to 0.0).
            reverse_kl_coef (float): Coefficient for reverse KL divergence penalty (defaults to 0.0).
            weighted_entropy_coef (float): Coefficient for the weighted entropy bonus.
            weighted_kl_coef (float): Coefficient for the weighted KL divergence penalty.
            weighted_reverse_kl_coef (float): Coefficient for the weighted reverse KL divergence penalty.
            weighted_ce_coef (float): Coefficient for the weighted cross entropy loss.
            normalize_values (bool): Whether to normalize values before computing the loss (default True).
            normalize_value_predictions (bool): Whether to normalize value predictions before computing the loss (default True).
            normalize_advantages (bool): Whether to normalize advantages before computing the loss (default True).
        """
        super().__init__()
        self.policy_coef = policy_coef
        self.unclipped_policy_coef = unclipped_policy_coef
        self.tanh_log_policy_coef = tanh_log_policy_coef
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.entropy_target = entropy_target
        self.entropy_target_coef = entropy_target_coef
        self.kl_coef = kl_coef
        self.reverse_kl_coef = reverse_kl_coef
        self.weighted_entropy_coef = weighted_entropy_coef
        self.weighted_kl_coef = weighted_kl_coef
        self.weighted_reverse_kl_coef = weighted_reverse_kl_coef
        self.weighted_ce_coef = weighted_ce_coef
        self.normalize_values = normalize_values
        self.normalize_value_predictions = normalize_value_predictions
        self.normalize_advantages = normalize_advantages

    def forward(
        self,
        *,
        logits: Union[torch.Tensor, list[torch.Tensor]],
        tokens: torch.Tensor,
        values: torch.Tensor,
        advantages: torch.Tensor,
        logprobs: torch.Tensor,
        reference_logprobs: torch.Tensor,
        weights: torch.Tensor,
        bos_id: int,
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

            values (Tensor):
                Shape: (batch_size, sequence_length)
                Value estimates for each token.

            advantages (Tensor):
                Shape: (batch_size, sequence_length)
                Advantage estimates for each token.

            logprobs (Tensor):
                Shape: (batch_size, sequence_length)
                Log probabilities of the sampled tokens under the old policy.

            reference_logprobs (Tensor):
                Shape: (batch_size, sequence_length)
                Log probabilities of the sampled tokens under the reference policy.

            weights (Tensor):
                Shape: (batch_size, sequence_length)
                Weights for each token in the sequence.

            bos_id (int):
                Index of the beginning of sequence token in the vocabulary.

        Returns:
            PPOResult: The combined loss results across all chunks.
        """
        if isinstance(logits, list):
            result = PPOResult().to(logits[0].device)
            num_chunks = len(logits)
            for chunked_args in zip(
                logits,
                tokens.chunk(num_chunks, dim=1),
                values.chunk(num_chunks, dim=1),
                advantages.chunk(num_chunks, dim=1),
                logprobs.chunk(num_chunks, dim=1),
                reference_logprobs.chunk(num_chunks, dim=1),
                weights.chunk(num_chunks, dim=1),
            ):
                result += self._forward_chunk(*chunked_args, bos_id=bos_id)
            return result

        return self._forward_chunk(
            logits,
            tokens,
            values,
            advantages,
            logprobs,
            reference_logprobs,
            weights,
            bos_id=bos_id,
        )

    def _forward_chunk(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        values: torch.Tensor,
        advantages: torch.Tensor,
        logprobs: torch.Tensor,
        reference_logprobs: torch.Tensor,
        weights: torch.Tensor,
        bos_id: int,
    ) -> PPOResult:
        """
        Processes a single chunk of the PPO loss computation.
        """
        # Flatten logits tensor to shape (batch_size * sequence_length, vocab_size)
        logits = logits.view(-1, logits.size(-1))
        # Shape: (batch_size * sequence_length,)
        tokens = shift(tokens, ignore_label=bos_id).view(-1)
        values = shift(values).view(-1)
        advantages = shift(advantages).view(-1)
        logprobs = shift(logprobs).view(-1)
        reference_logprobs = shift(reference_logprobs).view(-1)
        weights = shift(weights).view(-1)

        # Create a Categorical distribution from logits
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
        entropy = dist.entropy()

        # Create mask where advantages and logprobs are not NaN
        mask = ~torch.isnan(advantages) & ~torch.isnan(logprobs)
        num_tokens = mask.sum()

        # Apply mask to all tensors
        # Shape: (num_tokens,)
        new_logprobs = new_logprobs[mask]
        logprobs = logprobs[mask]
        values = values[mask]
        advantages = advantages[mask]
        entropy = entropy[mask]
        reference_logprobs = reference_logprobs[mask]
        weights = weights[mask]

        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Calculate the probability ratio (π_θ(a|s) / π_θ_old(a|s))
        log_ratio = new_logprobs - logprobs  # Shape: (num_tokens,)
        ratio = torch.exp(log_ratio)  # Shape: (num_tokens,)

        # Calculate the surrogate losses
        surrogate1 = ratio * advantages  # Shape: (num_tokens,)
        surrogate2 = (
            torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
            * advantages
        )  # Shape: (num_tokens,)

        # Take the minimum of the two surrogate losses for clipped version
        policy_loss = -torch.min(surrogate1, surrogate2).mul(weights).sum()  # Scalar

        # Calculate unclipped policy loss
        unclipped_policy_loss = -surrogate1.mul(weights).sum()  # Scalar

        # Calculate tanh log policy loss
        tanh_log_policy_loss = (
            -torch.tanh(log_ratio).mul(advantages).mul(weights).sum()
        )  # Scalar

        # Calculate the value loss
        # Shape: (num_tokens,)
        value_preds = logits.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)[mask]
        if self.normalize_values:
            values = (values - values.mean()) / (values.std() + 1e-8)
        if self.normalize_value_predictions:
            value_preds = (value_preds - value_preds.mean()) / (
                value_preds.std() + 1e-8
            )
        value_loss = (value_preds - values).pow(2).mul(weights).sum()

        # Entropy bonus
        entropy_bonus = entropy.mul(weights)
        weighted_entropy_bonus = entropy_bonus.mul(-advantages).sum()  # Scalar
        entropy_bonus = entropy_bonus.sum()  # Scalar

        # Calculate KL divergence between the old and new policies
        kl_divergence = torch.nn.functional.kl_div(
            new_logprobs,
            reference_logprobs if reference_logprobs is not None else logprobs,
            reduction="none",
            log_target=True,
        ).mul(
            weights
        )  # Shape: (num_tokens,)

        weighted_kl_divergence = kl_divergence.mul(-advantages).sum()  # Scalar
        kl_divergence = kl_divergence.sum()  # Scalar

        # Calculate reverse KL divergence between the old and new policies
        reverse_kl_divergence = torch.nn.functional.kl_div(
            reference_logprobs if reference_logprobs is not None else logprobs,
            new_logprobs,
            reduction="none",
            log_target=True,
        ).mul(
            weights
        )  # Shape: (num_tokens,)

        weighted_reverse_kl_divergence = reverse_kl_divergence.mul(
            -advantages
        ).sum()  # Scalar
        reverse_kl_divergence = reverse_kl_divergence.sum()  # Scalar

        # Calculate cross entropy loss using log probabilities, weighted by advantages
        weighted_ce_loss = -new_logprobs.mul(advantages).mul(weights).sum()  # Scalar

        return PPOResult(
            policy_weight=self.policy_coef * num_tokens,
            unclipped_policy_weight=self.unclipped_policy_coef * num_tokens,
            tanh_log_policy_weight=self.tanh_log_policy_coef * num_tokens,
            value_weight=self.value_coef * num_tokens,
            entropy_weight=self.entropy_coef * num_tokens,
            entropy_target_weight=self.entropy_target_coef * num_tokens,
            kl_weight=self.kl_coef * num_tokens,
            reverse_kl_weight=self.reverse_kl_coef * num_tokens,
            weighted_entropy_weight=self.weighted_entropy_coef * num_tokens,
            weighted_kl_weight=self.weighted_kl_coef * num_tokens,
            weighted_reverse_kl_weight=self.weighted_reverse_kl_coef * num_tokens,
            weighted_ce_weight=self.weighted_ce_coef * num_tokens,
            policy_loss=policy_loss,
            unclipped_policy_loss=unclipped_policy_loss,
            tanh_log_policy_loss=tanh_log_policy_loss,
            value_loss=value_loss,
            entropy_bonus=entropy_bonus,
            entropy_target=self.entropy_target * num_tokens,
            kl_divergence=kl_divergence,
            reverse_kl_divergence=reverse_kl_divergence,
            weighted_entropy_bonus=weighted_entropy_bonus,
            weighted_kl_divergence=weighted_kl_divergence,
            weighted_reverse_kl_divergence=weighted_reverse_kl_divergence,
            weighted_ce_loss=weighted_ce_loss,
            num_tokens=num_tokens,
        )
