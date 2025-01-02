import torch.nn as nn
from torch import Tensor
from typing import Optional


class MLPHead(nn.Module):
    """
    MLP head for PPO that estimates the value or advantage for each position in the sequence.

    The head takes the hidden states from the transformer and projects them to scalar values
    for each token position, allowing for per-token value/advantage estimation in PPO training.

    Args:
        hidden_size: Dimension of the transformer's hidden states
        intermediate_size: Size of the intermediate layer (if used)
        dropout_rate: Dropout probability
        use_intermediate_layer: Whether to use an intermediate layer before final projection
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        dropout_rate: float = 0.1,
        use_intermediate_layer: bool = True,
    ) -> None:
        super().__init__()

        # Build the MLP head layers
        if use_intermediate_layer:
            if intermediate_size is None:
                intermediate_size = hidden_size // 4

            self.head = nn.Sequential(
                nn.Linear(hidden_size, intermediate_size),
                nn.Tanh(),  # Tanh tends to work better than ReLU for value/advantage estimation
                nn.Dropout(dropout_rate),
                nn.Linear(intermediate_size, 1),
            )
        else:
            self.head = nn.Linear(hidden_size, 1)

    def forward(
        self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute value/advantage predictions for each position in the sequence.

        Args:
            hidden_states: Transformer hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask [batch_size, seq_len] where 1 indicates valid tokens
                          and 0 indicates masked tokens. Can be irregular.

        Returns:
            predictions: Predictions [batch_size, seq_len] with values/advantages for each token position
        """
        # Project each position's hidden state to a value
        predictions = self.head(hidden_states).squeeze(-1)  # [batch_size, seq_len]

        # Mask out invalid positions if mask provided
        if attention_mask is not None:
            predictions = predictions * attention_mask

        return predictions
