"""Core LSTM model for next-operation classification and time regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch  # type: ignore[import-not-found]
from torch import Tensor, nn  # type: ignore[attr-defined]


@dataclass
class LSTMState:
    """Container for hidden and cell states of an LSTM."""

    hidden: Tensor
    cell: Tensor


class LSTMEventPredictor(nn.Module):
    """Predict next operation distribution and delta-time using an LSTM backbone.

    Args:
        input_dim: Feature dimension of each time step (embedding + z value).
        hidden_dim: Hidden state size of the LSTM layers.
        num_op_categories: Number of operation categories for classification output.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability applied between LSTM layers.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_op_categories: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if num_op_categories <= 0:
            raise ValueError("num_op_categories must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_op_categories = num_op_categories
        self.num_layers = num_layers

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.op_head = nn.Linear(hidden_dim, num_op_categories)
        self.time_head = nn.Linear(hidden_dim, 1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize model weights using Xavier initialization for stability."""

        for name, param in self.named_parameters():
            if "weight" in name:
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif "bias" in name:
                nn.init.zeros_(param)

    def init_hidden(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Return zero-initialized hidden and cell states for a batch."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        hidden = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_dim,
            device=device,
            dtype=dtype or torch.float32,
        )
        cell = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_dim,
            device=device,
            dtype=dtype or torch.float32,
        )
        return hidden, cell

    def forward(
        self,
        inputs: Tensor,
        state: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Run a forward pass through the network.

        Args:
            inputs: Tensor of shape ``(batch, seq_len, input_dim)``.
            state: Optional tuple ``(hidden, cell)`` with shapes
                ``(num_layers, batch, hidden_dim)``.

        Returns:
            Tuple of tensors ``(op_probs, z_hat)`` where ``op_probs`` has shape
            ``(batch, seq_len, num_op_categories)`` and provides a probability
            distribution over the next operation categories (summing to 1 over
            the last dimension), while ``z_hat`` has shape ``(batch, seq_len)``.
        """

        if inputs.dim() != 3:
            raise ValueError(
                "inputs must be a 3D tensor of shape (batch, seq_len, input_dim)."
            )
        batch_size, _, feature_dim = inputs.shape
        if feature_dim != self.input_dim:
            raise ValueError("inputs feature dimension does not match model input_dim.")

        if state is None:
            h_0, c_0 = self.init_hidden(
                batch_size, device=inputs.device, dtype=inputs.dtype
            )
        else:
            h_0, c_0 = state

        outputs, _ = self.lstm(inputs, (h_0, c_0))
        op_logits = self.op_head(outputs)
        op_probs = torch.softmax(op_logits, dim=-1)
        z_hat = self.time_head(outputs).squeeze(-1)
        return op_probs, z_hat


__all__ = ["LSTMEventPredictor", "LSTMState"]
