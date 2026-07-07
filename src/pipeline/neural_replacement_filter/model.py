# mypy: ignore-errors
"""PyTorch model for replacement-level accept/reject scoring."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only when torch is absent
    raise RuntimeError(
        "PyTorch is required for the neural replacement filter. "
        "Install project dependencies or run: uv sync"
    ) from exc


class NeuralReplacementFilter(nn.Module):
    """Small feed-forward binary classifier with sigmoid output."""

    def __init__(self, input_size: int, hidden_size: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)
