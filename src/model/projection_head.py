"""
Projection Head for FutureL1 (similar to RoT CoTCompressor).
Maps LLM hidden states to latent/vision embedding space for latent token prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SwiGLU(nn.Module):
    """
    SwiGLU: gate(x) * value(x) -> Linear
    SwiGLU(x) = (SiLU(W1 @ x) ⊙ (W2 @ x)) @ W3
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=False)  # gate projection
        self.w2 = nn.Linear(input_dim, hidden_dim, bias=False)  # value projection
        self.w3 = nn.Linear(hidden_dim, input_dim, bias=False)  # output projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., input_dim)
        gate = F.silu(self.w1(x))      # (..., hidden_dim)
        value = self.w2(x)             # (..., hidden_dim)
        hidden = gate * value          # element-wise product
        return self.w3(hidden)          # (..., input_dim)


class ProjectionHead(nn.Module):
    """
    Projection head that maps LLM hidden states to latent embedding space.
    Structure: LayerNorm -> Linear(up) -> SwiGLU -> Linear(down)
    Same as RoT CoTCompressor projection_head.
    """
    def __init__(
        self,
        hidden_dim: int,
        projection_hidden_dim: int = 2048,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection_hidden_dim = projection_hidden_dim
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=eps),
            nn.Linear(hidden_dim, projection_hidden_dim),
            SwiGLU(input_dim=projection_hidden_dim, hidden_dim=projection_hidden_dim),
            nn.Linear(projection_hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim) or (batch, hidden_dim)
        Returns:
            Projected tensor with same shape as x
        """
        return self.projection(x)


class LVRHead(nn.Module):
    """
    Simpler MLP projection head (from lvr).
    LayerNorm -> Linear -> GELU -> Linear, no up-projection.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.ln_q = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.ln_q(x))
