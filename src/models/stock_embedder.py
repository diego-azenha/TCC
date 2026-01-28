import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class StockEmbedder(nn.Module):
    """Stock Embedder

    Inputs (per day, batch of assets):
      - S: Tensor[N, L, d_ts]  (time-series window per asset)
      - S_static: Tensor[N, d_static]  (static features per asset at time t)

    Outputs:
      - alpha: Tensor[N]
      - beta: Tensor[N, F]
      - sigma: Tensor[N] (positive)
      - nu: Tensor[N] (greater than offset, default +4)

    Implementation follows the specified pipeline:
      1) per-timestep projection d_ts -> h
      2) TransformerEncoder over time (batch_first=True)
      3) concat last temporal state with static features
      4) MLP head -> intermediate H3
      5) linear heads + softplus offsets for stability
    """

    def __init__(
        self,
        d_ts: int,
        d_static: int,
        h: int = 64,
        F: int = 8,
        nhead: int = 4,
        num_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.1,
        sigma_eps: float = 1e-6,
        nu_offset: float = 4.0,
    ):
        super().__init__()
        self.d_ts = d_ts
        self.d_static = d_static
        self.h = h
        self.F = F
        self.activation = activation
        self.sigma_eps = sigma_eps
        self.nu_offset = nu_offset

        # Step 1: per-timestep projection
        self.proj = nn.Linear(d_ts, h)

        # Step 2: Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=h, nhead=nhead, dim_feedforward=4 * h, dropout=dropout, activation=("gelu" if activation == "gelu" else "relu"), batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Step 4: MLP final (after concatenation with static features)
        self.fc1 = nn.Linear(h + d_static, h)
        self.fc2 = nn.Linear(h, h)
        self.dropout = nn.Dropout(dropout)

        # Heads
        self.alpha_head = nn.Linear(h, 1)
        self.beta_head = nn.Linear(h, F)
        self.sigma_head = nn.Linear(h, 1)
        self.nu_head = nn.Linear(h, 1)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "gelu":
            return F.gelu(x)
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "silu":
            return F.silu(x)
        return F.gelu(x)

    def forward(self, S: torch.Tensor, S_static: torch.Tensor):
        """
        Forward pass.

        Args:
            S: Tensor[N, L, d_ts]
            S_static: Tensor[N, d_static]

        Returns:
            alpha: Tensor[N]
            beta: Tensor[N, F]
            sigma: Tensor[N]
            nu: Tensor[N]
        """
        if S.dim() != 3:
            raise ValueError(f"S must be shape (N,L,d_ts), got {tuple(S.shape)}")
        if S_static.dim() not in (1, 2):
            raise ValueError(f"S_static must be shape (N,d_static) or (N,), got {tuple(S_static.shape)}")

        N, L, d_ts = S.shape
        if d_ts != self.d_ts:
            raise ValueError(f"d_ts mismatch: module d_ts={self.d_ts}, input d_ts={d_ts}")

        device = S.device
        S_static = S_static.to(device)
        if S_static.dim() == 1:
            S_static = S_static.unsqueeze(-1)
        if S_static.shape[0] != N:
            raise ValueError("Batch size mismatch between S and S_static")

        # Step 1: per-timestep projection (applied to last dim)
        M = self.proj(S)  # (N, L, h)
        M = self._act(M)

        # Step 2: sequence encoder (Transformer)
        H = self.encoder(M)  # (N, L, h)
        h_seq = H[:, -1, :]  # (N, h) -- last temporal state

        # Step 3: concat with static
        U = torch.cat([h_seq, S_static], dim=-1)  # (N, h + d_static)

        # Step 4: MLP final
        A = self._act(self.fc1(U))
        A = self.dropout(A)
        H3 = self._act(self.fc2(A))
        H3 = self.dropout(H3)

        # Step 5: heads
        alpha = self.alpha_head(H3).squeeze(-1)  # (N,)
        beta = self.beta_head(H3)  # (N, F)
        sigma = F.softplus(self.sigma_head(H3)).squeeze(-1) + self.sigma_eps  # (N,)
        nu = F.softplus(self.nu_head(H3)).squeeze(-1) + self.nu_offset  # (N,)

        return alpha, beta, sigma, nu


__all__ = ["StockEmbedder"]
