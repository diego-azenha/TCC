import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
from ..utils.config import ModelConfig


class StockEmbedder(nn.Module):
    """Outputs stock-specific parameters (alpha, beta, sigma, nu) from features.
    
    Input: S[N,L,d_ts] (time-series), S_static[N,d_static] (static features)
    Output: alpha[N], beta[N,F], sigma[N], nu[N]
    
    Note: N stocks from ONE day (not batched over days). Paper defaults: F=64, h=256, L=256.
    """

    def __init__(
        self,
        d_ts: int = None,
        d_static: int = None,
        h: int = 256,  # Paper uses 256 (Section 3.5)
        F: int = 64,  # Paper uses 64 factors (Table 3)
        nhead: int = 4,
        num_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.25,  # Paper uses 0.25 (Section 3.5)
        sigma_eps: float = 1e-6,
        nu_offset: float = 4.0,
        lookback: int = 256,  # Paper uses 256 (Table 3, Section 5.1.4)
        config: Optional[ModelConfig] = None,
    ):
        """Args: d_ts, d_static (required); h=256, F=64, dropout=0.25, lookback=256; config overrides all."""
        super().__init__()
        
        # If config provided, use its values
        if config is not None:
            d_ts = config.d_ts
            d_static = config.d_static
            h = config.hidden_size
            F = config.num_factors
            nhead = config.nhead
            num_layers = config.num_layers
            activation = config.activation
            dropout = config.dropout
            sigma_eps = config.sigma_eps
            nu_offset = config.nu_offset
            lookback = config.lookback
        
        # Validate required parameters
        if d_ts is None:
            raise ValueError("d_ts must be specified (either directly or via config)")
        if d_static is None:
            raise ValueError("d_static must be specified (either directly or via config)")
        
        self.d_ts = d_ts
        self.d_static = d_static
        self.h = h
        self.F = F
        self.activation = activation
        self.sigma_eps = sigma_eps
        self.nu_offset = nu_offset
        self.lookback = lookback  # Store for validation

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
        """Apply activation function."""
        if self.activation == "gelu":
            return F.gelu(x)
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "silu":
            return F.silu(x)
        raise ValueError(f"Unknown activation function: {self.activation}. Must be 'gelu', 'relu', or 'silu'.")

    def forward(self, S: torch.Tensor, S_static: torch.Tensor):
        """S[N,L,d_ts], S_static[N,d_static] -> alpha[N], beta[N,F], sigma[N], nu[N]"""
        if S.dim() != 3:
            raise ValueError(f"S must be (N,L,d_ts), got {tuple(S.shape)}")
        
        N, L, d_ts = S.shape
        
        if d_ts != self.d_ts:
            raise ValueError(f"d_ts mismatch: expected {self.d_ts}, got {d_ts}")
        
        if L != self.lookback:
            raise ValueError(f"Lookback mismatch: expected {self.lookback}, got {L}")
        
        device = S.device
        
        if S_static.dim() == 1:
            S_static = S_static.unsqueeze(-1)
        elif S_static.dim() == 2:
            if S_static.shape[-1] != self.d_static:
                raise ValueError(f"d_static mismatch: expected {self.d_static}, got {S_static.shape[-1]}")
        else:
            raise ValueError(f"S_static must be (N,d_static) or (N,), got {tuple(S_static.shape)}")
        
        S_static = S_static.to(device)
        
        if S_static.shape[0] != N:
            raise ValueError(f"Stock count mismatch: S has {N}, S_static has {S_static.shape[0]}")

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
