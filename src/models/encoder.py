"""DeepSet encoder for the variational posterior q(z|r).

Replaces the analytical (Cholesky + FP64 + jitter) encoder with a learned
DeepSet architecture that handles variable N across days naturally.

Architecture:
  Per-stock φ:  (r_i, α_i, β_i[F], σ_i, r_i−α_i) → h_i  ∈ ℝ^{phi_hidden}
  Aggregation:  masked mean(h_1 ... h_N) → h_agg
  Post-agg ρ:   h_agg → hidden                             ∈ ℝ^{rho_hidden}
  Heads:        μ_q[F], log_σ_q[F]

Initialization ensures σ_q ≈ 1.0 at the start → KL(q||N(0,I)) ≈ 0.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


class DeepSetEncoder(nn.Module):
    """Variational posterior q(z|r) implemented as a DeepSet network."""

    def __init__(self, num_factors: int, phi_hidden: int = 64, rho_hidden: int = 128):
        super().__init__()
        self.num_factors = num_factors

        # Per-stock network φ: input dim = 4 + F  (r_i, α_i, σ_i, residual_i, β_i[F])
        in_dim = 4 + num_factors
        self.phi = nn.Sequential(
            nn.Linear(in_dim, phi_hidden),
            nn.GELU(),
            nn.Linear(phi_hidden, phi_hidden),
            nn.GELU(),
        )

        # Post-aggregation network ρ
        self.rho = nn.Sequential(
            nn.Linear(phi_hidden, rho_hidden),
            nn.GELU(),
            nn.Linear(rho_hidden, rho_hidden),
            nn.GELU(),
        )

        # Output heads
        self.mu_head = nn.Linear(rho_hidden, num_factors)
        self.log_sigma_head = nn.Linear(rho_hidden, num_factors)

        # Small weight init on log_sigma_head → log_σ_q ≈ 0 → σ_q ≈ 1 at init
        nn.init.normal_(self.log_sigma_head.weight, std=0.01)
        nn.init.zeros_(self.log_sigma_head.bias)

    def forward(
        self,
        r: torch.Tensor,
        alpha: torch.Tensor,
        B: torch.Tensor,
        sigma: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode per-day cross-section into posterior parameters.

        Args:
            r:     [batch, N] observed returns
            alpha: [batch, N] intercepts from StockEmbedder
            B:     [batch, N, F] factor loadings
            sigma: [batch, N] idiosyncratic vol
            mask:  [batch, N] bool — True for valid stocks

        Returns:
            mu_q:       [batch, F] posterior mean
            log_sigma_q:[batch, F] log posterior std  (σ_q = exp(log_σ_q))
        """
        batch, N, F = B.shape

        residual = r - alpha  # [batch, N]  what the factors need to explain

        # Build per-stock input: (r_i, α_i, σ_i, residual_i, β_i[F])
        phi_in = torch.cat([
            r.unsqueeze(-1),       # [batch, N, 1]
            alpha.unsqueeze(-1),   # [batch, N, 1]
            sigma.unsqueeze(-1),   # [batch, N, 1]
            residual.unsqueeze(-1),# [batch, N, 1]
            B,                     # [batch, N, F]
        ], dim=-1)  # [batch, N, 4+F]

        # Per-stock embeddings
        h = self.phi(phi_in)  # [batch, N, phi_hidden]

        # Masked mean pooling
        if mask is not None:
            valid = mask.float().unsqueeze(-1)  # [batch, N, 1]
            h = h * valid
            count = valid.sum(dim=1).clamp(min=1.0)  # [batch, 1]
            h_agg = h.sum(dim=1) / count  # [batch, phi_hidden]
        else:
            h_agg = h.mean(dim=1)  # [batch, phi_hidden]

        # Post-aggregation
        h_rho = self.rho(h_agg)  # [batch, rho_hidden]

        mu_q = self.mu_head(h_rho)            # [batch, F]
        log_sigma_q = self.log_sigma_head(h_rho)  # [batch, F]

        return mu_q, log_sigma_q

    def sample(
        self,
        mu_q: torch.Tensor,
        log_sigma_q: torch.Tensor,
    ) -> torch.Tensor:
        """Reparameterization trick: z = μ_q + σ_q ⊙ ε, ε ~ N(0, I).

        Args:
            mu_q:        [batch, F]
            log_sigma_q: [batch, F]

        Returns:
            z: [batch, 1, F]  (K=1 sample, unsqueezed for decoder compatibility)
        """
        sigma_q = torch.exp(log_sigma_q)
        eps = torch.randn_like(sigma_q)
        z = mu_q + sigma_q * eps
        return z.unsqueeze(1)  # [batch, 1, F]



__all__ = ["DeepSetEncoder"]

