"""NeuralFactors VAE with DeepSet encoder, Gaussian decoder, and ELBO loss.

Architecture:
  StockEmbedder → (alpha, B, sigma)
  DeepSetEncoder → (mu_q, log_sigma_q)
  z = mu_q + sigma_q ⊙ ε,  ε ~ N(0,I)  [K=1 sample]
  p(r_i|z) = N(alpha_i + beta_i' z, sigma_i^2)
  ELBO = E[log p(r|z)] - KL(q || N(0,I)) + free_bits_penalty
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

from .stock_embedder import StockEmbedder
from .encoder import DeepSetEncoder
from . import decoder as dec
from ..utils.config import ModelConfig


class NeuralFactors(nn.Module):
    """NeuralFactors VAE: DeepSet encoder + Gaussian decoder + ELBO.

    Training:  compute_elbo_loss(S, S_static, r, mask)
    Inference: predict(S, S_static)
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config

        self.embedder = StockEmbedder(config=config)
        self.encoder = DeepSetEncoder(
            num_factors=config.num_factors,
            phi_hidden=config.encoder_config.phi_hidden,
            rho_hidden=config.encoder_config.rho_hidden,
        )

    # ── Encode ────────────────────────────────────────────────────────────────

    def encode(
        self,
        S: torch.Tensor,
        S_static: torch.Tensor,
        r: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run embedder + DeepSet encoder for one trading day.

        Args:
            S:        [batch, N, L, d_ts]
            S_static: [batch, N, d_static]
            r:        [batch, N]
            mask:     [batch, N] bool

        Returns:
            alpha:       [batch, N]
            B:           [batch, N, F]
            sigma:       [batch, N]
            mu_q:        [batch, F]
            log_sigma_q: [batch, F]
        """
        batch = S.shape[0]
        if batch != 1:
            raise ValueError(f"encode expects batch_size=1, got {batch}")

        S_nb = S.squeeze(0)        # [N, L, d_ts]
        S_st_nb = S_static.squeeze(0)  # [N, d_static]
        r_nb = r.squeeze(0)        # [N]
        mask_nb = mask.squeeze(0) if mask is not None else None  # [N]

        # Embedder
        alpha, B, sigma = self.embedder(S_nb, S_st_nb)  # [N], [N,F], [N]

        # Add batch dim for encoder
        alpha_b = alpha.unsqueeze(0)   # [1, N]
        B_b = B.unsqueeze(0)           # [1, N, F]
        sigma_b = sigma.unsqueeze(0)   # [1, N]
        r_b = r_nb.unsqueeze(0)        # [1, N]
        mask_b = mask_nb.unsqueeze(0) if mask_nb is not None else None  # [1, N]

        mu_q, log_sigma_q = self.encoder(r_b, alpha_b, B_b, sigma_b, mask_b)  # [1,F], [1,F]

        return alpha_b, B_b, sigma_b, mu_q, log_sigma_q

    # ── ELBO Loss ─────────────────────────────────────────────────────────────

    def compute_elbo_loss(
        self,
        S: torch.Tensor,
        S_static: torch.Tensor,
        r: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        free_bits_lambda: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute ELBO loss for one trading day.

        ELBO = E_q[log p(r|z)] - KL(q(z|r) || N(0,I))
        With optional free-bits floor: penalty = Σ_f max(0, λ - KL_f)

        Args:
            S:                [batch, N, L, d_ts]
            S_static:         [batch, N, d_static]
            r:                [batch, N]
            mask:             [batch, N] bool
            free_bits_lambda: λ (nats/factor); 0 disables the floor

        Returns dict with:
            loss, log_likelihood, kl_divergence, kl_per_factor,
            free_bits_penalty, alpha, sigma
        """
        alpha, B, sigma, mu_q, log_sigma_q = self.encode(S, S_static, r, mask)

        batch, N, F = B.shape

        # ── Sample z (K=1, reparameterization) ───────────────────────────────
        z = self.encoder.sample(mu_q, log_sigma_q)  # [batch, 1, F]

        # ── Reconstruction log p(r|z) ────────────────────────────────────────
        r_exp = r.unsqueeze(1).expand(-1, 1, -1)        # [batch, 1, N]
        mask_exp = mask.unsqueeze(1).expand(-1, 1, -1) if mask is not None else None

        log_p_r_z = dec.log_pdf_r_given_z(
            alpha=alpha.unsqueeze(1).expand(-1, 1, -1),  # [batch, 1, N]
            B=B.unsqueeze(1).expand(-1, 1, -1, -1),      # [batch, 1, N, F]
            sigma=sigma.unsqueeze(1).expand(-1, 1, -1),  # [batch, 1, N]
            z=z,                                          # [batch, 1, F]
            r=r_exp,
            mask=mask_exp,
        )  # [batch, 1]
        log_p_r_z = log_p_r_z.squeeze(1)  # [batch]

        # ── KL(q || N(0,I)) — exact closed form ──────────────────────────────
        # KL = 0.5 * Σ_f (σ²_f + μ²_f - 1 - log σ²_f)
        sigma_q = torch.exp(log_sigma_q)  # [batch, F]
        kl_per_factor = 0.5 * (
            sigma_q ** 2 + mu_q ** 2 - 1.0 - 2.0 * log_sigma_q
        )  # [batch, F]
        kl = kl_per_factor.sum(dim=-1)  # [batch]

        # ── Free bits floor ───────────────────────────────────────────────────
        free_bits_penalty = torch.zeros(1, device=mu_q.device)
        if free_bits_lambda > 0.0:
            free_bits_penalty = torch.clamp(
                free_bits_lambda - kl_per_factor, min=0.0
            ).sum(dim=-1).mean()

        # ── ELBO loss (minimised → negate ELBO) ──────────────────────────────
        elbo = log_p_r_z - kl  # [batch]
        loss = -elbo.mean() + free_bits_penalty

        with torch.no_grad():
            log_likelihood = log_p_r_z.mean()
            kl_divergence = kl.mean()
            kl_pf = kl_per_factor.mean(dim=0)  # [F]

        return {
            'loss': loss,
            'log_likelihood': log_likelihood,
            'kl_divergence': kl_divergence,
            'kl_per_factor': kl_pf,
            'free_bits_penalty': free_bits_penalty.detach(),
            'alpha': alpha.detach(),
            'sigma': sigma.detach(),
            'mu_q': mu_q.detach(),
            'log_sigma_q': log_sigma_q.detach(),
        }

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        S: torch.Tensor,
        S_static: torch.Tensor,
        num_samples: int = 1,
        return_factors: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Generate return predictions by sampling z ~ N(0,I).

        Args:
            S:           [batch, N, L, d_ts]
            S_static:    [batch, N, d_static]
            num_samples: K — samples from prior
            return_factors: include sampled z in output

        Returns dict with r_samples, r_mean, r_std, (optionally factors).
        """
        batch = S.shape[0]
        S_nb = S.squeeze(0)
        S_st_nb = S_static.squeeze(0)

        alpha_n, B_n, sigma_n = self.embedder(S_nb, S_st_nb)  # [N], [N,F], [N]
        alpha = alpha_n.unsqueeze(0)   # [1, N]
        B = B_n.unsqueeze(0)           # [1, N, F]
        sigma = sigma_n.unsqueeze(0)   # [1, N]

        F_dim = B.shape[-1]
        z = torch.randn(batch, num_samples, F_dim, device=S.device)  # [1, K, F]

        r_samples = dec.sample_r_given_z(
            alpha=alpha.unsqueeze(1).expand(-1, num_samples, -1),  # [1, K, N]
            B=B.unsqueeze(1).expand(-1, num_samples, -1, -1),      # [1, K, N, F]
            sigma=sigma.unsqueeze(1).expand(-1, num_samples, -1),  # [1, K, N]
            z=z,
        )  # [1, N, K] or [1, N] if K=1

        # Marginal stats:  E[r] = alpha (mu_z=0),  Var[r] = sigma^2 + diag(B B^T)
        r_mean = dec.marginal_mean(alpha, B)  # [1, N]
        r_cov = dec.marginal_covariance(B, sigma)  # [1, N, N]
        r_std = torch.sqrt(torch.diagonal(r_cov, dim1=-2, dim2=-1))  # [1, N]

        result = {
            'r_samples': r_samples,
            'r_mean': r_mean,
            'r_std': r_std,
        }
        if return_factors:
            result['factors'] = z
        return result

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        S: torch.Tensor,
        S_static: torch.Tensor,
        r: Optional[torch.Tensor] = None,
        num_samples: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Auto-switch: ELBO loss when r provided, prediction otherwise."""
        if r is not None:
            return self.compute_elbo_loss(S, S_static, r, mask)
        return self.predict(S, S_static, num_samples=num_samples or 1)


__all__ = ["NeuralFactors"]
