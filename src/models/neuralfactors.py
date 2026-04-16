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

    def __init__(self, config: Optional[ModelConfig] = None, sigma_ref_ema: float = 0.99):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config
        self.sigma_ref_ema = sigma_ref_ema

        self.embedder = StockEmbedder(config=config)
        self.encoder = DeepSetEncoder(
            num_factors=config.num_factors,
            phi_hidden=config.encoder_config.phi_hidden,
            rho_hidden=config.encoder_config.rho_hidden,
        )

        # sigma_ref: scalar EMA of residual RMS, used as fixed scale in L_recon
        self.register_buffer('sigma_ref', torch.ones(1))

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
        lambda_sigma: float = 1.0,
        freeze_alpha: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Compute decomposed loss with detached sigma gradient.

        Prevents sigma collapse (alpha shortcut) by separating the loss into:
          L_recon: reconstruction quality using fixed sigma_ref (no sigma gradient)
          L_sigma: Gaussian NLL calibration loss with detached residuals (sigma gradient only)
          KL:      standard closed-form KL(q || N(0,I))

        loss = -L_recon + KL + lambda_sigma * L_sigma + free_bits_penalty

        When freeze_alpha=True, alpha is detached from the loss so no gradient
        flows through it — forcing B and the encoder to explain returns first.

        Reference: Lucas et al. 2019 "Don't Blame the ELBO" — fixed decoder
        variance recovers PPCA solution for factors.

        Args:
            S:                [batch, N, L, d_ts]
            S_static:         [batch, N, d_static]
            r:                [batch, N]
            mask:             [batch, N] bool
            free_bits_lambda: λ (nats/factor); 0 disables the floor
            lambda_sigma:     weight on L_sigma calibration loss
            freeze_alpha:     if True, detach alpha from loss (alpha_head gets no gradient)

        Returns dict with:
            loss, L_recon, L_sigma, log_likelihood, kl_divergence,
            kl_per_factor, free_bits_penalty, alpha, B, sigma,
            mu_q, log_sigma_q, sigma_ref
        """
        alpha, B, sigma, mu_q, log_sigma_q = self.encode(S, S_static, r, mask)

        batch, N, F = B.shape

        # ── Sample z (K=1, reparameterization) ───────────────────────────────
        z = self.encoder.sample(mu_q, log_sigma_q)  # [batch, 1, F]
        z_flat = z.squeeze(1)                        # [batch, F]

        # ── Reconstruction: loc = alpha + B'z ────────────────────────────────
        # During alpha freeze: detach alpha so no gradient flows through it.
        # B and the encoder must explain returns; alpha_head gets no signal.
        alpha_for_loss = alpha.detach() if freeze_alpha else alpha
        loc = alpha_for_loss + torch.einsum('bnf,bf->bn', B, z_flat)  # [batch, N]
        residual = r - loc                                              # [batch, N]

        # ── Masking ──────────────────────────────────────────────────────────
        if mask is None:
            mask = torch.ones(batch, N, dtype=torch.bool, device=r.device)
        mask_f = mask.float()                                  # [batch, N]
        n_valid = mask_f.sum().clamp(min=1.0)

        # ── L_recon: uses sigma_ref (buffer, no gradient) ────────────────────
        # -0.5 * Σ_i (residual_i² / sigma_ref²) / N_valid
        # No log(sigma) term → no perverse incentive for sigma to collapse.
        # sigma_ref scales the loss so gradient magnitudes stay stable.
        sigma_ref = self.sigma_ref.detach()  # [1], no gradient
        L_recon = -0.5 * ((residual ** 2 / sigma_ref ** 2) * mask_f).sum() / n_valid

        # ── L_sigma: Gaussian NLL with detached residuals ────────────────────
        # 0.5 * Σ_i (2*log(sigma_i) + residual_sg_i² / sigma_i²) / N_valid
        # Optimal sigma_i = RMS(residual_i). Only sigma_head receives gradient.
        residual_sg = residual.detach()       # stop gradient to alpha/B/encoder
        sigma_safe = sigma.clamp(min=1e-6)
        L_sigma = 0.5 * ((2.0 * torch.log(sigma_safe) + residual_sg ** 2 / sigma_safe ** 2) * mask_f).sum() / n_valid

        # ── KL(q || N(0,I)) — exact closed form ──────────────────────────────
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

        # ── Total loss ────────────────────────────────────────────────────────
        loss = -L_recon + kl.mean() + lambda_sigma * L_sigma + free_bits_penalty

        # ── Update sigma_ref (EMA of batch residual RMS, no gradient) ────────
        if self.training:
            with torch.no_grad():
                batch_rms = torch.sqrt((residual.detach() ** 2 * mask_f).sum() / n_valid)
                self.sigma_ref.mul_(self.sigma_ref_ema).add_(
                    batch_rms, alpha=(1.0 - self.sigma_ref_ema)
                )

        with torch.no_grad():
            kl_divergence = kl.mean()
            kl_pf = kl_per_factor.mean(dim=0)  # [F]

        return {
            'loss': loss,
            'L_recon': L_recon.detach(),
            'L_sigma': L_sigma.detach(),
            'log_likelihood': L_recon.detach(),  # backward-compat alias
            'kl_divergence': kl_divergence,
            'kl_per_factor': kl_pf,
            'free_bits_penalty': free_bits_penalty.detach(),
            'alpha': alpha.detach(),
            'B': B.detach(),
            'sigma': sigma.detach(),
            'mu_q': mu_q.detach(),
            'log_sigma_q': log_sigma_q.detach(),
            'sigma_ref': self.sigma_ref.detach().clone(),
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
