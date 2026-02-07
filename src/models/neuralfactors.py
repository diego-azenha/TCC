"""Main NeuralFactors model integrating all components.

Combines StockEmbedder, Prior, Encoder, and Decoder for training and inference.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
import math

from .stock_embedder import StockEmbedder
from .prior import StudentTPrior
from . import encoder as enc
from . import decoder as dec
from ..utils.config import ModelConfig


class NeuralFactors(nn.Module):
    """NeuralFactors VAE for factor learning from equity returns.
    
    Training mode: Uses encoder q(z|r) for posterior inference (CIWAE).
    Inference mode: Samples from prior p(z) without encoder.
    
    Paper: Achintya Gopal, "NeuralFactors" (2024, arXiv:2408.01499v1)
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Args:
            config: ModelConfig with all hyperparameters (F, h, dropout, etc.)
        """
        super().__init__()
        
        if config is None:
            config = ModelConfig()
        
        self.config = config
        
        # Components
        self.embedder = StockEmbedder(config=config)
        self.prior = StudentTPrior(num_factors=config.num_factors, config=config.prior_config)
    
    def encode(
        self,
        S: torch.Tensor,
        S_static: torch.Tensor,
        r: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode features and returns to get factor parameters and posterior.
        
        Training step: Generate factor model parameters and compute analytical posterior.
        
        Args:
            S: [batch, N, L, d_time] time-varying features
            S_static: [batch, N, d_static] static features
            r: [batch, N] observed returns
            mask: [batch, N] optional mask for valid stocks
            
        Returns:
            alpha: [batch, N] intercepts
            B: [batch, N, F] factor loadings
            sigma: [batch, N] idiosyncratic volatility
            nu: [batch, N] Student-T degrees of freedom
            mu_q: [batch, F] posterior mean
            L_q: [batch, F, F] posterior Cholesky factor
        """
        # Paper uses batch_size=1, so we squeeze/unsqueeze to match embedder expectations
        # S: [batch, N, L, d_ts] -> [N, L, d_ts] (batch must be 1)
        # S_static: [batch, N, d_static] -> [N, d_static]
        batch_size = S.shape[0]
        if batch_size != 1:
            raise ValueError(f"encode expects batch_size=1 (as per paper), got {batch_size}")
        
        S_no_batch = S.squeeze(0)  # [N, L, d_ts]
        S_static_no_batch = S_static.squeeze(0)  # [N, d_static]
        r_no_batch = r.squeeze(0) if r.dim() == 2 else r  # [N]
        mask_no_batch = mask.squeeze(0) if mask is not None and mask.dim() == 2 else mask  # [N] or None
        
        # Generate factor model parameters
        alpha, B, sigma, nu = self.embedder(S_no_batch, S_static_no_batch)
        
        # Get prior parameters for encoder (as Normal via moment matching)
        mu_z, Sigma_z = self.prior.to_normal_params()
        
        # Encoder expects no batch dimension (will add it internally if needed)
        # Compute analytical posterior q(z|r)
        mu_q, L_q, _, _ = enc.encoder_recon(
            alpha=alpha,
            B=B,
            sigma=sigma,
            r=r_no_batch,
            mu_z=mu_z,
            Sigma_z=Sigma_z,
            mask=mask_no_batch,
            eps=self.config.encoder_config.eps,
            jitter_init=self.config.encoder_config.jitter_init,
            jitter_max=self.config.encoder_config.jitter_max,
            jitter_multiplier=self.config.encoder_config.jitter_multiplier,
            use_fp64=self.config.encoder_config.use_fp64,
        )
        
        # Add batch dimension back for consistency: [N, ...] -> [batch, N, ...]
        alpha = alpha.unsqueeze(0)  # [1, N]
        B = B.unsqueeze(0)  # [1, N, F]
        sigma = sigma.unsqueeze(0)  # [1, N]
        nu = nu.unsqueeze(0)  # [1, N]
        
        # mu_q and L_q might already have batch dim from encoder (if it added it)
        if mu_q.dim() == 1:
            mu_q = mu_q.unsqueeze(0)  # [F] -> [1, F]
        if L_q.dim() == 2:
            L_q = L_q.unsqueeze(0)  # [F, F] -> [1, F, F]
        
        return alpha, B, sigma, nu, mu_q, L_q
    
    def compute_iwae_loss(
        self,
        S: torch.Tensor,
        S_static: torch.Tensor,
        r: torch.Tensor,
        num_samples: int,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute CIWAE loss for training (Paper Equation 7).
        
        CIWAE = E_q [ log p(r|z) + log p(z) - log q(z|r) ]
        Uses importance weighting with K samples.
        
        Args:
            S: [batch, N, L, d_time] time-varying features
            S_static: [batch, N, d_static] static features
            r: [batch, N] observed returns
            num_samples: K, number of importance samples
            mask: [batch, N] optional mask for valid stocks
            
        Returns:
            Dictionary with:
                - loss: scalar CIWAE loss (negative ELBO)
                - log_likelihood: E[log p(r|z)]
                - kl_divergence: E[KL(q||p)]
                - log_weights: [batch, K] importance weights (for diagnostics)
        """
        # Encode to get parameters and posterior
        alpha, B, sigma, nu, mu_q, L_q = self.encode(S, S_static, r, mask)
        
        batch, N, F = B.shape
        
        # Sample z from posterior q(z|r) using reparameterization
        # z = mu_q + L_q @ eps, where eps ~ N(0,I)
        # mu_q: [batch, F], L_q: [batch, F, F], eps: [batch, K, F]
        eps = torch.randn(batch, num_samples, F, device=mu_q.device, dtype=mu_q.dtype)  # [batch, K, F]
        
        # Use bmm for batched matrix multiplication: L_q @ eps^T, then transpose back
        # eps: [batch, K, F] -> [batch, F, K]
        # L_q @ eps^T: [batch, F, F] @ [batch, F, K] -> [batch, F, K]
        # Result^T: [batch, K, F]
        z = mu_q.unsqueeze(1) + torch.bmm(L_q, eps.transpose(1, 2)).transpose(1, 2)  # [batch, K, F]
        
        # Compute log p(r|z) - likelihood
        log_p_r_given_z = dec.log_pdf_r_given_z(
            alpha=alpha.unsqueeze(1).expand(-1, num_samples, -1),  # [batch, K, N]
            B=B.unsqueeze(1).expand(-1, num_samples, -1, -1),      # [batch, K, N, F]
            sigma=sigma.unsqueeze(1).expand(-1, num_samples, -1),  # [batch, K, N]
            nu=nu.unsqueeze(1).expand(-1, num_samples, -1),        # [batch, K, N]
            z=z,                                                     # [batch, K, F]
            r=r.unsqueeze(1).expand(-1, num_samples, -1),          # [batch, K, N]
            mask=mask.unsqueeze(1).expand(-1, num_samples, -1) if mask is not None else None
        )  # [batch, K]
        
        # Compute log p(z) - prior
        log_p_z = self.prior.log_prob(z)  # [batch, K]
        
        # Compute log q(z|r) - posterior
        # q(z|r) ~ N(mu_q, L_q L_q^T)
        # log q = -0.5 * (F*log(2pi) + log|Sigma_q| + (z-mu)^T Sigma^-1 (z-mu))
        z_centered = z - mu_q.unsqueeze(1)  # [batch, K, F]
        
        # Solve L_q @ y = z_centered for y, then compute ||y||^2
        # This is equivalent to (z-mu)^T Sigma^-1 (z-mu)
        L_q_expanded = L_q.unsqueeze(1).expand(-1, num_samples, -1, -1)  # [batch, K, F, F]
        y = torch.linalg.solve_triangular(L_q_expanded, z_centered.unsqueeze(-1), upper=False)  # [batch, K, F, 1]
        mahalanobis = torch.sum(y * y, dim=(-2, -1))  # [batch, K]
        
        # Log determinant of covariance (via Cholesky)
        log_det_Sigma_q = 2.0 * torch.sum(torch.log(torch.diagonal(L_q, dim1=-2, dim2=-1)), dim=-1)  # [batch]
        
        log_q_z = -0.5 * (F * math.log(2 * math.pi) + log_det_Sigma_q.unsqueeze(1) + mahalanobis)  # [batch, K]
        
        # Compute importance weights: log w = log p(r|z) + log p(z) - log q(z|r)
        log_weights = log_p_r_given_z + log_p_z - log_q_z  # [batch, K]
        
        # IWAE loss: -log mean_k exp(log w_k)
        # Use log-sum-exp for numerical stability
        log_mean_weight = torch.logsumexp(log_weights, dim=1) - math.log(num_samples)  # [batch]
        iwae_loss = -torch.mean(log_mean_weight)  # scalar
        
        # Diagnostics
        with torch.no_grad():
            log_likelihood = torch.mean(log_p_r_given_z)
            kl_divergence = torch.mean(log_q_z - log_p_z)
        
        return {
            'loss': iwae_loss,
            'log_likelihood': log_likelihood,
            'kl_divergence': kl_divergence,
            'log_weights': log_weights.detach()
        }
    
    def predict(
        self,
        S: torch.Tensor,
        S_static: torch.Tensor,
        num_samples: int = 1,
        return_factors: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Generate predictions by sampling from prior (inference mode).
        
        Args:
            S: [batch, N, L, d_time] time-varying features
            S_static: [batch, N, d_static] static features
            num_samples: K, number of samples to draw
            return_factors: If True, return sampled z
            
        Returns:
            Dictionary with:
                - r_samples: [batch, N, K] or [batch, N] return samples
                - r_mean: [batch, N] marginal mean E[r]
                - r_std: [batch, N] marginal std sqrt(Var[r])
                - factors: [batch, K, F] sampled factors (if return_factors=True)
        """
        batch, N = S.shape[0], S.shape[1]
        
        # Generate factor model parameters
        alpha, B, sigma, nu = self.embedder(S, S_static)
        
        # Sample z from prior p(z)
        z = self.prior.sample(batch_size=batch, num_samples=num_samples, device=S.device)  # [batch, K, F]
        
        # Sample returns r|z
        r_samples = dec.sample_r_given_z(
            alpha=alpha.unsqueeze(1).expand(-1, num_samples, -1),  # [batch, K, N]
            B=B.unsqueeze(1).expand(-1, num_samples, -1, -1),      # [batch, K, N, F]
            sigma=sigma.unsqueeze(1).expand(-1, num_samples, -1),  # [batch, K, N]
            nu=nu.unsqueeze(1).expand(-1, num_samples, -1),        # [batch, K, N]
            z=z                                                     # [batch, K, F]
        )  # [batch, K, N]
        
        # Transpose to [batch, N, K] for consistency
        r_samples = r_samples.transpose(1, 2)  # [batch, N, K]
        
        # Compute marginal statistics (no sampling needed)
        mu_z, Sigma_z = self.prior.to_normal_params()
        r_mean = dec.marginal_mean(alpha, B, mu_z)  # [batch, N]
        r_cov = dec.marginal_covariance(B, Sigma_z, sigma)  # [batch, N, N]
        r_std = torch.sqrt(torch.diagonal(r_cov, dim1=-2, dim2=-1))  # [batch, N]
        
        result = {
            'r_samples': r_samples.squeeze(-1) if num_samples == 1 else r_samples,
            'r_mean': r_mean,
            'r_std': r_std
        }
        
        if return_factors:
            result['factors'] = z
        
        return result
    
    def forward(
        self,
        S: torch.Tensor,
        S_static: torch.Tensor,
        r: Optional[torch.Tensor] = None,
        num_samples: Optional[int] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with automatic train/inference mode.
        
        Training (r provided): Computes CIWAE loss.
        Inference (r=None): Generates predictions from prior.
        
        Args:
            S: [batch, N, L, d_time] time-varying features
            S_static: [batch, N, d_static] static features
            r: [batch, N] observed returns (training only)
            num_samples: K samples (default: config.num_iwae_samples for train, 1 for inference)
            mask: [batch, N] optional mask for valid stocks
            
        Returns:
            Training: {'loss', 'log_likelihood', 'kl_divergence', 'log_weights'}
            Inference: {'r_samples', 'r_mean', 'r_std'}
        """
        if r is not None:
            # Training mode
            if num_samples is None:
                num_samples = self.config.num_iwae_samples
            return self.compute_iwae_loss(S, S_static, r, num_samples, mask)
        else:
            # Inference mode
            if num_samples is None:
                num_samples = 1
            return self.predict(S, S_static, num_samples)


__all__ = ["NeuralFactors"]
