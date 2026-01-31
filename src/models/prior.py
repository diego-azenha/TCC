"""Prior distribution p(z) for latent factors.

Time-homogeneous Student-T prior with learnable parameters (Paper Section 3.1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from ..utils.config import PriorConfig


class StudentTPrior(nn.Module):
    """Learnable Student-T prior p(z) ~ Student-T(nu_z, mu_z, sigma_z).
    
    Time-homogeneous prior over latent factors (Paper Section 3.5).
    Parameters are learnable via gradient descent during training.
    
    Paper sets mu_z = 0 without loss of generality (Section 3.2).
    """
    
    def __init__(self, num_factors: int, config: Optional[PriorConfig] = None):
        """
        Args:
            num_factors: F, number of latent factors (paper: 64)
            config: Optional PriorConfig with initial values
        """
        super().__init__()
        
        self.num_factors = num_factors
        
        if config is None:
            config = PriorConfig()
        
        # Learnable parameters (use log-parameterization for positivity)
        self.mu_z = nn.Parameter(torch.full((num_factors,), config.mu_z_init))
        self.log_sigma_z = nn.Parameter(torch.full((num_factors,), math.log(config.sigma_z_init)))
        self.log_nu_z_minus_4 = nn.Parameter(torch.tensor(math.log(config.nu_z_init - 4.0)))
    
    def get_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get constrained parameters (mu_z, sigma_z, nu_z).
        
        Returns:
            mu_z: [F] mean
            sigma_z: [F] scale (positive)
            nu_z: scalar degrees of freedom (>4)
        """
        sigma_z = torch.exp(self.log_sigma_z)
        nu_z = torch.exp(self.log_nu_z_minus_4) + 4.0
        return self.mu_z, sigma_z, nu_z
    
    def to_normal_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert Student-T to Normal via moment matching (for encoder).
        
        For Student-T with df=nu: variance = sigma^2 * nu/(nu-2)
        
        Returns:
            mu: [F] mean (same as Student-T)
            Sigma: [F,F] covariance matrix (diagonal, scaled by nu/(nu-2))
        """
        mu_z, sigma_z, nu_z = self.get_params()
        
        # Moment matching: scale variance
        variance_scale = nu_z / (nu_z - 2.0)
        scaled_var = (sigma_z * sigma_z) * variance_scale
        
        # Return as diagonal covariance matrix
        Sigma = torch.diag(scaled_var)
        
        return mu_z, Sigma
    
    def sample(self, batch_size: int, num_samples: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
        """Sample from Student-T prior.
        
        Args:
            batch_size: Number of independent batches
            num_samples: K, number of samples per batch
            device: Device to place samples on
            
        Returns:
            z: [batch_size, num_samples, F] samples from p(z)
        """
        if device is None:
            device = self.mu_z.device
        
        mu_z, sigma_z, nu_z = self.get_params()
        
        # Sample from Student-T: T = mu + sigma * (N(0,1) / sqrt(V / nu))
        # where V ~ Gamma(nu/2, 1/2)
        
        # Standard normal samples
        eps = torch.randn(batch_size, num_samples, self.num_factors, device=device)
        
        # Gamma samples for denominator
        gamma_dist = torch.distributions.Gamma(nu_z / 2.0, 0.5)
        v = gamma_dist.sample((batch_size, num_samples))  # [batch, K]
        
        # Student-T transformation
        scale_factor = torch.sqrt(nu_z / (2.0 * v))  # [batch, K]
        z = mu_z + sigma_z * eps * scale_factor.unsqueeze(-1)
        
        return z
    
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Compute log p(z) for given z samples.
        
        Args:
            z: [batch, K, F] or [batch, F] latent samples
            
        Returns:
            log_p: [batch, K] or [batch] log probabilities
        """
        mu_z, sigma_z, nu_z = self.get_params()
        
        # Handle both [batch,F] and [batch,K,F]
        if z.dim() == 2:
            z = z.unsqueeze(1)  # [batch,F] -> [batch,1,F]
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch, K, F = z.shape
        
        # Standardized residuals
        z_centered = z - mu_z  # [batch, K, F]
        z_std = z_centered / sigma_z  # [batch, K, F]
        
        # Student-T log-pdf per dimension (independent factors)
        # log p(z_f) = log Gamma((nu+1)/2) - log Gamma(nu/2) - 0.5*log(nu*pi) - log(sigma) 
        #              - ((nu+1)/2) * log(1 + z_std^2/nu)
        
        log_gamma_term = torch.lgamma((nu_z + 1.0) / 2.0) - torch.lgamma(nu_z / 2.0)
        log_const = -0.5 * (math.log(math.pi) + torch.log(nu_z))
        log_scale = -torch.log(sigma_z)  # [F]
        
        q = 1.0 + (z_std * z_std) / nu_z  # [batch, K, F]
        log_q = torch.log(q)
        power_term = -((nu_z + 1.0) / 2.0) * log_q  # [batch, K, F]
        
        # Sum over factors (independent)
        log_p_per_factor = log_gamma_term + log_const + log_scale + power_term  # [batch, K, F]
        log_p = torch.sum(log_p_per_factor, dim=-1)  # [batch, K]
        
        if squeeze_output:
            log_p = log_p.squeeze(1)  # [batch,1] -> [batch]
        
        return log_p
    
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returns current parameters (for inspection)."""
        return self.get_params()


__all__ = ["StudentTPrior"]
