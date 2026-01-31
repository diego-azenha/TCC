"""PyTorch Lightning module for NeuralFactors training.

Wraps the NeuralFactors model with training/validation logic, optimizer configuration,
and Polyak averaging as specified in paper Section 3.5.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Optional
from copy import deepcopy

from .neuralfactors import NeuralFactors
from ..utils.config import ModelConfig, TrainingConfig


class NeuralFactorsLightning(pl.LightningModule):
    """Lightning module for NeuralFactors training.
    
    Implements:
    - CIWAE loss training (Paper Equation 3)
    - Adam optimizer with weight decay (Paper Section 3.5)
    - Polyak averaging starting at step 50,000 (Paper Section 3.5)
    - Validation with NLL_joint metric
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
    ):
        """
        Args:
            model_config: Model architecture configuration
            training_config: Training hyperparameters
        """
        super().__init__()
        
        self.save_hyperparameters()
        
        self.model_config = model_config
        self.training_config = training_config
        
        # Initialize model
        self.model = NeuralFactors(config=model_config)
        
        # Polyak averaging (exponential moving average)
        self.use_polyak = training_config.use_polyak
        if self.use_polyak:
            self.polyak_model = None  # Initialized after first forward pass
            self.polyak_alpha = training_config.polyak_alpha
            self.polyak_start_step = training_config.polyak_start_step
    
    def forward(
        self,
        S: torch.Tensor,
        S_static: torch.Tensor,
        r: Optional[torch.Tensor] = None,
        num_samples: Optional[int] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass delegates to model."""
        return self.model(S, S_static, r, num_samples, mask)
    
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step computes CIWAE loss.
        
        Args:
            batch: Tuple of (S, S_static, r, mask)
            batch_idx: Batch index
            
        Returns:
            Loss tensor for backpropagation
        """
        S, S_static, r, mask = batch
        
        # Compute CIWAE loss with k importance samples
        output = self.model.compute_iwae_loss(
            S=S,
            S_static=S_static,
            r=r,
            num_samples=self.training_config.num_iwae_samples,
            mask=mask
        )
        
        loss = output['loss']
        log_likelihood = output['log_likelihood']
        kl_divergence = output['kl_divergence']
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/log_likelihood', log_likelihood, on_step=True, on_epoch=True)
        self.log('train/kl_divergence', kl_divergence, on_step=True, on_epoch=True)
        
        # Compute effective sample size from importance weights
        log_weights = output['log_weights']  # [batch, K]
        weights = torch.exp(log_weights - torch.logsumexp(log_weights, dim=1, keepdim=True))
        ess = 1.0 / torch.sum(weights ** 2, dim=1).mean()
        self.log('train/ess', ess, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Validation step computes NLL_joint.
        
        Args:
            batch: Tuple of (S, S_static, r, mask)
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        S, S_static, r, mask = batch
        
        # Use more samples for validation (paper: 100 for NLL_joint)
        num_val_samples = 100
        
        output = self.model.compute_iwae_loss(
            S=S,
            S_static=S_static,
            r=r,
            num_samples=num_val_samples,
            mask=mask
        )
        
        loss = output['loss']
        log_likelihood = output['log_likelihood']
        kl_divergence = output['kl_divergence']
        
        # Compute NLL_joint: -log p(r|F) averaged over batch
        # loss is already negative ELBO, which approximates NLL
        N_valid = mask.sum(dim=1).float()  # [batch]
        nll_joint_per_stock = loss  # Already averaged
        
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/nll_joint', loss, on_step=False, on_epoch=True)
        self.log('val/log_likelihood', log_likelihood, on_step=False, on_epoch=True)
        self.log('val/kl_divergence', kl_divergence, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure Adam optimizer with weight decay (paper Section 3.5)."""
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay
        )
        return optimizer
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update Polyak averaged model after each training step.
        
        Paper Section 3.5: Polyak averaging starts at step 50,000
        """
        if not self.use_polyak:
            return
        
        current_step = self.global_step
        
        # Initialize Polyak model on first step after polyak_start_step
        if current_step == self.polyak_start_step and self.polyak_model is None:
            print(f"\nInitializing Polyak averaging at step {current_step}")
            self.polyak_model = deepcopy(self.model)
            for param in self.polyak_model.parameters():
                param.requires_grad = False
            return
        
        # Update Polyak model if past start step
        if current_step >= self.polyak_start_step and self.polyak_model is not None:
            with torch.no_grad():
                for param_current, param_polyak in zip(
                    self.model.parameters(),
                    self.polyak_model.parameters()
                ):
                    # EMA update: θ_polyak = α * θ_polyak + (1 - α) * θ_current
                    param_polyak.data.mul_(self.polyak_alpha).add_(
                        param_current.data, alpha=(1.0 - self.polyak_alpha)
                    )
    
    def get_polyak_model(self) -> Optional[nn.Module]:
        """Get Polyak-averaged model for inference/evaluation."""
        return self.polyak_model
    
    def predict_step(self, batch: tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Prediction step for inference.
        
        Args:
            batch: Tuple of (S, S_static, r, mask) - r may be None
            batch_idx: Batch index
            
        Returns:
            Dictionary with predictions
        """
        S, S_static, r, mask = batch
        
        # Use Polyak model if available
        model_to_use = self.polyak_model if self.polyak_model is not None else self.model
        
        # Generate predictions (samples from prior)
        output = model_to_use.predict(
            S=S,
            S_static=S_static,
            num_samples=1,
            return_factors=False
        )
        
        return output


__all__ = ["NeuralFactorsLightning"]
