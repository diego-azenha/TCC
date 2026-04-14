"""PyTorch Lightning module for NeuralFactors (simplified ELBO version).

Single Adam optimizer group — no sigma freeze, no prior params.
Training calls compute_elbo_loss() (K=1, Gaussian, exact KL).
Health metrics logged every 100 steps to watch for posterior collapse.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Optional
from copy import deepcopy

from .neuralfactors import NeuralFactors
from ..utils.config import ModelConfig, TrainingConfig


class NeuralFactorsLightning(pl.LightningModule):
    """Lightning wrapper for NeuralFactors training with ELBO and Polyak averaging."""

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_config = model_config
        self.training_config = training_config

        self.model = NeuralFactors(config=model_config)

        self.use_polyak = training_config.use_polyak
        if self.use_polyak:
            self.polyak_model = None
            self.polyak_alpha = training_config.polyak_alpha
            self.polyak_start_step = training_config.polyak_start_step

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        S: torch.Tensor,
        S_static: torch.Tensor,
        r: Optional[torch.Tensor] = None,
        num_samples: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.model(S, S_static, r, num_samples, mask)

    # ── Training ──────────────────────────────────────────────────────────────

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        S, S_static, r, mask = batch

        output = self.model.compute_elbo_loss(
            S=S,
            S_static=S_static,
            r=r,
            mask=mask,
            free_bits_lambda=self.training_config.free_bits_lambda,
        )

        loss = output['loss']

        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/log_likelihood', output['log_likelihood'], on_step=True, on_epoch=True)
        self.log('train/kl_divergence', output['kl_divergence'], on_step=True, on_epoch=True)

        if self.training_config.free_bits_lambda > 0.0:
            self.log('train/free_bits_penalty', output['free_bits_penalty'],
                     on_step=True, on_epoch=False)

        # Health metrics every 100 steps: watch for posterior collapse
        if self.global_step % 100 == 0:
            mu_q = output['mu_q']           # [batch, F]
            log_sigma_q = output['log_sigma_q']  # [batch, F]
            sigma_q = torch.exp(log_sigma_q)

            # σ_q_mean ≈ 1.0 at init; collapses toward 0 if prior dominates
            self.log('train/sigma_q_mean', sigma_q.mean(), on_step=True, on_epoch=False)
            self.log('train/sigma_q_min', sigma_q.min(), on_step=True, on_epoch=False)
            # μ_q_norm: should be non-zero when factors carry signal
            self.log('train/mu_q_norm', mu_q.norm(dim=-1).mean(), on_step=True, on_epoch=False)
            # β_norm: loading matrix; should grow during training
            alpha = output['alpha']  # [batch, N]
            sigma = output['sigma']  # [batch, N]
            self.log('train/alpha_mean', alpha.mean(), on_step=True, on_epoch=False)
            self.log('train/alpha_std', alpha.std(), on_step=True, on_epoch=False)
            self.log('train/sigma_mean', sigma.mean(), on_step=True, on_epoch=False)
            self.log('train/kl_min_factor', output['kl_per_factor'].min(), on_step=True, on_epoch=False)
            self.log('train/kl_max_factor', output['kl_per_factor'].max(), on_step=True, on_epoch=False)

        return loss

    # ── Validation ────────────────────────────────────────────────────────────

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        S, S_static, r, mask = batch

        output = self.model.compute_elbo_loss(
            S=S,
            S_static=S_static,
            r=r,
            mask=mask,
        )

        loss = output['loss']
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/log_likelihood', output['log_likelihood'], on_step=False, on_epoch=True)
        self.log('val/kl_divergence', output['kl_divergence'], on_step=False, on_epoch=True)

        return loss

    # ── Optimizer ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        """Single Adam param group — all parameters at the same base LR."""
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        return optimizer

    # ── Polyak averaging ──────────────────────────────────────────────────────

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if not self.use_polyak:
            return
        current_step = self.global_step

        if current_step == self.polyak_start_step and self.polyak_model is None:
            print(f"\nInitializing Polyak averaging at step {current_step}")
            self.polyak_model = deepcopy(self.model)
            for param in self.polyak_model.parameters():
                param.requires_grad = False
            return

        if current_step >= self.polyak_start_step and self.polyak_model is not None:
            with torch.no_grad():
                for p_cur, p_poly in zip(
                    self.model.parameters(),
                    self.polyak_model.parameters(),
                ):
                    p_poly.data.mul_(self.polyak_alpha).add_(
                        p_cur.data, alpha=(1.0 - self.polyak_alpha)
                    )

    def get_polyak_model(self) -> Optional[nn.Module]:
        return self.polyak_model

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_step(self, batch: tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        S, S_static, r, mask = batch
        model_to_use = self.polyak_model if self.polyak_model is not None else self.model
        return model_to_use.predict(S=S, S_static=S_static, num_samples=1, return_factors=False)


__all__ = ["NeuralFactorsLightning"]
