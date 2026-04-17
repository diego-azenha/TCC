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

        self.model = NeuralFactors(
            config=model_config,
            sigma_ref_ema=getattr(training_config, 'sigma_ref_ema', 0.99),
        )

        self.use_polyak = training_config.use_polyak
        if self.use_polyak:
            self.polyak_model = None
            self.polyak_alpha = training_config.polyak_alpha
            self.polyak_start_step = training_config.polyak_start_step

        # Alpha freeze: prevent alpha shortcut by freezing alpha_head
        self.alpha_freeze_steps = getattr(training_config, 'alpha_freeze_steps', 0) or 0
        self._alpha_frozen = self.alpha_freeze_steps > 0

    # ── Alpha freeze helpers ─────────────────────────────────────────────────

    def _check_alpha_freeze(self):
        """Unfreeze alpha once alpha_freeze_steps is reached."""
        if self._alpha_frozen and self.global_step >= self.alpha_freeze_steps:
            self._alpha_frozen = False
            print(f"\n[Step {self.global_step}] Alpha head unfrozen")

    def on_train_batch_start(self, batch, batch_idx):
        self._check_alpha_freeze()

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

        _kl_warmup = getattr(self.training_config, 'kl_warmup_steps', 0)
        kl_weight = min(1.0, self.global_step / _kl_warmup) if _kl_warmup > 0 else 1.0

        output = self.model.compute_elbo_loss(
            S=S,
            S_static=S_static,
            r=r,
            mask=mask,
            free_bits_lambda=self.training_config.free_bits_lambda,
            lambda_sigma=getattr(self.training_config, 'lambda_sigma', 1.0),
            freeze_alpha=self._alpha_frozen,
            kl_weight=kl_weight,
        )

        loss = output['loss']

        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/L_recon', output['L_recon'], on_step=True, on_epoch=True)
        self.log('train/L_sigma', output['L_sigma'], on_step=True, on_epoch=True)
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
            self.log('train/alpha_frozen', float(self._alpha_frozen), on_step=True, on_epoch=False)
            self.log('train/sigma_mean', sigma.mean(), on_step=True, on_epoch=False)
            self.log('train/sigma_ref', output['sigma_ref'].item(), on_step=True, on_epoch=False)
            self.log('train/kl_min_factor', output['kl_per_factor'].min(), on_step=True, on_epoch=False)
            self.log('train/kl_max_factor', output['kl_per_factor'].max(), on_step=True, on_epoch=False)
            # Per-factor KL — 16 individual scalars for Image 1 panel 3
            kl_pf = output['kl_per_factor']  # [F]
            for f in range(kl_pf.shape[0]):
                self.log(f'train/kl_factor_{f}', kl_pf[f], on_step=True, on_epoch=False)
            # Beta-norm mean — for Image 1 panel 4 variance competition
            B_log = output['B']  # [batch, N, F]
            self.log('train/beta_norm_mean', B_log.norm(dim=-1).mean(), on_step=True, on_epoch=False)
            self.log('train/kl_weight', kl_weight, on_step=True, on_epoch=False)

        # Bootstrap diagnostics every 1000 steps (heavier computations)
        if self.global_step % 1000 == 0:
            with torch.no_grad():
                alpha_d = output['alpha']   # [batch, N]
                B_d = output['B']           # [batch, N, F]
                mu_q_d = output['mu_q']     # [batch, F]
                mask_f = mask.float() if mask is not None else torch.ones_like(r, dtype=torch.float)
                n_valid = mask_f.sum().clamp(min=1.0)

                # Metric 1: R²(α, r) — how much of return variance does α alone explain?
                r_masked = r * mask_f
                alpha_masked = alpha_d * mask_f
                r_mean = r_masked.sum() / n_valid
                ss_tot = ((r - r_mean) ** 2 * mask_f).sum().clamp(min=1e-9)
                ss_res = ((r - alpha_d) ** 2 * mask_f).sum()
                r2_alpha = 1.0 - ss_res / ss_tot
                self.log('train/r2_alpha', r2_alpha, on_step=True, on_epoch=False)

                # Metric 2: signal magnitude comparison — std(β·μ_q) vs std(α)
                beta_mu = torch.einsum('bnf,bf->bn', B_d, mu_q_d)  # [batch, N]
                beta_mu_std = (beta_mu * mask_f).std()
                alpha_std_d = (alpha_d * mask_f).std().clamp(min=1e-9)
                signal_ratio = beta_mu_std / alpha_std_d
                self.log('train/beta_mu_std', beta_mu_std, on_step=True, on_epoch=False)
                self.log('train/signal_ratio', signal_ratio, on_step=True, on_epoch=False)

        return loss

    # ── Gradient norm diagnostics ─────────────────────────────────────────────

    def on_before_optimizer_step(self, optimizer):
        """Log gradient norms for alpha_head and beta_head every 100 steps."""
        if self.global_step % 100 != 0:
            return
        a_grad = self.model.embedder.alpha_head.weight.grad
        b_grad = self.model.embedder.beta_head.weight.grad
        if a_grad is not None and b_grad is not None:
            a_norm = a_grad.norm().item()
            b_norm = b_grad.norm().item()
            self.log('train/grad_norm_alpha', a_norm, on_step=True, on_epoch=False)
            self.log('train/grad_norm_beta', b_norm, on_step=True, on_epoch=False)
            ratio = b_norm / max(a_norm, 1e-9)
            self.log('train/grad_norm_ratio_beta_alpha', ratio, on_step=True, on_epoch=False)

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
