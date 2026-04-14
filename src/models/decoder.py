"""Gaussian decoder for the observation model p(r|z).

Linear factor model: r_i | z ~ N(alpha_i + beta_i' z, sigma_i^2)

Provides:
  log_pdf_r_given_z  — Gaussian log-likelihood given z samples
  sample_r_given_z   — Sample r ~ N(alpha + B'z, sigma^2)
  marginal_mean      — E[r] = alpha + B @ mu_z   (mu_z = 0 with N(0,I) prior)
  marginal_covariance— Cov[r] = diag(sigma^2) + B B^T   (Sigma_z = I)
  marginal_cov_actionable — Portfolio variance w^T Cov[r] w without NxN matrix
"""

import math
from typing import Optional, Tuple

import torch

_LOG_2PI = math.log(2 * math.pi)
_EPS_SIGMA = 1e-6


# ─── shape helpers ────────────────────────────────────────────────────────────

def _ensure_batch(x: Optional[torch.Tensor], ndim: int) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return x.unsqueeze(0) if x.dim() == ndim - 1 else x


def _canonicalize_B_alpha_sigma(alpha, B, sigma):
    """Promote (N,F) → (1,N,F) and matching batch dims for alpha/sigma."""
    if B.dim() == 2:
        B = B.unsqueeze(0)
    if B.dim() == 4:
        # (batch, K, N, F) → merge into (batch*K, N, F)
        batch, K, N, F = B.shape
        B = B.reshape(batch * K, N, F)
        if alpha is not None and alpha.dim() == 3:
            alpha = alpha.reshape(batch * K, N)
        if sigma is not None and sigma.dim() == 3:
            sigma = sigma.reshape(batch * K, N)
    else:
        if alpha is not None and alpha.dim() == 1:
            alpha = alpha.unsqueeze(0)
        if sigma is not None and sigma.dim() == 1:
            sigma = sigma.unsqueeze(0)
    return alpha, B, sigma


def _canonicalize_z(z: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Return z as (batch, K, F) and K."""
    if z.dim() == 1:
        return z.unsqueeze(0).unsqueeze(0), 1
    if z.dim() == 2:
        raise ValueError(
            f"z shape {tuple(z.shape)} is ambiguous. Use explicit (batch,K,F) or (F,)."
        )
    batch, K, F = z.shape
    return z, K


# ─── log p(r | z) ─────────────────────────────────────────────────────────────

def log_pdf_r_given_z(
    alpha: torch.Tensor,
    B: torch.Tensor,
    sigma: torch.Tensor,
    z: torch.Tensor,
    r: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    return_per_asset: bool = False,
) -> torch.Tensor:
    """Gaussian log-likelihood: log p(r | z) = sum_i log N(r_i; alpha_i + beta_i'z, sigma_i^2).

    Args:
        alpha: [batch, N] or [N]
        B:     [batch, N, F] or [N, F]
        sigma: [batch, N] or [N]
        z:     [batch, K, F] or [F]
        r:     [batch, N]  (no K dim — same target for all samples)
        mask:  [batch, N] bool
        return_per_asset: if True, also return per-asset logpdf [batch, K, N] or [batch, N]

    Returns:
        joint: [batch, K]  sum over valid assets
        (optionally) per_asset: [batch, K, N]
    """
    device = B.device

    # Canonicalize z → (batch, K, F);  record original K for reshape after
    z_raw, K_orig = _canonicalize_z(z)
    z_raw = z_raw.to(device=device)
    batch_z, K, F = z_raw.shape

    # Canonicalize B/alpha/sigma — may merge batch*K if 4D
    alpha, B, sigma = _canonicalize_B_alpha_sigma(alpha, B, sigma)
    batch_B = B.shape[0]

    # If B came in as 4D it was already merged; z must also be merged
    merged = (batch_B == batch_z * K)
    if merged:
        # Reshape z from (batch_z, K, F) → (batch_z*K, 1, F) so einsum works
        z_use = z_raw.reshape(batch_z * K, 1, F)
        K_use = 1
    else:
        z_use = z_raw
        K_use = K

    B = B.to(device=device)
    alpha = (alpha if alpha is not None else torch.zeros(B.shape[0], B.shape[1], device=device)).to(device)
    sigma = torch.clamp(sigma.to(device=device), min=_EPS_SIGMA)

    if alpha.dim() == 1:
        alpha = alpha.unsqueeze(0)
    if sigma.dim() == 1:
        sigma = sigma.unsqueeze(0)

    # Handle r shape
    if r.dim() == 1:
        r = r.unsqueeze(0)
    if merged and r.dim() == 3 and r.shape[1] == K:
        r = r.reshape(batch_z * K, r.shape[2])
    r = r.to(device=device)

    # Mask
    if mask is None:
        mask = torch.ones(B.shape[0], B.shape[1], dtype=torch.bool, device=device)
    else:
        if merged and mask.dim() == 3 and mask.shape[1] == K:
            mask = mask.reshape(batch_z * K, mask.shape[2])
        mask = mask.to(device=device)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)

    batch_size, N, _ = B.shape

    # loc = alpha + B z   →  (batch_size, N, K_use)
    loc = alpha.unsqueeze(-1) + torch.einsum("bnf,bkf->bnk", B, z_use)

    # Gaussian log-pdf per asset:
    # log p = -0.5 * [log(2π) + 2*log(σ) + ((r - loc)/σ)^2]
    r_exp = r.unsqueeze(-1)                     # (batch_size, N, 1)
    sigma_exp = sigma.unsqueeze(-1)             # (batch_size, N, 1)
    mask_exp = mask.unsqueeze(-1).float()       # (batch_size, N, 1)

    standardized = (r_exp - loc) / sigma_exp   # (batch_size, N, K_use)
    logpdf = -0.5 * (_LOG_2PI + 2.0 * torch.log(sigma_exp) + standardized ** 2)
    logpdf = logpdf * mask_exp                 # zero out invalid assets

    joint = logpdf.sum(dim=1)                  # (batch_size, K_use)

    # Reshape back from merged batch*K → (batch_z, K)
    if merged:
        joint = joint.reshape(batch_z, K)
        if return_per_asset:
            per_asset = logpdf.squeeze(-1).reshape(batch_z, K, N)

    if K_use == 1 and not merged:
        joint = joint.squeeze(-1)
        if return_per_asset:
            per_asset = logpdf.squeeze(-1)

    if return_per_asset:
        return joint, per_asset
    return joint


# ─── sample r | z ─────────────────────────────────────────────────────────────

def sample_r_given_z(
    alpha: torch.Tensor,
    B: torch.Tensor,
    sigma: torch.Tensor,
    z: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample r | z ~ N(alpha + B'z, sigma^2).

    Returns:
        r_samples: (batch, N, K) or (batch, N) when K=1
    """
    device = B.device
    alpha, B, sigma = _canonicalize_B_alpha_sigma(alpha, B, sigma)
    z, K = _canonicalize_z(z)

    B = B.to(device=device)
    alpha = (alpha if alpha is not None else torch.zeros(B.shape[0], B.shape[1], device=device)).to(device)
    sigma = torch.clamp(sigma.to(device=device), min=_EPS_SIGMA)
    z = z.to(device=device)

    if alpha.dim() == 1:
        alpha = alpha.unsqueeze(0)
    if sigma.dim() == 1:
        sigma = sigma.unsqueeze(0)

    batch, N, F = B.shape
    # loc: (batch, N, K)
    loc = alpha.unsqueeze(-1) + torch.einsum("bnf,bkf->bnk", B, z)
    eps = torch.randn_like(loc)
    r = loc + sigma.unsqueeze(-1) * eps  # (batch, N, K)

    if K == 1:
        return r.squeeze(-1)
    return r


# ─── marginal statistics ──────────────────────────────────────────────────────

def marginal_mean(
    alpha: torch.Tensor,
    B: torch.Tensor,
    mu_z: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """E[r] = alpha + B @ mu_z.

    With N(0,I) prior, mu_z = 0 so E[r] = alpha.

    Args:
        alpha: [batch, N] or [N]
        B:     [batch, N, F] or [N, F]
        mu_z:  [F] or None (defaults to zeros — correct for N(0,I) prior)
    """
    if B.dim() == 2:
        B = B.unsqueeze(0)
    batch, N, F = B.shape
    device = B.device
    if alpha.dim() == 1:
        alpha = alpha.unsqueeze(0)

    if mu_z is None:
        return alpha.to(device)

    mu_z = mu_z.to(device)
    if mu_z.dim() == 1:
        mu_z = mu_z.unsqueeze(0)
    return alpha.to(device) + torch.einsum("bnf,bf->bn", B, mu_z)


def marginal_covariance(
    B: torch.Tensor,
    sigma: torch.Tensor,
    Sigma_z: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cov[r] = diag(sigma^2) + B Sigma_z B^T.

    With N(0,I) prior, Sigma_z = I so Cov[r] = diag(sigma^2) + B B^T.

    Args:
        B:       [batch, N, F] or [N, F]
        sigma:   [batch, N] or [N]
        Sigma_z: [F, F] or [batch, F, F] — if None, uses identity
    """
    if B.dim() == 2:
        B = B.unsqueeze(0)
    batch, N, F = B.shape
    device = B.device
    if sigma.dim() == 1:
        sigma = sigma.unsqueeze(0)

    if Sigma_z is None:
        factor_cov = torch.bmm(B, B.transpose(-2, -1))  # (batch, N, N)
    else:
        if Sigma_z.dim() == 2:
            Sigma_z = Sigma_z.unsqueeze(0).expand(batch, -1, -1)
        Sigma_z = Sigma_z.to(device)
        factor_cov = torch.bmm(torch.bmm(B, Sigma_z), B.transpose(-2, -1))

    idio_var = (sigma * sigma).to(device)  # (batch, N)
    diag_idx = torch.arange(N, device=device)
    factor_cov = factor_cov.clone()
    factor_cov[:, diag_idx, diag_idx] = factor_cov[:, diag_idx, diag_idx] + idio_var
    return factor_cov


def marginal_cov_actionable(
    B: torch.Tensor,
    sigma: torch.Tensor,
    w: Optional[torch.Tensor] = None,
    Sigma_z: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Portfolio variance w^T Cov[r] w without forming full NxN matrix.

    If w is None, returns (B, Sigma_z, sigma) for deferred use.
    """
    if B.dim() == 2:
        B = B.unsqueeze(0)
    batch, N, F = B.shape
    device = B.device
    if sigma.dim() == 1:
        sigma = sigma.unsqueeze(0)

    if w is not None:
        if w.dim() == 1:
            w = w.unsqueeze(0)
        if Sigma_z is None:
            temp = torch.einsum("bnf,bn->bf", B, w)
            var = torch.einsum("bf,bf->b", temp, temp)
        else:
            if Sigma_z.dim() == 2:
                Sigma_z = Sigma_z.unsqueeze(0).expand(batch, -1, -1)
            temp = torch.einsum("bnf,bn->bf", B, w)
            var = torch.einsum("bf,bfg,bg->b", temp, Sigma_z, temp)
        var = var + torch.sum((w * w) * (sigma * sigma), dim=1)
        return var

    return B, Sigma_z, sigma


# ─── backward-compat alias ───────────────────────────────────────────────────

log_pdf_multiple_z = log_pdf_r_given_z


__all__ = [
    "log_pdf_r_given_z",
    "log_pdf_multiple_z",
    "sample_r_given_z",
    "marginal_mean",
    "marginal_covariance",
    "marginal_cov_actionable",
]
