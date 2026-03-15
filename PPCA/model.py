"""PPCA core model: closed-form fitting and inference.

Probabilistic PCA (Tipping & Bishop, 1999) with all computations
performed via the Woodbury identity so N×N matrices are never inverted.

Generative process
------------------
  z ~ N(0, I_F)
  r | z ~ N(W z + mu, sigma2 I_N)

Marginal distribution
---------------------
  r ~ N(mu, W W^T + sigma2 I_N)

All public functions are pure (no side effects).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PPCAModel:
    """Fitted PPCA model parameters.

    Attributes
    ----------
    W        : (N, F) loading matrix
    mu       : (N,)  mean return vector
    sigma2   : scalar  isotropic noise variance
    F        : number of retained factors
    tickers  : list of ticker strings — defines row ordering of W / mu
    """
    W: np.ndarray
    mu: np.ndarray
    sigma2: float
    F: int
    tickers: list[str]

    # ------------------------------------------------------------------
    # Small derived quantities cached on first access
    # ------------------------------------------------------------------
    _M: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _M_inv: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _log_det_C: Optional[float] = field(default=None, repr=False, compare=False)

    def _ensure_M(self) -> None:
        if self._M is None:
            self._M = self.W.T @ self.W + self.sigma2 * np.eye(self.F)
            self._M_inv = np.linalg.inv(self._M)

    def _ensure_log_det_C(self) -> None:
        """log |W W^T + sigma2 I_N| using the matrix determinant lemma.

        log|C| = log|M| - F*log(sigma2) + N*log(sigma2)
        where M = W^T W + sigma2 I_F.
        """
        if self._log_det_C is None:
            self._ensure_M()
            _, log_det_M = np.linalg.slogdet(self._M)
            N = len(self.mu)
            self._log_det_C = (
                log_det_M
                - self.F * np.log(self.sigma2)
                + N * np.log(self.sigma2)
            )


# =============================================================================
# Fitting
# =============================================================================

def fit(R: np.ndarray, F: int, tickers: list[str]) -> PPCAModel:
    """Fit PPCA by closed-form MLE (Tipping & Bishop, 1999).

    Parameters
    ----------
    R       : (T, N) returns matrix (rows = time, cols = stocks); no NaNs allowed.
    F       : number of latent factors to retain.  Must be < N.
    tickers : list of N ticker names (same column ordering as R).

    Returns
    -------
    PPCAModel with fields W, mu, sigma2, F, tickers.

    Raises
    ------
    ValueError  if F >= N (degenerate model).
    """
    T, N = R.shape
    if F >= N:
        raise ValueError(
            f"Number of factors F={F} must be strictly less than "
            f"number of stocks N={N}."
        )
    if F < 1:
        raise ValueError(f"F must be >= 1, got {F}.")

    # Step 1: mean and sample covariance
    mu = R.mean(axis=0)                          # (N,)
    R_c = R - mu                                 # (T, N) centred
    S = (R_c.T @ R_c) / T                        # (N, N) sample cov

    # Step 2: eigendecomposition (eigh: symmetric, ascending eigenvalues)
    eigenvalues, eigenvectors = np.linalg.eigh(S)

    # Flip to descending order
    eigenvalues = eigenvalues[::-1].copy()
    eigenvectors = eigenvectors[:, ::-1].copy()

    # Clip negative eigenvalues (numerical noise)
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    delta_top = eigenvalues[:F]          # (F,)
    delta_rest = eigenvalues[F:]         # (N-F,)

    # Step 3: isotropic noise variance = mean of discarded eigenvalues
    sigma2 = float(np.mean(delta_rest)) if len(delta_rest) > 0 else 1e-6
    sigma2 = max(sigma2, 1e-6)           # floor for numerical safety

    # Step 4: loading matrix  W = U_F * diag(sqrt(delta_j - sigma2))
    scales = np.sqrt(np.maximum(delta_top - sigma2, 1e-8))  # (F,)
    W = eigenvectors[:, :F] * scales[np.newaxis, :]          # (N, F)

    return PPCAModel(W=W, mu=mu, sigma2=sigma2, F=F, tickers=list(tickers))


# =============================================================================
# Log-probability (Woodbury — never materialises C^-1)
# =============================================================================

def log_prob(model: PPCAModel, r: np.ndarray) -> float:
    """Log-likelihood log p(r) under the fitted PPCA model.

    Uses the Woodbury / matrix-determinant-lemma identities:

      log|C| = log|M| - F*log(σ²) + N*log(σ²)

      (r-μ)^T C^-1 (r-μ) = (1/σ²)[‖r-μ‖² - v^T M^-1 v]
                            where v = W^T (r-μ)   [F-dim]

    Parameters
    ----------
    model : fitted PPCAModel
    r     : (N,) return vector (same ordering as model.tickers)

    Returns
    -------
    float: log p(r)
    """
    model._ensure_M()
    model._ensure_log_det_C()

    N = len(model.mu)
    res = r - model.mu                            # (N,)
    v = model.W.T @ res                           # (F,)

    quad = (np.dot(res, res) - v @ (model._M_inv @ v)) / model.sigma2

    return -0.5 * (N * np.log(2.0 * np.pi) + model._log_det_C + quad)


# =============================================================================
# Posterior over latent factors
# =============================================================================

def posterior(model: PPCAModel, r: np.ndarray):
    """Gaussian posterior q(z | r) = N(mu_z, Sigma_z).

    mu_z    = M^-1 W^T (r - mu)
    Sigma_z = sigma2 M^-1

    Both computed with the F×F matrix M — no N×N inversion.

    Parameters
    ----------
    model : fitted PPCAModel
    r     : (N,) return vector

    Returns
    -------
    mu_z    : (F,) posterior mean
    Sigma_z : (F, F) posterior covariance
    """
    model._ensure_M()
    res = r - model.mu                    # (N,)
    v = model.W.T @ res                   # (F,)
    mu_z = model._M_inv @ v              # (F,)
    Sigma_z = model.sigma2 * model._M_inv  # (F, F)
    return mu_z, Sigma_z


# =============================================================================
# Sampling
# =============================================================================

def sample(model: PPCAModel, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Sample return vectors from p(r) = ∫ p(r|z) p(z) dz.

    Equivalent to the ancestral sampler:
      z   ~ N(0, I_F)
      eps ~ N(0, sigma2 I_N)
      r   = z @ W^T + mu + eps

    Parameters
    ----------
    model     : fitted PPCAModel
    n_samples : number of draws
    rng       : numpy random Generator (for reproducibility)

    Returns
    -------
    (n_samples, N) array of sampled returns
    """
    N = len(model.mu)
    z = rng.standard_normal((n_samples, model.F))              # (K, F)
    eps = rng.standard_normal((n_samples, N)) * np.sqrt(model.sigma2)  # (K, N)
    return z @ model.W.T + model.mu[np.newaxis, :] + eps       # (K, N)


# =============================================================================
# Covariance matrix
# =============================================================================

def covariance(model: PPCAModel) -> np.ndarray:
    """Model covariance Sigma = W W^T + sigma2 I_N.

    Materialises the full N×N matrix.  With N~100 this is trivial.

    Returns
    -------
    (N, N) covariance matrix
    """
    N = len(model.mu)
    return model.W @ model.W.T + model.sigma2 * np.eye(N)
