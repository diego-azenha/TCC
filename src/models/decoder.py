"""Decoder for Student-T observation model p(r|z).

Linear factor model: r ~ Student-T(alpha + B^T z, sigma, nu)

Provides:
- log_pdf_r_given_z: Compute log p(r|z)
- sample_r_given_z: Sample r|z
- marginal_mean: E[r] = alpha + B @ mu_z
- marginal_covariance: Cov[r] = diag(sigma^2) + B Sigma_z B^T
- marginal_cov_actionable: Portfolio variance w^T Cov[r] w

USED IN BOTH TRAINING AND INFERENCE.
"""

from typing import Optional, Tuple
import math

import torch

_EPS_SIGMA = 1e-6
_MIN_NU = 4.01
_MAX_NU = 1e4


def _promote_batch(x: Optional[torch.Tensor], expected_ndim: int):
	if x is None:
		return None
	if x.dim() == expected_ndim - 1:
		return x.unsqueeze(0)
	return x


def _canonicalize_B_alpha_sigma_nu(alpha, B, sigma, nu):
	# B: (N,F) or (batch,N,F) or (batch,K,N,F)
	if B.dim() == 2:
		B = B.unsqueeze(0)
	
	# If B is 4D: (batch, K, N, F), reshape to (batch*K, N, F)
	if B.dim() == 4:
		batch, K, N, F = B.shape
		B = B.reshape(batch * K, N, F)
		# Need to expand alpha, sigma, nu as well
		if alpha is not None:
			if alpha.dim() == 3:  # (batch, K, N)
				alpha = alpha.reshape(batch * K, N)
			elif alpha.dim() == 1:
				alpha = alpha.unsqueeze(0)
		if sigma is not None:
			if sigma.dim() == 3:  # (batch, K, N)
				sigma = sigma.reshape(batch * K, N)
			elif sigma.dim() == 1:
				sigma = sigma.unsqueeze(0)
		if nu is not None:
			if nu.dim() == 3:  # (batch, K, N)
				nu = nu.reshape(batch * K, N)
			elif nu.dim() == 1:
				nu = nu.unsqueeze(0)
	else:
		batch_size, N, F = B.shape
		# promote 1D to batch
		if alpha is not None and alpha.dim() == 1:
			alpha = alpha.unsqueeze(0)
		if sigma is not None and sigma.dim() == 1:
			sigma = sigma.unsqueeze(0)
		if nu is not None and nu.dim() == 1:
			nu = nu.unsqueeze(0)

	return alpha, B, sigma, nu


def _canonicalize_z(z: torch.Tensor) -> Tuple[torch.Tensor, int]:
	"""Return z in shape (batch, K, F) and K value.

	Accepts:
	- (F,) -> (1,1,F) for single sample of single batch
	- (batch,K,F) -> (batch,K,F) explicit shape (PREFERRED)
	
	REJECTS (K,F) and (batch,F) as ambiguous - use explicit 3D shape.
	"""
	if z.dim() == 1:
		# Single latent vector -> single sample, single batch
		z = z.unsqueeze(0).unsqueeze(0)
	elif z.dim() == 2:
		# AMBIGUOUS: could be (K,F) samples or (batch,F) batches
		raise ValueError(
			f"z shape {tuple(z.shape)} is ambiguous (could be K samples or batch). "
			f"Use explicit 3D shape (batch,K,F) or 1D (F,) for single sample."
		)
	elif z.dim() == 3:
		pass
	else:
		raise ValueError(f"z must be 1D (F,) or 3D (batch,K,F), got {z.dim()}D: {tuple(z.shape)}")
	batch, K, F = z.shape
	return z, K


def log_pdf_r_given_z(
	alpha: torch.Tensor,
	B: torch.Tensor,
	sigma: torch.Tensor,
	nu: torch.Tensor,
	z: torch.Tensor,
	r: Optional[torch.Tensor] = None,
	mask: Optional[torch.Tensor] = None,
	return_per_asset: bool = False,
	use_fp64: bool = False,
):
	"""Compute Student-t logpdf per asset and joint sums for given z samples.

	Returns:
	- log_p: Tensor shape (batch, K) if multiple z samples, else (batch,)
	- per_asset_logpdf: if return_per_asset True returns tensor (batch, N, K) or (batch, N)
	"""
	device = B.device
	target_dtype = torch.float64 if use_fp64 else torch.get_default_dtype()

	# z -> (batch, K, F)
	z, K = _canonicalize_z(z)
	original_batch = z.shape[0]
	original_K = K
	original_F = z.shape[2]  # Capture F before any reshaping
	z = z.to(device=device, dtype=target_dtype)

	# canonicalize B/alpha/sigma/nu (may reshape from [batch,K,N,F] to [batch*K,N,F])
	alpha, B, sigma, nu = _canonicalize_B_alpha_sigma_nu(alpha, B, sigma, nu)
	
	# If B was reshaped to [batch*K, N, F], also reshape z and r
	if B.shape[0] == original_batch * original_K:
		# Reshape z from [batch, K, F] to [batch*K, 1, F]
		z = z.reshape(original_batch * original_K, 1, original_F)
		K = 1  # Now K dimension is merged into batch
		
		# Reshape r from [batch, K, N] to [batch*K, N]
		if r is not None:
			if r.dim() == 3 and r.shape[1] == original_K:
				r = r.reshape(original_batch * original_K, r.shape[2])

	# cast arrays
	B = B.to(device=device, dtype=target_dtype)
	if alpha is None:
		alpha = torch.zeros((B.shape[0], B.shape[1]), device=device, dtype=target_dtype)
	else:
		alpha = alpha.to(device=device, dtype=target_dtype)
	sigma = sigma.to(device=device, dtype=target_dtype)
	nu = nu.to(device=device, dtype=target_dtype)

	# mask
	if mask is None:
		mask = torch.ones_like(alpha, dtype=torch.bool, device=device)
	else:
		# Reshape mask from [batch, K, N] to [batch*K, N] if needed
		if mask.dim() == 3 and mask.shape[1] == original_K:
			mask = mask.reshape(original_batch * original_K, mask.shape[2])
		mask = mask.to(device=device)

	# r: (batch,N) -> promote
	if r is None:
		raise ValueError("r must be provided for log-likelihood evaluation")
	if r.dim() == 1:
		r = r.unsqueeze(0)
	r = r.to(device=device, dtype=target_dtype)

	batch_size, N, F = B.shape

	# ensure batch dims
	if alpha.dim() == 1:
		alpha = alpha.unsqueeze(0)
	if sigma.dim() == 1:
		sigma = sigma.unsqueeze(0)
	if nu.dim() == 1:
		nu = nu.unsqueeze(0)
	if mask.dim() == 1:
		mask = mask.unsqueeze(0)

	# clamps
	sigma = torch.clamp(sigma, min=_EPS_SIGMA)
	nu = torch.clamp(nu, min=_MIN_NU, max=_MAX_NU)

	# compute loc: alpha + B @ z -> (batch, N, K)
	# B: (batch,N,F); z: (batch,K,F) -> need z_k transposed
	# use einsum for clarity
	# loc[b,n,k] = alpha[b,n] + sum_f B[b,n,f]*z[b,k,f]
	loc = alpha.unsqueeze(-1) + torch.einsum("bnf,bkf->bnk", B, z)

	# expand r,sigma,nu to (batch,N,1)
	r_exp = r.unsqueeze(-1)  # (batch,N,1)
	sigma_exp = sigma.unsqueeze(-1)
	nu_exp = nu.unsqueeze(-1)
	mask_exp = mask.unsqueeze(-1)

	# standardized t
	t = (r_exp - loc) / sigma_exp

	# q = 1 + (t**2)/nu; use log1p for stability on log(q)
	q = 1.0 + (t * t) / nu_exp
	logq = torch.log1p((t * t) / nu_exp)

	# Student-T log-pdf formula
	lgamma = torch.lgamma
	log_term = lgamma((nu_exp + 1.0) / 2.0) - lgamma(nu_exp / 2.0)
	log_pi = math.log(math.pi)
	const_term = -0.5 * (torch.log(nu_exp) + log_pi)
	log_sigma = -torch.log(sigma_exp)
	power = -((nu_exp + 1.0) / 2.0) * logq

	logpdf = log_term + const_term + log_sigma + power

	# mask invalid assets by zeroing their contribution (logpdf->0) when summing
	logpdf = logpdf * mask_exp.to(dtype=target_dtype)

	# sum over assets -> (batch, K)
	joint = torch.sum(logpdf, dim=1)

	# Reshape back from [batch*K, ...] to [batch, K, ...]
	if original_K > 1 and K == 1:
		# We merged batch and K dimensions earlier, reshape back
		joint = joint.squeeze(-1) if joint.shape[-1] == 1 else joint
		joint = joint.reshape(original_batch, original_K)
		if return_per_asset:
			per_asset = logpdf.reshape(original_batch, original_K, logpdf.shape[1])
	# squeeze K dim if K==1 for ease of use
	elif K == 1:
		joint = joint.squeeze(-1)
		per_asset = logpdf.squeeze(-1) if return_per_asset else None
	else:
		per_asset = logpdf if return_per_asset else None

	if return_per_asset:
		return joint, per_asset
	return joint


def log_pdf_multiple_z(*args, **kwargs):
	# convenient alias
	return log_pdf_r_given_z(*args, **kwargs)


def sample_r_given_z(
	alpha: torch.Tensor,
	B: torch.Tensor,
	sigma: torch.Tensor,
	nu: torch.Tensor,
	z: torch.Tensor,
	mask: Optional[torch.Tensor] = None,
	rng: Optional[torch.Generator] = None,
	reparam_mode: Optional[str] = None,
):
	"""Sample r conditional on z. Returns samples with shape (batch,N,K) or (batch,N) if K==1.

	reparam_mode: None (default) or 'normal_approx' to approximate Student-t by normal with same mean and variance.
	"""
	device = B.device

	alpha, B, sigma, nu = _canonicalize_B_alpha_sigma_nu(alpha, B, sigma, nu)
	z, K = _canonicalize_z(z)
	batch, K, F = z.shape

	# cast
	B = B.to(device=device)
	alpha = alpha.to(device=device)
	sigma = torch.clamp(sigma.to(device=device), min=_EPS_SIGMA)
	nu = torch.clamp(nu.to(device=device), min=_MIN_NU, max=_MAX_NU)
	z = z.to(device=device)

	# compute loc
	loc = alpha.unsqueeze(-1) + torch.einsum("bnf,bkf->bnk", B, z)

	# sample u ~ StudentT( df=nu ) standardized per-asset
	# use torch.distributions for clarity and broadcasting
	df = nu.unsqueeze(-1)
	if reparam_mode is None:
		# StudentT sampling
		student = torch.distributions.StudentT(df)
		# sample shape (batch,N,K)
		try:
			u = student.rsample(sample_shape=(K,))
		except Exception:
			u = student.sample(sample_shape=(K,))
		# normalize u to (batch,N,K)
		if u.dim() == 4:
			# (K,batch,N,1)
			u = u.squeeze(-1).permute(1, 2, 0)
		elif u.dim() == 3:
			# (K,batch,N) -> permute
			u = u.permute(1, 2, 0)
		else:
			# Fallback: manual Student-T sampling
			# T = N(0,1) / sqrt(V / nu) where V ~ Gamma(nu/2, 1/2)
			g = torch.randn((batch, B.shape[1], K), device=device)
			v = torch.distributions.Gamma(nu.unsqueeze(-1) / 2.0, 0.5).sample((K,)).permute(1, 2, 0)
			u = g / torch.sqrt(2.0 * v / nu.unsqueeze(-1))
	elif reparam_mode == "normal_approx":
		var_factor = (nu / (nu - 2.0)).unsqueeze(-1)
		sigma_eff = sigma.unsqueeze(-1) * torch.sqrt(var_factor)
		eps = torch.randn((batch, B.shape[1], K), device=device)
		r = loc + sigma_eff * eps
		if K == 1:
			return r.squeeze(-1)
		return r
	else:
		raise ValueError(f"Unknown reparam_mode {reparam_mode}")

	# now u shape (batch,N,K)
	sigma_exp = sigma.unsqueeze(-1)
	r = loc + sigma_exp * u

	if K == 1:
		return r.squeeze(-1)
	return r


def marginal_mean(alpha: torch.Tensor, B: torch.Tensor, mu_z: Optional[torch.Tensor] = None):
	"""Return marginal mean E[r] = alpha + B @ mu_z. Accepts batched or non-batched inputs."""
	if B.dim() == 2:
		B = B.unsqueeze(0)
	batch, N, F = B.shape
	device = B.device
	if mu_z is None:
		mu_z = torch.zeros((F,), device=device)
	if mu_z.dim() == 1:
		mu_z = mu_z.unsqueeze(0)
	if mu_z.dim() == 2 and mu_z.shape[0] == 1 and batch > 1:
		mu_z = mu_z.expand(batch, -1)
	alpha = alpha.to(device=device)
	if alpha.dim() == 1:
		alpha = alpha.unsqueeze(0)
	mu = alpha + torch.einsum("bnf,bf->bn", B, mu_z)
	return mu


def marginal_covariance(B: torch.Tensor, Sigma_z: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
	"""Compute full marginal covariance: Cov[r] = diag(sigma^2) + B Sigma_z B^T.
	
	Args:
		B: Factor exposures (N,F) or (batch,N,F)
		Sigma_z: Prior covariance (F,F) or (batch,F,F)
		sigma: Idiosyncratic scale (N,) or (batch,N)
		
	Returns:
		Cov[r]: (batch,N,N) covariance matrix
	"""
	if B.dim() == 2:
		B = B.unsqueeze(0)
	batch, N, F = B.shape
	device = B.device
	
	if Sigma_z.dim() == 2:
		Sigma_z = Sigma_z.unsqueeze(0).expand(batch, -1, -1)
	if sigma.dim() == 1:
		sigma = sigma.unsqueeze(0)
	
	# Compute B Sigma_z B^T efficiently
	B_Sigma = torch.matmul(B, Sigma_z)  # (batch, N, F)
	factor_cov = torch.matmul(B_Sigma, B.transpose(-2, -1))  # (batch, N, N)
	
	# Add idiosyncratic variance on diagonal
	idio_var = sigma * sigma  # (batch, N)
	diag_idx = torch.arange(N, device=device)
	factor_cov[..., diag_idx, diag_idx] = factor_cov[..., diag_idx, diag_idx] + idio_var
	
	return factor_cov


def marginal_cov_actionable(B: torch.Tensor, Sigma_z: torch.Tensor, sigma: torch.Tensor, w: Optional[torch.Tensor] = None):
	"""Compute portfolio variance w^T Cov[r] w without forming full NxN.

	If `w` provided: returns portfolio variance.
	If `w` is None: returns low-rank factors (B, Sigma_z, sigma) for later use.
	
	Cov[r] = diag(sigma^2) + B Sigma_z B^T
	"""
	if B.dim() == 2:
		B = B.unsqueeze(0)
	batch, N, F = B.shape
	device = B.device
	if Sigma_z.dim() == 2:
		Sigma_z = Sigma_z.unsqueeze(0).expand(batch, -1, -1)
	if sigma.dim() == 1:
		sigma = sigma.unsqueeze(0)

	if w is not None:
		if w.dim() == 1:
			w = w.unsqueeze(0)
		temp = torch.einsum("bnf,bn->bf", B, w)
		var = torch.einsum("bf,bfg,bf->b", temp, Sigma_z, temp)
		var = var + torch.sum((w * w) * (sigma * sigma), dim=1)
		return var

	return B, Sigma_z, sigma


__all__ = [
	"log_pdf_r_given_z",
	"log_pdf_multiple_z",
	"sample_r_given_z",
	"marginal_mean",
	"marginal_covariance",
	"marginal_cov_actionable",
]

