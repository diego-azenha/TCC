import torch
from typing import Optional, Tuple

"""Analytical encoder for q(z|r,F) ≈ N(mu_q, Sigma_q).

Computes posterior using closed-form solution (Section 3.3, Eq 8).
Student-T prior converted to Normal via moment matching before applying formula.

TRAINING ONLY: Not used during inference.

Returns: (mu_q, L_q, Sigma_q, prec) where L_q is Cholesky of Sigma_q
"""

def encoder_recon(
	alpha: torch.Tensor,
	B: torch.Tensor,
	sigma: torch.Tensor,
	r: torch.Tensor,
	mask: Optional[torch.Tensor] = None,
	mu_z: Optional[torch.Tensor] = None,
	Sigma_z: Optional[torch.Tensor] = None,
	nu_z: Optional[torch.Tensor] = None,
	eps: float = 1e-8,
	jitter_init: float = 1e-6,
	jitter_max: float = 1e-1,
	jitter_multiplier: float = 2.0,
	use_fp64: bool = False,
	return_full_cov: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
	"""Compute q(z|r) posterior analytically.
	
	Args:
		nu_z: If provided, applies Student-T to Normal moment matching
		jitter_multiplier: Jitter increase factor (default 2.0, paper uses adaptive)
	"""

	device = B.device
	target_dtype = torch.float64 if use_fp64 else torch.get_default_dtype()
	
	def _ensure_batch(x: Optional[torch.Tensor], expected_ndim: int):
		"""Ensure tensor has batch dimension without double-unsqueezing."""
		if x is None:
			return None
		if x.dim() < expected_ndim - 1:
			raise ValueError(f"Tensor has too few dimensions: {x.dim()} < {expected_ndim - 1}")
		if x.dim() == expected_ndim - 1:
			return x.unsqueeze(0)
		return x

	# Process B first to get dimensions
	if B.dim() == 2:
		B = B.unsqueeze(0)
	elif B.dim() != 3:
		raise ValueError(f"B must be (N,F) or (batch,N,F), got {B.dim()} dims")
	
	batch_size, N, F = B.shape

	# Ensure batch dims (no double-unsqueeze)
	alpha = _ensure_batch(alpha, 2)
	sigma = _ensure_batch(sigma, 2)
	r = _ensure_batch(r, 2)
	if mask is not None:
		mask = _ensure_batch(mask, 2)

	# Cast to target dtype
	alpha = alpha.to(device=device, dtype=target_dtype)
	B = B.to(device=device, dtype=target_dtype)
	sigma = sigma.to(device=device, dtype=target_dtype)
	r = r.to(device=device, dtype=target_dtype)
	

	if mask is None:
		mask = torch.ones_like(alpha, dtype=torch.bool, device=device)
	else:
		mask = mask.to(device=device)

	sigma = torch.clamp(sigma, min=1e-6)

	# inv_sigma with masking by zeroing contributions of invalid assets
	inv_sigma = 1.0 / (sigma * sigma + eps)
	inv_sigma = inv_sigma * mask.to(dtype=target_dtype)

	# Weighted B and W = B^T diag(inv_sigma) B
	B_weighted = inv_sigma.unsqueeze(-1) * B  # (batch,N,F)
	W = torch.matmul(B.transpose(-2, -1), B_weighted)  # (batch,F,F)

	# v = B^T (inv_sigma * (r - alpha))
	resid = r - alpha
	weighted_resid = inv_sigma * resid
	
	v = torch.matmul(B.transpose(-2, -1), weighted_resid.unsqueeze(-1)).squeeze(-1)  # (batch,F)

	# Prior mu_z, Sigma_z handling
	if mu_z is None:
		mu_z = torch.zeros((batch_size, F), dtype=target_dtype, device=device)
	else:
		mu_z = _ensure_batch(mu_z, 2).to(device=device, dtype=target_dtype)

	if Sigma_z is None:
		Sigma_z = torch.eye(F, dtype=target_dtype, device=device).expand(batch_size, F, F).contiguous()
	else:
		# Ensure Sigma_z has correct dimensions and dtype
		if Sigma_z.dim() == 2:
			Sigma_z = Sigma_z.unsqueeze(0).expand(batch_size, F, F).contiguous().to(device=device, dtype=target_dtype)
		elif Sigma_z.dim() == 3:
			# Ensure it's on the right device/dtype and has the right batch size
			Sigma_z = Sigma_z.to(device=device, dtype=target_dtype)
			if Sigma_z.shape[0] == 1 and batch_size > 1:
				Sigma_z = Sigma_z.expand(batch_size, F, F).contiguous()
		else:
			raise ValueError(f"Sigma_z must have 2 or 3 dimensions, got {Sigma_z.dim()}")
	
	# Student-T to Normal moment matching (Section 3.3)
	# For Student-T with df=nu: variance = sigma^2 * nu/(nu-2)
	if nu_z is not None:
		if isinstance(nu_z, (int, float)):
			nu_z = torch.tensor(nu_z, dtype=target_dtype, device=device)
		else:
			nu_z = nu_z.to(device=device, dtype=target_dtype)
		
		if torch.any(nu_z <= 2.0):
			raise ValueError(f"nu_z must be > 2 for finite variance, got min={nu_z.min().item()}")
		
		# Apply variance scaling
		variance_scale = nu_z / (nu_z - 2.0)
		if Sigma_z.dim() == 3:
			Sigma_z = Sigma_z * variance_scale.view(-1, 1, 1)
		else:
			Sigma_z = Sigma_z * variance_scale

	# Precision matrix
	I_F = torch.eye(F, dtype=target_dtype, device=device).expand(batch_size, F, F).contiguous()
	Sigma_z_inv = torch.linalg.solve(Sigma_z, I_F)
	posterior_prec = Sigma_z_inv + W
	posterior_prec = 0.5 * (posterior_prec + posterior_prec.transpose(-2, -1))  # Symmetrize once

	# Cholesky with adaptive jitter
	jitter = torch.tensor(jitter_init, dtype=target_dtype, device=device)
	L = None
	while True:
		try:
			prec_jittered = posterior_prec.clone()
			diag_idx = torch.arange(F, device=device)
			prec_jittered[..., diag_idx, diag_idx] = prec_jittered[..., diag_idx, diag_idx] + jitter
			L_try, info = torch.linalg.cholesky_ex(prec_jittered)
			if isinstance(info, torch.Tensor):
				if (info == 0).all():
					L = L_try
					break
			else:
				if info == 0:
					L = L_try
					break
			jitter = jitter * jitter_multiplier
			if jitter.item() > jitter_max:
				raise RuntimeError(f"Cholesky failed with jitter up to {jitter.item()}")
		except RuntimeError as e:
			if "Cholesky" not in str(e):
				raise
			jitter = jitter * jitter_multiplier
			if jitter.item() > jitter_max:
				raise

	# Compute posterior mean and covariance
	b_term = torch.matmul(Sigma_z_inv, mu_z.unsqueeze(-1)).squeeze(-1) + v
	mu_q = torch.cholesky_solve(b_term.unsqueeze(-1), L).squeeze(-1)
	
	# Compute Sigma_q and its Cholesky
	Sigma_q = torch.cholesky_solve(I_F, L) if return_full_cov or True else None
	if Sigma_q is not None:
		eps_diag = 1e-12
		Sigma_q_j = Sigma_q.clone()
		diag_idx = torch.arange(F, device=device)
		Sigma_q_j[..., diag_idx, diag_idx] = Sigma_q_j[..., diag_idx, diag_idx] + eps_diag
		L_q = torch.linalg.cholesky(Sigma_q_j)
	else:
		L_q = None

	# Cast back if needed (once at end)
	if not use_fp64 and target_dtype != torch.get_default_dtype():
		default_dtype = torch.get_default_dtype()
		mu_q = mu_q.to(dtype=default_dtype)
		L_q = L_q.to(dtype=default_dtype) if L_q is not None else None
		Sigma_q = Sigma_q.to(dtype=default_dtype) if Sigma_q is not None else None
		posterior_prec = posterior_prec.to(dtype=default_dtype)

	if return_full_cov:
		return mu_q, L_q, Sigma_q, posterior_prec
	return mu_q, L_q, None, None


__all__ = ["encoder_recon"]

