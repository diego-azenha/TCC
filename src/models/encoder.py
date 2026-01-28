import torch
from typing import Optional, Tuple

"""Analytical encoder that computes q(z|r,F) ≈ N(mu_q, Sigma_q).

Parameters
- alpha: (N,) or (batch,N)
- B: (N,F) or (batch,N,F)
- sigma: (N,) or (batch,N)  -- interpreted as std/scale
- r: (N,) or (batch,N)
- mask: optional boolean tensor same shape as alpha/r (True=valid)
- mu_z: (F,) or (batch,F) prior mean (defaults to zeros)
- Sigma_z: (F,F) or (batch,F,F) prior covariance (defaults to I)
- eps: small epsilon for inv_sigma
- jitter_init, jitter_max: cholesky jitter loop parameters
- use_fp64: do LA in float64 for stability
- return_full_cov: if True also return full Sigma_q and prec

Returns (mu_q, L_q, Sigma_q_opt, prec_opt)
- mu_q: (batch,F)
- L_q: (batch,F,F) lower-triangular cholesky of Sigma_q
- Sigma_q_opt: (batch,F,F) if return_full_cov else None
- prec_opt: (batch,F,F) precision (Sigma_q^{-1}) if return_full_cov else None
"""

def encoder_recon(
	alpha: torch.Tensor,
	B: torch.Tensor,
	sigma: torch.Tensor,
	r: torch.Tensor,
	mask: Optional[torch.Tensor] = None,
	mu_z: Optional[torch.Tensor] = None,
	Sigma_z: Optional[torch.Tensor] = None,
	eps: float = 1e-8,
	jitter_init: float = 1e-6,
	jitter_max: float = 1e-1,
	use_fp64: bool = False,
	return_full_cov: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:

	# --- helper: make tensors and batch dims consistent ---
	device = B.device
	# promote 1D -> batch dim
	def _promote_batch(x: Optional[torch.Tensor], expected_ndim: int):
		if x is None:
			return None
		if x.dim() == expected_ndim - 1:
			return x.unsqueeze(0)
		return x

	# Ensure minimal dtypes
	target_dtype = torch.float64 if use_fp64 else torch.get_default_dtype()

	alpha = _promote_batch(alpha, 2)
	sigma = _promote_batch(sigma, 2)
	r = _promote_batch(r, 2)
	if mask is not None:
		mask = _promote_batch(mask, 2)

	# B may be (N,F) or (batch,N,F)
	if B.dim() == 2:
		B = B.unsqueeze(0)

	# Now shapes: (batch, N, F), (batch, N)
	batch_size, N, F = B.shape

	# cast
	alpha = alpha.to(device=device, dtype=target_dtype)
	B = B.to(device=device, dtype=target_dtype)
	sigma = sigma.to(device=device, dtype=target_dtype)
	r = r.to(device=device, dtype=target_dtype)
	if mask is None:
		mask = torch.ones_like(alpha, dtype=torch.bool, device=device)
	else:
		mask = mask.to(device=device)

	# Broadcast mask/sigma/alpha/r to (batch,N)
	if alpha.dim() == 1:
		alpha = alpha.unsqueeze(0)
	if sigma.dim() == 1:
		sigma = sigma.unsqueeze(0)
	if r.dim() == 1:
		r = r.unsqueeze(0)
	if mask.dim() == 1:
		mask = mask.unsqueeze(0)

	# clamp sigma to avoid division by zero
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

	# Prior mu_z, Sigma_z handling and inversion via solve
	if mu_z is None:
		mu_z = torch.zeros((batch_size, F), dtype=target_dtype, device=device)
	else:
		mu_z = _promote_batch(mu_z, 2).to(device=device, dtype=target_dtype)
		if mu_z.dim() == 1:
			mu_z = mu_z.unsqueeze(0)

	if Sigma_z is None:
		Sigma_z = torch.eye(F, dtype=target_dtype, device=device).expand(batch_size, F, F).contiguous()
	else:
		# allow (F,F) or (batch,F,F)
		if Sigma_z.dim() == 2:
			Sigma_z = Sigma_z.unsqueeze(0).expand(batch_size, F, F).contiguous()
		else:
			Sigma_z = Sigma_z.to(device=device, dtype=target_dtype)

	# Compute Sigma_z^{-1} via solve for numerical stability
	I_F = torch.eye(F, dtype=target_dtype, device=device).expand(batch_size, F, F).contiguous()
	# torch.linalg.solve supports batched input
	Sigma_z_inv = torch.linalg.solve(Sigma_z, I_F)

	# precision and cholesky with jitter adaptive
	prec = Sigma_z_inv + W

	# ensure symmetry
	prec = 0.5 * (prec + prec.transpose(-2, -1))

	jitter = torch.tensor(jitter_init, dtype=target_dtype, device=device)
	L = None
	info = None
	# loop with increasing jitter until cholesky succeeds or jitter exceeds max
	while True:
		try:
			# add jitter to diagonal
			prec_j = prec.clone()
			diag_idx = torch.arange(F, device=device)
			prec_j[..., diag_idx, diag_idx] = prec_j[..., diag_idx, diag_idx] + jitter
			# try cholesky
			L_try, info = torch.linalg.cholesky_ex(prec_j)
			# info is 0 for success per-batch
			if isinstance(info, torch.Tensor):
				if (info == 0).all():
					L = L_try
					break
			else:
				# info scalar
				if info == 0:
					L = L_try
					break
			# otherwise increase jitter
			jitter = jitter * 10.0
			if jitter.item() > jitter_max:
				raise RuntimeError(f"Cholesky failed for precision even after jitter up to {jitter.item()}")
		except RuntimeError:
			jitter = jitter * 10.0
			if jitter.item() > jitter_max:
				raise

	# compute mu_q without inverting: solve prec @ mu_q = (Sigma_z^{-1} @ mu_z + v)
	b_term = torch.matmul(Sigma_z_inv, mu_z.unsqueeze(-1)).squeeze(-1) + v  # (batch,F)
	# use cholesky_solve to compute mu_q = prec^{-1} @ b_term
	mu_q = torch.cholesky_solve(b_term.unsqueeze(-1), L).squeeze(-1)

	# compute Sigma_q optionally and its cholesky L_q
	Sigma_q = torch.cholesky_solve(I_F, L) if return_full_cov or True else None
	# ensure symmetry
	if Sigma_q is not None:
		Sigma_q = 0.5 * (Sigma_q + Sigma_q.transpose(-2, -1))
	# compute L_q (cholesky of Sigma_q) — useful for reparam
	# add tiny jitter to Sigma_q diagonal for safety
	eps_diag = 1e-12
	Sigma_q_j = Sigma_q.clone()
	diag_idx = torch.arange(F, device=device)
	Sigma_q_j[..., diag_idx, diag_idx] = Sigma_q_j[..., diag_idx, diag_idx] + eps_diag
	L_q = torch.linalg.cholesky(Sigma_q_j)

	# Return precision as well if requested
	prec_out = prec

	# Cast back to float32 if not using fp64 and original inputs were float32
	if not use_fp64:
		mu_q = mu_q.to(dtype=torch.get_default_dtype())
		L_q = L_q.to(dtype=torch.get_default_dtype())
		Sigma_q = Sigma_q.to(dtype=torch.get_default_dtype()) if Sigma_q is not None else None
		prec_out = prec_out.to(dtype=torch.get_default_dtype())

	if return_full_cov:
		return mu_q, L_q, Sigma_q, prec_out
	return mu_q, L_q, None, None


__all__ = ["encoder_recon"]

