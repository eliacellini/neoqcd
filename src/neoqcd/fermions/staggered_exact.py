"""Exact (dense-determinant) tools for one staggered field (N_f = 2^(D/2)) on tiny lattices.

Conventions are those of neoqcd.fermions.staggered; dense fields are flattened as
(site, colour) in C order, with the sites of a parity block in lexicographic order (the order of
`staggered_dense_matrix` on a `StaggeredEvenOdd`).

- `assemble_staggered_hopping`, `assemble_staggered_even_odd`: K and M_ee = m^2 - K_eo K_oe
  built directly from the links, batched over B (cross-checked against `staggered_dense_matrix`).
- `log_det_staggered`: log det M_ee = log det D (> 0), the log weight of N_f = 2^(D/2) flavours.
- `ExactStaggeredMetropolis`: random-walk Metropolis for
      weight = det M_ee exp(-S_g) = exp(-S_eff),   S_eff = S_g - log det M_ee,
  with exact dense determinants.
"""

from types import SimpleNamespace

import torch

from neoqcd.fermions._gauge import SUN_dagger, SUN_mul, Wilson_action
from neoqcd.fermions.algebra import expi, random_algebra
from neoqcd.fermions.exact import _site_tables
from neoqcd.fermions.lattice import fermion_boundary_links
from neoqcd.fermions.staggered import StaggeredDirac, _mass_tensor, staggered_dense_matrix, staggered_phases

__all__ = ["assemble_staggered_hopping", "assemble_staggered_even_odd", "log_det_staggered",
           "ExactStaggeredMetropolis"]


def assemble_staggered_hopping(U, geometry, rows=None, cols=None, time_sign=-1.0):
    """Dense staggered hopping matrix K (batched) restricted to row sites `rows`, column sites `cols`.

    K[x, x + mu] += +1/2 eta_mu(x) L_mu(x),   K[x, x - mu] += -1/2 eta_mu(x) L_mu(x - mu)^dagger,
    with L the links including the fermion boundary sign. Returns (B, len(rows) N, len(cols) N).
    """
    device = U.device
    B, D, N, V = U.shape[0], geometry.D, geometry.N, geometry.volume
    rows = torch.arange(V, device=device) if rows is None else rows
    cols = torch.arange(V, device=device) if cols is None else cols
    col_pos = torch.full((V,), -1, dtype=torch.long, device=device)
    col_pos[cols] = torch.arange(cols.numel(), device=device)
    fwd, bwd, _ = _site_tables(geometry, device)
    links = fermion_boundary_links(U, time_sign).reshape(B, D, V, N, N)
    eta = staggered_phases(geometry, device=device, dtype=U.real.dtype).reshape(D, V)
    R, Cn = rows.numel(), cols.numel()
    Kb = torch.zeros((R, Cn, B, N, N), dtype=U.dtype, device=device)
    r_index = torch.arange(R, device=device)
    for mu in range(D):
        e = eta[mu, rows].to(U.dtype)[:, None, None, None]  # (R, 1, 1, 1)
        pieces = ((fwd[mu][rows], 0.5 * e * links[:, mu, rows].transpose(0, 1)),
                  (bwd[mu][rows], -0.5 * e * SUN_dagger(links[:, mu, bwd[mu][rows]]).transpose(0, 1)))
        for neighbour, block in pieces:
            pos = col_pos[neighbour]
            keep = pos >= 0
            if not bool(keep.any()):
                continue
            Kb.index_put_((r_index[keep], pos[keep]), block[keep], accumulate=True)
    return Kb.permute(2, 0, 3, 1, 4).reshape(B, R * N, Cn * N)


def assemble_staggered_even_odd(U, mass, geometry, time_sign=-1.0):
    """Dense M_ee = m^2 - K_eo K_oe, batched, shape (B, n, n), n = V/2 * N, even sites in
    lexicographic order (as `staggered_dense_matrix(StaggeredEvenOdd)`)."""
    _, _, par = _site_tables(geometry, U.device)
    even = (par == 0).nonzero().squeeze(-1)
    odd = (par == 1).nonzero().squeeze(-1)
    K_eo = assemble_staggered_hopping(U, geometry, even, odd, time_sign)
    K_oe = assemble_staggered_hopping(U, geometry, odd, even, time_sign)
    m = _mass_tensor(mass, U.shape[0], U.device).to(U.dtype).view(-1, 1, 1)
    eye = torch.eye(K_eo.shape[1], dtype=U.dtype, device=U.device)
    return m ** 2 * eye - torch.matmul(K_eo, K_oe)


def log_det_staggered(U, mass, geometry, time_sign=-1.0, method="assembled"):
    """log det M_ee = log det D per gauge field, shape (B,), float64.

    method "assembled" (batched, used by the sampler) or "dense_matrix" (`staggered_dense_matrix`
    of the sparse `StaggeredEvenOdd`, one batch entry at a time, the reference). Raises if a
    determinant is not positive (M_ee is hermitian positive definite for m > 0).
    """
    if method == "assembled":
        sign, ld = torch.linalg.slogdet(assemble_staggered_even_odd(U, mass, geometry, time_sign))
    elif method == "dense_matrix":
        eo = StaggeredDirac(U, mass, geometry, time_sign).even_odd()
        pairs = [torch.linalg.slogdet(staggered_dense_matrix(eo, geometry, batch_index=b)) for b in range(U.shape[0])]
        sign = torch.stack([p[0] for p in pairs])
        ld = torch.stack([p[1] for p in pairs])
    else:
        raise ValueError(f"unknown method {method!r}")
    if bool(((sign - 1.0).abs() > 1e-8).any()):
        raise RuntimeError(f"det M_ee is not positive: sign {sign}")
    return ld


class ExactStaggeredMetropolis:
    """Exact random-walk Metropolis for one staggered field (N_f = 2^(D/2)) plus Wilson gauge action.

    Target exp(-S_eff), S_eff = S_g - log det M_ee, S_g = `neoqcd.theory.Wilson_action`. Proposal
    and acceptance as `neoqcd.fermions.exact.ExactMetropolis`: `links_per_update` distinct links
    chosen uniformly, each multiplied by exp(i eps P), P Gaussian hermitian traceless (symmetric,
    Haar-invariant), accepted with min(1, exp(-dS_eff)); dense determinant at every proposal.

    Batched over B chains; `beta`, `mass` scalars or (B,) tensors. Usage: `reset(U)`, then `step`.
    """

    def __init__(self, beta, mass, geometry, time_sign=-1.0, eps=0.5, links_per_update=1):
        self.beta = beta
        self.mass = mass
        self.geometry = geometry
        self.time_sign = float(time_sign)
        self.eps = float(eps)
        self.links_per_update = int(links_per_update)
        pars = SimpleNamespace(D=geometry.D, N=geometry.N, device=torch.device("cpu"))
        self._gauge = Wilson_action(pars, beta)
        self.U = None
        self.n_proposals = 0
        self.n_accepted = None

    def gauge_action(self, U):
        return self._gauge(U, None)

    def log_det(self, U):
        return log_det_staggered(U, self.mass, self.geometry, self.time_sign)

    def effective_action(self, U):
        """S_eff = S_g - log det M_ee, shape (B,)."""
        return self.gauge_action(U) - self.log_det(U)

    def reset(self, U):
        self.U = U
        self.S_g = self.gauge_action(U)
        self.log_det_value = self.log_det(U)
        self.n_proposals = 0
        self.n_accepted = torch.zeros(U.shape[0], dtype=torch.long, device=U.device)

    @property
    def acceptance(self):
        return (self.n_accepted.double().mean() / max(self.n_proposals, 1)).item()

    @torch.no_grad()
    def step(self, generator=None):
        """One proposal per chain; returns the (B,) accepted mask."""
        U = self.U
        B, D, N = U.shape[0], self.geometry.D, self.geometry.N
        n_links = D * self.geometry.volume
        k = self.links_per_update
        choice = torch.rand((B, n_links), generator=generator, dtype=torch.float64, device=U.device)
        picked = torch.topk(choice, k, dim=1).indices
        P = random_algebra((B, k, N, N), generator=generator, dtype=U.dtype, device=U.device)
        flat = U.reshape(B, n_links, N, N)
        index = picked[..., None, None].expand(B, k, N, N)
        moved = SUN_mul(expi(self.eps * P), torch.gather(flat, 1, index))
        U_new = flat.scatter(1, index, moved).reshape(U.shape)
        S_g_new = self.gauge_action(U_new)
        ld_new = self.log_det(U_new)
        dS = (S_g_new - self.S_g) - (ld_new - self.log_det_value)
        r = torch.rand(B, generator=generator, dtype=dS.dtype, device=U.device)
        acc = torch.log(r) < -dS
        view = (B,) + (1,) * (U.dim() - 1)
        self.U = torch.where(acc.reshape(view), U_new, U)
        self.S_g = torch.where(acc, S_g_new, self.S_g)
        self.log_det_value = torch.where(acc, ld_new, self.log_det_value)
        self.n_proposals += 1
        self.n_accepted += acc.long()
        return acc
