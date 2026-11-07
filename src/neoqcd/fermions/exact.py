"""Exact (dense-determinant) tools for N_f = 2 Wilson fermions on tiny lattices.

Only for lattices where the even-odd matrix fits comfortably in memory (2^3 x 4 SU(3): 192 x 192;
2D 4 x 4 SU(3): 48 x 48). Conventions are those of neoqcd.fermions.dirac; the dense flattening
is (site, spin, colour) in C order with the sites of a parity block in lexicographic order.

- `log_det_two_flavour`: log det(D^dagger D) = 2 log|det D_hat| from `dense_matrix` of the
  sparse even-odd operator, one gauge field at a time.
- `assemble_hopping`, `assemble_even_odd`: the same matrices built directly from the links,
  batched over B, independently of the sparse operator (cross-checked in the tests).
- `ExactMetropolis`: a random-walk Metropolis for the full theory
      weight = det(D^dagger D) exp(-S_g) = exp(-S_eff),   S_eff = S_g - log det(D^dagger D),
  with the exact S_eff difference from dense determinants.
"""

from types import SimpleNamespace

import torch

from neoqcd.fermions._gauge import SUN_dagger, SUN_mul, Wilson_action
from neoqcd.fermions.algebra import expi, random_algebra
from neoqcd.fermions.dirac import WilsonDirac, _kappa_tensor, dense_matrix, gamma_matrices
from neoqcd.fermions.lattice import fermion_boundary_links, parity

__all__ = ["log_det_two_flavour", "assemble_hopping", "assemble_even_odd", "ExactMetropolis"]


def log_det_two_flavour(U, kappa, geometry, time_sign=-1.0, method="dense_matrix"):
    """log det(D^dagger D) = 2 log|det D_hat| per gauge field, shape (B,), float64.

    method "dense_matrix": `dense_matrix` of the sparse `EvenOddWilsonDirac`, one batch entry at
    a time (the reference); "assembled": `assemble_even_odd`, batched (fast, used by the sampler).
    """
    if method == "assembled":
        return 2.0 * torch.linalg.slogdet(assemble_even_odd(U, kappa, geometry, time_sign))[1]
    if method != "dense_matrix":
        raise ValueError(f"unknown method {method!r}")
    eo = WilsonDirac(U, kappa, geometry, time_sign).even_odd()
    out = [torch.linalg.slogdet(dense_matrix(eo, geometry, batch_index=b))[1] for b in range(U.shape[0])]
    return 2.0 * torch.stack(out)


def _site_tables(geometry, device):
    """Flat lexicographic site index, forward/backward neighbours per direction, parity."""
    V = geometry.volume
    idx = torch.arange(V, device=device).reshape(geometry.lattice)
    # "x + mu" is array index n - 1 along the axis of mu: roll by +1 brings f[n - 1] to n
    fwd = [torch.roll(idx, 1, dims=mu).reshape(-1) for mu in range(geometry.D)]
    bwd = [torch.roll(idx, -1, dims=mu).reshape(-1) for mu in range(geometry.D)]
    par = parity(geometry, device=device).reshape(-1)
    return fwd, bwd, par


def assemble_hopping(U, geometry, rows=None, cols=None, time_sign=-1.0):
    """Dense hopping matrix H (batched) restricted to row sites `rows` and column sites `cols`.

    H[x, x + mu] += (1 - gamma_mu) (x) L_mu(x),   H[x, x - mu] += (1 + gamma_mu) (x) L_mu(x - mu)^dagger,
    with L the links including the fermion boundary sign and (x) the Kronecker product
    (spin, colour). `rows`, `cols` are 1D tensors of flat site indices (default: all sites, in
    lexicographic order). Returns (B, len(rows) * S * N, len(cols) * S * N).
    """
    device = U.device
    B, D = U.shape[0], geometry.D
    S, N = geometry.spin, geometry.N
    SN = S * N
    V = geometry.volume
    rows = torch.arange(V, device=device) if rows is None else rows
    cols = torch.arange(V, device=device) if cols is None else cols
    col_pos = torch.full((V,), -1, dtype=torch.long, device=device)
    col_pos[cols] = torch.arange(cols.numel(), device=device)
    fwd, bwd, _ = _site_tables(geometry, device)
    links = fermion_boundary_links(U, time_sign).reshape(B, D, V, N, N)
    gammas, _ = gamma_matrices(D, dtype=U.dtype, device=device)
    one = torch.eye(S, dtype=U.dtype, device=device)
    R, C = rows.numel(), cols.numel()
    Hb = torch.zeros((R, C, B, SN, SN), dtype=U.dtype, device=device)
    r_index = torch.arange(R, device=device)
    for mu in range(D):
        pieces = ((fwd[mu][rows], one - gammas[mu], links[:, mu, rows]),
                  (bwd[mu][rows], one + gammas[mu], SUN_dagger(links[:, mu, bwd[mu][rows]])))
        for neighbour, spin, colour in pieces:
            pos = col_pos[neighbour]
            keep = pos >= 0
            if not bool(keep.any()):
                continue
            block = torch.einsum("ij,brkl->rbikjl", spin, colour[:, keep]).reshape(-1, B, SN, SN)
            Hb.index_put_((r_index[keep], pos[keep]), block, accumulate=True)
    return Hb.permute(2, 0, 3, 1, 4).reshape(B, R * SN, C * SN)


def assemble_even_odd(U, kappa, geometry, time_sign=-1.0):
    """Dense D_hat = 1 - kappa^2 H_eo H_oe, batched, shape (B, n, n) with n = V/2 * S * N.

    Even sites in lexicographic order, as `dense_matrix(EvenOddWilsonDirac)`.
    """
    _, _, par = _site_tables(geometry, U.device)
    even = (par == 0).nonzero().squeeze(-1)
    odd = (par == 1).nonzero().squeeze(-1)
    H_eo = assemble_hopping(U, geometry, even, odd, time_sign)
    H_oe = assemble_hopping(U, geometry, odd, even, time_sign)
    k = _kappa_tensor(kappa, U.shape[0], U.device).to(U.dtype).view(-1, 1, 1)
    eye = torch.eye(H_eo.shape[1], dtype=U.dtype, device=U.device)
    return eye - k ** 2 * torch.matmul(H_eo, H_oe)


class ExactMetropolis:
    """Exact random-walk Metropolis for N_f = 2 Wilson fermions plus Wilson gauge action.

    Target: exp(-S_eff), S_eff = S_g - log det(D^dagger D) (det(D^dagger D) = |det D_hat|^2 > 0).
    S_g is `neoqcd.theory.Wilson_action` (no defect). Proposal: in each chain pick
    `links_per_update` distinct links uniformly and multiply each by exp(i eps P), P Gaussian
    hermitian traceless (density ~ exp(-tr P^2 / 2)). The proposal is symmetric (P and -P are
    equally likely, the link choice is uniform) and left multiplication preserves the Haar
    measure, so accepting with min(1, exp(-dS_eff)) has the target as stationary distribution.
    The determinant is recomputed densely (`assemble_even_odd`, slogdet) at every proposal.

    Batched over B independent chains; `beta`, `kappa` scalars or (B,) tensors.
    Usage: `m = ExactMetropolis(...); m.reset(U); for ...: m.step(generator)`; the current
    links are `m.U`, `m.log_det` and `m.S_g` the cached terms of S_eff.
    """

    def __init__(self, beta, kappa, geometry, time_sign=-1.0, eps=0.5, links_per_update=1):
        self.beta = beta
        self.kappa = kappa
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
        return log_det_two_flavour(U, self.kappa, self.geometry, self.time_sign, method="assembled")

    def effective_action(self, U):
        """S_eff = S_g - log det(D^dagger D), shape (B,)."""
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
        B, D = U.shape[0], self.geometry.D
        N = self.geometry.N
        n_links = D * self.geometry.volume
        k = self.links_per_update
        choice = torch.rand((B, n_links), generator=generator, dtype=torch.float64, device=U.device)
        picked = torch.topk(choice, k, dim=1).indices  # k distinct links per chain, uniformly
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
