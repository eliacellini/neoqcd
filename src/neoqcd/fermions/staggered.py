"""Staggered (Kogut-Susskind) Dirac operator, even-odd normal operator and free-field spectrum.

Conventions (fixed here, checked in test/test_fermions_staggered.py):

- Fields. Links `U` (B, D, T, L, ..., L, N, N) as in neoqcd.sun_utils; a staggered field `chi`
  has shape (B, T, L, ..., L, N), one colour vector per site, complex128. "x + mu" is
  `torch.roll(., +1)` along the axis of mu (neoqcd.fermions.lattice), i.e. array index n - 1.
- Phases. eta_mu(n) = (-1)^(n_0 + ... + n_{mu-1}) in ARRAY coordinates n, direction 0 = time
  first, so eta_0 = 1. Reading the lattice in the code's "physical" orientation (x = -n mod L)
  gives the same phases, because (-1)^(-n) = (-1)^n: the convention does not depend on the
  direction in which the code runs. They satisfy eta_mu^2 = 1, eta_mu(x +- mu) = eta_mu(x) and,
  for mu != nu, eta_mu(x) eta_nu(x + mu) = -eta_nu(x) eta_mu(x + nu) (needs even extents);
  the last relation is what makes D^dagger D a nearest-neighbour-squared (taste-diagonal)
  operator. Other common choices (ordering nu > mu, an overall sign of the hopping term, a
  staggered sign eps(x) = (-1)^(sum n) on one side) are unitarily equivalent: they differ by a
  site-diagonal sign matrix or by K -> -K = eps K eps, so spectra and determinants agree.
- Operator.
      D = m + K,
      (K chi)(x) = 1/2 sum_mu eta_mu(x) [ U_mu(x) chi(x + mu) - U_mu(x - mu)^dagger chi(x - mu) ].
  K is anti-hermitian and, with even extents, connects only sites of opposite parity, so
  eps K eps = -K with eps(x) = (-1)^(sum_mu n_mu); hence D^dagger = m - K = eps D eps = 2m - D
  and D^dagger D = D D^dagger = m^2 - K^2, which is block diagonal in parity.
- Boundary conditions: periodic in space; in time `time_sign` (-1 antiperiodic, default, +1
  periodic) multiplies the time links at array index t = 0 (`fermion_boundary_links`).
- Even-odd. With (even, odd) block order D = [[m, K_eo], [K_oe, m]] and
      M_ee = (D^dagger D)_ee = m^2 - K_eo K_oe,   M_oo = m^2 - K_oe K_eo.
  K_eo K_oe and K_oe K_eo have the same spectrum (square blocks of equal size V/2), so
  det M_ee = det M_oo and det(D^dagger D) = det(M_ee)^2; also det D = det M_ee (Schur
  complement on the odd block, m != 0). D x = b is solved through the Schur system
      M_ee x_e = m b_e - K_eo b_o          (`prepare_rhs`; this is (D^dagger b)_e),
      x_o = (b_o - K_oe x_e) / m           (`reconstruct`),
  which is x = D^dagger (D D^dagger)^-1 b restricted to the even sites (D^dagger commutes with
  D D^dagger = D^dagger D) followed by the odd rows of D x = b. M_ee is hermitian positive
  definite for m > 0: CG runs on `StaggeredEvenOdd.apply` directly (`solve_staggered`).
- Free spectrum (U = 1). On a plane wave chi(n) = exp(-i p.n) u, chi(x + mu) = e^{i p_mu} chi(x),
  so the translation (T_mu chi)(x) = chi(x + mu) has eigenvalue e^{i p_mu} and
  Delta_mu = T_mu - T_mu^-1 has eigenvalue 2 i sin p_mu. Write K = 1/2 sum_mu eta_mu Delta_mu.
  Because eta_nu(x +- mu) = -eta_nu(x) for mu < nu and +eta_nu(x) for mu > nu, the cross
  terms of K^2 cancel pairwise and the diagonal ones give eta_mu Delta_mu eta_mu Delta_mu =
  Delta_mu^2, so K^2 = 1/4 sum_mu Delta_mu^2 is translation invariant with unit spacing:
      D^dagger D chi_p = (m^2 + s(p)) chi_p,   s(p) = sum_mu sin^2 p_mu,
  exactly, for every momentum p of the full finite-lattice set (p_k = 2 pi n_k / L, p_0 =
  (2 n_0 + 1) pi / T antiperiodic or 2 pi n_0 / T periodic), each with multiplicity N (colour).
  Spectrum of D itself: D is normal with eigenvalues m + i lambda, lambda real, lambda^2 an
  eigenvalue of -K^2. On each eigenspace E_s of -K^2 with s > 0, eps maps the +i sqrt(s)
  eigenvectors of K to the -i sqrt(s) ones (eps K eps = -K), so they split evenly; for s = 0
  K vanishes on E_s (K normal, K^2 v = 0). Since sin^2 is invariant under p_mu -> p_mu + pi and
  the momentum set is closed under it (even extents, both boundary conditions), every E_s is a
  union of 2^D-tuples {p + pi A, A subset of directions}. Hence, with the reduced zone
  0 <= p_mu < pi (V / 2^D momenta), the eigenvalues of D are
      m + i sqrt(s(p)) and m - i sqrt(s(p)), each with multiplicity 2^(D-1) N,  p in the reduced zone.
  This is the familiar statement that the 2^D corners of the Brillouin zone are the
  2^(D/2) tastes x 2^(D/2) spin components of a naive fermion on the blocked lattice.
- Dense matrices: a field (lattice, N) is flattened in C order, index = site * N + c, with
  the same lexicographic site order as in neoqcd.fermions.dirac.
"""

import math

import torch

from neoqcd.fermions.dirac import free_momenta
from neoqcd.fermions.lattice import Geometry, even_odd_masks, fermion_boundary_links, parity, shift
from neoqcd.fermions.solver import SolverResult, cg

__all__ = [
    "staggered_phases",
    "staggered_epsilon",
    "StaggeredDirac",
    "StaggeredEvenOdd",
    "staggered_dense_matrix",
    "solve_staggered",
    "free_staggered_normal_eigenvalues",
    "free_staggered_eigenvalues",
]


def staggered_phases(geometry, device=None, dtype=torch.float64):
    """eta_mu(n) = (-1)^(n_0 + ... + n_{mu-1}) in array coordinates, shape (D,) + lattice.

    Direction 0 is time, so eta_0 = 1. See the module docstring for why other sign conventions
    are unitarily equivalent.
    """
    grids = torch.meshgrid(*[torch.arange(n, device=device) for n in geometry.lattice], indexing="ij")
    out = []
    partial = torch.zeros(geometry.lattice, dtype=torch.long, device=device)
    for mu in range(geometry.D):
        out.append(1 - 2 * (partial % 2))
        partial = partial + grids[mu]
    return torch.stack(out).to(dtype)


def staggered_epsilon(geometry, device=None, dtype=torch.float64):
    """eps(n) = (-1)^(sum_mu n_mu), shape `geometry.lattice`; eps D eps = D^dagger (even extents)."""
    return (1 - 2 * parity(geometry, device=device)).to(dtype)


def _mass_tensor(mass, batch_size, device):
    """mass as a float64 tensor of shape (B,) (or (1,) for a scalar, broadcast over B)."""
    m = torch.as_tensor(mass, dtype=torch.float64, device=device)
    if m.dim() == 0:
        m = m.reshape(1)
    if m.dim() != 1 or m.shape[0] not in (1, batch_size):
        raise ValueError(f"mass must be a scalar or have shape ({batch_size},), got {tuple(m.shape)}")
    return m


def _colour_mul(V, chi):
    """V chi on the colour index: V (..., N, N), chi (..., N), broadcasting the batch."""
    return torch.matmul(V, chi.unsqueeze(-1)).squeeze(-1)


class StaggeredDirac:
    """Staggered operator D = m + K on a batch of gauge fields (conventions: module docstring).

    Args:
        U: links (B, D, lattice, N, N); entry b of the batch is an independent gauge field.
        mass: bare mass m, a float or a tensor of shape (B,).
        geometry: `Geometry`; inferred from `U` if omitted.
        time_sign: fermion boundary phase in time, -1 antiperiodic (default), +1 periodic.

    Fields have shape (B, lattice, N); a field with batch size B' is accepted when the links have
    B = 1 (one gauge field applied to many vectors) or B' = B.
    """

    def __init__(self, U, mass, geometry=None, time_sign=-1.0):
        if geometry is None:
            D = U.shape[1]
            geometry = Geometry(D, U.shape[2], U.shape[3] if D > 1 else 1, U.shape[-1])
        if tuple(U.shape[1:]) != geometry.link_shape(1)[1:]:
            raise ValueError(f"links of shape {tuple(U.shape)} do not match geometry {geometry}")
        self.geometry = geometry
        self.U = U
        self.time_sign = float(time_sign)
        self.mass = _mass_tensor(mass, U.shape[0], U.device)
        # eta_mu(x) as (D, lattice, 1), broadcast against (B, lattice, N)
        self.eta = staggered_phases(geometry, device=U.device, dtype=U.real.dtype).unsqueeze(-1)
        links = fermion_boundary_links(U, self.time_sign)
        self._fwd = [links[:, mu] for mu in range(geometry.D)]
        self._bwd = [links[:, mu].conj().transpose(-1, -2) for mu in range(geometry.D)]

    # ---- protocol attributes -------------------------------------------------------------
    @property
    def batch_size(self):
        return self.U.shape[0]

    @property
    def dtype(self):
        return self.U.dtype

    @property
    def device(self):
        return self.U.device

    @property
    def field_shape(self):
        return self.geometry.staggered_shape(self.batch_size)

    def batch_select(self, index):
        """The operator of the single gauge field `index`, with links of batch size 1."""
        m = self.mass if self.mass.shape[0] == 1 else self.mass[index:index + 1]
        return StaggeredDirac(self.U[index:index + 1], m, self.geometry, self.time_sign)

    def _mass_view(self, chi):
        return self.mass.view((-1,) + (1,) * (chi.dim() - 1))

    # ---- operator ------------------------------------------------------------------------
    def epsilon(self, chi):
        """eps(x) chi(x), eps = (-1)^(sum n)."""
        eps = staggered_epsilon(self.geometry, device=chi.device, dtype=self.eta.dtype)
        return eps.unsqueeze(-1) * chi

    def hopping(self, chi):
        """K chi = 1/2 sum_mu eta_mu(x) [U_mu(x) chi(x+mu) - U_mu(x-mu)^dag chi(x-mu)] (anti-hermitian)."""
        out = None
        for mu in range(self.geometry.D):
            fwd = _colour_mul(self._fwd[mu], shift(chi, mu, 1))
            bwd = shift(_colour_mul(self._bwd[mu], chi), mu, -1)
            term = self.eta[mu] * (fwd - bwd)
            out = term if out is None else out + term
        return 0.5 * out

    def apply(self, chi):
        """D chi = m chi + K chi."""
        return self._mass_view(chi) * chi + self.hopping(chi)

    def apply_dagger(self, chi):
        """D^dagger chi = m chi - K chi (= 2m - D)."""
        return self._mass_view(chi) * chi - self.hopping(chi)

    def normal(self, chi):
        """D^dagger D chi = m^2 chi - K^2 chi."""
        return self.apply_dagger(self.apply(chi))

    def even_odd(self):
        """The even-site normal operator M_ee = m^2 - K_eo K_oe built on this one."""
        return StaggeredEvenOdd(self)


class StaggeredEvenOdd:
    """Even-site block M_ee = (D^dagger D)_ee = m^2 - K_eo K_oe of the staggered normal operator.

    Hermitian, positive definite for m > 0. Fields are full-lattice tensors (B, lattice, N);
    `apply` reads only the even sites of its input and returns a field supported on even sites.
    `apply_dagger` is `apply`; `normal` is M_ee^2 (protocol completeness: CG should run on
    `apply`, as `solve_staggered` does). `support` is the even site mask used by
    `staggered_dense_matrix`. Needs even extents.
    """

    def __init__(self, staggered):
        self.staggered = staggered
        self.geometry = staggered.geometry
        even, odd = even_odd_masks(self.geometry, device=staggered.device, dtype=staggered.U.real.dtype)
        shape = self.geometry.lattice + (1,)
        self.even_mask = even.reshape(shape)
        self.odd_mask = odd.reshape(shape)
        self.support = self.even_mask.reshape(self.geometry.lattice) > 0

    @property
    def mass(self):
        return self.staggered.mass

    @property
    def batch_size(self):
        return self.staggered.batch_size

    @property
    def dtype(self):
        return self.staggered.dtype

    @property
    def device(self):
        return self.staggered.device

    @property
    def field_shape(self):
        return self.staggered.field_shape

    def batch_select(self, index):
        return StaggeredEvenOdd(self.staggered.batch_select(index))

    def even(self, chi):
        return self.even_mask * chi

    def odd(self, chi):
        return self.odd_mask * chi

    def apply(self, chi):
        """M_ee chi_e = m^2 chi_e - K K chi_e, on even sites."""
        chi_e = self.even(chi)
        m2 = self.staggered._mass_view(chi) ** 2
        hop = self.staggered.hopping
        return self.even(m2 * chi_e - hop(hop(chi_e)))

    def apply_dagger(self, chi):
        """M_ee is hermitian."""
        return self.apply(chi)

    def normal(self, chi):
        """M_ee^2 chi_e."""
        return self.apply(self.apply(chi))

    def prepare_rhs(self, b):
        """Even-site right-hand side m b_e - K_eo b_o = (D^dagger b)_e of the Schur system."""
        m = self.staggered._mass_view(b)
        return self.even(m * b - self.staggered.hopping(self.odd(b)))

    def reconstruct(self, x_even, b):
        """Full solution of D x = b from the even part: x_o = (b_o - K_oe x_e) / m."""
        x_e = self.even(x_even)
        m = self.staggered._mass_view(b)
        return x_e + self.odd(b - self.staggered.hopping(x_e)) / m


def solve_staggered(op, b, tol=1e-10, max_iter=1000, x0=None):
    """Solve D x = b for a `StaggeredDirac` (or its `StaggeredEvenOdd`) with even-odd CG.

    CG on M_ee x_e = m b_e - K_eo b_o, then x_o = (b_o - K_oe x_e) / m (module docstring).
    Samples (and per-sample masses) are independent. The returned residual is the true relative
    residual |b - D x| / |b| of the full system; `iterations` counts applications of M_ee.
    """
    eo = op if isinstance(op, StaggeredEvenOdd) else op.even_odd()
    rhs = eo.prepare_rhs(b)
    inner = cg(eo.apply, rhs, x0=None if x0 is None else eo.even(x0), tol=tol, max_iter=max_iter)
    x = eo.reconstruct(inner.x, b)
    full = eo.staggered
    r2 = (b - full.apply(x)).abs().pow(2).reshape(b.shape[0], -1).sum(1)
    b2 = b.abs().pow(2).reshape(b.shape[0], -1).sum(1)
    residual = torch.sqrt(r2 / torch.where(b2 > 0, b2, torch.ones_like(b2)))
    return SolverResult(x, inner.iterations, residual)


def staggered_dense_matrix(op, geometry=None, batch_index=0, chunk=512):
    """Dense matrix of `op.apply` on staggered fields for gauge field `batch_index`.

    Basis vectors are flattened as index = site * N + c (C order). If the operator has a
    `support` site mask (StaggeredEvenOdd), rows and columns are restricted to those sites, kept
    in lexicographic order. Returns an (n, n) tensor, n = sites * N.
    """
    geometry = op.geometry if geometry is None else geometry
    single = op.batch_select(batch_index) if hasattr(op, "batch_select") else op
    N = geometry.N
    full_dim = geometry.volume * N
    support = getattr(op, "support", None)
    if support is None:
        index = torch.arange(full_dim, device=op.device)
    else:
        sites = support.reshape(-1).nonzero().squeeze(-1)
        index = (sites[:, None] * N + torch.arange(N, device=sites.device)[None, :]).reshape(-1)
    n = index.numel()
    M = torch.empty((n, n), dtype=op.dtype, device=op.device)
    for start in range(0, n, chunk):
        cols = index[start:start + chunk]
        c = cols.numel()
        basis = torch.zeros((c, full_dim), dtype=op.dtype, device=op.device)
        basis[torch.arange(c, device=op.device), cols] = 1.0
        out = single.apply(basis.reshape((c,) + geometry.lattice + (N,))).reshape(c, full_dim)
        M[:, start:start + c] = out[:, index].transpose(0, 1)
    return M


def free_staggered_normal_eigenvalues(geometry, mass, time_sign=-1.0):
    """Eigenvalues of the free D^dagger D: m^2 + sum_mu sin^2 p_mu over the full momentum set,
    each repeated N times (colour), shape (V * N,), float64, in the order of `free_momenta`."""
    p = free_momenta(geometry, time_sign)
    s = torch.sin(p).pow(2).sum(-1)
    return (float(mass) ** 2 + s).repeat_interleave(geometry.N)


def free_staggered_eigenvalues(geometry, mass, time_sign=-1.0):
    """Eigenvalues of the free (U = 1) staggered operator D, shape (V * N,), complex128.

    m +- i sqrt(sum_mu sin^2 p_mu) for p in the reduced zone 0 <= p_mu < pi, each sign with
    multiplicity 2^(D-1) N (derivation in the module docstring). Needs even extents.
    """
    if not geometry.has_even_extents:
        raise ValueError(f"the staggered free spectrum needs even extents, got {geometry.lattice}")
    p = free_momenta(geometry, time_sign)
    reduced = (p < math.pi - 1e-12).all(dim=-1)
    p = p[reduced]
    if p.shape[0] * 2 ** geometry.D != geometry.volume:
        raise RuntimeError("reduced zone does not contain V / 2^D momenta")
    lam = torch.sqrt(torch.sin(p).pow(2).sum(-1))
    mult = 2 ** (geometry.D - 1) * geometry.N
    m = float(mass)
    plus = torch.complex(torch.full_like(lam, m), lam)
    minus = torch.complex(torch.full_like(lam, m), -lam)
    return torch.cat((plus.repeat_interleave(mult), minus.repeat_interleave(mult)))
