"""Wilson-Dirac operator, even-odd preconditioning and dense-matrix tools for exact tests.

Conventions (fixed here, checked in test/test_fermions_dirac.py):

- Fields. Links `U` (B, D, T, L, ..., L, N, N) as in neoqcd.sun_utils; a Wilson fermion field
  `psi` has shape (B, T, L, ..., L, S, N), S = 2^(D // 2), complex128. "x + mu" is
  `torch.roll(., +1)` along the axis of mu (see neoqcd.fermions.lattice), i.e. array index
  n - 1.
- Operator (Hasenbusch / Luescher normalisation, c_sw = 0):
      D = 1 - kappa H,   kappa = 1 / (2 m0 + 2 D),
      (H psi)(x) = sum_mu [ (1 - gamma_mu) U_mu(x) psi(x + mu)
                            + (1 + gamma_mu) U_mu(x - mu)^dagger psi(x - mu) ].
- Boundary conditions: periodic in space; in time the phase `time_sign` (-1 antiperiodic,
  +1 periodic) multiplies the time links at array index t = 0, which are exactly the links
  crossing the time boundary in both hops (`fermion_boundary_links`). The gauge field held by
  the caller is never modified.
- Gamma matrices (Euclidean, hermitian, {gamma_mu, gamma_nu} = 2 delta_mu,nu), index 0 = time.
  D = 4, chiral representation of openQCD (doc/dirac.pdf):
      gamma_0 = [[0, -1], [-1, 0]],  gamma_k = [[0, -i sigma_k], [i sigma_k, 0]]  (k = 1, 2, 3),
      gamma_5 = gamma_0 gamma_1 gamma_2 gamma_3 = diag(1, 1, -1, -1).
  D = 2: gamma_0 = sigma_1, gamma_1 = sigma_2, gamma_5 = -i gamma_0 gamma_1 = sigma_3.
- Free theory: the eigenvectors of D at U = 1 are psi(n) = exp(-i p.n) u, n being the array
  index, because "x + mu" is array index n - 1, so psi(x + mu) = exp(+i p_mu) psi(x). On them
  D acts as the spin matrix
      D(p) = 1 - kappa sum_mu [ (1 - gamma_mu) e^{i p_mu} + (1 + gamma_mu) e^{-i p_mu} ]
           = 1 - 2 kappa sum_mu cos p_mu + 2 i kappa sum_mu gamma_mu sin p_mu,
  with p_k = 2 pi n_k / L and p_0 = (2 n_0 + 1) pi / T (antiperiodic) or 2 pi n_0 / T.
- Dense matrices: a field (lattice, S, N) is flattened in C (row-major) order of
  (t, x_1, ..., x_{D-1}, spin, colour), i.e. index = (site * S + s) * N + c with
  site = ((t * L + x_1) * L + x_2) * L + ...; an operator restricted to a site subset (even-odd)
  keeps the selected sites in the same lexicographic order.
"""

import math

import torch

from neoqcd.fermions.lattice import Geometry, even_odd_masks, fermion_boundary_links, shift

__all__ = [
    "gamma_matrices",
    "WilsonDirac",
    "EvenOddWilsonDirac",
    "dense_matrix",
    "free_momenta",
    "free_wilson_matrix",
    "free_wilson_eigenvalues",
]


def gamma_matrices(D, dtype=torch.complex128, device=None):
    """Euclidean gamma matrices indexed by lattice direction (0 = time) and gamma_5.

    Returns `(gammas, gamma5)` with `gammas` of shape (D, S, S) and `gamma5` of shape (S, S);
    the representation is given in the module docstring.
    """
    D = int(D)
    i = 1j
    s1 = [[0, 1], [1, 0]]
    s2 = [[0, -i], [i, 0]]
    s3 = [[1, 0], [0, -1]]
    if D == 2:
        gammas = torch.tensor([s1, s2], dtype=dtype, device=device)
        gamma5 = torch.tensor(s3, dtype=dtype, device=device)
        return gammas, gamma5
    if D == 4:
        sig = torch.tensor([s1, s2, s3], dtype=dtype, device=device)
        zero = torch.zeros(2, 2, dtype=dtype, device=device)
        one = torch.eye(2, dtype=dtype, device=device)

        def block(a, b, c, d):
            return torch.cat((torch.cat((a, b), dim=1), torch.cat((c, d), dim=1)), dim=0)

        g0 = block(zero, -one, -one, zero)
        gk = [block(zero, -i * sig[k], i * sig[k], zero) for k in range(3)]
        gammas = torch.stack([g0] + gk)
        gamma5 = gammas[0] @ gammas[1] @ gammas[2] @ gammas[3]
        return gammas, gamma5
    raise NotImplementedError(f"gamma matrices are implemented for D = 2 and D = 4, got D = {D}")


def _kappa_tensor(kappa, batch_size, device):
    """kappa as a float64 tensor of shape (B,) (or (1,) for a scalar, broadcast over B)."""
    k = torch.as_tensor(kappa, dtype=torch.float64, device=device)
    if k.dim() == 0:
        k = k.reshape(1)
    if k.dim() != 1 or k.shape[0] not in (1, batch_size):
        raise ValueError(f"kappa must be a scalar or have shape ({batch_size},), got {tuple(k.shape)}")
    return k


class WilsonDirac:
    """Wilson-Dirac operator D = 1 - kappa H on a batch of gauge fields (conventions: module doc).

    Args:
        U: links (B, D, lattice, N, N); entry b of the batch is an independent gauge field.
        kappa: hopping parameter, a float or a tensor of shape (B,).
        geometry: `Geometry`; inferred from `U` if omitted.
        time_sign: fermion boundary phase in time, -1 antiperiodic (default), +1 periodic.

    Fields have shape (B, lattice, S, N); a field with batch size B' is accepted when the links
    have B = 1 (one gauge field applied to many vectors) or B' = B.
    """

    def __init__(self, U, kappa, geometry=None, time_sign=-1.0):
        if geometry is None:
            D = U.shape[1]
            geometry = Geometry(D, U.shape[2], U.shape[3] if D > 1 else 1, U.shape[-1])
        if tuple(U.shape[1:]) != geometry.link_shape(1)[1:]:
            raise ValueError(f"links of shape {tuple(U.shape)} do not match geometry {geometry}")
        self.geometry = geometry
        self.U = U
        self.time_sign = float(time_sign)
        self.kappa = _kappa_tensor(kappa, U.shape[0], U.device)
        self.gammas, self.gamma5_matrix = gamma_matrices(geometry.D, dtype=U.dtype, device=U.device)
        links = fermion_boundary_links(U, self.time_sign)
        # psi @ U^T gives U psi on the colour index; psi @ conj(U) gives U^dagger psi.
        self._fwd = [links[:, mu].transpose(-1, -2) for mu in range(geometry.D)]
        self._bwd = [links[:, mu].conj() for mu in range(geometry.D)]

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
        return self.geometry.wilson_shape(self.batch_size)

    def batch_select(self, index):
        """The operator of the single gauge field `index`, with links of batch size 1."""
        k = self.kappa if self.kappa.shape[0] == 1 else self.kappa[index:index + 1]
        return WilsonDirac(self.U[index:index + 1], k, self.geometry, self.time_sign)

    def _kappa_view(self, psi):
        return self.kappa.view((-1,) + (1,) * (psi.dim() - 1))

    # ---- operator ------------------------------------------------------------------------
    def gamma5(self, psi):
        """gamma_5 psi on the spin index."""
        return torch.matmul(self.gamma5_matrix, psi)

    def hopping(self, psi):
        """H psi = sum_mu [(1 - gamma_mu) U_mu(x) psi(x+mu) + (1 + gamma_mu) U_mu(x-mu)^dag psi(x-mu)]."""
        out = None
        for mu in range(self.geometry.D):
            fwd = torch.matmul(shift(psi, mu, 1), self._fwd[mu])
            bwd = shift(torch.matmul(psi, self._bwd[mu]), mu, -1)
            term = (fwd + bwd) - torch.matmul(self.gammas[mu], fwd - bwd)
            out = term if out is None else out + term
        return out

    def apply(self, psi):
        """D psi = psi - kappa H psi."""
        return psi - self._kappa_view(psi) * self.hopping(psi)

    def apply_dagger(self, psi):
        """D^dagger psi = gamma_5 D gamma_5 psi."""
        return self.gamma5(self.apply(self.gamma5(psi)))

    def normal(self, psi):
        """D^dagger D psi."""
        return self.apply_dagger(self.apply(psi))

    def even_odd(self):
        """The even-odd preconditioned operator built on this one."""
        return EvenOddWilsonDirac(self)


class EvenOddWilsonDirac:
    """Even-odd (Schur) preconditioned Wilson operator D_hat = 1_ee - kappa^2 H_eo H_oe.

    With D = [[1, -kappa H_eo], [-kappa H_oe, 1]] in the (even, odd) block order,
    det D = det D_hat and D x = b is solved by
        D_hat x_e = b_e + kappa H_eo b_o          (`prepare_rhs`),
        x_o = b_o + kappa H_oe x_e                (`reconstruct`).
    Fields are full-lattice tensors (B, lattice, S, N); `apply` reads only the even sites of its
    input and returns a field supported on even sites (zero on odd sites). `support` is the even
    site mask, used by `dense_matrix` to restrict to the even subspace. Needs even extents.
    """

    def __init__(self, wilson):
        self.wilson = wilson
        self.geometry = wilson.geometry
        even, odd = even_odd_masks(self.geometry, device=wilson.device, dtype=wilson.U.real.dtype)
        self.even_mask = even
        self.odd_mask = odd
        self.support = even.reshape(self.geometry.lattice) > 0

    @property
    def kappa(self):
        return self.wilson.kappa

    @property
    def batch_size(self):
        return self.wilson.batch_size

    @property
    def dtype(self):
        return self.wilson.dtype

    @property
    def device(self):
        return self.wilson.device

    @property
    def field_shape(self):
        return self.wilson.field_shape

    def batch_select(self, index):
        return EvenOddWilsonDirac(self.wilson.batch_select(index))

    def gamma5(self, psi):
        return self.wilson.gamma5(psi)

    def even(self, psi):
        return self.even_mask * psi

    def odd(self, psi):
        return self.odd_mask * psi

    def apply(self, psi):
        """D_hat psi_e = psi_e - kappa^2 H_eo H_oe psi_e, on even sites."""
        psi_e = self.even(psi)
        k2 = self.wilson._kappa_view(psi) ** 2
        return self.even(psi_e - k2 * self.wilson.hopping(self.wilson.hopping(psi_e)))

    def apply_dagger(self, psi):
        """D_hat^dagger = gamma_5 D_hat gamma_5 (gamma_5 H gamma_5 = H^dagger, gamma_5 is site-diagonal)."""
        return self.gamma5(self.apply(self.gamma5(psi)))

    def normal(self, psi):
        """D_hat^dagger D_hat psi on even sites."""
        return self.apply_dagger(self.apply(psi))

    def prepare_rhs(self, b):
        """Even-site right-hand side b_e + kappa H_eo b_o of the Schur system for D x = b."""
        k = self.wilson._kappa_view(b)
        return self.even(b + k * self.wilson.hopping(self.odd(b)))

    def reconstruct(self, x_even, b):
        """Full solution of D x = b from the even part: x_o = b_o + kappa H_oe x_e."""
        x_e = self.even(x_even)
        k = self.wilson._kappa_view(b)
        return x_e + self.odd(b + k * self.wilson.hopping(x_e))


def dense_matrix(op, geometry=None, batch_index=0, chunk=512):
    """Dense matrix of `op.apply` for gauge field `batch_index`, built column by column.

    Basis vectors are flattened in C order of (lattice, spin, colour) (module docstring). If the
    operator has a `support` site mask (EvenOddWilsonDirac), rows and columns are restricted to
    those sites, kept in lexicographic order. Returns a (n, n) tensor with n = sites * S * N.
    """
    geometry = op.geometry if geometry is None else geometry
    single = op.batch_select(batch_index) if hasattr(op, "batch_select") else op
    S, N = geometry.spin, geometry.N
    inner = S * N
    full_dim = geometry.volume * inner
    support = getattr(op, "support", None)
    if support is None:
        index = torch.arange(full_dim, device=op.device)
    else:
        sites = support.reshape(-1).nonzero().squeeze(-1)
        index = (sites[:, None] * inner + torch.arange(inner, device=sites.device)[None, :]).reshape(-1)
    n = index.numel()
    M = torch.empty((n, n), dtype=op.dtype, device=op.device)
    for start in range(0, n, chunk):
        cols = index[start:start + chunk]
        c = cols.numel()
        basis = torch.zeros((c, full_dim), dtype=op.dtype, device=op.device)
        basis[torch.arange(c, device=op.device), cols] = 1.0
        out = single.apply(basis.reshape((c,) + geometry.lattice + (S, N))).reshape(c, full_dim)
        M[:, start:start + c] = out[:, index].transpose(0, 1)
    return M


def free_momenta(geometry, time_sign=-1.0, device=None):
    """Allowed momenta, shape (V, D), in the C order of the lattice sites (p_0 first)."""
    sign = float(time_sign)
    if sign not in (1.0, -1.0):
        raise ValueError(f"time_sign must be +1 or -1, got {time_sign}")
    T, L = geometry.T, geometry.L
    shift0 = 1.0 if sign == -1.0 else 0.0
    axes = [(2 * torch.arange(T, dtype=torch.float64, device=device) + shift0) * math.pi / T]
    axes += [2 * torch.arange(L, dtype=torch.float64, device=device) * math.pi / L] * (geometry.D - 1)
    grids = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([g.reshape(-1) for g in grids], dim=-1)


def free_wilson_matrix(geometry, kappa, p, dtype=torch.complex128):
    """Spin matrix D(p) of the free Wilson operator for momenta `p` (..., D): shape (..., S, S).

    D(p) = 1 - kappa sum_mu [(1 - gamma_mu) e^{i p_mu} + (1 + gamma_mu) e^{-i p_mu}]; it is the
    action of D on psi(n) = exp(-i p.n) u (module docstring).
    """
    gammas, _ = gamma_matrices(geometry.D, dtype=dtype, device=p.device)
    S = geometry.spin
    one = torch.eye(S, dtype=dtype, device=p.device)
    phase = torch.exp(1j * p.to(dtype))  # (..., D)
    hop = 0
    for mu in range(geometry.D):
        e = phase[..., mu, None, None]
        hop = hop + (one - gammas[mu]) * e + (one + gammas[mu]) / e
    return one - kappa * hop


def free_wilson_eigenvalues(geometry, kappa, time_sign=-1.0):
    """Eigenvalues of the free (U = 1) Wilson operator on the finite lattice, shape (V * S * N,).

    Each momentum contributes the S eigenvalues of D(p), each repeated N times (colour);
    analytically 1 - 2 kappa sum cos p_mu +- 2 i kappa (sum sin^2 p_mu)^(1/2).
    """
    p = free_momenta(geometry, time_sign)
    lam = torch.linalg.eigvals(free_wilson_matrix(geometry, kappa, p))  # (V, S)
    return lam.reshape(-1).repeat_interleave(geometry.N)
