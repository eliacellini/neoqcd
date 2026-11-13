"""Sheikholeslami-Wohlert (clover) improved N_f = 2 Wilson fermions.

Conventions are those of neoqcd.fermions.dirac (doc/fermions.md): links (B, D, lattice, N, N),
"x + mu" = `shift(., mu, +1)` (array index n - 1), D = 1 - kappa H, gamma matrices in the chiral
representation of openQCD with gamma_5 = diag(1, 1, -1, -1), algebra force tr(F T) =
d/dt S(e^{itT} U)|_{t=0} with F hermitian traceless.

Normalisation (derived from openQCD, doc/dirac.pdf, and pinned by test_fermions_clover):

- Lattice-unit operator (openQCD):
      D_W = (1/2) sum_mu [gamma_mu (nabla_mu + nabla*_mu) - nabla*_mu nabla_mu]
            + c_sw (i/4) sum_{mu,nu} sigma_{mu nu} Fhat_{mu nu} + m0,
  with sigma_{mu nu} = (i/2) [gamma_mu, gamma_nu] and the sum over ALL mu, nu, i.e.
  c_sw (i/2) sum_{mu<nu} sigma_{mu nu} Fhat_{mu nu}. Its hopping part is m0 + D - H/2, with H the
  hopping operator of dirac.py, so D_W = (2 kappa)^-1 (1 - kappa H + T) with kappa = 1/(2 m0 + 2D).
- Hopping normalisation used here:
      D = 2 kappa D_W = 1 + T - kappa H,
      T(x) = i kappa c_sw sum_{mu<nu} sigma_{mu nu} (x) Fhat_{mu nu}(x)
           = (i/2) kappa c_sw sum_{mu,nu} sigma_{mu nu} (x) Fhat_{mu nu}(x),
  site-diagonal, hermitian (sigma hermitian, Fhat anti-hermitian), commuting with gamma_5.
- Field strength: Fhat_{mu nu}(x) = (1/8) [Q_{mu nu}(x) - Q_{nu mu}(x)], projected on the traceless
  part (openQCD stores it as an su(3) algebra element), with Q_{mu nu}(x) the sum of the four
  plaquette leaves based at x, all oriented mu -> nu (counter-clockwise in the (mu, nu) plane):
      L1 = U_mu(x) U_nu(x+mu) U_mu(x+nu)^+ U_nu(x)^+            (the sun_utils plaquette at x)
      L2 = U_nu(x) U_mu(x-mu+nu)^+ U_nu(x-mu)^+ U_mu(x-mu)
      L3 = U_mu(x-mu)^+ U_nu(x-mu-nu)^+ U_mu(x-mu-nu) U_nu(x-nu)
      L4 = U_nu(x-nu)^+ U_mu(x-nu) U_nu(x+mu-nu) U_mu(x)^+
  Q_{nu mu} = Q_{mu nu}^+, so Fhat = P_tl[(Q - Q^+) / 8]. For U_mu(x) = exp(a A_mu) with
  anti-hermitian A (the same reading of the links as the hopping term), Fhat = a^2 [D_mu, D_nu]
  + O(a^4), and c_sw = 1 cancels the O(a) Pauli term of the Wilson term at tree level: the
  identity D_n^2 - sum_mu nabla*_mu nabla_mu = -i sum_{mu<nu} sigma_{mu nu} Fhat_{mu nu} + O(a^4)
  on smooth fields (D_n the naive operator) is checked numerically in the tests.
- Boundary phase: every leaf contains either no time link or two time links at the same time
  slice, one of them daggered, so the antiperiodic sign of the t = 0 time links cancels in every
  leaf for any T (including T = 2). `clover_term` builds Fhat from `fermion_boundary_links` anyway
  (the same links as the hopping term); the tests check that the result does not depend on
  `time_sign`.
- Chiral blocks: sigma_{mu nu} commutes with gamma_5 = diag(1, 1, -1, -1) (D = 4) or sigma_3
  (D = 2), so A = 1 + T is block diagonal in the flattened (spin, colour) index s * N + c: two
  hermitian (S N / 2) x (S N / 2) blocks per site (6 x 6 for SU(3) in 4D). In 2D there is a single
  plane and sigma_01 = -sigma_3.

Even-odd preconditioning. In (even, odd) block order D = [[A_ee, -kappa H_eo], [-kappa H_oe, A_oo]]
with A = 1 + T. The Schur complement on the even sites is
    D_hat = A_ee - kappa^2 H_eo A_oo^-1 H_oe,    det D = det A_oo det D_hat,
and D x = b is solved by D_hat x_e = b_e + kappa H_eo A_oo^-1 b_o, x_o = A_oo^-1 (b_o + kappa H_oe x_e).
D_hat^+ = gamma_5 D_hat gamma_5.

N_f = 2 action (even-odd): det(D^+ D) = det(A_oo)^2 |det D_hat|^2, written as
    S = S_pf + S_det,   S_pf = phi^+ (D_hat^+ D_hat)^-1 phi,   S_det = -2 log|det A_oo|
      = -2 sum_{x odd} log|det A(x)|,
with phi = D_hat^+ eta at the refresh. S_det is local and needs no solve (`CloverDeterminantTerm`,
meant for the gauge level of the integrator).

Forces. All link dependence through T is collected by `clover_derivative`: for an arbitrary
complex weight field X_{mu nu}(x) (B, planes, lattice, N, N) it returns the algebra gradient of
    f(U) = Re sum_x sum_{mu<nu} tr[X_{mu nu}(x) Fhat_{mu nu}(x)].
For a bilinear, Re[a^+ T b] = kappa c_sw f(U) with X_{mu nu} = i sum_{s s'} sigma_{mu nu, s s'} b_{s'} a_s^+
(`clover_bilinear_weights`); for the small determinant, tr[A^-1 dT] = kappa c_sw df with
X_{mu nu} = i sum_{s s'} sigma_{mu nu, s' s} (A^-1)_{s, s'} (colour blocks, `clover_determinant_weights`).
Pseudofermion force: with X = M^-1 phi, Y = D_hat X, W = A_oo^-1 H_oe X, Z = A_oo^-1 (H^+)_oe Y,
    dS_pf = -2 Re[Y^+ dD_hat X]
          = 2 kappa^2 Re[(Y + Z)^+ dH (X + W)] - 2 Re[(Y + kappa Z)^+ dT (X + kappa W)]
(the cross terms vanish because dH changes parity and dT is site-diagonal), and
    dS_det = -2 tr[A_oo^-1 dT_oo].
Route of `clover_derivative`: torch autograd. Fhat(U) is rebuilt differentiably, the
vector-Jacobian product gives the Wirtinger gradient G = dL/dRe U + i dL/dIm U, and the algebra
force is F = P_ht(-i G U^+) (convention of snf_layers.py). It is exact (no truncation) and costs
one extra forward of Fhat plus its backward, about 3x the Fhat construction: O(V * planes) 3x3
products, small against a CG solve (tests print the timings).
"""

import torch

from neoqcd.fermions._gauge import SUN_dagger, SUN_mul
from neoqcd.fermions.algebra import project_hermitian_traceless
from neoqcd.fermions.dirac import (
    EvenOddWilsonDirac,
    WilsonDirac,
    _kappa_tensor,
    dense_matrix,
    gamma_matrices,
)
from neoqcd.fermions.exact import ExactMetropolis, _site_tables, assemble_hopping
from neoqcd.fermions.hmc import ActionTerm, Hamiltonian, WilsonGaugeAction
from neoqcd.fermions.lattice import Geometry, fermion_boundary_links, shift
from neoqcd.fermions.pseudofermion import TwoFlavourWilson, hopping_derivative

__all__ = [
    "planes",
    "sigma_matrices",
    "clover_leaves",
    "clover_field_strength",
    "clover_term",
    "clover_derivative",
    "clover_bilinear_weights",
    "clover_determinant_weights",
    "CloverWilsonDirac",
    "EvenOddCloverDirac",
    "TwoFlavourClover",
    "CloverDeterminantTerm",
    "CloverHamiltonian",
    "assemble_clover_even_odd",
    "log_det_clover_two_flavour",
    "ExactCloverMetropolis",
]


# ---- geometry of the planes and spin matrices -----------------------------------------------
def planes(D):
    """The planes (mu, nu), mu < nu, in the order of `sun_utils.plaquette_fields`."""
    return [(mu, nu) for nu in range(1, D) for mu in range(nu)]


def sigma_matrices(D, dtype=torch.complex128, device=None):
    """sigma_{mu nu} = (i/2) [gamma_mu, gamma_nu] for the planes of `planes(D)`, shape (P, S, S)."""
    gammas, _ = gamma_matrices(D, dtype=dtype, device=device)
    out = [0.5j * (gammas[mu] @ gammas[nu] - gammas[nu] @ gammas[mu]) for mu, nu in planes(D)]
    return torch.stack(out)


def _geometry(U, geometry):
    if geometry is not None:
        return geometry
    D = U.shape[1]
    return Geometry(D, U.shape[2], U.shape[3] if D > 1 else 1, U.shape[-1])


def _coupling(value, batch_size, device, name):
    """A scalar or (B,) coupling as a float64 tensor of shape (B,) or (1,)."""
    try:
        return _kappa_tensor(value, batch_size, device)
    except ValueError as err:
        raise ValueError(str(err).replace("kappa", name)) from None


def _bview(c, x):
    return c.view((-1,) + (1,) * (x.dim() - 1)).to(x.dtype)


# ---- field strength and clover term ---------------------------------------------------------
def clover_leaves(U, geometry=None, time_sign=1.0):
    """Q_{mu nu}(x) = L1 + L2 + L3 + L4 (module docstring), shape (B, P, lattice, N, N).

    `time_sign` multiplies the t = 0 time links first (`fermion_boundary_links`).
    """
    geometry = _geometry(U, geometry)
    links = U if float(time_sign) == 1.0 else fermion_boundary_links(U, time_sign)
    out = []
    for mu, nu in planes(geometry.D):
        Um, Un = links[:, mu], links[:, nu]
        Um_m = shift(Um, mu, -1)            # U_mu(x - mu)
        Un_m = shift(Un, nu, -1)            # U_nu(x - nu)
        L1 = SUN_mul(SUN_mul(Um, shift(Un, mu, 1)), SUN_mul(SUN_dagger(shift(Um, nu, 1)), SUN_dagger(Un)))
        L2 = SUN_mul(SUN_mul(Un, SUN_dagger(shift(Um_m, nu, 1))),
                     SUN_mul(SUN_dagger(shift(Un, mu, -1)), Um_m))
        L3 = SUN_mul(SUN_mul(SUN_dagger(Um_m), SUN_dagger(shift(shift(Un, mu, -1), nu, -1))),
                     SUN_mul(shift(Um_m, nu, -1), Un_m))
        L4 = SUN_mul(SUN_mul(SUN_dagger(Un_m), shift(Um, nu, -1)),
                     SUN_mul(shift(Un_m, mu, 1), SUN_dagger(Um)))
        out.append(L1 + L2 + L3 + L4)
    return torch.stack(out, dim=1)


def clover_field_strength(U, geometry=None, time_sign=1.0):
    """Clover-leaf field strength Fhat_{mu nu}(x), shape (B, P, lattice, N, N), P = D (D - 1) / 2.

    Fhat = traceless part of (Q_{mu nu} - Q_{mu nu}^+) / 8, Q from `clover_leaves`;
    anti-hermitian traceless. `time_sign` multiplies the t = 0 time links first
    (`fermion_boundary_links`); the result does not depend on it (module docstring). The
    function is differentiable in U (used by `clover_derivative`).
    """
    geometry = _geometry(U, geometry)
    Q = clover_leaves(U, geometry, time_sign)
    F = (Q - SUN_dagger(Q)) / 8.0
    tr = torch.einsum("...ii->...", F)
    eye = torch.eye(geometry.N, dtype=U.dtype, device=U.device)
    return F - (tr / geometry.N)[..., None, None] * eye


def clover_term(U, kappa, csw, geometry=None, time_sign=-1.0, field_strength=None):
    """T(x) = i kappa c_sw sum_{mu<nu} sigma_{mu nu} (x) Fhat_{mu nu}(x), shape (B, lattice, S N, S N).

    The (spin, colour) index is flattened as s * N + c (as the dense matrices). `kappa`, `csw` are
    floats or (B,) tensors. Hermitian and block diagonal in chirality (module docstring).
    """
    geometry = _geometry(U, geometry)
    F = clover_field_strength(U, geometry, time_sign) if field_strength is None else field_strength
    sig = sigma_matrices(geometry.D, dtype=U.dtype, device=U.device)
    S, N = geometry.spin, geometry.N
    T = torch.einsum("pst,bp...cd->b...sctd", sig, F)
    T = T.reshape(T.shape[:-4] + (S * N, S * N))
    c = _coupling(kappa, U.shape[0], U.device, "kappa") * _coupling(csw, U.shape[0], U.device, "csw")
    return 1j * _bview(c, T) * T


def clover_derivative(U, X, geometry=None):
    """Algebra gradient of f(U) = Re sum_x sum_{mu<nu} tr[X_{mu nu}(x) Fhat_{mu nu}(U)(x)].

    X: complex weights (B, P, lattice, N, N) (planes in the order of `planes(D)`), treated as
    constants. Returns G like U (hermitian traceless) with tr(G_mu(x) T) = d/dt f(e^{itT} U_mu(x))
    for every hermitian traceless T. Route: autograd vector-Jacobian product through
    `clover_field_strength`, F = P_ht(-i G_W U^+) with G_W the Wirtinger gradient (module doc).
    Works under `torch.no_grad()` (it enables grad locally) and never touches `U.grad`.
    """
    geometry = _geometry(U, geometry)
    Xc = X.detach()
    with torch.enable_grad():
        V = U.detach().clone().requires_grad_(True)
        F = clover_field_strength(V, geometry)
        f = (Xc * F.transpose(-1, -2)).real.sum()
        (G,) = torch.autograd.grad(f, V)
    return project_hermitian_traceless(-1j * SUN_mul(G, SUN_dagger(U.detach())))


def _outer(b, c):
    """O_ab = sum_s b_{s a} conj(c_{s b}) of two Wilson fields."""
    return torch.einsum("...sa,...sb->...ab", b, c.conj())


def clover_bilinear_weights(a, b, geometry):
    """Weights X with Re[a^+ T b] = kappa c_sw Re sum tr[X Fhat]: X_{mu nu} = i sum_s (sigma b)_s a_s^+.

    a, b: Wilson fields (B, lattice, S, N). Returns (B, P, lattice, N, N).
    """
    sig = sigma_matrices(geometry.D, dtype=a.dtype, device=a.device)
    out = [1j * _outer(torch.matmul(sig[p], b), a) for p in range(sig.shape[0])]
    return torch.stack(out, dim=1)


def clover_determinant_weights(A_inv, geometry):
    """Weights X with tr[A^-1 dT] = kappa c_sw Re sum tr[X dFhat]: X_{mu nu, cd} =
    i sum_{s s'} sigma_{mu nu, s' s} (A^-1)_{(s c), (s' d)}.

    A_inv: (B, lattice, S N, S N) site matrices (zero on the sites that do not contribute).
    Returns (B, P, lattice, N, N).
    """
    S, N = geometry.spin, geometry.N
    sig = sigma_matrices(geometry.D, dtype=A_inv.dtype, device=A_inv.device)
    Ai = A_inv.reshape(A_inv.shape[:-2] + (S, N, S, N))
    X = 1j * torch.einsum("pts,b...sctd->bp...cd", sig, Ai)
    return X


def _chiral_blocks(M):
    """The two diagonal chiral blocks of site matrices (..., S N, S N): (..., 2, S N/2, S N/2)."""
    h = M.shape[-1] // 2
    return torch.stack((M[..., :h, :h], M[..., h:, h:]), dim=-3)


def _from_chiral_blocks(blocks):
    """Inverse of `_chiral_blocks` for block-diagonal matrices."""
    h = blocks.shape[-1]
    out = blocks.new_zeros(blocks.shape[:-3] + (2 * h, 2 * h))
    out[..., :h, :h] = blocks[..., 0, :, :]
    out[..., h:, h:] = blocks[..., 1, :, :]
    return out


# ---- operators ------------------------------------------------------------------------------
class CloverWilsonDirac(WilsonDirac):
    """Clover-improved Wilson operator D = 1 + T - kappa H (conventions: module doc).

    Args:
        U, kappa, geometry, time_sign: as `WilsonDirac`.
        csw: Sheikholeslami-Wohlert coefficient, a float or a (B,) tensor (c_sw = 0 is WilsonDirac).
    Attributes: `T` (B, lattice, S N, S N), `A_inv` = (1 + T)^-1 site by site (from the chiral
    blocks), `log_det_A` (B, lattice) = log|det(1 + T)(x)|.
    """

    def __init__(self, U, kappa, csw, geometry=None, time_sign=-1.0):
        super().__init__(U, kappa, geometry, time_sign)
        self.csw = _coupling(csw, U.shape[0], U.device, "csw")
        self.field_strength = clover_field_strength(U, self.geometry)
        self.T = clover_term(U, self.kappa, self.csw, self.geometry, self.time_sign,
                             field_strength=self.field_strength)
        SN = self.T.shape[-1]
        eye = torch.eye(SN, dtype=U.dtype, device=U.device)
        blocks = _chiral_blocks(self.T + eye)
        self.A_inv = _from_chiral_blocks(torch.linalg.inv(blocks))
        self.log_det_A = torch.linalg.slogdet(blocks)[1].sum(-1)

    def batch_select(self, index):
        k = self.kappa if self.kappa.shape[0] == 1 else self.kappa[index:index + 1]
        c = self.csw if self.csw.shape[0] == 1 else self.csw[index:index + 1]
        return CloverWilsonDirac(self.U[index:index + 1], k, c, self.geometry, self.time_sign)

    def _site(self, M, psi):
        """Site-diagonal (S N) x (S N) matrices M applied to a Wilson field psi (B, lattice, S, N)."""
        flat = psi.reshape(psi.shape[:-2] + (-1, 1))
        return torch.matmul(M, flat).reshape(psi.shape)

    def clover(self, psi):
        """T psi."""
        return self._site(self.T, psi)

    def clover_inverse(self, psi):
        """(1 + T)^-1 psi."""
        return self._site(self.A_inv, psi)

    def apply(self, psi):
        """D psi = psi + T psi - kappa H psi."""
        return psi + self.clover(psi) - self._kappa_view(psi) * self.hopping(psi)

    def even_odd(self):
        return EvenOddCloverDirac(self)


class EvenOddCloverDirac(EvenOddWilsonDirac):
    """D_hat = A_ee - kappa^2 H_eo A_oo^-1 H_oe on even sites, A = 1 + T (module doc).

    Same protocol as `EvenOddWilsonDirac` (apply, apply_dagger, normal, prepare_rhs,
    reconstruct, support, even, odd, `wilson`), so `solver.solve` and `dense_matrix` accept it.
    """

    def batch_select(self, index):
        return EvenOddCloverDirac(self.wilson.batch_select(index))

    def apply(self, psi):
        w = self.wilson
        psi_e = self.even(psi)
        k2 = w._kappa_view(psi) ** 2
        inner = self.odd(w.clover_inverse(w.hopping(psi_e)))
        return self.even(psi_e + w.clover(psi_e) - k2 * w.hopping(inner))

    def prepare_rhs(self, b):
        """b_e + kappa H_eo A_oo^-1 b_o."""
        w = self.wilson
        k = w._kappa_view(b)
        return self.even(b + k * w.hopping(self.odd(w.clover_inverse(self.odd(b)))))

    def reconstruct(self, x_even, b):
        """x_o = A_oo^-1 (b_o + kappa H_oe x_e)."""
        w = self.wilson
        x_e = self.even(x_even)
        k = w._kappa_view(b)
        return x_e + self.odd(w.clover_inverse(self.odd(b + k * w.hopping(x_e))))

    def log_det_odd(self):
        """log|det A_oo| per gauge field, shape (B,)."""
        odd = self.odd_mask.reshape(self.geometry.lattice).to(self.wilson.log_det_A.dtype)
        return (self.wilson.log_det_A * odd).reshape(self.batch_size, -1).sum(1)


# ---- actions --------------------------------------------------------------------------------
class TwoFlavourClover(TwoFlavourWilson):
    """N_f = 2 clover pseudofermion term S_pf = phi^+ (D_hat^+ D_hat)^-1 phi (module doc).

    Same arguments, state, refresh, action, action_difference and stats as `TwoFlavourWilson`,
    plus `csw`. With `even_odd=True` (default) the weight is det(D_hat^+ D_hat) only: the small
    determinant det(A_oo)^2 is a separate term (`CloverDeterminantTerm`). With `even_odd=False`
    S_pf = phi^+ (D^+ D)^-1 phi is the whole N_f = 2 weight.
    """

    def __init__(self, kappa, csw, geometry, **kwargs):
        self.csw = csw
        super().__init__(kappa, geometry, **kwargs)

    def wilson(self, U):
        return CloverWilsonDirac(U, self.kappa, self.csw, self.geometry, self.time_sign)

    def force(self, U):
        """Algebra gradient of S_pf (module doc), hermitian traceless, shape like U."""
        self.stats["force_calls"] += 1
        op = self.operator(U)
        X = self._solve(op, self.solver_tol)
        if self.even_odd:
            w = op.wilson
            Y = op.apply(X)
            W = op.odd(w.clover_inverse(w.hopping(X)))                       # A_oo^-1 H_oe X
            Z = op.odd(w.clover_inverse(w.gamma5(w.hopping(w.gamma5(Y)))))   # A_oo^-1 (H^+)_oe Y
            k = _bview(w.kappa, Y)
            G_hop = hopping_derivative(w, Y + Z, X + W)
            weights = clover_bilinear_weights(Y + k * Z, X + k * W, self.geometry)
            G_clv = clover_derivative(U, weights, self.geometry)
            return (_bview(2.0 * w.kappa ** 2, G_hop) * G_hop
                    - _bview(2.0 * w.kappa * w.csw, G_clv) * G_clv)
        Y = op.apply(X)
        G_hop = hopping_derivative(op, Y, X)
        G_clv = clover_derivative(U, clover_bilinear_weights(Y, X, self.geometry), self.geometry)
        return _bview(2.0 * op.kappa, G_hop) * G_hop - _bview(2.0 * op.kappa * op.csw, G_clv) * G_clv


class CloverDeterminantTerm(ActionTerm):
    """S_det = -2 log|det A_oo| = -2 sum_{x odd} log|det(1 + T(x))| (no solve; module doc).

    Together with `TwoFlavourClover(even_odd=True)` it gives the N_f = 2 weight
    det(A_oo)^2 det(D_hat^+ D_hat) = det(D^+ D). Force: tr(F T') = -2 kappa c_sw d/dt f with the
    weights `clover_determinant_weights(A_inv on odd sites)`. `stats` counts calls.
    """

    def __init__(self, kappa, csw, geometry, time_sign=-1.0):
        self.kappa = kappa
        self.csw = csw
        self.geometry = geometry
        self.time_sign = float(time_sign)
        self.reset_stats()

    def reset_stats(self):
        self.stats = {"action_calls": 0, "force_calls": 0}

    def operator(self, U):
        return CloverWilsonDirac(U, self.kappa, self.csw, self.geometry, self.time_sign).even_odd()

    def site_action(self, U):
        """-2 log|det A(x)| on odd sites, 0 on even sites, shape (B, lattice)."""
        eo = self.operator(U)
        odd = eo.odd_mask.reshape(self.geometry.lattice).to(eo.wilson.log_det_A.dtype)
        return -2.0 * eo.wilson.log_det_A * odd

    def action(self, U):
        self.stats["action_calls"] += 1
        s = self.site_action(U)
        return s.reshape(s.shape[0], -1).sum(1)

    def action_difference(self, U_old, U_new):
        self.stats["action_calls"] += 2
        d = self.site_action(U_new) - self.site_action(U_old)
        return d.reshape(d.shape[0], -1).sum(1)

    def force(self, U):
        self.stats["force_calls"] += 1
        eo = self.operator(U)
        w = eo.wilson
        odd = eo.odd_mask.reshape(self.geometry.lattice + (1, 1)).to(w.A_inv.dtype)
        weights = clover_determinant_weights(w.A_inv * odd, self.geometry)
        G = clover_derivative(U, weights, self.geometry)
        return -_bview(2.0 * w.kappa * w.csw, G) * G


class CloverHamiltonian(Hamiltonian):
    """N_f = 2 clover fermions plus Wilson gauge action, two time scales.

    Levels: [pseudofermion] outside, [gauge, small determinant] inside with `n_gauge_substeps`
    (`det_level="gauge"`, the default; "fermion" puts S_det on the outer level). With
    `even_odd=False` the full-lattice pseudofermion carries the whole weight and there is no
    determinant term. Extra keyword arguments go to `TwoFlavourClover`. Attributes `fermion`,
    `determinant` (or None) and `gauge`.
    """

    def __init__(self, beta, kappa, csw, geometry, n_gauge_substeps=1, plaquette_weights=None,
                 det_level="gauge", **fermion_kwargs):
        self.fermion = TwoFlavourClover(kappa, csw, geometry, **fermion_kwargs)
        self.gauge = WilsonGaugeAction(beta, geometry, plaquette_weights)
        self.determinant = None
        outer, inner = [self.fermion], [self.gauge]
        if self.fermion.even_odd:
            self.determinant = CloverDeterminantTerm(kappa, csw, geometry, self.fermion.time_sign)
            if det_level == "gauge":
                inner.append(self.determinant)
            elif det_level == "fermion":
                outer.append(self.determinant)
            else:
                raise ValueError(f"det_level must be 'gauge' or 'fermion', got {det_level!r}")
        super().__init__([(outer, 1), (inner, n_gauge_substeps)])


# ---- exact tools ----------------------------------------------------------------------------
def _block_diagonal(blocks):
    """Dense block-diagonal matrices from site blocks (B, n, m, m): shape (B, n m, n m)."""
    B, n, m, _ = blocks.shape
    eye = torch.eye(n, dtype=blocks.dtype, device=blocks.device)
    return torch.einsum("ij,bixy->bixjy", eye, blocks).reshape(B, n * m, n * m)


def assemble_clover_even_odd(U, kappa, csw, geometry, time_sign=-1.0):
    """Dense D_hat = A_ee - kappa^2 H_eo A_oo^-1 H_oe (B, n, n) and log|det A_oo| (B,), batched.

    Built from `exact.assemble_hopping` and the site blocks of `CloverWilsonDirac`; even sites in
    lexicographic order, as `dense_matrix(EvenOddCloverDirac)`.
    """
    _, _, par = _site_tables(geometry, U.device)
    even = (par == 0).nonzero().squeeze(-1)
    odd = (par == 1).nonzero().squeeze(-1)
    w = CloverWilsonDirac(U, kappa, csw, geometry, time_sign)
    B, V = U.shape[0], geometry.volume
    SN = w.T.shape[-1]
    eye = torch.eye(SN, dtype=U.dtype, device=U.device)
    A = (w.T + eye).reshape(B, V, SN, SN)
    A_inv = w.A_inv.reshape(B, V, SN, SN)
    H_eo = assemble_hopping(U, geometry, even, odd, time_sign)
    H_oe = assemble_hopping(U, geometry, odd, even, time_sign)
    k = _bview(w.kappa, H_eo)
    D_hat = _block_diagonal(A[:, even]) - k ** 2 * torch.matmul(H_eo, torch.matmul(_block_diagonal(A_inv[:, odd]), H_oe))
    log_det_odd = w.log_det_A.reshape(B, V)[:, odd].sum(1)
    return D_hat, log_det_odd


def log_det_clover_two_flavour(U, kappa, csw, geometry, time_sign=-1.0, method="dense_matrix"):
    """log det(D^+ D) = 2 log|det D| per gauge field, shape (B,), float64.

    method "dense_matrix": `dense_matrix` of the full sparse `CloverWilsonDirac` (the reference, no
    even-odd); "even_odd": 2 (log|det A_oo| + log|det D_hat|) with D_hat from `dense_matrix` of the
    sparse even-odd operator; "assembled": the same from `assemble_clover_even_odd`, batched (used
    by the sampler).
    """
    if method == "assembled":
        D_hat, ld_odd = assemble_clover_even_odd(U, kappa, csw, geometry, time_sign)
        return 2.0 * (ld_odd + torch.linalg.slogdet(D_hat)[1])
    w = CloverWilsonDirac(U, kappa, csw, geometry, time_sign)
    if method == "dense_matrix":
        out = [torch.linalg.slogdet(dense_matrix(w, geometry, batch_index=b))[1] for b in range(U.shape[0])]
        return 2.0 * torch.stack(out)
    if method == "even_odd":
        eo = w.even_odd()
        out = [torch.linalg.slogdet(dense_matrix(eo, geometry, batch_index=b))[1] for b in range(U.shape[0])]
        return 2.0 * (eo.log_det_odd() + torch.stack(out))
    raise ValueError(f"unknown method {method!r}")


class ExactCloverMetropolis(ExactMetropolis):
    """`ExactMetropolis` for N_f = 2 clover fermions: S_eff = S_g - log det(D^+ D), with
    log det(D^+ D) = 2 (log|det A_oo| + log|det D_hat|) from `assemble_clover_even_odd`."""

    def __init__(self, beta, kappa, csw, geometry, time_sign=-1.0, eps=0.5, links_per_update=1):
        super().__init__(beta, kappa, geometry, time_sign, eps, links_per_update)
        self.csw = csw

    def log_det(self, U):
        return log_det_clover_two_flavour(U, self.kappa, self.csw, self.geometry, self.time_sign,
                                          method="assembled")
