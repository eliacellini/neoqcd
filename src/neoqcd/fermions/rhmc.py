"""Rational HMC for rooted staggered fermions: N_f = gamma * 2^(D/2) flavours (N_f = 2 in 4D).

Conventions are those of neoqcd.fermions.staggered and neoqcd.fermions.staggered_pseudofermion:
D = m + K, M = M_ee = m^2 - K_eo K_oe hermitian positive definite on the even sites, and det M_ee
is the weight of 2^(D/2) tastes. Rooting gives the weight

    det(M_ee)^gamma,    gamma = N_f / 2^(D/2) = p/q in (0, 1]
    (gamma = 1/2: N_f = 2 in 4D, N_f = 1 in 2D; gamma = 1 is StaggeredPseudofermion).

Pseudofermions. n_pf even-site fields phi_j, each carrying det(M)^(gamma/n_pf). With
g = gamma / n_pf and three rational approximations on the spectral interval [lambda_lo, lambda_hi]
of M (neoqcd.fermions.rational, partial-fraction form r(x) = a_0 + sum_k a_k / (x + b_k)):

    r_S(x)  ~ x^-g       action and Metropolis step (accurate, `action_approx_tol`),
    r_F(x)  ~ x^-g       molecular-dynamics force (cheaper, `force_approx_tol`),
    r_H(x)  ~ x^(+g/2)   heat bath (accurate),

    S_pf = sum_j phi_j^dagger r_S(M) phi_j
         = sum_j [ a_0 |phi_j|^2 + sum_k a_k Re phi_j^dagger X_jk ],   X_jk = (M + b_k)^-1 phi_j,
    refresh: phi_j = r_H(M) eta_j, eta_j complex Gaussian on the even sites (density exp(-eta^dagger eta)),

so phi_j has covariance r_H(M)^2 ~ M^g and density ~ exp(-phi^dagger M^-g phi): an exact heat bath up
to the approximation error. All X_jk of one field come from one multishift CG (solver.multishift_cg).
Integrating each phi_j gives det r_S(M)^-1 ~ det M^g; the product over j gives det M^gamma.

The MD force is the exact algebra gradient of the force action S_F = sum_j phi_j^dagger r_F(M) phi_j.
Since d(M + b)^-1 = -(M + b)^-1 dM (M + b)^-1 and dM is the same as for StaggeredPseudofermion
(module docstring there: -X^dagger dM X = 2 Re[X^dagger dK W], W = K X),

    dS_F = - sum_jk a_k X_jk^dagger dM X_jk = sum_jk a_k 2 Re[X_jk^dagger dK W_jk],   W_jk = K_oe X_jk,
    F = 2 sum_jk a_k G(X_jk, W_jk),

G = staggered_hopping_derivative (the colour outer products are summed over k before the link
multiplication, `staggered_rational_force`). The MD therefore integrates the Hamiltonian with S_F
(reversible, area preserving), and the Metropolis test uses S_pf with r_S: the algorithm is exact
for the weight det r_S(M)^-1 ~ det M^gamma, independently of the force degree.

Spectral interval. Default [lambda_lo, lambda_hi] = [m_min^2 / (1 + margin), (m_max^2 + D^2)(1 + margin)]
(m_min, m_max over the batch). The lower bound m^2 is rigorous (K_eo K_eo^dagger >= 0). The upper
bound m^2 + D, the free-field value (max sum_mu sin^2 p_mu = D), does NOT hold on rough fields:
on Haar-random links lambda_max(M) - m^2 is about 2D - 1 (3.0 in 2D, 7.0-7.3 in 4D). The rigorous
bound is D^2: K = sum_mu eta_mu (V_mu - V_mu^dagger)/2 with V_mu unitary, so ||K|| <= D and
||K_eo K_oe|| <= D^2. That is the default; `spectral_bounds` overrides it.
"""

from fractions import Fraction

import torch

from neoqcd.fermions._gauge import SUN_mul
from neoqcd.fermions.algebra import expi, project_hermitian_traceless, random_algebra
from neoqcd.fermions.hmc import ActionTerm, Hamiltonian, WilsonGaugeAction
from neoqcd.fermions.lattice import fermion_boundary_links
from neoqcd.fermions.rational import RationalApproximation, rational_approximation
from neoqcd.fermions.solver import multishift_cg
from neoqcd.fermions.staggered import StaggeredDirac
from neoqcd.fermions.staggered_exact import ExactStaggeredMetropolis

__all__ = [
    "as_exponent",
    "staggered_spectral_bounds",
    "rational_apply",
    "staggered_rational_force",
    "RootedStaggeredPseudofermion",
    "RootedStaggeredHamiltonian",
    "rooted_effective_action",
    "ExactRootedMetropolis",
]


def as_exponent(gamma):
    """gamma as a Fraction (floats via limit_denominator(10^6))."""
    if isinstance(gamma, float):
        return Fraction(gamma).limit_denominator(10 ** 6)
    return Fraction(gamma)


def staggered_spectral_bounds(mass, D, margin=0.01):
    """[m_min^2 / (1 + margin), (m_max^2 + D^2)(1 + margin)]: contains the spectrum of M_ee on every
    gauge field (module docstring)."""
    m = torch.as_tensor(mass, dtype=torch.float64).reshape(-1).abs()
    lo = float(m.min()) ** 2
    hi = float(m.max()) ** 2 + float(D) ** 2
    if lo <= 0:
        raise ValueError("the rooted staggered action needs m > 0")
    return lo / (1.0 + margin), hi * (1.0 + margin)


def rational_apply(approx, apply, phi, tol=1e-10, max_iter=5000):
    """r(A) phi = a_0 phi + sum_k a_k (A + b_k)^-1 phi with one multishift CG.

    Returns (r(A) phi, solutions X (K,) + phi.shape, MultishiftResult).
    """
    res = multishift_cg(apply, phi, approx.shifts.tolist(), tol=tol, max_iter=max_iter)
    a = torch.as_tensor(approx.residues, dtype=phi.real.dtype, device=phi.device)
    out = approx.constant * phi + (a.view((-1,) + (1,) * phi.dim()).to(phi.dtype) * res.x).sum(0)
    return out, res.x, res


def staggered_rational_force(op, X, coefficients):
    """F = 2 sum_k c_k staggered_hopping_derivative(op, X_k, K_oe X_k), shape like the links.

    X: (K, B, lattice, N) even-site fields, coefficients: K real numbers. The colour outer products
    O1 = sum_k c_k W_k(x + mu) X_k(x)^dagger and O2 = sum_k c_k W_k(x) X_k(x + mu)^dagger are summed
    over k first, then G_mu(x) = P_TA(i eta/2 (L O1 + O2 L^dagger)) as in
    staggered_pseudofermion.staggered_hopping_derivative (P_TA: hermitian traceless projection).
    """
    stag = getattr(op, "staggered", op)
    eo = stag.even_odd() if not hasattr(op, "odd") else op
    c = torch.as_tensor(coefficients, dtype=X.real.dtype, device=X.device).to(X.dtype)
    W = torch.stack([eo.odd(stag.hopping(X[k])) for k in range(X.shape[0])])
    links = fermion_boundary_links(stag.U, stag.time_sign)
    out = []
    for mu in range(stag.geometry.D):
        L = links[:, mu]
        Xs = torch.roll(X, 1, dims=2 + mu)  # shift(., mu, +1) on the stacked (K, B, ...) fields
        Ws = torch.roll(W, 1, dims=2 + mu)
        O1 = torch.einsum("k,k...i,k...j->...ij", c, Ws, X.conj())
        O2 = torch.einsum("k,k...i,k...j->...ij", c, W, Xs.conj())
        Q = torch.matmul(L, O1) + torch.matmul(O2, L.conj().transpose(-1, -2))
        Q = (0.5 * stag.eta[mu]).unsqueeze(-1).to(Q.dtype) * Q
        out.append(project_hermitian_traceless(1j * Q))
    return 2.0 * torch.stack(out, dim=1)


class RootedStaggeredPseudofermion(ActionTerm):
    """det(M_ee)^gamma through n_pf even-site pseudofermions with rational functions (module docstring).

    Args:
        mass: bare mass m > 0, float or (B,) tensor.
        geometry: `Geometry` (even extents).
        exponent: gamma = N_f / 2^(D/2) in (0, 1], a Fraction (default 1/2).
        n_pf: number of pseudofermion fields, each with exponent gamma / n_pf.
        time_sign: fermion boundary phase in time (-1 antiperiodic).
        spectral_bounds: (lambda_lo, lambda_hi) of M_ee; default `staggered_spectral_bounds`.
        spectral_margin: relative margin of the default bounds.
        action_approx_tol, force_approx_tol: certified max relative errors of r_S, r_H and of r_F;
            ignored for an approximation whose degree is given.
        action_degree, heatbath_degree, force_degree: explicit numbers of poles.
        solver_tol: multishift CG tolerance of the force solves;
        action_solver_tol: of the action and heat-bath solves, default min(solver_tol, 1e-12).
        max_iter: CG iteration cap.

    The approximations are attributes `action_approx`, `force_approx`, `heatbath_approx`
    (`RationalApproximation`). State: `phi` (list of n_pf fields), `eta` (the Gaussian draws).
    `stats` as in StaggeredPseudofermion; `solves` counts multishift solves, `cg_iterations` the
    sum over solves of the largest iteration count (= applications of M_ee).
    The CG always starts from zero, so the force is a deterministic function of U.
    """

    def __init__(self, mass, geometry, exponent=Fraction(1, 2), n_pf=1, time_sign=-1.0,
                 spectral_bounds=None, spectral_margin=0.01, action_approx_tol=1e-12,
                 force_approx_tol=1e-7, action_degree=None, heatbath_degree=None, force_degree=None,
                 solver_tol=1e-10, action_solver_tol=None, max_iter=5000):
        self.mass = mass
        self.geometry = geometry
        self.exponent = as_exponent(exponent)
        if not (0 < self.exponent <= 1):
            raise ValueError(f"the exponent must be in (0, 1], got {self.exponent}")
        self.n_pf = int(n_pf)
        if self.n_pf < 1:
            raise ValueError("n_pf must be >= 1")
        self.time_sign = float(time_sign)
        if spectral_bounds is None:
            spectral_bounds = staggered_spectral_bounds(mass, geometry.D, spectral_margin)
        self.spectral_bounds = (float(spectral_bounds[0]), float(spectral_bounds[1]))
        lo, hi = self.spectral_bounds
        g = self.exponent / self.n_pf
        self.action_approx = rational_approximation(-g, lo, hi, degree=action_degree, tol=action_approx_tol)
        self.heatbath_approx = rational_approximation(g / 2, lo, hi, degree=heatbath_degree, tol=action_approx_tol)
        self.force_approx = rational_approximation(-g, lo, hi, degree=force_degree, tol=force_approx_tol)
        self.solver_tol = float(solver_tol)
        self.action_solver_tol = min(self.solver_tol, 1e-12) if action_solver_tol is None else float(action_solver_tol)
        self.max_iter = int(max_iter)
        self.phi = None
        self.eta = None
        self._refresh_links = None
        self._refresh_version = None
        self._refresh_action = None
        self.reset_stats()

    # ---- bookkeeping ---------------------------------------------------------------------
    def reset_stats(self):
        self.stats = {"refresh_calls": 0, "action_calls": 0, "force_calls": 0, "solves": 0,
                      "cg_iterations": 0, "last_iterations": None, "last_residual": None}

    def _count(self, result):
        it = result.iterations.max(dim=0).values  # (B,) applications of M_ee
        self.stats["solves"] += 1
        self.stats["cg_iterations"] += int(it.max().item())
        self.stats["last_iterations"] = it
        self.stats["last_residual"] = result.residual.max(dim=0).values

    @property
    def n_components(self):
        """Complex components of one phi: V/2 * N."""
        return self.geometry.volume * self.geometry.N // 2

    @property
    def n_flavours(self):
        """gamma * 2^(D/2) (a Fraction)."""
        return self.exponent * 2 ** (self.geometry.D // 2)

    # ---- operators -----------------------------------------------------------------------
    def dirac(self, U):
        return StaggeredDirac(U, self.mass, self.geometry, self.time_sign)

    def operator(self, U):
        """M_ee at links U (`StaggeredEvenOdd`)."""
        return self.dirac(U).even_odd()

    def _approx(self, approximation):
        if approximation is None or approximation == "action":
            return self.action_approx
        if approximation == "force":
            return self.force_approx
        if isinstance(approximation, RationalApproximation):
            return approximation
        raise ValueError(f"unknown approximation {approximation!r}")

    def _check_phi(self):
        if self.phi is None:
            raise RuntimeError("the pseudofermion fields are not set: call refresh(U, generator) first")

    # ---- ActionTerm interface -------------------------------------------------------------
    def refresh(self, U, generator=None):
        """phi_j = r_H(M_ee) eta_j, eta_j complex Gaussian on the even sites, j = 1 .. n_pf."""
        op = self.operator(U)
        self.phi, self.eta = [], []
        for _ in range(self.n_pf):
            eta = op.even(torch.randn(self.geometry.staggered_shape(U.shape[0]), generator=generator,
                                      dtype=U.dtype, device=U.device))
            phi, _, res = rational_apply(self.heatbath_approx, op.apply, eta, self.action_solver_tol, self.max_iter)
            self._count(res)
            self.eta.append(eta)
            self.phi.append(op.even(phi))
        self._refresh_links = U
        self._refresh_version = U._version
        self._refresh_action = None
        self.stats["refresh_calls"] += 1

    def action(self, U, approximation=None):
        """S_pf = sum_j phi_j^dagger r(M_ee) phi_j per configuration, shape (B,).

        `approximation`: None/"action" (r_S, the default, at `action_solver_tol`), "force" (r_F, the
        function whose exact gradient is `force`, at `solver_tol`) or a RationalApproximation.
        """
        self._check_phi()
        self.stats["action_calls"] += 1
        approx = self._approx(approximation)
        tol = self.solver_tol if approximation == "force" else self.action_solver_tol
        op = self.operator(U)
        total = None
        for phi in self.phi:
            rphi, _, res = rational_apply(approx, op.apply, phi, tol, self.max_iter)
            self._count(res)
            s = (phi.conj() * rphi).real.reshape(phi.shape[0], -1).sum(1)
            total = s if total is None else total + s
        return total

    def action_difference(self, U_old, U_new):
        """S_pf(U_new) - S_pf(U_old) at fixed phi, with S_pf at the refresh links cached (solved once,
        used only while U_old is the refresh tensor and its `_version` is unchanged)."""
        S_new = self.action(U_new)
        if U_old is self._refresh_links and U_old._version == self._refresh_version:
            if self._refresh_action is None:
                self._refresh_action = self.action(U_old)
            S_old = self._refresh_action
        else:
            S_old = self.action(U_old)
        return S_new - S_old

    def force(self, U, approximation="force"):
        """Algebra gradient of sum_j phi_j^dagger r(M_ee) phi_j, r = r_F by default (module docstring)."""
        self._check_phi()
        self.stats["force_calls"] += 1
        approx = self._approx(approximation)
        op = self.operator(U)
        F = None
        for phi in self.phi:
            res = multishift_cg(op.apply, phi, approx.shifts.tolist(), tol=self.solver_tol, max_iter=self.max_iter)
            self._count(res)
            f = staggered_rational_force(op, op.even(res.x), approx.residues)
            F = f if F is None else F + f
        return F


class RootedStaggeredHamiltonian(Hamiltonian):
    """Rooted staggered fermions plus Wilson gauge action on two time scales.

    Hamiltonian([(fermion, 1), (gauge, n_gauge_substeps)]): fermion force outside, gauge force
    inside. Extra keyword arguments go to `RootedStaggeredPseudofermion`. Attributes `fermion`, `gauge`.
    """

    def __init__(self, beta, mass, geometry, exponent=Fraction(1, 2), n_gauge_substeps=1,
                 plaquette_weights=None, **fermion_kwargs):
        self.fermion = RootedStaggeredPseudofermion(mass, geometry, exponent=exponent, **fermion_kwargs)
        self.gauge = WilsonGaugeAction(beta, geometry, plaquette_weights)
        super().__init__([([self.fermion], 1), ([self.gauge], n_gauge_substeps)])


class ExactRootedMetropolis(ExactStaggeredMetropolis):
    """Exact random-walk Metropolis for det(M_ee)^gamma exp(-S_g): S_eff = S_g - gamma log det M_ee.

    Same proposal as ExactStaggeredMetropolis (whose S_eff is the gamma = 1 case). gamma = 0 is the
    pure-gauge Metropolis (no determinant is computed; `log_det_value` is then None). Otherwise
    `log_det_value` is log det M_ee (not multiplied by gamma) of the current links.
    """

    def __init__(self, beta, mass, geometry, exponent=Fraction(1, 2), time_sign=-1.0, eps=0.5,
                 links_per_update=1):
        super().__init__(beta, mass, geometry, time_sign=time_sign, eps=eps, links_per_update=links_per_update)
        self.exponent = as_exponent(exponent)
        if self.exponent < 0:
            raise ValueError("the exponent must be >= 0")

    def effective_action(self, U):
        """S_eff = S_g - gamma log det M_ee, shape (B,)."""
        S = self.gauge_action(U)
        if self.exponent == 0:
            return S
        return S - float(self.exponent) * self.log_det(U)

    def reset(self, U):
        self.U = U
        self.S_g = self.gauge_action(U)
        self.log_det_value = None if self.exponent == 0 else self.log_det(U)
        self.n_proposals = 0
        self.n_accepted = torch.zeros(U.shape[0], dtype=torch.long, device=U.device)

    @torch.no_grad()
    def step(self, generator=None):
        """One proposal per chain; returns the (B,) accepted mask."""
        U = self.U
        B, N = U.shape[0], self.geometry.N
        n_links = self.geometry.D * self.geometry.volume
        k = self.links_per_update
        choice = torch.rand((B, n_links), generator=generator, dtype=torch.float64, device=U.device)
        picked = torch.topk(choice, k, dim=1).indices
        P = random_algebra((B, k, N, N), generator=generator, dtype=U.dtype, device=U.device)
        flat = U.reshape(B, n_links, N, N)
        index = picked[..., None, None].expand(B, k, N, N)
        moved = SUN_mul(expi(self.eps * P), torch.gather(flat, 1, index))
        U_new = flat.scatter(1, index, moved).reshape(U.shape)
        S_g_new = self.gauge_action(U_new)
        dS = S_g_new - self.S_g
        if self.exponent != 0:
            ld_new = self.log_det(U_new)
            dS = dS - float(self.exponent) * (ld_new - self.log_det_value)
        r = torch.rand(B, generator=generator, dtype=dS.dtype, device=U.device)
        acc = torch.log(r) < -dS
        self.U = torch.where(acc.reshape((B,) + (1,) * (U.dim() - 1)), U_new, U)
        self.S_g = torch.where(acc, S_g_new, self.S_g)
        if self.exponent != 0:
            self.log_det_value = torch.where(acc, ld_new, self.log_det_value)
        self.n_proposals += 1
        self.n_accepted += acc.long()
        return acc


def rooted_effective_action(U, beta, mass, geometry, exponent=Fraction(1, 2), time_sign=-1.0):
    """S_eff = S_g - gamma log det M_ee per configuration (exact, dense determinant), shape (B,)."""
    return ExactRootedMetropolis(beta, mass, geometry, exponent, time_sign).effective_action(U)
