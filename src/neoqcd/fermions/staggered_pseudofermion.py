"""Staggered pseudofermions (one even-site field): action, heat-bath refresh, MD force.

Conventions (shared with neoqcd.fermions.staggered and neoqcd.fermions.hmc):

- D = m + K, K anti-hermitian and parity-changing (neoqcd.fermions.staggered), antiperiodic in
  time by default. M_ee = (D^dagger D)_ee = m^2 - K_eo K_oe = m^2 + K_eo K_eo^dagger (since
  K_oe = -K_eo^dagger), hermitian positive definite for m > 0, and det M_ee = det D > 0.
- Number of flavours. One staggered field describes 2^(D/2) tastes, and the weight det D =
  det M_ee of one even-site pseudofermion is the weight of all of them. Hence
      N_f = 2^(D/2):  N_f = 4 in 4D,  N_f = 2 in 2D
  (`n_flavours`); no rooting anywhere in this module.
- Pseudofermion action:
      S_pf = phi^dagger M_ee^-1 phi,      phi on even sites,
  with integral d phi d phi^* exp(-S_pf) = pi^n det M_ee (n = V/2 * N complex components).
- Refresh: chi complex Gaussian on ALL sites with density ~ exp(-chi^dagger chi) (<|chi_i|^2> =
  1, `torch.randn`), and
      phi = (D^dagger chi)_e = m chi_e - K_eo chi_o.
  Since chi_e and chi_o are independent, <phi phi^dagger> = m^2 + K_eo K_eo^dagger = M_ee, so
  phi has density ~ exp(-phi^dagger M_ee^-1 phi) at the current links: an exact heat bath.
  Difference from the Wilson case (neoqcd.fermions.pseudofermion): there phi = D_hat^dagger eta
  with D_hat square on the even sites, so S_pf(U) = eta^dagger eta; here (D^dagger)_{e, all} =
  [m, -K_eo] is rectangular (V/2 x V sites), S_pf(U) = chi^dagger P chi with P the orthogonal
  projector onto the row space of [m, -K_eo], which is not chi^dagger chi. S_pf at the refresh
  links therefore needs one solve; it is computed once, the first time it is asked for, and
  cached (`action_difference`). <S_pf> = n still holds.
- Force: the algebra gradient of hmc.py, tr(F_mu(x) T) = d/dt S_pf(e^{i t T} U_mu(x))|_{t=0}.

Force formula. With X = M_ee^-1 phi (even) and M_ee = m^2 - P_e K K P_e (P_e the even projector):
    dS = -X^dagger dM_ee X = X^dagger (dK K + K dK) X = X^dagger dK W + (X^dagger K) dK X,
    W = K X (odd, = K_oe X).
K is anti-hermitian for every U, so X^dagger K = -(K X)^dagger = -W^dagger and dK^dagger = -dK:
    W^dagger dK X = (X^dagger dK^dagger W)^* = -(X^dagger dK W)^*,
hence
    dS = X^dagger dK W + (X^dagger dK W)^* = 2 Re[X^dagger dK W],
    F = 2 * staggered_hopping_derivative(op, X, W).
The mass enters only through X.
"""

import torch

from neoqcd.fermions.algebra import project_hermitian_traceless
from neoqcd.fermions.hmc import Hamiltonian, ActionTerm, WilsonGaugeAction
from neoqcd.fermions.lattice import fermion_boundary_links, shift
from neoqcd.fermions.solver import cg
from neoqcd.fermions.staggered import StaggeredDirac

__all__ = ["staggered_hopping_derivative", "StaggeredPseudofermion", "StaggeredHamiltonian"]


def _outer(b, c):
    """Colour matrix O_ab = b_a conj(c_b) of two staggered fields, shape (B, lattice, N, N)."""
    return b.unsqueeze(-1) * c.conj().unsqueeze(-2)


def staggered_hopping_derivative(op, a, b):
    """Algebra gradient G of Re[a^dagger K b] with respect to the links, shape like `op.U`.

    `op` is a `StaggeredDirac` (or a `StaggeredEvenOdd`, its `.staggered` is used). For every
    link U_mu(x) and hermitian traceless T,
        tr(G_mu(x) T) = d/dt Re[a^dagger K(U_mu(x) -> e^{i t T} U_mu(x)) b]|_{t=0}.

    K contains L = L_mu(x) (the link including the fermion boundary sign, which only rescales it
    by +-1, so dL = i T L dt as for U) in two places, with eta = eta_mu(x) = eta_mu(x + mu):
        forward:   (K b)(x)      += +1/2 eta L b(x + mu),
        backward:  (K b)(x + mu) += -1/2 eta L^dagger b(x).
    Their variations (colour matrices, "x + mu" = `shift(., mu, +1)`):
        a(x)^dagger (1/2 eta) i T L b(x + mu)            = (i/2) eta tr[T L O1],   O1 = b(x + mu) a(x)^dagger,
        -(1/2 eta) a(x + mu)^dagger (-i L^dagger T) b(x) = (i/2) eta tr[T O2 L^dagger], O2 = b(x) a(x + mu)^dagger.
    With Q = (eta/2) (L O1 + O2 L^dagger), Re[i tr(T Q)] = tr(T * i (Q - Q^dagger) / 2), hence
        G_mu(x) = project_hermitian_traceless(i Q).
    """
    stag = getattr(op, "staggered", op)
    links = fermion_boundary_links(stag.U, stag.time_sign)
    eta = stag.eta  # (D, lattice, 1)
    out = []
    for mu in range(stag.geometry.D):
        L = links[:, mu]
        O1 = _outer(shift(b, mu, 1), a)
        O2 = _outer(b, shift(a, mu, 1))
        Q = torch.matmul(L, O1) + torch.matmul(O2, L.conj().transpose(-1, -2))
        Q = (0.5 * eta[mu]).unsqueeze(-1).to(Q.dtype) * Q
        out.append(project_hermitian_traceless(1j * Q))
    return torch.stack(out, dim=1)


class StaggeredPseudofermion(ActionTerm):
    """N_f = 2^(D/2) degenerate staggered flavours (4 in 4D, 2 in 2D) through one even-site
    pseudofermion field (conventions: module docstring).

    Args:
        mass: bare mass m > 0, a float or a (B,) tensor.
        geometry: `Geometry` (even extents).
        time_sign: fermion boundary phase in time, -1 antiperiodic (default).
        solver_tol: CG relative residual for the force solves.
        action_tol: CG relative residual for the action solves; default min(solver_tol, 1e-12).
        max_iter: CG iteration cap per solve.
        reuse_guess: start each CG from the last solution X. The trajectory is then reversible
            only up to the solver tolerance; with False the force is a deterministic function of U.

    State: `phi` (set by `refresh`), `X` (last solution of M_ee X = phi). `stats` as in
    `TwoFlavourWilson`. Per HMC trajectory there are two action solves (S_pf at the refresh links,
    cached, and at the end), one more than for Wilson (module docstring).
    """

    def __init__(self, mass, geometry, time_sign=-1.0, solver_tol=1e-10, action_tol=None,
                 max_iter=2000, reuse_guess=True):
        self.mass = mass
        self.geometry = geometry
        self.time_sign = float(time_sign)
        self.solver_tol = float(solver_tol)
        self.action_tol = min(self.solver_tol, 1e-12) if action_tol is None else float(action_tol)
        self.max_iter = int(max_iter)
        self.reuse_guess = bool(reuse_guess)
        self.phi = None
        self.X = None
        self._refresh_links = None
        self._refresh_action = None
        self.reset_stats()

    # ---- bookkeeping ---------------------------------------------------------------------
    def reset_stats(self):
        self.stats = {"refresh_calls": 0, "action_calls": 0, "force_calls": 0, "solves": 0,
                      "cg_iterations": 0, "cg_iterations_per_sample": None, "last_iterations": None,
                      "last_residual": None}

    def _count(self, result):
        it = result.iterations
        self.stats["solves"] += 1
        self.stats["cg_iterations"] += int(it.max().item())
        tot = self.stats["cg_iterations_per_sample"]
        self.stats["cg_iterations_per_sample"] = it.clone() if tot is None or tot.shape != it.shape else tot + it
        self.stats["last_iterations"] = it
        self.stats["last_residual"] = result.residual

    @property
    def n_components(self):
        """Complex components of phi: V/2 * N."""
        return self.geometry.volume * self.geometry.N // 2

    @property
    def n_flavours(self):
        """Continuum flavours (tastes) described: 2^(D/2)."""
        return 2 ** (self.geometry.D // 2)

    # ---- operators -----------------------------------------------------------------------
    def dirac(self, U):
        return StaggeredDirac(U, self.mass, self.geometry, self.time_sign)

    def operator(self, U):
        """M_ee at links U (`StaggeredEvenOdd`)."""
        return self.dirac(U).even_odd()

    def _solve(self, op, tol):
        if self.phi is None:
            raise RuntimeError("the pseudofermion field is not set: call refresh(U, generator) first")
        x0 = self.X if (self.reuse_guess and self.X is not None) else None
        res = cg(op.apply, self.phi, x0=x0, tol=tol, max_iter=self.max_iter)
        self._count(res)
        self.X = op.even(res.x)
        return self.X

    # ---- ActionTerm interface -------------------------------------------------------------
    def refresh(self, U, generator=None):
        """phi = (D^dagger chi)_e = m chi_e - K_eo chi_o, chi complex Gaussian on all sites."""
        op = self.operator(U)
        chi = torch.randn(self.geometry.staggered_shape(U.shape[0]), generator=generator,
                          dtype=U.dtype, device=U.device)
        self.phi = op.prepare_rhs(chi)
        self.X = None
        self._refresh_links = U
        self._refresh_action = None
        self.stats["refresh_calls"] += 1

    def action(self, U):
        """S_pf = phi^dagger M_ee^-1 phi per configuration, shape (B,), one CG solve at `action_tol`."""
        self.stats["action_calls"] += 1
        X = self._solve(self.operator(U), self.action_tol)
        return (self.phi.conj() * X).real.reshape(X.shape[0], -1).sum(1)

    def action_difference(self, U_old, U_new):
        """S_pf(U_new) - S_pf(U_old) at fixed phi; S_pf at the refresh links is solved once and cached."""
        if U_old is self._refresh_links:
            if self._refresh_action is None:
                self._refresh_action = self.action(U_old)
            S_old = self._refresh_action
        else:
            S_old = self.action(U_old)
        return self.action(U_new) - S_old

    def force(self, U):
        """Algebra gradient of S_pf: F = 2 * staggered_hopping_derivative(X, K X), X = M_ee^-1 phi."""
        self.stats["force_calls"] += 1
        op = self.operator(U)
        X = self._solve(op, self.solver_tol)
        W = op.odd(op.staggered.hopping(X))
        return 2.0 * staggered_hopping_derivative(op, X, W)


class StaggeredHamiltonian(Hamiltonian):
    """N_f = 2^(D/2) staggered fermions plus Wilson gauge action, two time scales.

    Hamiltonian([(fermion, 1), (gauge, n_gauge_substeps)]): pseudofermion force on the outer
    level, gauge force on the inner one. Extra keyword arguments go to `StaggeredPseudofermion`.
    Attributes `fermion` and `gauge`.
    """

    def __init__(self, beta, mass, geometry, n_gauge_substeps=1, plaquette_weights=None, **fermion_kwargs):
        self.fermion = StaggeredPseudofermion(mass, geometry, **fermion_kwargs)
        self.gauge = WilsonGaugeAction(beta, geometry, plaquette_weights)
        super().__init__([([self.fermion], 1), ([self.gauge], n_gauge_substeps)])
