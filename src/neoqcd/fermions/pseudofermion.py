"""N_f = 2 Wilson pseudofermions: action, heat-bath refresh and molecular-dynamics force.

Conventions (shared with neoqcd.fermions.dirac and neoqcd.fermions.hmc):

- D = 1 - kappa H (Hasenbusch normalisation, c_sw = 0), antiperiodic in time by default;
  even-odd: D_hat = 1_ee - kappa^2 H_eo H_oe, det D = det D_hat, so the two-flavour weight is
  det(D^dagger D) = det(D_hat^dagger D_hat).
- Pseudofermion action (even-odd, the default):
      S_pf = phi^dagger (D_hat^dagger D_hat)^-1 phi,      phi on even sites,
  with integral d phi d phi^* exp(-S_pf) = pi^n det(D_hat^dagger D_hat) (n complex components).
  With `even_odd=False` the same theory is written on the full lattice,
  S_pf = phi^dagger (D^dagger D)^-1 phi, phi on all sites.
- Refresh: phi = D_hat^dagger eta, with eta a complex Gaussian of density ~ exp(-eta^dagger eta),
  i.e. independent components with <|eta_i|^2> = 1 (real and imaginary parts of variance 1/2,
  as `torch.randn` draws complex numbers). Then S_pf(U) = eta^dagger eta exactly and phi has
  density ~ exp(-S_pf) at the current links; <S_pf> = n, the number of complex components
  (V/2 * S * N even-odd, V * S * N full).
- Force: the algebra gradient of hmc.py: for every hermitian traceless T,
      tr(F_mu(x) T) = d/dt S_pf(e^{i t T} U_mu(x), other links fixed)|_{t=0}.

Force formula. Let M = D_hat^dagger D_hat, X = M^-1 phi (even), Y = D_hat X (even). Then
    dS = -X^dagger dM X = -2 Re[Y^dagger dD_hat X],
    dD_hat = -kappa^2 (dH_eo H_oe + H_eo dH_oe),
so, with W = H_oe X (odd) and Z = (H^dagger)_oe Y = gamma_5 H gamma_5 Y (odd),
    dS = 2 kappa^2 Re[Y^dagger dH W + Z^dagger dH X] = 2 kappa^2 Re[(Y + Z)^dagger dH (X + W)]
(the cross terms Y^dagger dH X and Z^dagger dH W vanish: dH changes parity). For the full-lattice
variant dS = 2 kappa Re[Y^dagger dH X] with X = (D^dagger D)^-1 phi, Y = D X.
The bilinear Re[a^dagger dH b] is `hopping_derivative(wilson, a, b)`; see its docstring for the
projectors and shifts.
"""

import torch

from neoqcd.fermions.algebra import project_hermitian_traceless
from neoqcd.fermions.dirac import WilsonDirac
from neoqcd.fermions.hmc import ActionTerm, Hamiltonian, WilsonGaugeAction
from neoqcd.fermions.lattice import fermion_boundary_links, shift
from neoqcd.fermions.solver import cg

__all__ = ["hopping_derivative", "TwoFlavourWilson", "TwoFlavourHamiltonian"]


def _outer(b, c):
    """Colour matrix O_ab = sum_s b_{s a} conj(c_{s b}) of two Wilson fields, shape (B, lattice, N, N)."""
    return torch.einsum("...sa,...sb->...ab", b, c.conj())


def hopping_derivative(wilson, a, b):
    """Algebra gradient G of Re[a^dagger H b] with respect to the links, shape like `wilson.U`.

    For every link U_mu(x) and hermitian traceless T,
        tr(G_mu(x) T) = d/dt Re[a^dagger H(U_mu(x) -> e^{i t T} U_mu(x)) b]|_{t=0}.

    H contains U_mu(x) in two places (with L = the link including the fermion boundary sign,
    which only rescales it by +-1, so dL = i T L dt as for U):
        forward:   (H b)(x)      += (1 - gamma_mu) L_mu(x) b(x + mu),
        backward:  (H b)(x + mu) += (1 + gamma_mu) L_mu(x)^dagger b(x).
    Their variations give, summing over spin s (colour matrices, "x + mu" = `shift(., mu, +1)`):
        a(x)^dagger (1 - gamma_mu) i T L b(x + mu)          = i tr[T L O1],
            O1 = sum_s b(x + mu)_s [(1 - gamma_mu) a(x)]_s^dagger,
        -a(x + mu)^dagger (1 + gamma_mu) i L^dagger T b(x)   = -i tr[T O2 L^dagger],
            O2 = sum_s b(x)_s [(1 + gamma_mu) a(x + mu)]_s^dagger,
    (gamma_mu hermitian, so the projectors pass through the dagger unchanged). With
    Q = L O1 - O2 L^dagger, Re[i tr(T Q)] = tr(T * i (Q - Q^dagger) / 2), hence
        G_mu(x) = traceless hermitian part of (i Q) = project_hermitian_traceless(i Q).
    """
    links = fermion_boundary_links(wilson.U, wilson.time_sign)
    out = []
    for mu in range(wilson.geometry.D):
        L = links[:, mu]
        g = wilson.gammas[mu]
        a_fwd = shift(a, mu, 1)
        b_fwd = shift(b, mu, 1)
        O1 = _outer(b_fwd, a - torch.matmul(g, a))
        O2 = _outer(b, a_fwd + torch.matmul(g, a_fwd))
        Q = torch.matmul(L, O1) - torch.matmul(O2, L.conj().transpose(-1, -2))
        out.append(project_hermitian_traceless(1j * Q))
    return torch.stack(out, dim=1)


class TwoFlavourWilson(ActionTerm):
    """Two degenerate Wilson flavours through one pseudofermion field (conventions: module doc).

    Args:
        kappa: hopping parameter, a float or a (B,) tensor.
        geometry: `Geometry` (even extents for the even-odd form).
        time_sign: fermion boundary phase in time, -1 antiperiodic (default).
        even_odd: True (default) for the even-odd preconditioned action, False for the
            full-lattice one (same theory; used to cross-check the even-odd force).
        solver_tol: CG relative residual for the force solves.
        action_tol: CG relative residual for the action solves (Metropolis step); defaults to
            min(solver_tol, 1e-12).
        max_iter: CG iteration cap per solve.
        reuse_guess: start each CG from the last solution X (chronological guess of order one).
            The trajectory is then reversible only up to the solver tolerance; with False the
            force is a deterministic function of U and the integrator is reversible to rounding.

    State: `phi` (the pseudofermion field, set by `refresh`) and `X` (the last solution of
    M X = phi). `stats` counts calls, solves and CG iterations; `cg_iterations` adds the
    largest per-sample count of each solve (the batch costs as its slowest sample), and
    `cg_iterations_per_sample` keeps the (B,) per-sample totals. `reset_stats()` zeroes them.
    """

    def __init__(self, kappa, geometry, time_sign=-1.0, even_odd=True, solver_tol=1e-10,
                 action_tol=None, max_iter=2000, reuse_guess=True):
        self.kappa = kappa
        self.geometry = geometry
        self.time_sign = float(time_sign)
        self.even_odd = bool(even_odd)
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
        """Complex components of phi: V/2 * S * N (even-odd) or V * S * N (full)."""
        n = self.geometry.volume * self.geometry.spin * self.geometry.N
        return n // 2 if self.even_odd else n

    # ---- operators -----------------------------------------------------------------------
    def wilson(self, U):
        return WilsonDirac(U, self.kappa, self.geometry, self.time_sign)

    def operator(self, U):
        """D_hat (even-odd) or D (full) at links U."""
        w = self.wilson(U)
        return w.even_odd() if self.even_odd else w

    def _support(self, op, psi):
        return op.even(psi) if self.even_odd else psi

    def _solve(self, op, tol):
        if self.phi is None:
            raise RuntimeError("the pseudofermion field is not set: call refresh(U, generator) first")
        x0 = self.X if (self.reuse_guess and self.X is not None) else None
        res = cg(op.normal, self.phi, x0=x0, tol=tol, max_iter=self.max_iter)
        self._count(res)
        self.X = res.x
        return res.x

    # ---- ActionTerm interface -------------------------------------------------------------
    def refresh(self, U, generator=None):
        """phi = D_hat^dagger eta, eta complex Gaussian with <|eta_i|^2> = 1 (module doc)."""
        op = self.operator(U)
        eta = torch.randn(self.geometry.wilson_shape(U.shape[0]), generator=generator,
                          dtype=U.dtype, device=U.device)
        eta = self._support(op, eta)
        self.phi = op.apply_dagger(eta)
        self.X = None  # no useful guess for the new phi
        self._refresh_links = U
        self._refresh_action = (eta.conj() * eta).real.reshape(eta.shape[0], -1).sum(1)
        self.stats["refresh_calls"] += 1

    def action(self, U):
        """S_pf = phi^dagger M^-1 phi per configuration, shape (B,), one CG solve at `action_tol`."""
        self.stats["action_calls"] += 1
        X = self._solve(self.operator(U), self.action_tol)
        return (self.phi.conj() * X).real.reshape(X.shape[0], -1).sum(1)

    def action_difference(self, U_old, U_new):
        """S_pf(U_new) - S_pf(U_old) at fixed phi.

        If U_old is (the same tensor object as) the links of the last refresh, S_pf(U_old) is
        eta^dagger eta from the refresh and no solve is needed for it.
        """
        if U_old is self._refresh_links and self._refresh_action is not None:
            S_old = self._refresh_action
        else:
            S_old = self.action(U_old)
        return self.action(U_new) - S_old

    def force(self, U):
        """Algebra gradient of S_pf (module doc), hermitian traceless, shape like U."""
        self.stats["force_calls"] += 1
        op = self.operator(U)
        X = self._solve(op, self.solver_tol)
        if self.even_odd:
            w = op.wilson
            Y = op.apply(X)
            W = op.odd(w.hopping(X))  # H_oe X
            Z = op.odd(w.gamma5(w.hopping(w.gamma5(Y))))  # (H^dagger)_oe Y
            k2 = w.kappa ** 2
            G = hopping_derivative(w, Y + Z, X + W)
            return (2.0 * k2).view((-1,) + (1,) * (G.dim() - 1)).to(G.dtype) * G
        Y = op.apply(X)
        G = hopping_derivative(op, Y, X)
        return (2.0 * op.kappa).view((-1,) + (1,) * (G.dim() - 1)).to(G.dtype) * G


class TwoFlavourHamiltonian(Hamiltonian):
    """N_f = 2 Wilson fermions plus Wilson gauge action, two time scales.

    Hamiltonian([(fermion, 1), (gauge, n_gauge_substeps)]): the pseudofermion force on the outer
    level, the gauge force on the inner one with `n_gauge_substeps` steps per outer position
    update. `HMC.trajectory` calls `refresh` at the start of every trajectory, which draws a new
    phi. Extra keyword arguments go to `TwoFlavourWilson`. Attributes `fermion` and `gauge`.
    """

    def __init__(self, beta, kappa, geometry, n_gauge_substeps=1, plaquette_weights=None, **fermion_kwargs):
        self.fermion = TwoFlavourWilson(kappa, geometry, **fermion_kwargs)
        self.gauge = WilsonGaugeAction(beta, geometry, plaquette_weights)
        super().__init__([([self.fermion], 1), ([self.gauge], n_gauge_substeps)])
