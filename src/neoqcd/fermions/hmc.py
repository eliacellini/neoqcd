"""Hybrid Monte Carlo: momenta, nested (multi-time-scale) integrators, Metropolis step.

Conventions (see also neoqcd.fermions.algebra):
- links U (B, D, T, L, ..., L, N, N) as in neoqcd.sun_utils;
- momenta P are hermitian traceless, one per link, with density ~ exp(-(1/2) sum_links tr P^2):
  P = sum_a p_a T^a, p_a ~ N(0, 1), tr(T^a T^b) = delta_ab, so <tr P^2> = N^2 - 1 per link;
- H(U, P) = (1/2) sum_links tr P^2 + sum_terms S_term(U);
- equations of motion: dU/dt = i P U, dP/dt = -F(U);
- the force of an action term is the algebra gradient F = sum_a T^a d/dt S(exp(i t T^a) U)|_{t=0},
  i.e. tr(F X) = d/dt S(exp(i t X) U)|_{t=0} for every hermitian traceless X (per link);
- a position step is U <- exp(i h P) U (closed form for SU(2), Cayley-Hamilton for SU(3), see
  `algebra.expi`); the links are reunitarised once, at the end of the trajectory.

An action term is any object with
    action(U) -> (B,)                          the action per configuration,
    force(U) -> like U                         the algebra gradient above (hermitian traceless),
optionally
    action_difference(U_old, U_new) -> (B,)    S(U_new) - S(U_old), summed site by site,
    refresh(U, generator)                      a heat-bath refresh of auxiliary fields
                                               (pseudofermions) at the start of a trajectory.
The terms are grouped in levels of a `Hamiltonian`: level 0 is the outermost (coarsest) time scale
and level k + 1 runs `n_substeps` integrator steps inside each position update of level k
(Sexton-Weingarten). Phase 1 will put the fermion term on level 0 and the gauge term on level 1.
"""


import torch

from neoqcd.fermions._gauge import (
    SUN_dagger,
    SUN_mul,
    SUN_trace,
    mean_plaquette,
    nstaple,
    plaquette_fields,
    plaquette_weight,
    pstaple,
)
from neoqcd.fermions.algebra import expi, random_algebra, trace_product, unitarize

# Omelyan, Mryglod, Folk, Comput. Phys. Commun. 146 (2002) 188, eq. (31): the second-order
# minimum-norm scheme; Takaishi and de Forcrand, hep-lat/0505020
OMELYAN_LAMBDA = 0.1931833275037836

INTEGRATORS = ("leapfrog", "omelyan")


def _per_batch(value, U):
    """A scalar or a (B,) coupling broadcast against link-shaped tensors (B, D, lattice, N, N)."""
    if torch.is_tensor(value) and value.dim() > 0:
        return value.to(device=U.device, dtype=U.real.dtype).reshape((-1,) + (1,) * (U.dim() - 1))
    return float(value)


def link_sum(x, start=1):
    """Sum over every dimension from `start` on: per-configuration totals of a (B, ...) field."""
    return x.sum(tuple(range(start, x.dim())))


def kinetic_energy(P):
    """(1/2) sum_links tr P^2 per configuration, shape (B,)."""
    return 0.5 * link_sum(trace_product(P, P))


def random_momenta(shape, generator=None, dtype=torch.cdouble, device=None):
    """Momenta with density ~ exp(-(1/2) sum tr P^2): hermitian traceless, one per link."""
    return random_algebra(shape, generator=generator, dtype=dtype, device=device)


def defect_plaquette_weights(defect_mask, defect_par, D):
    """Plaquette weights {(mu, nu): field} of `Wilson_action` with a defect, for a scalar defect_par.

    kappa = defect_mask[0] + defect_mask[1] * defect_par on the links, and the weight of a plaquette
    is the product of the kappa of its four links (`theory.plaquette_weight`).
    """
    kappa = defect_mask[0] + defect_mask[1] * float(defect_par)
    return {(mu, nu): plaquette_weight(kappa, mu, nu, D) for nu in range(1, D) for mu in range(nu)}


def batched_defect_plaquette_weights(defect_mask, defect_par, D):
    """Plaquette weights {(mu, nu): field} for a (B,) tensor of defect parameters.

    Entry b of each field (shape (B,) + lattice) equals the field of
    `defect_plaquette_weights(defect_mask, defect_par[b], D)`, i.e. `theory.plaquette_weight` of
    kappa_b = defect_mask[0] + defect_mask[1] * defect_par[b]; this is the per-configuration layout
    accepted by `WilsonGaugeAction`. A scalar (0-dim) defect_par gives B = 1.
    """
    c = torch.as_tensor(defect_par, dtype=defect_mask.dtype, device=defect_mask.device).reshape(-1)
    n_lattice = defect_mask.dim() - 2  # defect_mask is (2, D) + lattice
    # kappa of shape (D, B) + lattice, so that plaquette_weight's kappa[mu] is (B,) + lattice and its
    # rolls act on the trailing lattice dimensions
    kappa = defect_mask[0][:, None] + defect_mask[1][:, None] * c.reshape((1, -1) + (1,) * n_lattice)
    return {(mu, nu): plaquette_weight(kappa, mu, nu, D) for nu in range(1, D) for mu in range(nu)}


class ActionTerm:
    """Base class of the action terms; subclasses define action(U) and force(U)."""

    def action(self, U):
        raise NotImplementedError

    def force(self, U):
        raise NotImplementedError

    def action_difference(self, U_old, U_new):
        return self.action(U_new) - self.action(U_old)

    def refresh(self, U, generator=None):
        return None


class WilsonGaugeAction(ActionTerm):
    """S_g = beta sum_x sum_{mu<nu} [1 - w_{mu nu}(x) Re tr U_{mu nu}(x) / N].

    This is `neoqcd.theory.Wilson_action` exactly (same constant, same sign): without a defect
    w = 1; with `plaquette_weights` = {(mu, nu): w_{mu nu}} (mu < nu, fields of shape lattice or
    (B,) + lattice) the plaquettes are weighted as in `Wilson_action` with a defect, see
    `defect_plaquette_weights`. `beta` is a float or a (B,) tensor (one coupling per configuration).

    Force: for the link U = U_mu(x) let A be the sum of its (weighted) staples, so that the part of
    S_g that depends on U is -(beta/N) Re tr(U A). With Omega = U A,
        F = -i beta/(2N) [Omega - Omega^dagger - tr(Omega - Omega^dagger)/N],
    hermitian traceless, and tr(F X) = d/dt S_g(exp(i t X) U)|_{t=0}.
    """

    def __init__(self, beta, geometry, plaquette_weights=None):
        self.beta = beta
        self.geometry = geometry
        self.D = geometry.D
        self.N = geometry.N
        self.n_planes = self.D * (self.D - 1) // 2
        self.plaquette_weights = plaquette_weights

    def _weight(self, mu, nu):
        if self.plaquette_weights is None:
            return None
        return self.plaquette_weights[(min(mu, nu), max(mu, nu))]

    def _weighted_retr(self, U):
        """sum_{mu<nu} w_{mu nu}(x) Re tr U_{mu nu}(x), shape (B,) + lattice."""
        planes = ((mu, nu) for nu in range(1, self.D) for mu in range(nu))
        total = None
        for (mu, nu), retr in zip(planes, plaquette_fields(U, self.D)):
            w = self._weight(mu, nu)
            if w is not None:
                retr = w.to(device=retr.device, dtype=retr.dtype) * retr
            total = retr if total is None else total + retr
        return total

    def _beta_vector(self, U):
        beta = self.beta
        if torch.is_tensor(beta) and beta.dim() > 0:
            return beta.to(device=U.device, dtype=U.real.dtype).reshape(-1)
        return float(beta)

    def action(self, U):
        retr = self._weighted_retr(U)
        return self._beta_vector(U) * link_sum(float(self.n_planes) - retr / float(self.N))

    def action_difference(self, U_old, U_new):
        diff = self._weighted_retr(U_old) - self._weighted_retr(U_new)
        return self._beta_vector(U_new) * link_sum(diff) / float(self.N)

    def staple_sum(self, U, mu):
        """A_mu(x) = sum_{nu != mu} [w(x) pstaple + w(x - nu) nstaple], so that the local action is
        -(beta/N) Re tr(U_mu(x) A_mu(x))."""
        A = None
        for nu in range(self.D):
            if nu == mu:
                continue
            up = pstaple(U, mu, nu, self.D)
            down = nstaple(U, mu, nu, self.D)
            w = self._weight(mu, nu)
            if w is not None:
                w = w.to(device=U.device, dtype=U.real.dtype)
                # the backward plaquette of the link is based at x - nu: roll by -1 (code convention)
                w_down = torch.roll(w, -1, dims=-self.D + nu)
                up = w[..., None, None] * up
                down = w_down[..., None, None] * down
            A = up + down if A is None else A + up + down
        return A

    def force(self, U):
        Omega = torch.stack([SUN_mul(U[:, mu], self.staple_sum(U, mu)) for mu in range(self.D)], dim=1)
        Z = Omega - SUN_dagger(Omega)
        eye = torch.eye(self.N, dtype=U.dtype, device=U.device)
        Z = Z - (SUN_trace(Z) / self.N)[..., None, None] * eye
        return (-0.5j / self.N) * _per_batch(self.beta, U) * Z


class Hamiltonian:
    """H = (1/2) sum tr P^2 + sum of the action terms, with the terms grouped in time-scale levels.

    `levels` is a sequence of (terms, n_substeps), outermost level first; `terms` is an action term
    or a sequence of them. Level k + 1 performs n_substeps integrator steps (of its own scheme) in
    each position update of level k; the n_substeps of level 0 multiplies the HMC `n_steps`.
    Example for phase 1: Hamiltonian([(fermion_term, 1), (gauge_term, 4)]).
    """

    def __init__(self, levels):
        self.levels = []
        for terms, n_substeps in levels:
            if isinstance(terms, ActionTerm) or not isinstance(terms, (list, tuple)):
                terms = [terms]
            n_substeps = int(n_substeps)
            if n_substeps < 1:
                raise ValueError("n_substeps must be >= 1")
            self.levels.append((list(terms), n_substeps))
        if not self.levels:
            raise ValueError("a Hamiltonian needs at least one level")

    @property
    def terms(self):
        return [t for terms, _ in self.levels for t in terms]

    def level_force(self, U, level):
        F = None
        for term in self.levels[level][0]:
            f = term.force(U)
            F = f if F is None else F + f
        return F

    def action(self, U):
        return sum(term.action(U) for term in self.terms)

    def action_difference(self, U_old, U_new):
        return sum(term.action_difference(U_old, U_new) for term in self.terms)

    def energy(self, U, P):
        return kinetic_energy(P) + self.action(U)

    def energy_difference(self, U_old, P_old, U_new, P_new):
        """H(U_new, P_new) - H(U_old, P_old), with the differences summed site by site."""
        dK = 0.5 * link_sum(trace_product(P_new, P_new) - trace_product(P_old, P_old))
        return dK + self.action_difference(U_old, U_new)

    def refresh(self, U, generator=None):
        for term in self.terms:
            term.refresh(U, generator)


class WilsonGaugeHamiltonian(Hamiltonian):
    """Pure-gauge Hamiltonian: one level holding the Wilson gauge action."""

    def __init__(self, beta, geometry, plaquette_weights=None):
        self.gauge = WilsonGaugeAction(beta, geometry, plaquette_weights)
        super().__init__([([self.gauge], 1)])


class _MDState:
    """Molecular-dynamics state with lazy momentum kicks.

    kick(level, c) records P <- P - c F_level(U); the kicks are applied (flushed) just before the
    next position update or at the end, so consecutive kicks of one level at the same U (the last
    of a step and the first of the next one) cost a single force evaluation. Kicks of different
    levels at the same U commute, so this is exactly the textbook sequence of updates.
    """

    def __init__(self, hamiltonian, U, P, exp_method):
        self.hamiltonian = hamiltonian
        self.U = U
        self.P = P
        self.exp_method = exp_method
        n = len(hamiltonian.levels)
        self.pending = [0.0] * n
        self.force_evals = [0] * n

    def kick(self, level, coefficient):
        self.pending[level] += coefficient

    def flush(self):
        for level, c in enumerate(self.pending):
            if c != 0.0:
                self.P = self.P - c * self.hamiltonian.level_force(self.U, level)
                self.force_evals[level] += 1
                self.pending[level] = 0.0

    def move(self, h):
        self.flush()
        self.U = SUN_mul(expi(h * self.P, self.exp_method), self.U)


class HMC:
    """Hybrid Monte Carlo over a batch of B independent configurations.

    geometry:     `neoqcd.fermions.lattice.Geometry`;
    hamiltonian:  a `Hamiltonian` (e.g. `WilsonGaugeHamiltonian`);
    tau:          trajectory length;
    n_steps:      integrator steps of the outermost level per trajectory (times its n_substeps);
    integrator:   "leapfrog" or "omelyan", or one name per level;
    exp_method:   see `algebra.expi`.
    """

    def __init__(self, geometry, hamiltonian, tau, n_steps, integrator="omelyan", exp_method="auto",
                 omelyan_lambda=OMELYAN_LAMBDA):
        self.geometry = geometry
        self.hamiltonian = hamiltonian
        self.tau = float(tau)
        self.n_steps = int(n_steps)
        n_levels = len(hamiltonian.levels)
        if isinstance(integrator, str):
            integrator = [integrator] * n_levels
        integrator = [str(s).lower() for s in integrator]
        if len(integrator) != n_levels or any(s not in INTEGRATORS for s in integrator):
            raise ValueError(f"integrator must be one of {INTEGRATORS} (or one per level), got {integrator}")
        self.integrator = integrator
        self.exp_method = exp_method
        self.omelyan_lambda = float(omelyan_lambda)

    # ---- molecular dynamics -------------------------------------------------------------------
    def _step(self, state, level, h):
        if self.integrator[level] == "leapfrog":
            state.kick(level, 0.5 * h)
            self._drift(state, level, h)
            state.kick(level, 0.5 * h)
        else:
            lam = self.omelyan_lambda
            state.kick(level, lam * h)
            self._drift(state, level, 0.5 * h)
            state.kick(level, (1.0 - 2.0 * lam) * h)
            self._drift(state, level, 0.5 * h)
            state.kick(level, lam * h)

    def _drift(self, state, level, h):
        if level + 1 == len(self.hamiltonian.levels):
            state.move(h)
            return
        n = self.hamiltonian.levels[level + 1][1]
        for _ in range(n):
            self._step(state, level + 1, h / n)

    @torch.no_grad()
    def integrate(self, U, P, tau=None, n_steps=None):
        """Molecular-dynamics evolution (U, P) -> (U', P') over time `tau` (negative runs backwards).

        Returns (U', P', force_evals per level). No reunitarisation here.
        """
        tau = self.tau if tau is None else float(tau)
        n = (self.n_steps if n_steps is None else int(n_steps)) * self.hamiltonian.levels[0][1]
        state = _MDState(self.hamiltonian, U, P, self.exp_method)
        h = tau / n
        for _ in range(n):
            self._step(state, 0, h)
        state.flush()
        return state.U, state.P, state.force_evals

    # ---- Monte Carlo ---------------------------------------------------------------------------
    @torch.no_grad()
    def trajectory(self, U, generator=None):
        """One HMC trajectory with a Metropolis step per configuration.

        Returns (U_new, accepted (B,) bool, delta_H (B,), stats). stats holds exp(-dH), the
        acceptance rate, the force evaluations per level and the mean plaquette of U_new.

        `generator` is used for the pseudofermion refresh, the momenta and the Metropolis uniforms,
        all drawn on U.device: on CUDA it must be a `torch.Generator(device=U.device)` (a CPU
        generator makes torch.rand/randn on a CUDA device raise). stats["acceptance"] is a Python
        float obtained with `.item()`, i.e. one host-device synchronisation per trajectory; the
        returned `accepted` tensor is the sync-free alternative.
        """
        B = U.shape[0]
        self.hamiltonian.refresh(U, generator)
        P = random_momenta(U.shape, generator=generator, dtype=U.dtype, device=U.device)
        U1, P1, force_evals = self.integrate(U, P)
        U1 = unitarize(U1)
        dH = self.hamiltonian.energy_difference(U, P, U1, P1)
        r = torch.rand(B, generator=generator, dtype=dH.dtype, device=dH.device)
        accepted = torch.log(r) < -dH
        mask = accepted.reshape((B,) + (1,) * (U.dim() - 1))
        U_new = torch.where(mask, U1, U)
        stats = {
            "exp_minus_dH": torch.exp(-dH),
            "acceptance": accepted.to(dH.dtype).mean().item(),
            "force_evals": force_evals,
            "plaquette": mean_plaquette(plaquette_fields(U_new, self.geometry.D), self.geometry.N),
        }
        return U_new, accepted, dH, stats
