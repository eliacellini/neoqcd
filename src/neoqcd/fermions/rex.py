"""Parallel tempering on the boundary-condition defect (REX) with N_f = 2 Wilson fermions.

Phase 5 of fermions_plan.md. The defect acts on the gauge action only (Bonanno, Clemente, D'Elia,
Maio, Parente, arXiv:2404.14151): replica r samples

    p_r(U) ~ exp(-S_g(U; beta, bc_r)) det(D^dagger D)(U; kappa),

with the plaquette weights of the defect in S_g and a determinant that does not see bc. The
replicas are updated with the fermionic HMC (`hmc.HMC` on `QCDAction.hamiltonian(bc)`, pseudofermion
on the outer level, gauge on the inner one, a fresh phi at every trajectory).

Swap of rungs i < j between configurations U_a (at bc_i) and U_b (at bc_j): the Metropolis ratio is

    p_i(U_b) p_j(U_a) / (p_i(U_a) p_j(U_b)) = exp(-dS),
    dS = [S_g(U_a; bc_j) - S_g(U_a; bc_i)] + [S_g(U_b; bc_i) - S_g(U_b; bc_j)],

because the determinant is the same function of U on every rung and cancels configuration by
configuration. dS is the bc jump of `QCDAction.action_difference` (beta/N sum_p (w_old - w_new)
Re tr U_p, nonzero only on the plaquettes that touch the defect): no Dirac solve, no pseudofermion.
The same holds at fixed phi (S_pf does not depend on bc), so the swap is exact whether phi is kept or
refreshed; it is refreshed by the next trajectory anyway.

As in `distributedPT.ParallelTempering`, configurations never move: the swaps exchange the bc labels.

Two drivers:

- `FermionREXTarget`: the object a rank of `distributedPT.ParallelTempering` talks to (one per rank,
  B local configurations, each at its own bc). `ParallelTempering.step(local_actions=...)` (rule
  "exact") wants a (B, n_rungs) table S(U_b; ladder[k]); `action_table` returns the relative table
  S(U_b; ladder[k]) - S(U_b; bc_b) from `action_difference`. The row offset S(U_b; bc_b) cancels in
  the swap dS of `_attempt_swaps_exact_actions` (it enters as s_ij - s_ii and s_ji - s_jj, each a
  difference on one configuration), so the relative table gives the same swaps as the absolute one
  and costs no solve. The base `ParallelTempering` never calls `action_difference` itself; the
  O-REX classes (`OutOfEquilibriumParallelTempering.action_difference`) do, with the signature
  `action_difference(x, parameter_old, parameter_new, parameter_name=...)` implemented here.
  `distributed_rex_step` is one REX step of a rank: trajectory, table, `pt.step`, new labels,
  optional periodic translation.
- `FermionREX`: one process, no distributed backend, for tests and small runs. n_chains independent
  ladders of n_rungs replicas are the batch of one `QCDAction` with a (B,) bc tensor (B = n_rungs *
  n_chains, entry b = slot * n_chains + chain), so one batched HMC trajectory updates every replica;
  then even/odd swap rounds between neighbouring rungs. The label history has the layout of
  `neoqcd.replica_history` (steps, world_size, B) with world_size -> slot within a ladder and
  B -> chain.

Ladder convention: increasing bc, rung 0 = the smallest bc (most open), top rung = bc = 1 (periodic),
as `ParallelTempering` (it sorts the parameters) and main/main_pt_obc.py.

Periodic translation (optional, `periodic_translation=True`): after the swaps the configuration that
carries bc = 1 is shifted by one site along a random direction (`utils.translate_periodic_replica`,
PTBC). Exact for fermions too: the gauge action at bc = 1 and det(D^dagger D) with antiperiodic time
boundary conditions are translation invariant (the antiperiodic sign moves with a change of
variables psi(t0) -> -psi(t0) on one time slice), and phi is drawn afresh at the next trajectory.
Between the translation and the next trajectory `qcd.phi` belongs to the untranslated U: callers
must not evaluate the full action (S_g + S_pf) in that window (the next trajectory refreshes phi).

Optimal ladder: `defect_action` is S_D = dS/dc per configuration, the five-point formula of
`mcmc.DefectMCMCAdapter.defect_action` on the gauge-only `action_difference` (S_pf does not depend
on bc, so dS/dc = dS_g/dc and costs no solve). Its per-rung variance is the metric of
`neoqcd.ladder` / main/optimal_ladder.py; `FermionREX` records it after every step and the drivers
store it as the `defect_action` array of replica_history.npz.

Swap uniforms of the distributed step: `ParallelTempering` draws them on rank 0 from the global
generator of pt.device; `GlobalRNGStream` turns that into a private, checkpointable stream.

Seeds: `derived_seed(seed, *keys)` maps (run seed, rank, stream tag) to an independent 63-bit seed
through `numpy.random.SeedSequence`, so runs with seeds s and s + 1 share no stream on any rank
(the pure-gauge rule seed + rank makes seed s, rank 1 equal to seed s + 1, rank 0).

Cost: the plaquette fields Re tr U_p are computed once per configuration set and reused by the
swap table (every ladder column), the defect action and the mean plaquette
(`bc_action_difference`, `swap_action_difference(plaquettes=...)`, `step_observables`); the
arithmetic is that of `QCDAction.action_difference`, so the numbers are bitwise the same.
"""

from contextlib import nullcontext

import numpy as np
import torch

from neoqcd.fermions._gauge import (
    mean_plaquette,
    plaquette_fields,
    replica_observable_series,
    summarize_history,
    translate_periodic_replica,
)
from neoqcd.fermions.action import QCDAction, bc_plaquette_weights
from neoqcd.fermions.hmc import HMC

__all__ = ["FermionREXTarget", "FermionREX", "distributed_rex_step", "swap_action_difference", "defect_action",
           "GlobalRNGStream", "derived_seed", "bc_action_difference", "step_observables"]

STREAM_RUN, STREAM_GLOBAL, STREAM_SWAP = 0, 1, 2  # stream tags of `derived_seed`


def derived_seed(seed, *keys):
    """A 63-bit seed for the stream (seed, *keys), e.g. (seed, rank, STREAM_RUN) (module doc).

    Distinct key tuples give statistically independent seeds (`numpy.random.SeedSequence`); in
    particular (s, r) and (s + 1, r') never coincide, unlike seed + rank.
    """
    entropy = [int(seed)] + [int(k) for k in keys]
    if any(v < 0 for v in entropy):
        raise ValueError(f"derived_seed needs non-negative integers, got {entropy}")
    state = np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint64)[0]
    return int(state) >> 1


def _check_name(parameter_name):
    if str(parameter_name) != "bc":
        raise ValueError(f"the fermionic REX tempers bc only, got parameter_name={parameter_name!r}")


def _as_ladder(ladder):
    ladder = torch.as_tensor(ladder, dtype=torch.float64).reshape(-1)
    if ladder.numel() < 2:
        raise ValueError("a ladder needs at least two rungs")
    if not bool((ladder[1:] > ladder[:-1]).all()):
        raise ValueError(f"the ladder must be strictly increasing (rung 0 = smallest bc), got {ladder.tolist()}")
    return ladder


def bc_action_difference(qcd, U, p_old, p_new, plaquettes=None):
    """`qcd.action_difference(U, p_old, p_new)` for parameter_name "bc", with the plaquette fields
    `plaquettes` (= `plaquette_fields(U, D)`) given or computed here. Same arithmetic, same order
    as `QCDAction.action_difference` (bitwise equal result); no solve."""
    if qcd.parameter_name != "bc":
        raise ValueError(f"bc_action_difference needs a QCDAction with parameter_name 'bc', got {qcd.parameter_name!r}")
    D, N = qcd.geometry.D, qcd.geometry.N
    if plaquettes is None:
        plaquettes = plaquette_fields(U, D)
    w_old = bc_plaquette_weights(qcd.defect_mask, p_old, D)
    w_new = bc_plaquette_weights(qcd.defect_mask, p_new, D)
    planes = ((mu, nu) for nu in range(1, D) for mu in range(nu))
    total = None
    for (mu, nu), retr in zip(planes, plaquettes):
        dw = (w_old[(mu, nu)] - w_new[(mu, nu)]).to(device=retr.device, dtype=retr.dtype)
        term = dw * retr
        total = term if total is None else total + term
    total = total.sum(tuple(range(1, total.dim())))
    beta = qcd.beta
    if torch.is_tensor(beta):
        beta = beta.to(device=U.device)
    return beta * total / float(N)


def swap_action_difference(qcd, U, parameters, ladder, plaquettes=None):
    """(B, n_rungs) table S(U_b; ladder[k]) - S(U_b; parameters[b]) from the gauge-only bc jump.

    `qcd` is a `QCDAction(parameter_name="bc")`; no solve (module doc). Column k at
    ladder[k] == parameters[b] is exactly zero. The plaquette fields of U are computed once for all
    columns (or taken from `plaquettes`).
    """
    B = U.shape[0]
    parameters = torch.as_tensor(parameters, dtype=torch.float64).reshape(-1)
    if parameters.numel() == 1:
        parameters = parameters.expand(B)
    if plaquettes is None:
        plaquettes = plaquette_fields(U, qcd.geometry.D)
    ladder = torch.as_tensor(ladder, dtype=torch.float64).reshape(-1)
    columns = [bc_action_difference(qcd, U, parameters, torch.full((B,), float(v), dtype=torch.float64),
                                    plaquettes=plaquettes)
               for v in ladder.tolist()]
    return torch.stack(columns, dim=1).to(torch.float64)


def step_observables(qcd, U, parameters, plaquettes=None):
    """(mean plaquette, S_D) per configuration, both (B,) float64, from one set of plaquette fields
    (computed here unless given): `mean_plaquette` and `defect_action` at `parameters`."""
    if plaquettes is None:
        plaquettes = plaquette_fields(U, qcd.geometry.D)
    plaq = mean_plaquette(plaquettes, qcd.geometry.N).to(torch.float64).reshape(-1)
    return plaq, defect_action(qcd, U, parameters, plaquettes=plaquettes)


def defect_action(qcd, U, parameters, step=0.125, plaquettes=None):
    """S_D = dS/dc per configuration at its own bc, (B,) float64: the metric of the optimal ladder.

    Same five-point formula as `mcmc.DefectMCMCAdapter.defect_action`, on the gauge-only
    `QCDAction.action_difference` (bc parameter): each plaquette weight is a product of four
    couplings linear in bc, a polynomial of degree <= 4, so the formula is exact for any `step`.
    With the gauge-only defect dS/dc = dS_g/dc (S_pf does not depend on bc): no solve.
    `plaquettes`: the plaquette fields of U if already computed (module doc).
    """
    B = U.shape[0]
    p = torch.as_tensor(parameters, dtype=torch.float64).reshape(-1).cpu()
    if p.numel() == 1:
        p = p.expand(B)
    h = float(step)
    if plaquettes is None:
        plaquettes = plaquette_fields(U, qcd.geometry.D)
    near = bc_action_difference(qcd, U, p - h, p + h, plaquettes=plaquettes)
    far = bc_action_difference(qcd, U, p - 2.0 * h, p + 2.0 * h, plaquettes=plaquettes)
    return ((8.0 * near - far) / (12.0 * h)).to(torch.float64)


class GlobalRNGStream:
    """A private stream of the default generator of one device, swapped in only inside `with`.

    `distributedPT.ParallelTempering` draws the swap uniforms of rank 0 with
    `torch.rand(B, device=pt.device)`, i.e. from the default (global) generator of pt.device, and
    takes no generator argument. To make those draws reproducible across a checkpoint and independent
    of anything else that touches the global generator, the driver runs `pt.step` inside
    `with stream:`: on entry the global state of the device is saved and replaced by the stream's
    state, on exit the stream's advanced state is kept and the global state put back.
    `get_state` / `set_state` checkpoint the stream (a CPU ByteTensor, the format of
    `torch.get_rng_state` or `torch.cuda.get_rng_state`).
    """

    def __init__(self, device, seed):
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            raise ValueError("GlobalRNGStream needs an explicit CUDA index (cuda:K)")
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError(f"unsupported device {self.device}")
        self.state = torch.Generator(device=self.device).manual_seed(int(seed)).get_state().cpu().clone()
        self._saved = None

    def _get(self):
        if self.device.type == "cuda":
            return torch.cuda.get_rng_state(self.device).clone()
        return torch.get_rng_state().clone()

    def _set(self, state):
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(state.cpu(), self.device)
        else:
            torch.set_rng_state(state.cpu())

    def __enter__(self):
        self._saved = self._get()
        self._set(self.state)
        return self

    def __exit__(self, *exc):
        self.state = self._get()
        self._set(self._saved)
        self._saved = None
        return False

    def get_state(self):
        return self.state.clone()

    def set_state(self, state):
        self.state = torch.as_tensor(state).cpu().clone()


class FermionREXTarget:
    """A `QCDAction(parameter_name="bc")` seen as a tempering target of `distributedPT`.

    Args:
        qcd: `QCDAction` with parameter_name "bc" (its defect_mask fixes the defect).
        tau, n_steps, integrator: the HMC of `trajectory` (`hmc.HMC`).

    Interface:
        set_parameter(bc, parameter_name="bc")   bc: scalar or (B,) tensor, one value per local
                                                  configuration (`pt.local_parameters`);
        parameter                                 the current bc;
        action(U, bc=None, parameter_name="bc")   full S_g + S_pf, (B,) (one solve; needs phi);
        gauge_action(U, bc=None)                  S_g only;
        action_difference(U, bc_old, bc_new, parameter_name="bc") -> (B,)
                                                  gauge-only, no solve (used by the O-REX classes);
        action_table(U, ladder) -> (B, n_rungs)   relative table for ParallelTempering.step;
        trajectory(U, generator)                  one HMC trajectory at the current bc.
    """

    parameter_name = "bc"

    def __init__(self, qcd, tau=1.0, n_steps=2, integrator="omelyan"):
        if not isinstance(qcd, QCDAction) or qcd.parameter_name != "bc":
            raise ValueError("FermionREXTarget needs a QCDAction with parameter_name='bc'")
        self.qcd = qcd
        self.tau = float(tau)
        self.n_steps = int(n_steps)
        self.integrator = integrator

    @property
    def parameter(self):
        return self.qcd.parameter

    def set_parameter(self, parameter, parameter_name="bc"):
        _check_name(parameter_name)
        self.qcd.set_parameter(parameter)

    def action(self, U, parameter=None, parameter_name="bc"):
        _check_name(parameter_name)
        return self.qcd.action(U, parameter)

    def gauge_action(self, U, parameter=None):
        return self.qcd.gauge_action(U, parameter)

    def action_difference(self, U, parameter_old, parameter_new, parameter_name="bc"):
        _check_name(parameter_name)
        return self.qcd.action_difference(U, parameter_old, parameter_new)

    def action_table(self, U, ladder, parameter=None, plaquettes=None):
        """S(U_b; ladder[k]) - S(U_b; bc_b), (B, n_rungs) float64, bc_b the current (or given) bc.

        `ParallelTempering._attempt_swaps_exact_actions` only uses differences of a row (module
        doc), so this replaces the absolute table without a solve.
        """
        ladder = torch.as_tensor(ladder, dtype=torch.float64).reshape(-1).cpu()
        current = self.qcd.parameter if parameter is None else parameter
        return swap_action_difference(self.qcd, U, current, ladder, plaquettes=plaquettes)

    def hmc(self):
        """`hmc.HMC` on the joint Hamiltonian at the current bc (cheap to build)."""
        return HMC(self.qcd.geometry, self.qcd.hamiltonian(), self.tau, self.n_steps, self.integrator)

    @torch.no_grad()
    def trajectory(self, U, generator=None):
        """One fermionic HMC trajectory of every local configuration at its own bc."""
        return self.hmc().trajectory(U, generator)


@torch.no_grad()
def distributed_rex_step(pt, target, U, generator=None, periodic_translation=False, swap_rng=None):
    """One REX step of a rank of `distributedPT.ParallelTempering` with a `FermionREXTarget`.

    trajectory at the current labels -> relative action table at pt.ladder (no solve) ->
    pt.step(local_actions=table) (gather, swaps on rank 0, scatter) -> the new labels are set on
    the target -> optional periodic translation of the configurations at bc = 1.
    Returns (U, swap stats of pt.step, HMC stats). pt.config.acceptance_rule must be "exact" or
    "auto". `periodic_translation=True` needs bc = 1 on the top rung of pt.ladder (ValueError
    otherwise); after the translation `target.qcd.phi` belongs to the untranslated U, so the full
    action must not be evaluated before the next trajectory refreshes phi.
    `swap_rng`: optional `GlobalRNGStream` of pt.device; pt.step then draws rank 0's swap uniforms
    from that stream (checkpointable) instead of the bare global generator.
    """
    if periodic_translation and float(pt.ladder[-1]) != 1.0:
        raise ValueError(f"periodic_translation needs bc = 1 on the top rung, got {float(pt.ladder[-1])}")
    target.set_parameter(pt.local_parameters.to(torch.float64).cpu())
    U, accepted, dH, hmc_stats = target.trajectory(U, generator)
    hmc_stats = dict(hmc_stats, accepted=accepted, dH=dH)
    table = target.action_table(U, pt.ladder.cpu())
    with (swap_rng if swap_rng is not None else nullcontext()):
        stats = pt.step(local_actions=table.to(device=pt.device))
    target.set_parameter(pt.local_parameters.to(torch.float64).cpu())
    if periodic_translation:
        U = translate_periodic_replica(U, pt.local_parameters.cpu(), 1.0, target.qcd.geometry.D, generator)
    return U, stats, hmc_stats


class FermionREX:
    """Single-process REX of n_chains independent ladders, one batched QCDAction (module doc).

    Args:
        geometry: `lattice.Geometry`.
        beta, kappa: couplings, common to all rungs.
        defect_mask: (2, D) + lattice, layout of `utils.Defect`.
        ladder: increasing bc values, rung 0 first (the periodic rung bc = 1 last).
        n_chains: independent ladders (the error of a rung observable comes from their spread).
        tau, n_steps, integrator, n_gauge_substeps: the HMC.
        swap_parity: "alternating" (even pairs at even steps, odd pairs at odd steps: DEO) or
            "even_odd" (both rounds every step), as `PTConfig.swap_parity`.
        periodic_translation: translate the bc = 1 configurations after the swaps (module doc).
        U: initial links (B = n_rungs * n_chains); default cold start.
        swap_generator: CPU `torch.Generator` for the swap uniforms (kept apart from the generator
            of the trajectories, which may live on the device). If None, a CPU generator is seeded
            with `swap_seed`, or, if that is None too, at the first `step` with one int64 drawn
            from the step's generator (from the global RNG if that is None). A resumable run
            passes its own generator and saves its state.
        Extra keyword arguments go to `TwoFlavourWilson` (via QCDAction).

    Device: U and the HMC may live on CUDA; labels, swap table, swap uniforms and history are kept
    on the CPU (per-rung reductions run on the device, values are moved to the CPU when stored).

    State: `U` (B, D, lattice, N, N), entry b = slot * n_chains + chain; `labels` (n_rungs,
    n_chains) rung carried by each slot; `labels_history` (after the swaps of each step),
    `trajectory_labels_history` (the rungs at which the trajectories ran), `plaquette_history`,
    `exp_minus_dH_history`, all lists of (n_rungs, n_chains) tensors.
    """

    def __init__(self, geometry, beta, kappa, defect_mask, ladder, n_chains=1, tau=1.0, n_steps=2,
                 integrator="omelyan", n_gauge_substeps=3, swap_parity="alternating",
                 periodic_translation=False, U=None, swap_generator=None, swap_seed=None, **fermion_kwargs):
        if swap_parity not in ("alternating", "even_odd"):
            raise ValueError(f"swap_parity must be 'alternating' or 'even_odd', got {swap_parity!r}")
        self.geometry = geometry
        self.ladder = _as_ladder(ladder)
        self.n_rungs = int(self.ladder.numel())
        self.n_chains = int(n_chains)
        self.B = self.n_rungs * self.n_chains
        self.swap_parity = swap_parity
        self.periodic_translation = bool(periodic_translation)
        if self.periodic_translation and float(self.ladder[-1]) != 1.0:
            raise ValueError("periodic_translation needs bc = 1 on the top rung")
        self.labels = torch.arange(self.n_rungs).view(-1, 1).repeat(1, self.n_chains)
        self.qcd = QCDAction(geometry, "bc", beta=beta, kappa=kappa, bc=self.parameters(),
                             defect_mask=defect_mask, n_gauge_substeps=n_gauge_substeps, **fermion_kwargs)
        self.target = FermionREXTarget(self.qcd, tau=tau, n_steps=n_steps, integrator=integrator)
        if U is None:
            eye = torch.eye(geometry.N, dtype=torch.cdouble)
            U = eye.expand(geometry.link_shape(self.B)).clone()
        if U.shape[0] != self.B:
            raise ValueError(f"U must hold n_rungs * n_chains = {self.B} configurations, got {U.shape[0]}")
        self.U = U
        if swap_generator is None and swap_seed is not None:
            swap_generator = torch.Generator().manual_seed(int(swap_seed))
        if swap_generator is not None and swap_generator.device.type != "cpu":
            raise ValueError("swap_generator must be a CPU generator")
        self.swap_generator = swap_generator
        self.n_steps_done = 0
        self.link_accepted = torch.zeros(self.n_rungs - 1, dtype=torch.long)
        self.link_proposed = torch.zeros(self.n_rungs - 1, dtype=torch.long)
        self.labels_history = []
        self.trajectory_labels_history = []
        self.plaquette_history = []
        self.exp_minus_dH_history = []
        self.defect_action_history = []  # S_D = dS/dc of each slot's configuration at its bc after the step

    # ---- labels ----------------------------------------------------------------------------
    def parameters(self):
        """(B,) bc of every batch entry (slot-major) from the current labels."""
        return self.ladder[self.labels].reshape(-1)

    def _slot_of_rung(self):
        """(n_rungs, n_chains): slot carrying rung k in chain c (inverse of `labels`)."""
        pos = torch.empty_like(self.labels)
        pos.scatter_(0, self.labels, torch.arange(self.n_rungs).view(-1, 1).expand_as(self.labels).contiguous())
        return pos

    def _parities(self):
        if self.swap_parity == "even_odd":
            return ("even", "odd")
        return ("even",) if self.n_steps_done % 2 == 0 else ("odd",)

    # ---- swaps -----------------------------------------------------------------------------
    def swap_round(self, table, parity, generator=None):
        """One round of swaps between rungs (k, k + 1), k even or odd, in every chain.

        `table` is the (B, n_rungs) relative action table (`swap_action_difference`) of the
        current configurations; it stays valid across rounds (the configurations do not move).
        The uniforms come from the CPU `generator` (default `self.swap_generator`).
        Returns per-pair (accepted, proposed) counts, length n_rungs - 1, and dS of the attempts.
        """
        if generator is None:
            generator = self.swap_generator
        table = table.cpu()
        C = self.n_chains
        chains = torch.arange(C)
        accepted = torch.zeros(self.n_rungs - 1, dtype=torch.long)
        proposed = torch.zeros(self.n_rungs - 1, dtype=torch.long)
        delta = {}
        start = 0 if parity == "even" else 1
        for k in range(start, self.n_rungs - 1, 2):
            pos = self._slot_of_rung()
            b_i = pos[k] * C + chains  # configuration at rung k
            b_j = pos[k + 1] * C + chains  # configuration at rung k + 1
            dS = (table[b_i, k + 1] - table[b_i, k]) + (table[b_j, k] - table[b_j, k + 1])
            u = torch.rand(C, generator=generator, dtype=torch.float64)
            accept = torch.log(u) < -dS
            new_i = torch.where(accept, torch.full_like(b_i, k + 1), torch.full_like(b_i, k))
            new_j = torch.where(accept, torch.full_like(b_j, k), torch.full_like(b_j, k + 1))
            self.labels[pos[k], chains] = new_i
            self.labels[pos[k + 1], chains] = new_j
            accepted[k] += int(accept.sum())
            proposed[k] += C
            delta[k] = dS
        return accepted, proposed, delta

    # ---- one REX step ----------------------------------------------------------------------
    @torch.no_grad()
    def step(self, generator=None):
        """Trajectory of every replica at its bc, swap round(s), optional translation; stats.

        `generator` drives the trajectory and the translation (on the device of U); the swaps use
        `self.swap_generator` (class doc). After the translation `self.qcd.phi` belongs to the
        untranslated U: do not evaluate the full action before the next step refreshes phi.
        """
        if self.swap_generator is None:
            device = generator.device if generator is not None else "cpu"
            seed = int(torch.randint(0, 2**62, (1,), generator=generator, device=device))
            self.swap_generator = torch.Generator().manual_seed(seed)
        traj_labels = self.labels.clone()
        self.target.set_parameter(self.parameters())
        U, accepted, dH, hmc_stats = self.target.trajectory(self.U, generator)
        plaquettes = plaquette_fields(U, self.geometry.D)  # once per step (twice with translation)
        table = swap_action_difference(self.qcd, U, self.parameters(), self.ladder, plaquettes=plaquettes).cpu()
        acc = torch.zeros(self.n_rungs - 1, dtype=torch.long)
        prop = torch.zeros(self.n_rungs - 1, dtype=torch.long)
        delta = {}
        for parity in self._parities():
            a, p, d = self.swap_round(table, parity)
            acc += a
            prop += p
            delta.update(d)
        self.target.set_parameter(self.parameters())
        if self.periodic_translation:
            U = translate_periodic_replica(U, self.parameters(), 1.0, self.geometry.D, generator)
            plaquettes = plaquette_fields(U, self.geometry.D)
        self.U = U
        self.n_steps_done += 1
        self.link_accepted += acc
        self.link_proposed += prop

        shape = (self.n_rungs, self.n_chains)
        plaq, sd = step_observables(self.qcd, U, self.parameters(), plaquettes=plaquettes)
        plaq, sd = plaq.view(shape), sd.view(shape)
        x = hmc_stats["exp_minus_dH"].to(torch.float64).view(shape)
        self.defect_action_history.append(sd.cpu().clone())
        self.labels_history.append(self.labels.clone())
        self.trajectory_labels_history.append(traj_labels)
        self.plaquette_history.append(plaq.cpu().clone())
        self.exp_minus_dH_history.append(x.cpu().clone())
        return {
            "hmc_acceptance": float(accepted.to(torch.float64).mean()),
            "link_accepted": acc.tolist(),
            "link_proposed": prop.tolist(),
            "swap_dS": delta,
            "plaquette_per_rung": self._per_rung(plaq, self.labels).cpu().tolist(),
            "exp_minus_dH_per_rung": self._per_rung(x, traj_labels).cpu().tolist(),
        }

    def _per_rung(self, values, labels):
        """Mean over chains of the value carried by each rung, (n_rungs,), on the device of values."""
        out = torch.zeros(self.n_rungs, dtype=torch.float64, device=values.device)
        out.scatter_add_(0, labels.reshape(-1).to(values.device), values.reshape(-1).to(torch.float64))
        return out / self.n_chains

    def run(self, n_steps, generator=None):
        return [self.step(generator) for _ in range(int(n_steps))]

    # ---- history ---------------------------------------------------------------------------
    @property
    def swap_acceptance(self):
        """Accepted / proposed swaps per pair of neighbouring rungs since the start."""
        return (self.link_accepted.to(torch.float64) / self.link_proposed.clamp(min=1).to(torch.float64)).tolist()

    def history_arrays(self, start=0):
        """numpy (labels, plaquette, trajectory_labels, exp_minus_dH), each (steps, n_rungs, n_chains),
        in the layout of `neoqcd.replica_history` (world_size -> slot, B -> chain)."""
        stack = lambda xs: torch.stack(xs[start:], dim=0).cpu().numpy()
        return (stack(self.labels_history), stack(self.plaquette_history),
                stack(self.trajectory_labels_history), stack(self.exp_minus_dH_history))

    def defect_action_array(self, start=0):
        """numpy (steps, n_rungs, n_chains) S_D = dS/dc after each step (`defect_action`), layout of
        `history_arrays`: the `defect_action` array of main/optimal_ladder.py."""
        return torch.stack(self.defect_action_history[start:], dim=0).cpu().numpy()

    def rung_series(self, values, labels, rung):
        """(steps, n_chains) series of the replica carrying `rung` (`replica_observable_series`)."""
        return replica_observable_series(labels, values, rung)

    def summarize(self, start=0):
        """`replica_history.summarize_history` of the run: round trips, flux, plaquette tau_int on
        the bottom (bc = ladder[0]) and top (bc = ladder[-1]) rungs, and the mean and variance of the
        defect action on every rung when its history covers every step."""
        labels, plaq, _, _ = self.history_arrays(start)
        sd = None
        if len(self.defect_action_history) == len(self.labels_history):
            sd = self.defect_action_array(start)
        return summarize_history(labels, plaq, self.n_rungs, defect_action=sd)
