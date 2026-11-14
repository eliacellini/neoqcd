"""Non-equilibrium MCMC (Jarzynski) with N_f = 2 Wilson fermions: phase 3 of fermions_plan.md.

The fermionic analogue of `neoqcd.flow.Flow` with flow_type='nemcmc' (no deterministic layers
yet): a protocol lambda_0 -> lambda_1 -> ... -> lambda_n of one parameter of a `QCDAction`
(beta, kappa or bc), with an HMC block (`HMCBlock`) at each new value. The state is the pair
x = (U, phi): links and pseudofermion field.

Bookkeeping (S(x; lambda) is the joint action of `QCDAction`, x_i the state after block i,
x_0 the initial state, drawn from p_0 ~ exp(-S(x; lambda_0))):

    step i = 1 .. n:
      jump at fixed fields:  dW_i = S(x_{i-1}; lambda_i) - S(x_{i-1}; lambda_{i-1})
      HMC block at lambda_i: dQ_i = S(x_i; lambda_i)     - S(x_{i-1}; lambda_i)

    W = sum_i dW_i,   Q = sum_i dQ_i,   so   W + Q = S(x_n; lambda_n) - S(x_0; lambda_0)

(the identity W = S_final - S_0 - Q of `Flow.forward`, logJ = 0 here). The heat dQ_i contains
everything that happens inside the block: the pseudofermion heat baths phi -> phi' (one at the
start of every HMC trajectory, `HMC.trajectory` calls `refresh`) and the molecular-dynamics +
Metropolis moves of U at fixed phi. Each trajectory (Gibbs step on phi, then HMC on U at fixed phi)
leaves exp(-S(U, phi; lambda_i)) invariant, so Crooks/Jarzynski holds on the joint (U, phi):

    <exp(-W)> = Z(lambda_n) / Z(lambda_0),   Z(lambda) = int dU dphi dphi^* exp(-S(U, phi; lambda)),

and Z(lambda) = pi^n_phi int dU det(D_hat^dagger D_hat)(kappa) exp(-S_g(U; beta, bc)), the gauge-
fermion partition function up to a lambda-independent constant.

How the jumps are computed: for beta and bc with `QCDAction.action_difference` (no solve; for bc
gauge only, the fermion part of the work is identically zero); for kappa as S(x; kappa_i) minus
the action S(x; kappa_{i-1}) already known from the end of the previous block (one solve per step,
not two). A zero jump (lambda_i == lambda_{i-1}) is exactly zero, so a constant protocol has W = 0
exactly. S(x_i; lambda_i) at the end of a block is always recomputed from scratch.

Staggered fermions: with a `QCDAction(..., fermions="staggered")` everything above holds with the
mass in place of kappa (protocol parameter "mass", one solve per jump as for kappa) and
Z(lambda) ~ int dU det M_ee(m) exp(-S_g(U; beta, bc)); FieldState.phi is then the even-site
staggered field. Nothing in this module depends on the fermion discretisation beyond that.

Fast path: with `QCDAction(..., fast=True)` (Wilson) nothing changes here: the fast fermion term
reads and writes phi as a full-lattice tensor (odd sites zero, as the reference even-odd term),
so `FieldState.phi` has the same layout on both paths and round-trips exactly through
`QCDAction.phi`; a kappa jump still costs one solve.

Accumulators: `FieldState` carries `work`, `heat` and `logJ` per configuration, so a run can be
continued from its final state; `FermionNEMCMC.forward` also returns W and Q of the run alone.
"""

from dataclasses import dataclass
from typing import Optional

import math

import torch

from neoqcd.fermions._gauge import mean_plaquette, plaquette_fields
from neoqcd.fermions.action import _as_parameter, _same_parameter
from neoqcd.fermions.hmc import HMC

__all__ = ["FieldState", "HMCBlock", "FermionNEMCMC", "OBSERVABLES", "effective_sample_size",
           "jarzynski_free_energy"]

# columns of int_obs, as in flow.Flow.forward
OBSERVABLES = ("work", "heat", "logJ", "action", "parameter", "plaquette", "plaquette_squared")


@dataclass
class FieldState:
    """Links U (B, D, lattice, N, N), pseudofermion field phi (None before the first refresh) and
    the per-configuration accumulators work, heat, logJ (float64 (B,), zero by default)."""

    U: torch.Tensor
    phi: Optional[torch.Tensor] = None
    work: Optional[torch.Tensor] = None
    heat: Optional[torch.Tensor] = None
    logJ: Optional[torch.Tensor] = None

    def __post_init__(self):
        B = self.U.shape[0]
        for name in ("work", "heat", "logJ"):
            if getattr(self, name) is None:
                setattr(self, name, torch.zeros(B, dtype=torch.float64, device=self.U.device))


class HMCBlock:
    """The stochastic block of the protocol: `n_traj` fermionic HMC trajectories at a fixed
    parameter value, each with its own pseudofermion heat bath (by `HMC.trajectory`).

    `__call__(U, parameter, generator, S_old)` returns (U_new, dQ, S_new, info) with
    S_new = S(U_new, phi_new; parameter) computed from scratch and dQ = S_new - S_old, where S_old
    = S(U, phi; parameter) for the incoming fields (computed if not given; phi must then be set).
    phi_new is left in `action.phi`. info: acceptance and <exp(-dH)> per trajectory (lists).
    """

    def __init__(self, action, tau=1.0, n_steps=2, n_traj=1, integrator="omelyan", exp_method="auto"):
        if int(n_traj) < 1:
            raise ValueError("n_traj must be >= 1")
        self.action = action
        self.tau = float(tau)
        self.n_steps = int(n_steps)
        self.n_traj = int(n_traj)
        self.integrator = integrator
        self.exp_method = exp_method

    @torch.no_grad()
    def __call__(self, U, parameter, generator=None, S_old=None):
        if S_old is None:
            S_old = self.action.action(U, parameter)
        hmc = HMC(self.action.geometry, self.action.hamiltonian(parameter), self.tau, self.n_steps,
                  integrator=self.integrator, exp_method=self.exp_method)
        info = {"acceptance": [], "exp_minus_dH": []}
        for _ in range(self.n_traj):
            U, _, _, stats = hmc.trajectory(U, generator)
            info["acceptance"].append(stats["acceptance"])
            info["exp_minus_dH"].append(stats["exp_minus_dH"])
        S_new = self.action.action(U)
        return U, S_new - S_old, S_new, info


class FermionNEMCMC:
    """NE-MCMC over the protocol initial_parameter -> protocol[0] -> ... -> protocol[-1].

    Args:
        action: `QCDAction` (its `parameter_name` is the protocol parameter).
        protocol: the values lambda_1 .. lambda_n (floats, or (B,) tensors).
        initial_parameter: lambda_0; default the current parameter of `action`.
        block: an `HMCBlock`, or None to build one from `hmc_kwargs` (tau, n_steps, n_traj,
            integrator, exp_method).

    `forward(state0, generator=None, int_meas=None)` -> (state, W, Q, int_obs):
        state0: `FieldState` drawn from equilibrium at lambda_0 for U; if state0.phi is None the
            pseudofermions are drawn here by a heat bath at lambda_0 (an exact Gibbs step, so
            (U, phi) is in joint equilibrium); otherwise phi is used as given;
        state: final `FieldState` (U, phi, accumulators = those of state0 plus this run);
        W, Q: work and heat of this run (B,), with W + Q = S_final - S_0 (module doc);
        int_obs: (B, int_meas, 7) float64, columns `OBSERVABLES` as in `Flow.forward` (running
            work and heat, logJ = 0, action S(x_i; lambda_i), parameter, mean plaquette and its
            square), measured after every n / int_meas blocks (default: every block).
    After a run `last_run` holds S_0, S_final (B,) and the per-block HMC info.
    """

    def __init__(self, action, protocol, initial_parameter=None, block=None, **hmc_kwargs):
        self.action = action
        self.protocol = [_as_parameter(p) for p in protocol]
        if not self.protocol:
            raise ValueError("the protocol needs at least one value")
        self.initial_parameter = action.parameter if initial_parameter is None else _as_parameter(initial_parameter)
        self.block = HMCBlock(action, **hmc_kwargs) if block is None else block
        self.last_run = None

    @property
    def n_step(self):
        return len(self.protocol)

    def _jump(self, U, p_old, p_new, S_cur):
        """(dW, S(x; p_new)) at fixed fields, given S_cur = S(x; p_old)."""
        if _same_parameter(p_old, p_new):
            return torch.zeros_like(S_cur), S_cur
        if self.action.parameter_name in ("kappa", "mass"):  # fermion parameter: one solve
            S_new = self.action.action(U, p_new)
            return S_new - S_cur, S_new
        dW = self.action.action_difference(U, p_old, p_new)
        return dW, S_cur + dW

    @torch.no_grad()
    def forward(self, state0, generator=None, int_meas=None):
        n = self.n_step
        int_meas = n if int_meas is None else int(int_meas)
        if int_meas < 1 or n % int_meas:
            raise ValueError("int_meas must be a positive divisor of the number of protocol steps")
        per_meas = n // int_meas
        act = self.action
        U = state0.U
        B = U.shape[0]
        act.set_parameter(self.initial_parameter)
        if state0.phi is None:
            act.refresh(U, generator)
        else:
            act.phi = state0.phi
        S0 = act.action(U)
        S_cur = S0
        W = torch.zeros(B, dtype=torch.float64, device=U.device)
        Q = torch.zeros_like(W)
        int_obs = torch.zeros((B, int_meas, len(OBSERVABLES)), dtype=torch.float64, device=U.device)
        p_old = self.initial_parameter
        infos = []
        m = 0
        for i, p_new in enumerate(self.protocol):
            dW, S_cur = self._jump(U, p_old, p_new, S_cur)
            W = W + dW
            U, dQ, S_cur, info = self.block(U, p_new, generator, S_old=S_cur)
            Q = Q + dQ
            infos.append(info)
            p_old = p_new
            if (i + 1) % per_meas == 0:
                pl = mean_plaquette(plaquette_fields(U, act.geometry.D), act.geometry.N)
                int_obs[:, m, 0] = W
                int_obs[:, m, 1] = Q
                int_obs[:, m, 3] = S_cur
                int_obs[:, m, 4] = torch.as_tensor(p_new, dtype=torch.float64)
                int_obs[:, m, 5] = pl
                int_obs[:, m, 6] = pl.square()
                m += 1
        self.last_run = {"S0": S0, "S_final": S_cur, "block_info": infos}
        state = FieldState(U, act.phi, state0.work + W, state0.heat + Q, state0.logJ.clone())
        return state, W, Q, int_obs

    __call__ = forward


def _log_mean_exp_minus(W):
    """log <exp(-W)> along the last axis, stable."""
    return torch.logsumexp(-W, dim=-1) - math.log(W.shape[-1])


def jarzynski_free_energy(W):
    """Jarzynski estimate dF = -log <exp(-W)> (= -log Z_n / Z_0) with a delete-one jackknife error.

    Returns (dF, error) as floats; exp(-dF) estimates Z_n / Z_0 with relative error `error`.
    """
    W = torch.as_tensor(W, dtype=torch.float64).reshape(-1)
    n = W.numel()
    if n < 2:
        raise ValueError("need at least two work values")
    dF = -_log_mean_exp_minus(W).item()
    shift = W.min()
    e = torch.exp(-(W - shift))
    loo = (e.sum() - e) / (n - 1)  # leave-one-out means of exp(-(W - shift))
    jk = -(torch.log(loo) - shift)
    err = math.sqrt((n - 1) / n * ((jk - jk.mean()) ** 2).sum().item())
    return dF, err


def effective_sample_size(W):
    """Normalised ESS = <exp(-W)>^2 / <exp(-2W)> in (0, 1] (as `Flow.compute_metrics`)."""
    W = torch.as_tensor(W, dtype=torch.float64).reshape(-1)
    Wc = W - W.mean()
    return (torch.exp(2.0 * _log_mean_exp_minus(Wc) - _log_mean_exp_minus(2.0 * Wc))).item()
