"""Stochastic normalizing flow with N_f = 2 Wilson fermions: phase 4, step 2 of fermions_plan.md.

`FermionSNF` is `snf.FermionNEMCMC` with a deterministic (trainable) gauge layer before every HMC
block, in the order of the pure-gauge `neoqcd.flow.Flow` with flow_type='snf': at step i the
layer conditioned on lambda_i, then the stochastic block at lambda_i. The layers are the
pure-gauge `CouplingLayer` (stout) or `ResidualCouplingLayer`: they act on U only; the
pseudofermion field phi is held fixed across a layer (it is resampled in every HMC trajectory, so
no conditional flow on phi is needed, see the plan, "Cosa cambia con i fermioni").

Bookkeeping (S(x; lambda) = S_g(U; beta, bc) + S_pf(U, phi; kappa) of `QCDAction`, x = (U, phi)):

    step i = 1 .. n, starting from x_{i-1}:
      jump at fixed fields:   dW_i^jump  = S(x_{i-1}; lambda_i) - S(x_{i-1}; lambda_{i-1})
      layer at lambda_i:      x'_i = (U'_i, phi_{i-1}),  U'_i = f_i(U_{i-1}),
                              dW_i^layer = S(x'_i; lambda_i) - S(x_{i-1}; lambda_i) - log J_i
      HMC block at lambda_i:  dQ_i       = S(x_i; lambda_i)  - S(x'_i; lambda_i)

    W = sum_i (dW_i^jump + dW_i^layer),  Q = sum_i dQ_i,  logJ = sum_i log J_i,
    W + Q = S(x_n; lambda_n) - S(x_0; lambda_0) - logJ,

the identity `w = st - s0 - Q - logJ` of `Flow.__call__`. log J_i is the log-Jacobian the pure-gauge
layer returns, treated as in `Flow._full_logdet`. For SU(3) it is the exact Haar log-Jacobian
log |det df_i/dU| (checked against the dense tangent-space Jacobian to ~1e-14, test_fermions_snf);
for SU(2) the stout value is doubled by the legacy rule of `Flow._full_logdet`, which is NOT the
exact Haar log-Jacobian (relative error ~1e-3 measured at rho = 0.01-0.1), so Jarzynski is only
approximate there (`FermionLayerBlock` warns). Jarzynski: <exp(-W)> = Z(lambda_n)/Z(lambda_0) on the joint (U, phi), since
each layer is a bijection of U at fixed phi with Jacobian J_i and each HMC trajectory leaves
exp(-S(x; lambda_i)) invariant.

How the gauge layer's own outputs are used (nothing is counted twice). The pure-gauge layers
return (U', dQ_layer, dlogJ). In the present code dQ_layer is identically 0 for both layer types
(`Smearing._forward_blocks` and `ResidualNormalizingFlows.forward` return 0): it is not the gauge
action change. `Flow` never uses an explicit per-layer action change either: its work is
st - s0 - Q - logJ with st the final action, so the gauge part of the layer work is implicit.
Here the layer work is explicit: dW^layer = [S_g(U') - S_g(U)] + [S_pf(U', phi) - S_pf(U, phi)]
- log J, both action differences at lambda_i computed by `QCDAction` (S_g by autograd through
`WilsonGaugeAction`, S_pf by `PseudofermionAction`). `FermionLayerBlock` raises if a layer ever
returns a nonzero dQ_layer, so that a future layer with its own action bookkeeping cannot be
added silently on top. dlogJ goes into the logJ accumulator (and into W with a minus sign).

Cost per step (one solve = one CG solve of D_hat^dagger D_hat):
- layer: one solve, S_pf(U', phi; kappa_i); the force for the gradient comes from the same solve;
  S(x_{i-1}; lambda_i) is known (from the end of the previous block and the jump);
- jump: beta and bc no solve, kappa (or staggered mass) one solve (as in FermionNEMCMC);
- HMC block: as in FermionNEMCMC (S at its end recomputed from scratch, one solve).

Gradient for training. With the HMC blocks under no_grad and their outputs detached, the work of
a run depends on the layer parameters theta only through U'_i = f_i(U_{i-1}; theta) at fixed
U_{i-1}, and
    dW/dtheta = sum_i d/dtheta [S(U'_i, phi_{i-1}; lambda_i) - log J_i],
which is the estimator of `main_nemcmc_snf.blockwise_training_forward` (x detached at every
block, loss = action after the layer minus its logdet, summed over blocks); the loss is mean(W)
(`FermionSNF.loss`), as `_backprop_local_loss` on the work there. The S_pf part of the gradient
goes through `PseudofermionAction` (one solve, no backpropagation through CG), the gauge part and
log J through autograd as in the pure-gauge code.

Wirtinger gradient of S_pf (`PseudofermionAction.backward`). PyTorch stores for a real loss L of
complex z the gradient G = dL/dRe z + i dL/dIm z, so that dL = Re sum_ab conj(G_ab) dz_ab
= Re tr(G^dagger dz) per link. The force F (hermitian traceless, `TwoFlavourWilson.force`) gives
the derivative along the group: tr(F T) = d/dt S(e^{itT} U)|_0 for hermitian traceless T, i.e.
along dU = i T U dt. Take G = i F U. Then for dU = i T U dt
    Re tr(G^dagger dU) = Re tr(-i U^dagger F i T U) dt = Re tr(F T) dt = tr(F T) dt,
the correct directional derivative for every tangent direction. G itself lies in the tangent
space at U (G U^dagger = i F is anti-hermitian traceless): it is the tangent projection of the
full Wirtinger gradient 2 dS/dU^* of S as a function of unconstrained complex U, and the normal
part (which a dense autograd would also produce) does not matter, because the layers' outputs are
exactly SU(N) as functions of their inputs and parameters, so every vector-Jacobian product
through them annihilates it. Conversely F = project_hermitian_traceless(-i G U^dagger) recovers
the force from any gradient G with the same tangent part (G = i F U + normal part).

Staggered fermions (`QCDAction(..., fermions="staggered", mass=m)`): the same construction with
S_pf = phi^dagger M_ee(m)^-1 phi of `StaggeredPseudofermion` (one even-site field, N_f = 2^(D/2)),
kappa replaced by the mass (`QCDAction.fermion_parameter_name`), and the force from the same solve,
F = 2 staggered_hopping_derivative(op, X, K_oe X) with X = M_ee^-1 phi (its algebra gradient
convention is that of `TwoFlavourWilson.force`, so the backward G = i F U is unchanged). A mass
jump costs one solve, as a kappa jump. The fermion term type is dispatched by isinstance.
"""

import warnings

import torch

from neoqcd.fermions._gauge import (CouplingLayer, ResidualCouplingLayer, mean_plaquette,
                                    plaquette_fields)
from neoqcd.fermions.action import _as_parameter, _same_parameter
from neoqcd.fermions.pseudofermion import TwoFlavourWilson, hopping_derivative
from neoqcd.fermions.staggered_pseudofermion import StaggeredPseudofermion, staggered_hopping_derivative
from neoqcd.fermions.snf import OBSERVABLES, FieldState, HMCBlock

__all__ = ["PseudofermionAction", "pseudofermion_action", "joint_action", "FermionLayerBlock",
           "FermionSNF", "make_gauge_layers"]


def _fermion_kind(fermion):
    """"wilson" or "staggered" from the type of the fermion term (dispatch of this module)."""
    from neoqcd.fermions.hasenbusch import RatioWilsonTerm  # lazy: avoids an import cycle
    if isinstance(fermion, RatioWilsonTerm):
        raise TypeError(f"{type(fermion).__name__}: Hasenbusch terms are not supported in the SNF layers "
                        "yet (PseudofermionAction would return the plain TwoFlavourWilson S_pf, which is wrong)")
    if isinstance(fermion, StaggeredPseudofermion):
        return "staggered"
    if isinstance(fermion, TwoFlavourWilson):
        return "wilson"
    raise TypeError(f"unsupported fermion term {type(fermion).__name__}: PseudofermionAction handles "
                    "TwoFlavourWilson and StaggeredPseudofermion (and their parametric subclasses)")


def _staggered_force_from_solution(op, X):
    """The force of `StaggeredPseudofermion.force` from X = M_ee^-1 phi (no new solve):
    F = 2 staggered_hopping_derivative(op, X, W), W = K_oe X."""
    W = op.odd(op.staggered.hopping(X))
    return 2.0 * staggered_hopping_derivative(op, X, W)


def _force_from_solution(fermion, op, X):
    """The force of `TwoFlavourWilson.force` (or `StaggeredPseudofermion.force`) from a given
    solution X = M^-1 phi (no new solve)."""
    if _fermion_kind(fermion) == "staggered":
        return _staggered_force_from_solution(op, X)
    if fermion.even_odd:
        w = op.wilson
        Y = op.apply(X)
        W = op.odd(w.hopping(X))  # H_oe X
        Z = op.odd(w.gamma5(w.hopping(w.gamma5(Y))))  # (H^dagger)_oe Y
        G = hopping_derivative(w, Y + Z, X + W)
        k = w.kappa ** 2
    else:
        Y = op.apply(X)
        G = hopping_derivative(op, Y, X)
        k = op.kappa
    return (2.0 * k).view((-1,) + (1,) * (G.dim() - 1)).to(G.dtype) * G


class PseudofermionAction(torch.autograd.Function):
    """S_pf(U, phi; kappa) = phi^dagger (D_hat^dagger D_hat)^-1 phi, differentiable in U.

    `PseudofermionAction.apply(U, fermion)`: `fermion` is a `TwoFlavourWilson` (or the
    `ParametricTwoFlavourWilson` of a `QCDAction`) with phi set; its current kappa and phi are
    used. Staggered: `fermion` a `StaggeredPseudofermion` (or `ParametricStaggeredPseudofermion`),
    S = phi^dagger M_ee(m)^-1 phi at its current mass, force F = 2 staggered_hopping_derivative(
    op, X, K_oe X) from the same solve X = M_ee^-1 phi (`staggered_pseudofermion` module doc); any
    other fermion type raises TypeError. Forward: one CG solve at `fermion.action_tol`, S = Re phi^dagger X, (B,) float64; if U
    requires grad, the force F at U is computed from the same X and saved (so later changes of
    fermion.phi or kappa do not affect the backward). Backward: dS/dU = grad_out * i F U
    (module doc), no backpropagation through CG.
    """

    @staticmethod
    def forward(ctx, U, fermion):
        _fermion_kind(fermion)  # TypeError for unknown fermion terms
        op = fermion.operator(U)
        fermion.stats["action_calls"] += 1
        X = fermion._solve(op, fermion.action_tol)
        S = (fermion.phi.conj() * X).real.reshape(X.shape[0], -1).sum(1)
        if ctx.needs_input_grad[0]:
            fermion.stats["force_calls"] += 1
            F = _force_from_solution(fermion, op, X)
            ctx.save_for_backward(U, F)
        return S

    @staticmethod
    def backward(ctx, grad_out):
        U, F = ctx.saved_tensors
        g = grad_out.to(U.dtype).view((-1,) + (1,) * (U.dim() - 1))
        return g * (1j * torch.matmul(F, U)), None


def pseudofermion_action(U, fermion):
    return PseudofermionAction.apply(U, fermion)


def _fermion_parameter_accessors(action):
    """(get, set) of the fermion parameter named `action.fermion_parameter_name` ("kappa" for
    Wilson, "mass" for staggered) on `action.fermion`."""
    name = getattr(action, "fermion_parameter_name", "kappa")
    fermion = action.fermion
    if name == "kappa":
        return (lambda: fermion.kappa), fermion.set_kappa
    if name == "mass":
        return (lambda: fermion.mass), fermion.set_mass
    raise ValueError(f"unknown fermion parameter {name!r}")


def joint_action(action, U, parameter=None):
    """S(U, phi; parameter) of a `QCDAction`, differentiable in U: S_g by autograd, S_pf by
    `PseudofermionAction` (one solve) at the kappa (Wilson) or mass (staggered) of `parameter`;
    the current kappa or mass is kept."""
    _, value, _ = action._couplings(parameter)
    get, set_ = _fermion_parameter_accessors(action)
    saved = get()
    set_(value)
    try:
        S_pf = pseudofermion_action(U, action.fermion)
    finally:
        set_(saved)
    return action.gauge_action(U, parameter) + S_pf


def _is_zero(value):
    if torch.is_tensor(value):
        return bool((value == 0).all())
    return value == 0


class FermionLayerBlock(torch.nn.Module):
    """A pure-gauge deterministic layer inside the fermionic flow (module doc for the bookkeeping).

    Args:
        layer: `CouplingLayer` or `ResidualCouplingLayer` (any module with the signature
            `layer(U, flow_pars, rho_layer, beta=..., delta_beta=...) -> (U', dQ, dlogJ)`).
        action: the `QCDAction` (holds phi and the couplings).
        flow_pars: the `FlowPars` the layer was built with (mask, ...).
        condition, delta_condition: what the layer receives as beta and delta_beta (its
            conditioning for hyper/residual parametrisations), as `Flow._forward_layer`: the
            protocol value lambda_i and the forward difference lambda_{i+1} - lambda_i.

    `forward(state, parameter, S_old=None)` -> (new_state, dW, S_new, dlogJ):
        state: `FieldState` (U, phi); phi must be the field held by `action`;
        parameter: lambda at which the work is measured;
        S_old: S(U, phi; parameter) if known (else computed, one solve, no grad);
        dW = S_new - S_old - dlogJ with S_new = S(U', phi; parameter) (differentiable), the layer
        dQ checked to be zero and not used; new_state = (U', phi, work + dW, heat, logJ + dlogJ).
    """

    def __init__(self, layer, action, flow_pars, condition=0.0, delta_condition=0.0):
        super().__init__()
        self.layer = layer
        self.action = action
        self.flow_pars = flow_pars
        self.condition = float(condition)
        self.delta_condition = float(delta_condition)
        if action.geometry.N == 2 and not isinstance(layer, ResidualCouplingLayer):
            warnings.warn("FermionLayerBlock with SU(2) and a stout layer: the log-Jacobian follows the "
                          "legacy 2x rule of Flow._full_logdet, which is not the exact Haar log-Jacobian "
                          "(relative error ~1e-3); the work is only approximately Jarzynski-exact",
                          RuntimeWarning, stacklevel=2)

    def _full_logdet(self, logdet):
        # As Flow._full_logdet (legacy SU(2) stout rule: logdet doubled). For SU(3) the layer's
        # dlogJ is the exact Haar log-Jacobian (to ~1e-14); for SU(2) the doubled value is not
        # (relative error ~1e-3 at rho = 0.01-0.1, measured), see the module doc.
        if self.action.geometry.N == 2 and not isinstance(self.layer, ResidualCouplingLayer):
            return 2.0 * logdet
        return logdet

    def forward(self, state, parameter, S_old=None):
        U = state.U
        if S_old is None:
            with torch.no_grad():
                S_old = self.action.action(U.detach(), parameter)
        U_new, dQ_layer, dlogJ = self.layer(U, self.flow_pars, None, beta=self.condition,
                                            delta_beta=self.delta_condition)
        if not _is_zero(dQ_layer):
            raise RuntimeError("the gauge layer returned a nonzero dQ; FermionLayerBlock computes the "
                               "layer's action change itself and would count it twice")
        dlogJ = self._full_logdet(torch.as_tensor(dlogJ, dtype=torch.float64, device=U.device)
                                  * torch.ones(U.shape[0], dtype=torch.float64, device=U.device))
        S_new = joint_action(self.action, U_new, parameter)
        dW = S_new - S_old - dlogJ
        new_state = FieldState(U_new, state.phi, state.work + dW, state.heat, state.logJ + dlogJ)
        return new_state, dW, S_new, dlogJ


def make_gauge_layers(flow_pars, kind="smearing", rho=1e-3):
    """One pure-gauge layer per protocol step, as `Flow.__init__` builds them for flow_type='snf'
    (time (i+1)/n): kind "smearing" -> `CouplingLayer` with rho_layer = rho * ones(rho_shape)
    (a float or a tensor of shape rho_shape), "residual" -> `ResidualCouplingLayer`.

    The default rho is small but nonzero: the stout layer uses |rho| (`Smearing._mask_rho`), whose
    gradient at rho = 0 exactly is zero, so layers built with rho = 0 are silently untrainable.
    Pass rho = 0.0 explicitly for the identity layers (the NE-MCMC limit)."""
    n = int(flow_pars.nstep)
    layers = []
    for i in range(n):
        r = torch.as_tensor(rho, dtype=torch.float64) * torch.ones(flow_pars.rho_shape, dtype=torch.float64)
        r = r.to(flow_pars.device)
        if kind == "smearing":
            layers.append(CouplingLayer(flow_pars, r, (i + 1) / n))
        elif kind == "residual":
            layers.append(ResidualCouplingLayer(flow_pars, r, (i + 1) / n))
        else:
            raise ValueError(f"unknown layer kind {kind!r}")
    return layers


class FermionSNF(torch.nn.Module):
    """SNF over initial_parameter -> protocol[0] -> ... -> protocol[-1] (module doc).

    Args:
        action: `QCDAction` (its `parameter_name` is the protocol parameter).
        protocol: lambda_1 .. lambda_n (floats).
        layers: n pure-gauge layers (e.g. `make_gauge_layers(flow_pars)`); their parameters are
            the trainable parameters of this module.
        flow_pars: the `FlowPars` of the layers.
        initial_parameter: lambda_0 (default: the current parameter of `action`).
        block: an `HMCBlock`, or None to build one from `hmc_kwargs`.

    `forward(state0, generator=None, int_meas=None)` -> (state, W, Q, logJ, int_obs):
        as `FermionNEMCMC.forward`, plus logJ; W carries the gradient with respect to the layer
        parameters when grad mode is on (Q, logJ in int_obs and the HMC blocks do not);
        W + Q = S_final - S_0 - logJ per chain; int_obs columns `OBSERVABLES` (logJ filled).
    `loss(W)` = mean(W). After a run `last_run` holds S0, S_final, the per-step jump and layer
    works and the HMC info.
    """

    def __init__(self, action, protocol, layers, flow_pars, initial_parameter=None, block=None,
                 **hmc_kwargs):
        super().__init__()
        self.action = action
        self.protocol = [_as_parameter(p) for p in protocol]
        n = len(self.protocol)
        if n == 0:
            raise ValueError("the protocol needs at least one value")
        if len(layers) != n:
            raise ValueError(f"need one layer per protocol step ({n}), got {len(layers)}")
        if any(torch.is_tensor(p) for p in self.protocol):
            raise ValueError("FermionSNF takes a scalar protocol (the layers are conditioned on it)")
        self.initial_parameter = action.parameter if initial_parameter is None else _as_parameter(initial_parameter)
        blocks = []
        for i, layer in enumerate(layers):
            if n == 1:
                delta = 0.0
            elif i < n - 1:
                delta = self.protocol[i + 1] - self.protocol[i]
            else:
                delta = self.protocol[i] - self.protocol[i - 1]
            blocks.append(FermionLayerBlock(layer, action, flow_pars, self.protocol[i], delta))
        self.blocks = torch.nn.ModuleList(blocks)
        self.block = HMCBlock(action, **hmc_kwargs) if block is None else block
        self.last_run = None

    @property
    def n_step(self):
        return len(self.protocol)

    @torch.no_grad()
    def _jump(self, U, p_old, p_new, S_cur):
        """(dW, S(x; p_new)) at fixed fields, as `FermionNEMCMC._jump`."""
        if _same_parameter(p_old, p_new):
            return torch.zeros_like(S_cur), S_cur
        if self.action.parameter_name in ("kappa", "mass"):
            S_new = self.action.action(U, p_new)
            return S_new - S_cur, S_new
        dW = self.action.action_difference(U, p_old, p_new)
        return dW, S_cur + dW

    def forward(self, state0, generator=None, int_meas=None):
        n = self.n_step
        int_meas = n if int_meas is None else int(int_meas)
        if int_meas < 1 or n % int_meas:
            raise ValueError("int_meas must be a positive divisor of the number of protocol steps")
        per_meas = n // int_meas
        act = self.action
        U = state0.U.detach()
        B = U.shape[0]
        act.set_parameter(self.initial_parameter)
        with torch.no_grad():
            if state0.phi is None:
                act.refresh(U, generator)
            else:
                act.phi = state0.phi
            S0 = act.action(U)
        S_cur = S0
        zeros = torch.zeros(B, dtype=torch.float64, device=U.device)
        W, Q, logJ = zeros, zeros.clone(), zeros.clone()
        int_obs = torch.zeros((B, int_meas, len(OBSERVABLES)), dtype=torch.float64, device=U.device)
        p_old = self.initial_parameter
        infos, jumps, layer_works = [], [], []
        m = 0
        for i, (p_new, layer_block) in enumerate(zip(self.protocol, self.blocks)):
            dW, S_cur = self._jump(U, p_old, p_new, S_cur)
            W = W + dW
            jumps.append(dW)
            act.set_parameter(p_new)
            st = FieldState(U, act.phi)
            st, dW_layer, S_layer, dlogJ = layer_block(st, p_new, S_old=S_cur)
            W = W + dW_layer
            logJ = logJ + dlogJ.detach()
            layer_works.append(dW_layer.detach())
            U, dQ, S_cur, info = self.block(st.U.detach(), p_new, generator, S_old=S_layer.detach())
            Q = Q + dQ
            infos.append(info)
            p_old = p_new
            if (i + 1) % per_meas == 0:
                with torch.no_grad():
                    pl = mean_plaquette(plaquette_fields(U, act.geometry.D), act.geometry.N)
                int_obs[:, m, 0] = W.detach()
                int_obs[:, m, 1] = Q
                int_obs[:, m, 2] = logJ
                int_obs[:, m, 3] = S_cur
                int_obs[:, m, 4] = float(p_new)
                int_obs[:, m, 5] = pl
                int_obs[:, m, 6] = pl.square()
                m += 1
        self.last_run = {"S0": S0, "S_final": S_cur, "jump_work": jumps, "layer_work": layer_works,
                         "block_info": infos}
        state = FieldState(U, act.phi, state0.work + W.detach(), state0.heat + Q, state0.logJ + logJ)
        return state, W, Q, logJ, int_obs

    __call__ = forward

    @staticmethod
    def loss(W):
        """mean(W): the work loss of the pure-gauge SNF training (`_backprop_local_loss`)."""
        return W.mean()
