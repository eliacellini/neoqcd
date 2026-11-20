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
jump costs one solve, as a kappa jump. The fermion term type is dispatched by EXACT type
(`_fermion_kind`): the reference and fast Wilson and staggered terms and their parametric
subclasses; every other term (Hasenbusch heavy/ratio, clover, DD-HMC blocks, the fused-kernel
term, a user subclass) raises TypeError rather than being taken for its base class.

Fast Wilson (`QCDAction(..., fast=True)`, `fast.FastTwoFlavourWilson`): `PseudofermionAction`
solves on the half lattice (phi_e, X_e), S = Re phi_e^dagger X_e, and the force from the same X_e
(`FastTwoFlavourWilson.force_from_solution`, built on `HalfLatticeWilson.hopping_derivative`); the
backward i F U is unchanged. Equal to the reference path to the solver tolerance (unit tests). The
fast staggered term (`fast_terms.FastStaggeredPseudofermion`, library only: `QCDAction` has no
fast staggered option) is handled the same way, checked against the reference staggered term.

Identity start (`make_gauge_layers(..., kind="smearing_signed", rho=0)`, `SignedCouplingLayer`):
the pure-gauge stout layer uses |rho|, whose gradient vanishes at rho = 0, so a layer starting at
the identity never trains; the signed variant passes rho itself to the stout map (linear in rho
through the staples, exact log-Jacobian unchanged), so rho = 0 is the identity and a regular point.

Residual layers (`make_gauge_layers(..., kind="residual"|"hyper-residual", residual={...})`, the
pure-gauge `ResidualCouplingLayer`, SU(3) only). Options (`residual_options`, names and defaults of
main_nemcmc_snf.py): include_imag, quadratic, coeff_init, coeff_max, n_pt (frozen-link
convolutions), mask_mod (2 or 4; extents multiples of it when n_pt > 0 or mod 4), hyper_* (sizes of
the hyper-network, hyper-residual only). "residual" has static coefficients, one layer per step;
"hyper-residual" a hyper-network of the conditioning value, one layer shared by all steps (the
pure-gauge `_nf_layer_sharing`; `sharing=` overrides). Each layer's dlogJ is the exact Haar
log-Jacobian (dense check to ~1e-13, test_fermions_snf_residual).

Conditioning (`layer_conditioning`, `FermionSNF(condition=..., delta=...)`). The pure-gauge layer
receives beta (input of the hyper-network: Fourier features sin/cos(2^k beta), k < emb_dim/2, and
beta itself) and delta_beta, which multiplies its generator: delta = 0 is exactly the identity.
Passing the raw protocol value and step for kappa or bc is badly scaled: kappa ~ 0.15 spans
~0.03 over a protocol (the Fourier features barely change: 8 * 0.03 = 0.24 rad at the highest
frequency) and dkappa ~ 1e-3 makes the generator ~1e-3 of what the same coefficients give for a
unit step, so the layer is a near-identity (reviewer: <log J> ~ -2e-3 at the default init). The
default here is the same for every protocol parameter: condition "time", c_i = normalised
protocol position in (0, 1] (i/n for a linear protocol), and delta "fraction",
d_i = (lambda_i - lambda_{i-1}) / sum_j |lambda_j - lambda_{j-1}| (1/n for a linear protocol; the
step INTO lambda_i, the jump this layer follows; exactly 0 for a zero step, so the identity limit
is kept). "parameter" / "step" reproduce the pure-gauge convention (lambda_i, forward difference,
0 for n = 1). Under "fraction" the coefficients are scaled per protocol fraction rather than per
unit of the parameter, the same for beta, kappa, mass and bc.

Identity start. coeff_init = 0 (static: zero linear and quadratic coefficients; hyper: zero output
layer) is the exact identity (U' = U bitwise) and a regular point: the gradient with respect to the
linear coefficients (static) or the output layer (hyper) is nonzero there. Only second-order
parameters have zero gradient at that point (conv_eta: the convolutions enter only through
coefficient-weighted staples; the inner hyper layers: the zero output weights), and they start
moving after the first step. There is no analogue of the stout |rho| dead point.
"""

import copy
import warnings

import torch

from neoqcd.fermions._gauge import (CouplingLayer, ResidualCouplingLayer, Smearing, mean_plaquette,
                                    plaquette_fields)
from neoqcd.fermions.action import (ParametricFastTwoFlavourWilson, ParametricStaggeredPseudofermion,
                                    ParametricTwoFlavourWilson, _as_parameter, _same_parameter)
from neoqcd.fermions.fast import FastTwoFlavourWilson
from neoqcd.fermions.pseudofermion import TwoFlavourWilson, hopping_derivative
from neoqcd.fermions.staggered_pseudofermion import StaggeredPseudofermion, staggered_hopping_derivative
from neoqcd.fermions.snf import OBSERVABLES, FieldState, HMCBlock

__all__ = ["PseudofermionAction", "pseudofermion_action", "joint_action", "FermionLayerBlock",
           "FermionSNF", "make_gauge_layers", "SignedSmearing", "SignedCouplingLayer", "LAYER_KINDS",
           "RESIDUAL_KINDS", "RESIDUAL_OPTIONS", "residual_options", "residual_flow_pars",
           "layer_conditioning", "CONDITION_MODES", "DELTA_MODES", "default_sharing", "stout_rho_bound",
           "layer_diagnostics", "layer_diagnostics_message"]


def _supported_terms():
    """{exact class: kind} of the fermion terms `PseudofermionAction` handles, each validated
    against its reference by the unit tests. The lookup is by exact type, not isinstance: a
    subclass may change the action (Hasenbusch ratio: chi^dagger M^-1 chi), the operator (clover,
    blocks) or the field layout, and would otherwise be taken for its base class silently."""
    from neoqcd.fermions.fast_terms import FastStaggeredPseudofermion  # lazy: optional fast path
    return {TwoFlavourWilson: "wilson", ParametricTwoFlavourWilson: "wilson",
            FastTwoFlavourWilson: "wilson_fast", ParametricFastTwoFlavourWilson: "wilson_fast",
            StaggeredPseudofermion: "staggered", ParametricStaggeredPseudofermion: "staggered",
            FastStaggeredPseudofermion: "staggered_fast"}


def _fermion_kind(fermion):
    """"wilson" (reference `TwoFlavourWilson`, even-odd or full lattice), "wilson_fast"
    (`fast.FastTwoFlavourWilson`, half-lattice phi and X), "staggered" (`StaggeredPseudofermion`)
    or "staggered_fast" (`fast_terms.FastStaggeredPseudofermion`, half-lattice phi and X), with
    the parametric subclasses of `action.py`: the dispatch of this module, by exact type
    (`_supported_terms`). Every other term raises TypeError: the Hasenbusch heavy and ratio terms
    (reference and fast) with a message naming them, any other subclass of a supported term
    (clover, DD-HMC blocks, the fused-kernel term, ...) and unknown types."""
    supported = _supported_terms()
    kind = supported.get(type(fermion))
    if kind is not None:
        return kind
    from neoqcd.fermions.fast_terms import FastHeavyWilsonTerm, FastRatioWilsonTerm  # lazy
    from neoqcd.fermions.hasenbusch import HeavyWilsonTerm, RatioWilsonTerm  # lazy: avoids an import cycle
    name = type(fermion).__name__
    names = ", ".join(sorted({c.__name__ for c in supported}))
    if isinstance(fermion, (HeavyWilsonTerm, RatioWilsonTerm, FastHeavyWilsonTerm, FastRatioWilsonTerm)):
        raise TypeError(f"{name}: Hasenbusch terms are not supported in the SNF layers (PseudofermionAction "
                        "would return the plain TwoFlavourWilson S_pf and force, which is wrong for the ratio "
                        f"term); supported terms: {names}")
    base = next((c for c in type(fermion).__mro__[1:] if c in supported), None)
    if base is not None:
        raise TypeError(f"unsupported fermion term {name}: a subclass of {base.__name__} that PseudofermionAction "
                        f"has not been validated for (its action, operator or field layout may differ from "
                        f"those of {base.__name__}); supported terms: {names}")
    raise TypeError(f"unsupported fermion term {name}: PseudofermionAction handles {names}")


def _staggered_force_from_solution(op, X):
    """The force of `StaggeredPseudofermion.force` from X = M_ee^-1 phi (no new solve):
    F = 2 staggered_hopping_derivative(op, X, W), W = K_oe X."""
    W = op.odd(op.staggered.hopping(X))
    return 2.0 * staggered_hopping_derivative(op, X, W)


def _force_from_solution(fermion, op, X):
    """The force of `TwoFlavourWilson.force` (or `StaggeredPseudofermion.force`) from a given
    solution X = M^-1 phi (no new solve). For the fast term X is the half-lattice solution and op
    its `HalfLatticeWilson` (`FastTwoFlavourWilson.force_from_solution`)."""
    kind = _fermion_kind(fermion)
    if kind == "staggered":
        return _staggered_force_from_solution(op, X)
    if kind in ("wilson_fast", "staggered_fast"):
        return fermion.force_from_solution(op, X)
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
    other fermion type raises TypeError. Fast Wilson (`fast.FastTwoFlavourWilson`, e.g. the term
    of `QCDAction(..., fast=True)`) and fast staggered (`fast_terms.FastStaggeredPseudofermion`):
    the same S and F from one half-lattice solve (phi and X on the even sites, force by the
    half-lattice operator's `hopping_derivative`, `force_from_solution`); the backward is
    unchanged. The accepted types are exactly those of `_supported_terms` (Hasenbusch, clover,
    DD-HMC block and fused-kernel terms raise TypeError).
    Forward: one CG solve at `fermion.action_tol` (the Metropolis accuracy: the SNF work is an
    action difference), S = Re phi^dagger X, (B,) float64; if U requires grad, the force F at U is
    computed from the same X and saved (so later changes of fermion.phi or kappa do not affect the
    backward). Backward: dS/dU = grad_out * i F U (module doc), no backpropagation through CG.
    First derivatives only: F is saved as a constant, so a second derivative would silently miss
    dF/dU (and `torch.autograd.function.once_differentiable` would still drop this term's
    contribution without an error when the gradient also flows through other terms). The backward
    therefore raises RuntimeError whenever it runs in grad mode, i.e. under `create_graph=True`.
    """

    @staticmethod
    def forward(ctx, U, fermion):
        kind = _fermion_kind(fermion)  # TypeError for unknown fermion terms
        half = kind in ("wilson_fast", "staggered_fast")
        # the fast terms cache their operator per links tensor: give them a detached view, so the
        # cache never holds the autograd graph of a layer output
        op = fermion.operator(U.detach() if half else U)
        fermion.stats["action_calls"] += 1
        X = fermion._solve(op, fermion.action_tol)
        phi = fermion._phi_h if half else fermion.phi  # the layout of X
        S = (phi.conj() * X).real.reshape(X.shape[0], -1).sum(1)
        if ctx.needs_input_grad[0]:
            fermion.stats["force_calls"] += 1
            F = _force_from_solution(fermion, op, X)
            ctx.save_for_backward(U, F)
        return S

    @staticmethod
    def backward(ctx, grad_out):
        if torch.is_grad_enabled():  # the engine runs backward in grad mode only for create_graph=True
            raise RuntimeError("PseudofermionAction is differentiable once (the saved force F is a constant, there "
                               "is no backpropagation through the solver): create_graph=True / second derivatives "
                               "of S_pf are not supported")
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
            conditioning for hyper/residual parametrisations); `FermionSNF` sets them from
            `layer_conditioning` (default: normalised position and fractional step; "parameter" /
            "step" give `Flow._forward_layer`'s lambda_i and lambda_{i+1} - lambda_i).

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


class SignedSmearing(Smearing):
    """`Smearing` with a signed rho: the stout coefficient of every mask block is rho itself, not
    |rho| (`Smearing._mask_rho`). The stout map and its exact log-Jacobian are linear in rho
    through the staples C = sum rho_p staple_p, so nothing else depends on the sign. Bijectivity:
    each masked update is guaranteed to be a bijection only for |rho| < 1/(2(D - 1))
    (`stout_rho_bound`: 1/6 in 4D), for either sign; above it two configurations can have the same
    image with a finite log J and no error (a cold 4D configuration at rho = 0.2). Nothing here
    clamps rho (that would change what is trained): `layer_diagnostics` reports max |rho| against
    the bound and the SNF driver warns when it is exceeded. Plain trainable rho only (no
    hyper-smearing, no rho functional, no fit_train)."""

    def _mask_rho(self, rho_runtime, batched_rho, sm, m):
        if self.use_hyper_smearing:
            return super()._mask_rho(rho_runtime, batched_rho, sm, m)
        return rho_runtime[sm, m, :]


class SignedCouplingLayer(CouplingLayer):
    """`CouplingLayer` with a signed stout parameter (`SignedSmearing`): rho = 0 is the identity
    map AND a regular point of the parametrisation (d U'/d rho != 0 there), so a layer can start
    exactly at the identity and still be trained. With the pure-gauge |rho| the gradient at
    rho = 0 vanishes identically (torch's d|x|/dx = sign(x) = 0 at 0) and a zero-initialised layer
    never moves. For rho > 0 it is the same map as `CouplingLayer`. The state_dict has the same
    keys ("rho_layer"); a signed checkpoint loaded into a `CouplingLayer` would act as |rho|."""

    def __init__(self, flow_pars, rho_layer, time):
        if (int(getattr(flow_pars, "fit_train", 0)) != 0 or getattr(flow_pars, "use_hyper_smearing", False)
                or getattr(flow_pars, "use_rho_functional", False) or getattr(flow_pars, "use_nn", False)):
            raise ValueError("SignedCouplingLayer needs a plain trainable rho (no fit_train, hyper-smearing, "
                             "rho functional or NN staples)")
        super().__init__(flow_pars, rho_layer, time)
        self.smearing = SignedSmearing(flow_pars, time)


RESIDUAL_KINDS = ("residual", "hyper-residual")
LAYER_KINDS = ("smearing", "smearing_signed") + RESIDUAL_KINDS
HYPER_ACTIVATIONS = ("silu", "gelu", "tanh", "relu")

# Options of the residual layers (`ResidualNormalizingFlows`), keyword -> FlowPars attribute. The
# defaults are those of the pure-gauge driver main_nemcmc_snf.py (--residual-*, --hyper-*).
RESIDUAL_OPTIONS = {
    "include_imag": ("residual_include_imag", True),
    "quadratic": ("residual_quadratic", True),
    "coeff_init": ("residual_coeff_init", 1e-3),
    "coeff_max": ("residual_coeff_max", 0.0),
    "n_pt": ("residual_n_pt", 0),
    "mask_mod": ("residual_mask_mod", 2),
    "hyper_time_embedding_dim": ("hyper_time_embedding_dim", 8),
    "hyper_hidden_dim": ("hyper_hidden_dim", 16),
    "hyper_depth": ("hyper_depth", 2),
    "hyper_activation": ("hyper_activation", "silu"),
}


def _exact_int(name, value):
    if isinstance(value, bool) or int(value) != value:
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(value)


def residual_options(kind="residual", lattice=None, **options):
    """The complete, validated option set of a residual layer kind ("residual": static trainable
    coefficients; "hyper-residual": coefficients from the hyper-network of the conditioning value,
    `ResidualNormalizingFlows`), as a plain dict (parameterization + `RESIDUAL_OPTIONS` keys, unset
    keys at their pure-gauge defaults); suitable for run_meta and the flow sidecar.

    Validation as main_nemcmc_snf.py: n_pt >= 0, mask_mod 2 or 4, coeff_max >= 0, and with
    `lattice` (the extents) every extent divisible by mask_mod when n_pt > 0 or mask_mod = 4
    (periodic colouring); hyper sizes positive, embedding dimension even. In D = 2 a single
    convolution with mask_mod = 2 adds no effective dependence (README): a RuntimeWarning."""
    if kind not in RESIDUAL_KINDS:
        raise ValueError(f"residual options need a residual layer kind {RESIDUAL_KINDS}, got {kind!r}")
    unknown = set(options) - set(RESIDUAL_OPTIONS)
    if unknown:
        raise ValueError(f"unknown residual options {sorted(unknown)} (known: {sorted(RESIDUAL_OPTIONS)})")
    opt = {k: options.get(k, default) for k, (_, default) in RESIDUAL_OPTIONS.items()}
    opt["include_imag"], opt["quadratic"] = bool(opt["include_imag"]), bool(opt["quadratic"])
    opt["coeff_init"], opt["coeff_max"] = float(opt["coeff_init"]), float(opt["coeff_max"])
    for name in ("n_pt", "mask_mod", "hyper_time_embedding_dim", "hyper_hidden_dim", "hyper_depth"):
        opt[name] = _exact_int(name, opt[name])
    opt["hyper_activation"] = str(opt["hyper_activation"]).lower()
    if opt["n_pt"] < 0:
        raise ValueError("n_pt (residual-n-pt) must be non-negative")
    if opt["mask_mod"] not in (2, 4):
        raise ValueError("mask_mod (residual-mask-mod) must be 2 or 4")
    if not opt["coeff_max"] >= 0.0:
        raise ValueError("coeff_max must be >= 0 (0: no tanh bound)")
    if opt["hyper_time_embedding_dim"] <= 0 or opt["hyper_time_embedding_dim"] % 2:
        raise ValueError("hyper_time_embedding_dim must be a positive even integer")
    if opt["hyper_hidden_dim"] < 1 or opt["hyper_depth"] < 1:
        raise ValueError("hyper_hidden_dim and hyper_depth must be >= 1")
    if opt["hyper_activation"] not in HYPER_ACTIVATIONS:
        raise ValueError(f"hyper_activation must be one of {HYPER_ACTIVATIONS}")
    if lattice is not None:
        lattice = tuple(int(x) for x in lattice)
        if (opt["n_pt"] or opt["mask_mod"] != 2) and any(x % opt["mask_mod"] for x in lattice):
            raise ValueError(f"lattice extents {lattice} must be multiples of mask_mod = {opt['mask_mod']} "
                             "for the periodic colouring (n_pt > 0 or mask_mod = 4)")
        if len(lattice) == 2 and opt["n_pt"] == 1 and opt["mask_mod"] == 2:
            warnings.warn("residual n_pt = 1 with mask_mod = 2 in D = 2 adds no effective dependence (README): "
                          "use n_pt >= 2 or mask_mod = 4", RuntimeWarning, stacklevel=2)
    opt["parameterization"] = "static" if kind == "residual" else "hyper"
    return opt


def residual_flow_pars(flow_pars, kind="residual", **options):
    """A shallow copy of `flow_pars` with the residual_* / hyper_* attributes of `residual_options`
    set (the layers read them at construction; the mask is shared)."""
    lattice = (int(flow_pars.T),) + (int(flow_pars.L),) * (int(flow_pars.D) - 1)
    opt = residual_options(kind, lattice=lattice, **options)
    if int(flow_pars.N) != 3:
        raise ValueError("the residual layers support SU(3) only")
    fp = copy.copy(flow_pars)
    fp.residual_parameterization = opt["parameterization"]
    for key, (attr, _) in RESIDUAL_OPTIONS.items():
        setattr(fp, attr, opt[key])
    return fp


def default_sharing(kind):
    """"shared" for hyper-residual, "per_step" otherwise: the rule of main_nemcmc_snf.py
    (`_nf_layer_sharing`: hyper architectures shared across the protocol, the others per block)."""
    return "shared" if kind == "hyper-residual" else "per_step"


def make_gauge_layers(flow_pars, kind="smearing", rho=1e-3, residual=None, sharing=None):
    """One pure-gauge layer per protocol step, as `Flow.__init__` builds them for flow_type='snf'
    (time (i+1)/n): kind "smearing" -> `CouplingLayer` with rho_layer = rho * ones(rho_shape)
    (a float or a tensor of shape rho_shape), "smearing_signed" -> `SignedCouplingLayer` (same,
    with a signed rho: trainable from rho = 0, the identity), "residual" / "hyper-residual" ->
    `ResidualCouplingLayer` with static / hyper-network coefficients (the pure-gauge spelling of
    main_nemcmc_snf.py --nf-architecture), configured by the dict `residual` of
    `residual_options` keywords (n_pt, mask_mod, include_imag, quadratic, coeff_init, coeff_max,
    hyper_*; unset keys at the pure-gauge defaults; rho is unused). The residual layer's
    generator is proportional to the delta_condition its `FermionLayerBlock` passes, so the
    conditioning of `FermionSNF` (condition, delta) decides its strength (module doc).

    sharing (residual kinds): "per_step" builds n independent layers; "shared" returns the same
    layer object n times (one set of parameters for the whole protocol, told apart only by the
    conditioning c_i, d_i). Default `default_sharing(kind)`: shared for hyper-residual (a
    hyper-network of a per-step constant input would be only a reparametrisation of static
    coefficients), per_step for residual. The stout kinds are always per_step.

    The default rho is small but nonzero: the stout layer of kind "smearing" uses |rho|
    (`Smearing._mask_rho`), whose gradient at rho = 0 exactly is zero, so "smearing" layers built
    with rho = 0 are silently untrainable (pass rho = 0.0 explicitly for fixed identity layers,
    the NE-MCMC limit). For an identity start that trains, use kind "smearing_signed" with rho = 0,
    or a residual kind with coeff_init = 0 (a regular point, see the module doc)."""
    if kind not in LAYER_KINDS:
        raise ValueError(f"unknown layer kind {kind!r} (one of {LAYER_KINDS})")
    if kind in RESIDUAL_KINDS:
        flow_pars = residual_flow_pars(flow_pars, kind, **(residual or {}))
    elif residual:
        raise ValueError(f"residual options given for layer kind {kind!r}")
    sharing = default_sharing(kind) if sharing is None else sharing
    if sharing not in ("shared", "per_step") or (sharing == "shared" and kind not in RESIDUAL_KINDS):
        raise ValueError(f"sharing must be 'per_step', or 'shared' for a residual kind; got {sharing!r} for {kind!r}")
    n = int(flow_pars.nstep)
    if sharing == "shared":
        r = torch.zeros(flow_pars.rho_shape, dtype=torch.float64).to(flow_pars.device)
        return [ResidualCouplingLayer(flow_pars, r, 1.0)] * n
    layers = []
    for i in range(n):
        r = torch.as_tensor(rho, dtype=torch.float64) * torch.ones(flow_pars.rho_shape, dtype=torch.float64)
        r = r.to(flow_pars.device)
        if kind == "smearing":
            layers.append(CouplingLayer(flow_pars, r, (i + 1) / n))
        elif kind == "smearing_signed":
            layers.append(SignedCouplingLayer(flow_pars, r, (i + 1) / n))
        else:
            layers.append(ResidualCouplingLayer(flow_pars, r, (i + 1) / n))
    return layers


def stout_rho_bound(D):
    """1 / (2 (D - 1)) (1/6 in 4D, 1/2 in 2D): the stout coefficient below which the map
    U -> exp(iQ(C U^dagger)) U of one masked update is guaranteed to be a bijection.

    The staple sum C = rho sum_k S_k^dagger of the 2(D - 1) staples of a link (fixed by the mask)
    has norm <= 2(D - 1) |rho|, so the generator is a contraction of U below this bound (the same
    argument as for the Wilson flow). Above it the map can fold: on a cold 4D configuration two
    inputs differing on one link have the same image at rho = 0.2 > 1/6, with a finite log J
    (review of 20 November). The bound is sufficient, and reached on a cold configuration."""
    return 1.0 / (2.0 * (int(D) - 1))


def _residual_step_ratio(layer, condition, delta, N):
    """Contraction estimate of one residual layer at (c, d): the largest over its sub-steps of
    |d| / N * sum_f (|lin_f| + 2 sum_j |H_fj|) (f, j the trace features; H the symmetrised
    quadratic coefficients, after the coeff_max tanh bound), and the largest |coefficient|."""
    rf = layer.residual_flow
    device = next(rf.parameters()).device
    lin, quad, dv = rf._coefficients(condition, delta, 1, torch.float64, device)  # (1, steps, nf), (1, steps, nf, nf)
    n_steps, nf = lin.shape[1], lin.shape[-1]
    lin, quad = lin.reshape(n_steps, nf).abs(), quad.reshape(n_steps, nf, nf).abs()
    d = float(dv.reshape(-1)[0].abs())
    per_step = d / float(N) * (lin.sum(-1) + 2.0 * quad.sum((-1, -2)))
    return _nan_max(per_step.tolist()), _nan_max([lin.max(), quad.max()])


@torch.no_grad()
def layer_diagnostics(blocks, D, N=3):
    """Bijectivity diagnostics of the layers of a flow (no effect on the map: they only read the
    parameters). `blocks`: the `FermionLayerBlock`s of a `FermionSNF` (`FermionSNF.layer_diagnostics`).

    Returns {"kind", "max_abs_coeff", "rho_bound", "injectivity_ratio", "exceeded", "per_step"}:
      stout kinds ("smearing", "smearing_signed"): max_abs_coeff = max |rho| over every layer and
          mask slot (the coefficient the map uses: |rho| or the signed rho), rho_bound =
          `stout_rho_bound(D)`, injectivity_ratio = max |rho| / rho_bound;
      residual kinds: max_abs_coeff = the largest |linear| or |quadratic| coefficient the layers
          use at their conditioning (after the coeff_max tanh bound), rho_bound None, and
          injectivity_ratio = the largest contraction estimate
              |d_i| / N * sum_f (|lin_f| + 2 sum_j |H_fj|)
          over steps and sub-steps (f, j the 2(D - 1) or 4(D - 1) trace features). A HEURISTIC:
          with real features and no quadratic term the layer is exactly a stout map with per-staple
          coefficients d lin_f / N, and the estimate is the stout ratio above; the quadratic term
          enters through the feature derivatives (|z| <= 1), counted twice; norm constants of
          that term and the non-unitary convolved staples of n_pt > 0 are not tracked.
    exceeded = not (injectivity_ratio < 1) (True for a NaN ratio): the layers are then not
    guaranteed to be bijections and the Jarzynski/Crooks identities may fail silently (a finite
    log J does not prove injectivity). per_step: the ratio of each protocol step."""
    blocks = list(blocks)
    if not blocks:
        raise ValueError("no layers")
    residual = [isinstance(b.layer, ResidualCouplingLayer) for b in blocks]
    if any(residual) and not all(residual):
        raise ValueError("mixed stout and residual layers")
    if residual[0]:
        ratios, coeffs = zip(*(_residual_step_ratio(b.layer, b.condition, b.delta_condition, N) for b in blocks))
        ratios, coeffs = list(ratios), list(coeffs)
        out = {"kind": "residual", "max_abs_coeff": _nan_max(coeffs), "rho_bound": None}
    else:
        bound = stout_rho_bound(D)
        coeffs = [float(b.layer.rho_layer.detach().abs().max()) for b in blocks]
        ratios = [c / bound for c in coeffs]
        out = {"kind": "stout", "max_abs_coeff": _nan_max(coeffs), "rho_bound": bound}
    ratio = _nan_max(ratios)
    out.update(injectivity_ratio=ratio, exceeded=not ratio < 1.0, per_step=ratios)
    return out


def _nan_max(values):
    """max of floats, NaN if any is NaN (python's max drops NaN depending on the order)."""
    values = [float(v) for v in values]
    return float("nan") if any(v != v for v in values) else max(values)


def layer_diagnostics_message(diag):
    """One line describing `layer_diagnostics` output (for logs and the RuntimeWarning)."""
    if diag["kind"] == "stout":
        return (f"stout layers: max |rho| = {diag['max_abs_coeff']:.4g}, bijectivity guaranteed only below "
                f"1/(2(D-1)) = {diag['rho_bound']:.4g} (ratio {diag['injectivity_ratio']:.3g})")
    return (f"residual layers: max |coefficient| = {diag['max_abs_coeff']:.4g}, contraction estimate "
            f"{diag['injectivity_ratio']:.3g} (heuristic; bijectivity guaranteed only below 1)")


CONDITION_MODES = ("time", "parameter")
DELTA_MODES = ("fraction", "step")


def layer_conditioning(initial, protocol, condition="time", delta="fraction"):
    """[(c_i, d_i)], i = 1 .. n: what the layer of step i receives as beta and delta_beta.

    condition "parameter": c_i = lambda_i (the pure-gauge convention, beta-conditioned);
              "time":      c_i = s_i / s_n in (0, 1], s_i = sum_{j <= i} |lambda_j - lambda_{j-1}|
                           (normalised protocol position; i/n for a linear protocol, and if
                           every step is zero).
    delta     "step":      d_i = lambda_{i+1} - lambda_i (last step: lambda_n - lambda_{n-1};
                           0 for n = 1), exactly `Flow._forward_layer`;
              "fraction":  d_i = (lambda_i - lambda_{i-1}) / s_n, the step INTO lambda_i (the jump
                           the layer follows, lambda_0 = initial) as a fraction of the protocol's
                           total variation: sum_i |d_i| = 1, 1/n for a linear protocol, whatever
                           the parameter and its scale. d_i = 0 exactly when lambda_i =
                           lambda_{i-1} (and all d_i = 0 if every step is zero), so a zero step
                           still gives the identity layer."""
    if condition not in CONDITION_MODES:
        raise ValueError(f"condition must be one of {CONDITION_MODES}, got {condition!r}")
    if delta not in DELTA_MODES:
        raise ValueError(f"delta must be one of {DELTA_MODES}, got {delta!r}")
    lam = [float(initial)] + [float(p) for p in protocol]
    n = len(lam) - 1
    steps = [lam[i] - lam[i - 1] for i in range(1, n + 1)]
    total = sum(abs(s) for s in steps)
    out = []
    cumulative = 0.0
    for i in range(1, n + 1):
        cumulative += abs(steps[i - 1])
        if condition == "parameter":
            c = lam[i]
        else:
            c = cumulative / total if total > 0 else i / n
        if delta == "fraction":
            d = steps[i - 1] / total if total > 0 else 0.0
        elif n == 1:
            d = 0.0
        elif i < n:
            d = lam[i + 1] - lam[i]
        else:
            d = lam[i] - lam[i - 1]
        out.append((float(c), float(d)))
    return out


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
        condition, delta: what every layer receives as beta and delta_beta
            (`layer_conditioning`; default "time" and "fraction", the normalised protocol position
            and the fractional step, the same scale for beta, kappa, mass and bc; "parameter" and
            "step" are the pure-gauge convention of `Flow._forward_layer`). Only the conditioned
            layers use them (residual, hyper-smearing); the plain stout layers ignore them.

    `forward(state0, generator=None, int_meas=None)` -> (state, W, Q, logJ, int_obs):
        as `FermionNEMCMC.forward`, plus logJ; W carries the gradient with respect to the layer
        parameters when grad mode is on (Q, logJ in int_obs and the HMC blocks do not);
        W + Q = S_final - S_0 - logJ per chain; int_obs columns `OBSERVABLES` (logJ filled).
    `loss(W)` = mean(W). After a run `last_run` holds S0, S_final, the per-step jump and layer
    works and the HMC info.
    """

    def __init__(self, action, protocol, layers, flow_pars, initial_parameter=None, block=None,
                 condition="time", delta="fraction", **hmc_kwargs):
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
        self.condition_mode, self.delta_mode = condition, delta
        self.conditioning = layer_conditioning(self.initial_parameter, self.protocol, condition, delta)
        blocks = [FermionLayerBlock(layer, action, flow_pars, c, d)
                  for layer, (c, d) in zip(layers, self.conditioning)]
        self.blocks = torch.nn.ModuleList(blocks)
        self.block = HMCBlock(action, **hmc_kwargs) if block is None else block
        self.last_run = None

    @property
    def n_step(self):
        return len(self.protocol)

    def layer_diagnostics(self):
        """`layer_diagnostics` of this flow's layers (max |rho| or coefficient, bijectivity bound)."""
        return layer_diagnostics(self.blocks, self.action.geometry.D, self.action.geometry.N)

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
