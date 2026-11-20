"""Path Gradient (PG) training of flow fields with N_f = 2 Wilson fermions: the pseudofermion term.

The pure-gauge PG training (`neoqcd.cnf`, `neoqcd.cnf_training`, doc/cnf.md) trains a velocity
field F(U, c) on SU(3) links, dU/dc = i F_a T_a U (T_a = lambda_a / 2), on a persistent REX
ensemble. Per chain the residual is

    xi = [d_c S + F . grad S - div_Haar F] / V,        F . grad S = Re <grad_U S, i F U>,

and the PG loss is Re <grad_U xi_frozen, i F_theta U> (parameters frozen inside xi, links
differentiated; `cnf.path_gradient_loss`). With dynamical fermions the action of a replica is the
joint S(U, phi; c) = S_g(U; beta, c) + S_pf(U, phi), S_pf = phi^dagger M^-1 phi, M = D_hat^dagger
D_hat (`pseudofermion.TwoFlavourWilson`, even-odd). The defect parameter c enters S_g only, so
d_c S and div F are the pure-gauge ones; the only new piece is the pseudofermion term with the
replica's own phi HELD FIXED (the convention of the fermionic SNF, `snf_layers`):

    xi_f = F . grad S_pf / V = s / V,     s = -X^dagger (d_F M) X,     X = M^-1 phi,

where d_F is the derivative along the tangent V = i F U (the hopping term is linear in the links,
boundary phase included, so (d_F D_hat) x is D_hat's hopping with the links replaced by V: no jvp).
With M = D_hat^dagger D_hat, s = -2 Re[(D_hat X)^dagger (d_F D_hat) X].

Its link gradient (the Hessian-vector product of S_pf along the flow; the loss is linear in xi, so
the gauge and fermion parts add) for any direction W:

    d_W s = [-X^dagger (d_W d_F M) X]_(X fixed)  +  2 Re[X^dagger (d_W M) Y],   Y = M^-1 (d_F M) X.

Proof: dX = -M^-1 (d_W M) X, and A = d_F M is hermitian, so the X-variation of -X^dagger A X is
-2 Re[X^dagger A dX] = 2 Re[(M^-1 A X)^dagger (d_W M) X]. Both terms are autograd at fixed fields
(no backpropagation through the solver): the first is autograd of s with X a constant (through the
sparse operator AND through the dependence of V = i F(U) U on the links, i.e. the frozen field), the
second autograd of 2 Re[(D_hat X)^dagger (D_hat Y)] with X, Y constants. Cost: two CG solves per
chain (X and Y) and a few operator applications. (d_F M) X = (d_F D_hat)^dagger D_hat X
+ D_hat^dagger (d_F D_hat) X with (d_F D_hat)^dagger = gamma_5 (d_F D_hat) gamma_5.

The gradient G is PyTorch's Wirtinger gradient of the unconstrained expression (dxi = Re sum
conj(G) dU, the convention of `cnf._residual_and_gradient`); only its contraction with tangent
vectors i H U is meaningful. Because s is computed as the exact directional derivative of S_pf in
the space of complex matrices (D_hat^dagger = gamma_5 D_hat gamma_5 for any complex links), xi_f and
G equal those of the dense phi^dagger (D_hat^dagger D_hat)^-1 phi with the same extension (unit tests).

Solves: Y is solved with `solver.cg` directly (not `fermion._solve`, which takes phi as its
right-hand side and overwrites the cached X); X is solved afresh from zero (the term's cached X is
the last solution of the trajectory, generally at other links), both at `tol` (default the term's
`solver_tol`: a gradient, like the MD force). After each solve the true relative residual is
checked against `pseudofermion.convergence_limit(tol)` with the term's `on_unconverged` policy
("raise" default, "warn" once per kind, "ignore" counts), as every solve of the package; a sample
with non-finite links or phi ends with a non-finite residual and is named by the check. The solves
are counted in a `LossSolveCounter`, not in the term's `stats` (the drivers split trajectory and
loss costs).

Supported terms (exact type): `TwoFlavourWilson` and `ParametricTwoFlavourWilson` even-odd (the
`QCDAction` default), and on the fast path (`QCDAction(..., fast=True)`) `fast.FastTwoFlavourWilson`
and `ParametricFastTwoFlavourWilson`: the same formulas on half-lattice fields with
`fast.HalfLatticeWilson` (the term's colour product; autograd through its gathered links) and the
term's solver (`lean_cg`, or `mixed_precision_cg` with precision "mixed", converging to the same
tolerance); equal to the reference to the solver tolerance (unit tests). Anything else (staggered,
Hasenbusch, clover, DD-HMC, the fused-kernel term, subclasses) raises TypeError.

Joint loss (`fermion_path_gradient_loss`): the pure-gauge `cnf.path_gradient_loss` with the gauge
action of the ladder, unchanged, plus Re <grad_U xi_f, i F_theta U> (the loss is linear in xi):
xi_f and its gradient at frozen field parameters, the live field only in the tangent, so the
parameter gradient is that of the pure-gauge construction applied to the joint action. d_c S comes
from `defect_ladder_action` (the pure-gauge `ladder_action`), never from `QCDAction`, whose
parameters are constants (a jvp through it is silently zero; unit test).
"""

import contextlib
import functools
import math
import time
import warnings
from types import SimpleNamespace

import torch
import torch.distributed as dist

from neoqcd.fermions._gauge import SUN_identity, is_uniform, mean_plaquette, plaquette_fields
from neoqcd.fermions._gauge_cnf import ParallelTempering, PTConfig, ladder_action, lie_tangent, path_gradient_loss
from neoqcd.fermions.action import ParametricFastTwoFlavourWilson, ParametricTwoFlavourWilson, QCDAction
from neoqcd.fermions.dirac import WilsonDirac
from neoqcd.fermions.fast import FastTwoFlavourWilson, HalfLatticeWilson, lean_cg, mixed_precision_cg, to_half
from neoqcd.fermions.pseudofermion import (
    ON_UNCONVERGED,
    TwoFlavourWilson,
    UnconvergedSolveWarning,
    convergence_limit,
)
from neoqcd.fermions.rex import (
    STREAM_RUN,
    STREAM_SWAP,
    FermionREX,
    FermionREXTarget,
    GlobalRNGStream,
    derived_seed,
    distributed_rex_step,
)
from neoqcd.fermions.solver import cg

__all__ = ["LossSolveCounter", "pseudofermion_pg_term", "pg_fermion_kind", "hermitian_tangent",
           "defect_ladder_action", "check_same_defect", "fermion_path_gradient_loss", "frozen_parameters",
           "FERMION_TERM", "TrajectorySolveWatch", "FermionCNFREXTrainer", "DistributedFermionCNFREXTrainer"]

# The fermion contribution of the residual: S_pf of the joint action with the replica's phi held fixed.
FERMION_TERM = "joint_pseudofermion"


# ---- bookkeeping ------------------------------------------------------------------------------
def _nan_max(a, b):
    """max of two floats or None, NaN-sticky (as `main_snf_fermions.nan_max`)."""
    if a is None:
        return b
    if b is None:
        return a
    if a != a or b != b:
        return float("nan")
    return max(a, b)


class LossSolveCounter:
    """Solves of the PG loss (kept apart from the fermion term's `stats`, which count the HMC).

    Attributes: `solves`, `cg_iterations` (sum over solves of the largest per-sample count: the
    batch costs as its slowest sample), `cg_loop_iterations` (operator applications actually
    performed, fast path; equal to cg_iterations on the reference path), `max_residual` (largest
    true relative residual, NaN-sticky; None before any solve), `unconverged_solves` (solves with a
    sample above `convergence_limit(tol)`), `seconds` (wall time of the pseudofermion term).
    `reset()` zeroes them; the "warn once per kind" memory survives a reset.
    """

    def __init__(self):
        self._warned = set()
        self.reset()

    def reset(self):
        self.solves = 0
        self.cg_iterations = 0
        self.cg_loop_iterations = 0
        self.max_residual = None
        self.unconverged_solves = 0
        self.seconds = 0.0

    def as_dict(self):
        return {"solves": self.solves, "cg_iterations": self.cg_iterations,
                "cg_loop_iterations": self.cg_loop_iterations, "max_residual": self.max_residual,
                "unconverged_solves": self.unconverged_solves, "seconds": self.seconds}

    def record(self, result, tol, kind, policy, max_iter, owner="pseudofermion_pg_term", loop_iterations=None):
        """Book-keeping of one solve and the convergence policy of `TwoFlavourWilson._check_converged`."""
        it = result.iterations
        self.solves += 1
        self.cg_iterations += int(it.max().item()) if it.numel() else 0
        self.cg_loop_iterations += int(it.max().item()) if loop_iterations is None else int(loop_iterations)
        res = result.residual.detach()
        if res.numel():
            self.max_residual = _nan_max(self.max_residual, float(res.to(torch.float64).max().item()))
        limit = convergence_limit(float(tol), res.dtype)
        bad = ~(res <= limit)  # NaN counts as unconverged
        if not bool(bad.any()):
            return
        self.unconverged_solves += 1
        if policy == "ignore":
            return
        samples = bad.nonzero().flatten().tolist()
        worst = float(res[bad].nan_to_num(nan=float("inf")).max().item())
        its = it[bad].tolist()
        msg = (f"{owner} {kind}: CG not converged, relative residual {worst:.3e} > {limit:.1e} "
               f"(tol {float(tol):.1e} x slack 10, eps floor 1e3 eps) for sample(s) {samples}, "
               f"iterations {its}, max_iter {max_iter}")
        if policy == "raise":
            raise RuntimeError(msg)
        if kind not in self._warned:
            self._warned.add(kind)
            warnings.warn(msg, UnconvergedSolveWarning, stacklevel=4)


# ---- dispatch ---------------------------------------------------------------------------------
_REFERENCE_TERMS = (TwoFlavourWilson, ParametricTwoFlavourWilson)
_FAST_TERMS = (FastTwoFlavourWilson, ParametricFastTwoFlavourWilson)


def pg_fermion_kind(fermion):
    """"wilson" (reference even-odd terms) or "wilson_fast" (`fast.FastTwoFlavourWilson` and its
    parametric variant), by exact type; TypeError for any other term (staggered, Hasenbusch, clover,
    DD-HMC, the fused-kernel term, subclasses) and ValueError for the full-lattice reference term."""
    if type(fermion) in _REFERENCE_TERMS:
        if not fermion.even_odd:
            raise ValueError("the PG pseudofermion term is even-odd only (TwoFlavourWilson(even_odd=True), the "
                             "QCDAction default)")
        return "wilson"
    if type(fermion) in _FAST_TERMS:
        return "wilson_fast"
    names = ", ".join(c.__name__ for c in _REFERENCE_TERMS + _FAST_TERMS)
    raise TypeError(f"unsupported fermion term {type(fermion).__name__}: the PG pseudofermion term handles "
                    f"{names} (N_f = 2 Wilson, even-odd)")


# ---- operators --------------------------------------------------------------------------------
class _ReferenceOps:
    """D_hat, D_hat^dagger, gamma_5 and d_V D_hat on full-lattice even fields (dirac.py operators),
    CG of `solver.cg` (per-sample convergence)."""

    def __init__(self, U, kappa, geometry, time_sign, fermion=None):
        self.geometry, self.kappa, self.time_sign = geometry, kappa, time_sign
        self.w = WilsonDirac(U, kappa, geometry, time_sign)
        self.eo = self.w.even_odd()

    def field(self, phi):
        return phi

    def apply(self, x):
        return self.eo.apply(x)

    def apply_dagger(self, x):
        return self.eo.apply_dagger(x)

    def gamma5(self, x):
        return self.eo.gamma5(x)

    def tangent(self, V):
        """d_V D_hat x = -kappa^2 even[H(V) H(U) x + H(U) H(V) x], x even (H linear in the links)."""
        wV = WilsonDirac(V, self.kappa, self.geometry, self.time_sign)
        w, eo = self.w, self.eo

        def apply(x):
            x = eo.even(x)
            k2 = w._kappa_view(x) ** 2
            return -k2 * eo.even(wV.hopping(w.hopping(x)) + w.hopping(wV.hopping(x)))
        return apply

    def solve(self, b, tol, max_iter):
        return cg(self.eo.normal, b, tol=tol, max_iter=max_iter), None


class _FastOps(_ReferenceOps):
    """The same on half-lattice even fields (`fast.HalfLatticeWilson`: the term's colour product), with
    the term's solver (`lean_cg`, or `mixed_precision_cg` with precision "mixed")."""

    def __init__(self, U, kappa, geometry, time_sign, fermion=None):
        self.geometry, self.kappa, self.time_sign = geometry, kappa, time_sign
        self.fermion = fermion
        self.colour_product = fermion.colour_product
        self.op = HalfLatticeWilson(U, kappa, geometry, time_sign, colour_product=self.colour_product)

    def field(self, phi):
        return to_half(phi, self.geometry, 0)

    def apply(self, x):
        return self.op.apply(x)

    def apply_dagger(self, x):
        return self.op.apply(x, dagger=True)

    def gamma5(self, x):
        return self.op.gamma5(x)

    def tangent(self, V):
        """d_V D_hat x = -kappa^2 [H_eo(V) H_oe(U) + H_eo(U) H_oe(V)] x on half fields."""
        opV = HalfLatticeWilson(V, self.kappa, self.geometry, self.time_sign, colour_product=self.colour_product)
        op = self.op

        def apply(x):
            return -op._k2 * (opV.hop(op.hop(x, 1), 0) + op.hop(opV.hop(x, 1), 0))
        return apply

    def solve(self, b, tol, max_iter):
        f, info = self.fermion, {}
        if f.precision == "double":
            res = lean_cg(self.op.normal, b, tol=tol, max_iter=max_iter, check_every=f.check_every, info=info)
        else:
            if getattr(self, "_low", None) is None:
                self._low = self.op.to(torch.complex64)
            res = mixed_precision_cg(self.op, self._low, b, tol=tol, inner_tol=f.inner_tol, max_outer=f.max_outer,
                                     max_iter=max_iter, check_every=f.check_every, info=info)
        return res, info["loop_iterations"]


_OPS = {"wilson": _ReferenceOps, "wilson_fast": _FastOps}


def _rdot(a, b):
    """Per-sample Re <a, b> = Re sum conj(a) b, shape (B,)."""
    return (a.conj() * b).real.reshape(a.shape[0], -1).sum(1)


def hermitian_tangent(H):
    """The tangent field U -> i H U of a fixed hermitian (traceless) field H, shape like U."""
    return lambda U: 1j * torch.matmul(H.to(U.dtype), U)


@contextlib.contextmanager
def _timed(counter):
    start = time.perf_counter()
    try:
        yield
    finally:
        if counter is not None:
            counter.seconds += time.perf_counter() - start


def pseudofermion_pg_term(U, fermion, velocity, *, phi=None, kappa=None, volume=None, tol=None, max_iter=None,
                          on_unconverged=None, counter=None):
    """xi_f = F . grad S_pf / V per chain and its link gradient (module doc), with two solves.

    Args:
        U: links (B, D, lattice, N, N) (not modified; differentiated on a detached copy).
        fermion: `TwoFlavourWilson` / `ParametricTwoFlavourWilson` (even-odd) or the fast
            `FastTwoFlavourWilson` / `ParametricFastTwoFlavourWilson` (half-lattice operator with the
            term's colour product, lean or mixed-precision CG as the term's `precision`), with phi
            set; its kappa, time_sign, geometry, solver_tol, max_iter and on_unconverged are the
            defaults.
        velocity: a callable U -> tangent field i F(U) U (e.g. the frozen flow field; it is
            differentiated through), or a hermitian (B, D, lattice, N, N) field H held fixed
            (tangent i H U).
        phi, kappa: override the term's phi (full-lattice, even sites) and kappa (float or (B,)),
            e.g. for a slice of the batch.
        volume: V of xi (default the number of sites, as `cnf.path_gradient_loss`).
        tol, max_iter, on_unconverged: of both solves (defaults: the term's solver_tol, max_iter,
            on_unconverged).
        counter: a `LossSolveCounter` to accumulate into (default a new one).
    Returns a dict:
        xi (B,) float (detached), gradient (like U, complex; Wirtinger, divided by V), S_pf (B,)
        = Re phi^dagger X (the action at U, free from the X solve), counter.
    """
    Ops = _OPS[pg_fermion_kind(fermion)]
    if counter is None:
        counter = LossSolveCounter()
    geometry = fermion.geometry
    phi = fermion.phi if phi is None else phi
    if phi is None:
        raise RuntimeError("the pseudofermion field is not set: the PG term needs the phi of the last trajectory")
    kappa = fermion.kappa if kappa is None else kappa
    tol = float(fermion.solver_tol if tol is None else tol)
    max_iter = int(fermion.max_iter if max_iter is None else max_iter)
    policy = getattr(fermion, "on_unconverged", "raise") if on_unconverged is None else on_unconverged
    if policy not in ON_UNCONVERGED:
        raise ValueError(f"on_unconverged must be one of {ON_UNCONVERGED}, got {policy!r}")
    if volume is None:
        volume = geometry.volume
    volume = int(volume)
    tangent_of = velocity if callable(velocity) else hermitian_tangent(velocity)
    if phi.shape[0] != U.shape[0]:
        raise ValueError(f"phi holds {phi.shape[0]} fields, U {U.shape[0]} configurations")

    with _timed(counter):
        U0 = U.detach()
        ops = Ops(U0, kappa, geometry, fermion.time_sign, fermion)
        phi = ops.field(phi)
        X_res, loops = ops.solve(phi, tol, max_iter)
        counter.record(X_res, tol, "pg_X", policy, max_iter, loop_iterations=loops)
        X = X_res.x
        S_pf = _rdot(phi, X)
        with torch.enable_grad():
            Ug = U0.clone().requires_grad_(True)
            V = tangent_of(Ug)
            if V.shape != U.shape:
                raise ValueError(f"the velocity must return a tangent of shape {tuple(U.shape)}, got {tuple(V.shape)}")
            g_ops = Ops(Ug, kappa, geometry, fermion.time_sign, fermion)
            dD = g_ops.tangent(V)
            DX = g_ops.apply(X)
            dDX = dD(X)
            s = -2.0 * _rdot(DX, dDX)
            with torch.no_grad():  # (d_V M) X = gamma_5 (d_V D_hat) gamma_5 D_hat X + D_hat^dagger (d_V D_hat) X
                rhs = ops.gamma5(dD(ops.gamma5(DX))) + ops.apply_dagger(dDX)
                rhs = rhs.detach()
            Y_res, loops = ops.solve(rhs, tol, max_iter)
            counter.record(Y_res, tol, "pg_Y", policy, max_iter, loop_iterations=loops)
            t2 = 2.0 * _rdot(DX, g_ops.apply(Y_res.x.detach()))
            (gradient,) = torch.autograd.grad((s + t2).sum(), Ug)
    return {"xi": s.detach() / volume, "gradient": gradient / volume, "S_pf": S_pf.detach(), "counter": counter}


# ---- the joint loss ---------------------------------------------------------------------------
@contextlib.contextmanager
def frozen_parameters(model):
    """Parameters of `model` without requires_grad inside the block (as `cnf._frozen_parameters`)."""
    flags = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in flags:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)


def defect_ladder_action(geometry, beta, defect_mask):
    """The gauge action S_g(U; beta, c) of the bc ladder as a function (U, c) -> (B,), differentiable
    in c: the pure-gauge `cnf_training.ladder_action` (domain "defect").

    d_c S of the PG residual must come from here: `QCDAction` stores its parameters as floats or
    detached tensors (`action._as_parameter`), so a derivative in c through it does not exist. The
    value equals `QCDAction.gauge_action(U, c)` (unit test).
    """
    pars = SimpleNamespace(D=int(geometry.D), N=int(geometry.N))
    return functools.partial(ladder_action, pars, "defect", float(beta), defect_mask, int(geometry.volume))


def check_same_defect(field_mask, action_mask, what="defect"):
    """ValueError unless two (2, D, lattice) defect masks agree link by link (the field's patch and
    the action's mask are built separately: `cnf_io.build_cnf` and the fermion drivers)."""
    a = torch.as_tensor(field_mask).detach().to("cpu", torch.float64)
    b = torch.as_tensor(action_mask).detach().to("cpu", torch.float64)
    if a.shape != b.shape:
        raise ValueError(f"{what}: the field's defect mask has shape {tuple(a.shape)}, the action's {tuple(b.shape)}")
    differ = int((a != b).any(dim=0).sum())
    if differ:
        raise ValueError(f"{what}: the field's defect mask and the action's differ on {differ} links "
                         f"(field {int(a[1].sum())} defect links, action {int(b[1].sum())})")


def _slice_kappa(kappa, sl, B):
    if torch.is_tensor(kappa) and kappa.dim() == 1 and kappa.numel() == B and B > 1:
        return kappa[sl]
    return kappa


def _pinn(xi, t, volume):
    """V Var(xi | t) within groups of equal t (`cnf._pinn`), 0-dim."""
    groups = torch.unique(t.detach())
    total, count = xi.new_zeros(()), 0
    for value in groups:
        sample = xi[t.detach() == value]
        if sample.numel() > 1:
            total = total + sample.var(unbiased=False) * sample.numel()
            count += sample.numel()
    return volume * total / max(count, 1)


def fermion_path_gradient_loss(field, action_fn, fermion, U, t, *, volume=None, chunk_size=None,
                               directions_per_term=1, tol=None, max_iter=None, on_unconverged=None,
                               counter=None):
    """PG loss of the joint action S_g(U; c) + S_pf(U, phi) per chain (module doc).

    loss = [unchanged pure-gauge `cnf.path_gradient_loss(field, action_fn, U, t)`]
           + Re <grad_U xi_f, lie_tangent(U, field.drift(U, t))>,
    the second with xi_f from `pseudofermion_pg_term` at frozen field parameters (links
    differentiated, also through the frozen field) and the live field in the tangent, so the
    parameter gradient reaches the live field only. `action_fn(U, c)` is the gauge action of the
    ladder (`defect_ladder_action`); `fermion` holds the phi of each chain (the phi of its last
    trajectory: S_pf does not depend on c). `chunk_size` splits the chains (memory only).

    Returns a dict: loss (B,) (differentiable in the field parameters), loss_gauge, loss_fermion,
    xi = xi_gauge + xi_fermion, xi_gauge, xi_fermion (B,) detached, S_pf (B,) (= Re phi^dagger X at
    U: <S_pf>/n = 1 in equilibrium), pinn (V Var(xi | t) of the total), F_norm (of the pure-gauge
    loss), solves (`LossSolveCounter.as_dict()` of the fermion part), gauge_seconds.
    """
    pg_fermion_kind(fermion)
    if volume is None:
        volume = fermion.geometry.volume
    volume = int(volume)
    t = torch.as_tensor(t, dtype=U.real.dtype, device=U.device).reshape(-1)
    if t.numel() == 1:
        t = t.expand(U.shape[0]).clone()
    if t.numel() != U.shape[0]:
        raise ValueError(f"t must be scalar or have one value per chain ({U.shape[0]})")
    counter = LossSolveCounter() if counter is None else counter
    start = time.perf_counter()
    gauge = path_gradient_loss(field, action_fn, U, t, volume=volume, chunk_size=chunk_size,
                               directions_per_term=directions_per_term)
    gauge_seconds = time.perf_counter() - start
    phi = fermion.phi
    if phi is None:
        raise RuntimeError("the pseudofermion field is not set: the PG term needs the phi of the last trajectory")
    B = U.shape[0]
    size = B if chunk_size is None else int(chunk_size)
    parts = []
    for begin in range(0, B, size):
        sl = slice(begin, begin + size)
        tc = t[sl]
        tangent = lambda value, tc=tc: lie_tangent(value, field.drift(value, tc))
        with frozen_parameters(field):
            parts.append(pseudofermion_pg_term(U[sl], fermion, tangent, phi=phi[sl], kappa=_slice_kappa(fermion.kappa, sl, B),
                                               volume=volume, tol=tol, max_iter=max_iter,
                                               on_unconverged=on_unconverged, counter=counter))
    xi_f = torch.cat([p["xi"] for p in parts])
    gradient = torch.cat([p["gradient"] for p in parts])
    S_pf = torch.cat([p["S_pf"] for p in parts])
    velocity = lie_tangent(U.detach(), field.drift(U.detach(), t))
    loss_f = (gradient.conj() * velocity).real.flatten(1).sum(1)
    xi_g = gauge["xi"]
    xi = xi_g + xi_f
    return {
        "loss": gauge["loss"] + loss_f,
        "loss_gauge": gauge["loss"].detach(),
        "loss_fermion": loss_f.detach(),
        "xi": xi,
        "xi_gauge": xi_g,
        "xi_fermion": xi_f,
        "S_pf": S_pf,
        "pinn": _pinn(xi, t, volume),
        "F_norm": gauge["F_norm"],
        "solves": counter.as_dict(),
        "gauge_seconds": gauge_seconds,
    }


# ---- trainers ---------------------------------------------------------------------------------
class TrajectorySolveWatch:
    """Largest true residual and unconverged count of the solves of a fermion term (the HMC ones).

    Wraps the term's `_solve(op, tol)` as an instance attribute (the class is untouched), as
    `main_snf_fermions.SolveMonitor` does: after every solve that reached the term's bookkeeping it
    reads `stats["last_residual"]`; `interval()` returns (max_residual or None, NaN-sticky;
    unconverged solves) since the previous call and resets them.
    """

    def __init__(self, fermion):
        self.fermion = fermion
        self._max, self._unconverged = None, 0
        original = fermion._solve

        def watched(op, tol):
            before = fermion.stats["solves"]
            try:
                return original(op, tol)
            finally:
                if fermion.stats["solves"] > before:
                    self._record(fermion.stats.get("last_residual"), tol)

        fermion._solve = watched

    def _record(self, residual, tol):
        if residual is None:
            return
        residual = residual.detach().reshape(-1).to("cpu", torch.float64)
        if residual.numel():
            self._max = _nan_max(self._max, float(residual.max()))
        self._unconverged += int(bool((~(residual <= convergence_limit(float(tol), torch.float64))).any()))

    def interval(self):
        out = (self._max, self._unconverged)
        self._max, self._unconverged = None, 0
        return out


def _cost(fermion):
    """(solves, cg_iterations, operator applications) of the term so far (fast terms count the loop)."""
    s = fermion.stats
    return s["solves"], s["cg_iterations"], s.get("cg_loop_iterations", s["cg_iterations"])


def _clock(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _finite_gradients(parameters):
    return all(bool(torch.isfinite(p.grad).all()) for p in parameters if p.grad is not None)


def _by_rung(values, labels, n_rungs, reduce):
    """[reduce(values[labels == k]) for each rung], floats (NaN for an empty rung)."""
    out = []
    for k in range(n_rungs):
        v = values[labels == k]
        out.append(float(reduce(v)) if v.numel() else float("nan"))
    return out


def _rung_statistics(volume, n_components, labels, n_rungs, loss, xi_g, xi_f, S_pf, plaquette):
    """Per-rung metrics of one epoch (lists over rungs) from per-chain (cpu, float64) tensors."""
    xi = xi_g + xi_f
    var = lambda v: volume * v.var(unbiased=False)
    cov = lambda a, b: volume * ((a - a.mean()) * (b - b.mean())).mean()
    return {
        "losses_by_rung": _by_rung(loss, labels, n_rungs, torch.mean),
        "pinn_by_rung": _by_rung(xi, labels, n_rungs, var),
        "pinn_gauge_by_rung": _by_rung(xi_g, labels, n_rungs, var),
        "pinn_fermion_by_rung": _by_rung(xi_f, labels, n_rungs, var),
        "pinn_cov_by_rung": [float(cov(xi_g[labels == k], xi_f[labels == k])) if bool((labels == k).any())
                             else float("nan") for k in range(n_rungs)],
        "residuals_by_rung": _by_rung(xi, labels, n_rungs, torch.mean),
        "residuals_gauge_by_rung": _by_rung(xi_g, labels, n_rungs, torch.mean),
        "residuals_fermion_by_rung": _by_rung(xi_f, labels, n_rungs, torch.mean),
        "spf_per_component_by_rung": _by_rung(S_pf / float(n_components), labels, n_rungs, torch.mean),
        "plaquette_by_rung": _by_rung(plaquette, labels, n_rungs, torch.mean),
    }


class _TrainerBase:
    """Optimizer step and gradient checks shared by both trainers (as `cnf_training`)."""

    def _optimizer_step(self, finite):
        parameters = [p for p in self.field.parameters() if p.requires_grad]
        grads = [p.grad for p in parameters if p.grad is not None]
        grad_norm, updated = float("nan"), False
        if finite and grads:
            if self.grad_clip_norm > 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, self.grad_clip_norm).item())
            else:
                grad_norm = float(torch.linalg.vector_norm(torch.stack([g.detach().norm() for g in grads])).item())
            if math.isfinite(grad_norm) and _finite_gradients(parameters):
                self.optimizer.step()
                updated = True
        self.n_updates += 1
        return grad_norm, updated

    def _loss(self, U, t):
        self.counter.reset()
        return fermion_path_gradient_loss(
            self.field, self.action, self.fermion, U, t, volume=self.volume, chunk_size=self.chunk_size,
            directions_per_term=self.directions_per_term, tol=self.loss_tol, counter=self.counter)


class FermionCNFREXTrainer(_TrainerBase):
    """PG training of a flow field on a single-process fermionic REX (`rex.FermionREX`).

    n_rungs = len(ladder) rungs x `batch_size` independent ladders, one batched `QCDAction`
    (entry b = slot * batch_size + chain; labels move, configurations and their phi stay). One epoch
    (`step`): `updates_per_replica` fermionic HMC trajectories of every replica at its bc (fresh phi
    per trajectory; the swaps follow the last one) -> swap rounds (`swap_parity`) -> PG loss of every
    chain at its post-swap label with the phi of its last trajectory (`fermion_path_gradient_loss`)
    -> one optimizer step on the mean loss. No periodic translation (phi would belong to the
    untranslated links). The defect enters the gauge action only: swaps cost no solve.

    Args: the flow `field` and its `optimizer`; physics `geometry`, `beta`, `kappa`, `defect_mask`
    ((2, D) + lattice), `ladder` (increasing bc); HMC `tau`, `n_steps`, `integrator`,
    `n_gauge_substeps`; `field_defect_mask` (the mask the field was built with, checked against
    `defect_mask`); `loss_tol` (tolerance of the two loss solves, default the term's solver_tol);
    `seed` (run generator `derived_seed(seed, 0, STREAM_RUN)`, swap generator `(seed, 0, STREAM_SWAP)`);
    `U` initial links (default cold); extra keywords go to `QCDAction` / the fermion term
    (time_sign, solver_tol, action_tol, max_iter, reuse_guess, on_unconverged, fast options).
    API of `cnf_training.DistributedCNFREXTrainer`: `step`, `warmup`, `thermalize`, `state_dict`,
    `shard_state_dict`, `load_state_dict(state, shard)`.
    """

    def __init__(self, field, optimizer, *, geometry, beta, kappa, defect_mask, ladder, batch_size=2,
                 tau=1.0, n_steps=2, integrator="omelyan", n_gauge_substeps=3, updates_per_replica=1,
                 swap_parity="alternating", grad_clip_norm=0.0, chunk_size=None, directions_per_term=1,
                 loss_tol=None, seed=1234, device="cpu", U=None, field_defect_mask=None, **fermion_kwargs):
        if field_defect_mask is not None:
            check_same_defect(field_defect_mask, defect_mask)
        self.field, self.optimizer = field, optimizer
        self.geometry = geometry
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.updates_per_replica = int(updates_per_replica)
        self.grad_clip_norm = float(grad_clip_norm)
        self.chunk_size = None if chunk_size is None else int(chunk_size)
        self.directions_per_term = int(directions_per_term)
        self.loss_tol = None if loss_tol is None else float(loss_tol)
        if self.batch_size < 2:
            raise ValueError("batch_size must be >= 2 to measure PINN within each replica")
        if self.updates_per_replica < 1:
            raise ValueError("updates_per_replica must be positive")
        ladder = torch.as_tensor(ladder, dtype=torch.float64).reshape(-1).cpu()
        if ladder.numel() < 2:
            raise ValueError("replica exchange requires at least two rungs")
        defect_mask = torch.as_tensor(defect_mask, dtype=torch.float64).to(self.device)
        B = ladder.numel() * self.batch_size
        if U is None:
            U = SUN_identity(geometry.link_shape(B)[:-1], dtype=torch.cdouble, device=self.device)
        self.rex = FermionREX(geometry, beta, kappa, defect_mask, ladder, n_chains=self.batch_size, tau=tau,
                              n_steps=n_steps, integrator=integrator, n_gauge_substeps=n_gauge_substeps,
                              swap_parity=swap_parity, periodic_translation=False, U=U.to(self.device),
                              swap_seed=derived_seed(seed, 0, STREAM_SWAP), **fermion_kwargs)
        self.fermion = self.rex.qcd.fermion
        pg_fermion_kind(self.fermion)
        self.generator = torch.Generator(device=self.device).manual_seed(derived_seed(seed, 0, STREAM_RUN))
        self.parameters = self.rex.ladder
        self.n_rungs = int(self.parameters.numel())
        self.volume = geometry.volume
        self.action = defect_ladder_action(geometry, beta, defect_mask)
        self.counter = LossSolveCounter()
        self.watch = TrajectorySolveWatch(self.fermion)
        self.n_updates = 0

    @property
    def x(self):
        """The links of every replica, (n_rungs * batch_size, D, lattice, N, N)."""
        return self.rex.U

    def labels(self):
        """Rung of every batch entry, (B,) long (cpu)."""
        return self.rex.labels.reshape(-1)

    def local_parameters(self):
        return self.rex.parameters()

    @torch.no_grad()
    def _trajectories(self, n):
        for _ in range(int(n)):
            self.rex.target.set_parameter(self.rex.parameters())
            self.rex.U, _, _, _ = self.rex.target.trajectory(self.rex.U, self.generator)

    def thermalize(self, n=1):
        """n trajectories of every replica at its current bc, no swaps."""
        if int(n) < 0:
            raise ValueError("thermalisation trajectories must be non-negative")
        self._trajectories(n)

    def _sweep(self):
        self._trajectories(self.updates_per_replica - 1)
        return self.rex.step(self.generator)

    def warmup(self, sweeps=1):
        sweeps = int(sweeps)
        if sweeps < 0:
            raise ValueError("warmup sweeps must be non-negative")
        for _ in range(sweeps):
            self._sweep()

    def _backward(self):
        t = self.rex.parameters().to(device=self.device, dtype=self.rex.U.real.dtype)
        result = self._loss(self.rex.U, t)
        result["loss"].mean().backward()
        return result

    def step(self):
        start = _clock(self.device)
        self.watch.interval()
        c0 = _cost(self.fermion)
        stats = self._sweep()
        c1 = _cost(self.fermion)
        traj_res, traj_bad = self.watch.interval()
        swept = _clock(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        result = self._backward()
        backward = _clock(self.device)
        loss, xi = result["loss"].detach(), result["xi"]
        finite = bool(torch.isfinite(loss).all()) and bool(torch.isfinite(xi).all())
        grad_norm, updated = self._optimizer_step(finite)
        labels = self.labels()
        with torch.no_grad():
            plaq = mean_plaquette(plaquette_fields(self.rex.U, self.geometry.D), self.geometry.N)
        cpu = lambda v: v.detach().to("cpu", torch.float64).reshape(-1)
        out = _rung_statistics(self.volume, self.fermion.n_components, labels, self.n_rungs, cpu(loss),
                               cpu(result["xi_gauge"]), cpu(result["xi_fermion"]), cpu(result["S_pf"]), cpu(plaq))
        acc = torch.tensor(stats["link_accepted"], dtype=torch.float64)
        prop = torch.tensor(stats["link_proposed"], dtype=torch.float64)
        loss_cost = result["solves"]
        out.update({
            "swap_acceptance_by_edge": [float(v) for v in acc / prop.clamp_min(1.0)],
            "swap_acceptance": float(acc.sum() / prop.sum().clamp_min(1.0)),
            "hmc_acceptance": stats["hmc_acceptance"],
            "grad_norm": grad_norm, "optimizer_updated": updated, "updates": self.n_updates,
            "trajectory_solves": c1[0] - c0[0], "trajectory_cg_iter": c1[1] - c0[1],
            "trajectory_cg_applications": c1[2] - c0[2],
            "loss_solves": loss_cost["solves"], "loss_cg_iter": loss_cost["cg_iterations"],
            "loss_cg_applications": loss_cost["cg_loop_iterations"],
            "trajectory_max_residual": traj_res, "loss_max_residual": loss_cost["max_residual"],
            "max_residual": _nan_max(traj_res, loss_cost["max_residual"]),
            "trajectory_unconverged_solves": traj_bad, "loss_unconverged_solves": loss_cost["unconverged_solves"],
            "unconverged_solves": traj_bad + loss_cost["unconverged_solves"],
            "sweep_s": round(swept - start, 4), "loss_s": round(backward - swept, 4),
            "gauge_loss_s": round(result["gauge_seconds"], 4), "fermion_loss_s": round(loss_cost["seconds"], 4),
        })
        return out

    def state_dict(self):
        """State shared by all ranks (here one), saved with the field."""
        return {"distributed": False, "world_size": 1, "parameters": self.parameters.detach().cpu(),
                "n_updates": self.n_updates, "optimizer": self.optimizer.state_dict()}

    def shard_state_dict(self):
        """The chains and their random streams (the only "rank")."""
        shard = {"rank": 0, "configs": self.rex.U.detach().cpu(), "labels": self.rex.labels.clone(),
                 "n_steps_done": int(self.rex.n_steps_done), "link_accepted": self.rex.link_accepted.clone(),
                 "link_proposed": self.rex.link_proposed.clone(), "generator": self.generator.get_state(),
                 "swap_generator": self.rex.swap_generator.get_state(), "torch_rng_state": torch.get_rng_state()}
        if self.device.type == "cuda":
            shard["cuda_rng_state"] = torch.cuda.get_rng_state(self.device)
        return shard

    def load_state_dict(self, state, shard):
        if state.get("distributed") or int(state.get("world_size", 1)) != 1:
            raise ValueError("checkpoint was written by a distributed run")
        if not torch.equal(state["parameters"].cpu(), self.parameters.detach().cpu()):
            raise ValueError("checkpoint replica ladder does not match")
        if int(shard["rank"]) != 0 or tuple(shard["configs"].shape) != tuple(self.rex.U.shape):
            raise ValueError("checkpoint chains do not match this trainer")
        labels = shard["labels"].to(torch.long)
        if tuple(labels.shape) != tuple(self.rex.labels.shape) or not bool(
                (labels.sort(dim=0).values == torch.arange(self.n_rungs).view(-1, 1)).all()):
            raise ValueError("checkpoint labels are not a permutation of the rungs in every chain")
        self.rex.U = shard["configs"].to(device=self.device, dtype=self.rex.U.dtype)
        self.rex.labels = labels.clone()
        self.rex.n_steps_done = int(shard["n_steps_done"])
        self.rex.link_accepted = shard["link_accepted"].clone()
        self.rex.link_proposed = shard["link_proposed"].clone()
        self.rex.target.set_parameter(self.rex.parameters())
        self.generator.set_state(shard["generator"])
        self.rex.swap_generator.set_state(shard["swap_generator"])
        self.n_updates = int(state["n_updates"])
        self.optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(shard["torch_rng_state"].cpu())
        if self.device.type == "cuda" and "cuda_rng_state" in shard:
            torch.cuda.set_rng_state(shard["cuda_rng_state"], self.device)


class DistributedFermionCNFREXTrainer(_TrainerBase):
    """PG training with one rung per rank: `distributedPT.ParallelTempering` + `rex.FermionREXTarget`.

    As `cnf_training.DistributedCNFREXTrainer` with the HBOR sweep replaced by the fermionic REX step
    of `rex.distributed_rex_step` (trajectory of every local chain at its bc label, gauge-only swap
    table, `pt.step` with the swap uniforms of rank 0 from a private `GlobalRNGStream`, new labels;
    no translation). The field is broadcast from rank 0 at construction; the loss of the local
    chains (at their post-swap labels, with the phi of their last trajectory) is backpropagated
    with weight 1/world_size and the parameter gradients are summed over the ranks (all_reduce), so
    every rank applies the same update to an identical field and the loss is the mean over all
    chains. Arguments as `FermionCNFREXTrainer`, plus `rank`, `world_size`; `batch_size` is the
    number of chains per rank; seeds `derived_seed(seed, rank, STREAM_RUN / STREAM_SWAP)`.
    """

    def __init__(self, field, optimizer, *, rank, world_size, geometry, beta, kappa, defect_mask, ladder,
                 batch_size=2, tau=1.0, n_steps=2, integrator="omelyan", n_gauge_substeps=3,
                 updates_per_replica=1, swap_parity="alternating", grad_clip_norm=0.0, chunk_size=None,
                 directions_per_term=1, loss_tol=None, seed=1234, device="cpu", U=None, field_defect_mask=None,
                 **fermion_kwargs):
        if field_defect_mask is not None:
            check_same_defect(field_defect_mask, defect_mask)
        self.field, self.optimizer = field, optimizer
        self.rank, self.world_size = int(rank), int(world_size)
        self.geometry = geometry
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.updates_per_replica = int(updates_per_replica)
        self.grad_clip_norm = float(grad_clip_norm)
        self.chunk_size = None if chunk_size is None else int(chunk_size)
        self.directions_per_term = int(directions_per_term)
        self.loss_tol = None if loss_tol is None else float(loss_tol)
        ladder = torch.as_tensor(ladder, dtype=torch.float64).reshape(-1).cpu()
        if ladder.numel() != self.world_size:
            raise ValueError(f"one rung per rank: {ladder.numel()} rungs on {self.world_size} ranks")
        if self.world_size < 2 or not bool((ladder[1:] > ladder[:-1]).all()):
            raise ValueError("the bc ladder must be strictly increasing, one rung per rank")
        if self.batch_size < 2:
            raise ValueError("batch_size must be >= 2")
        if self.updates_per_replica < 1:
            raise ValueError("updates_per_replica must be positive")
        # Same field on every rank, whatever each rank's random initialization.
        with torch.no_grad():
            for tensor in list(field.parameters()) + list(field.buffers()):
                dist.broadcast(tensor.data, src=0)
        explicit = None if is_uniform(ladder.tolist()) else ladder.tolist()
        self.pt = ParallelTempering(
            PTConfig(batch_size=self.batch_size, parameter_min=float(ladder[0]), parameter_max=float(ladder[-1]),
                     swap_parity=swap_parity, parameter_dtype=torch.float64, parameter_name="bc",
                     acceptance_rule="exact", parameter_ladder=explicit),
            self.rank, self.world_size, self.device)
        self.parameters = self.pt.ladder.detach().to(torch.float64)
        defect_mask = torch.as_tensor(defect_mask, dtype=torch.float64).to(self.device)
        self.qcd = QCDAction(geometry, "bc", beta=beta, kappa=kappa, bc=self.pt.local_parameters.to(torch.float64).cpu(),
                             defect_mask=defect_mask, n_gauge_substeps=n_gauge_substeps, **fermion_kwargs)
        self.target = FermionREXTarget(self.qcd, tau=tau, n_steps=n_steps, integrator=integrator)
        self.fermion = self.qcd.fermion
        pg_fermion_kind(self.fermion)
        if U is None:
            U = SUN_identity(geometry.link_shape(self.batch_size)[:-1], dtype=torch.cdouble, device=self.device)
        if tuple(U.shape) != geometry.link_shape(self.batch_size):
            raise ValueError(f"U must have shape {geometry.link_shape(self.batch_size)}, got {tuple(U.shape)}")
        self.x = U.to(self.device)
        self.generator = torch.Generator(device=self.device).manual_seed(derived_seed(seed, self.rank, STREAM_RUN))
        self.swap_rng = GlobalRNGStream(self.device, derived_seed(seed, self.rank, STREAM_SWAP))
        self.volume = geometry.volume
        self.action = defect_ladder_action(geometry, beta, defect_mask)
        self.counter = LossSolveCounter()
        self.watch = TrajectorySolveWatch(self.fermion)
        self.n_updates = 0

    def _labels(self):
        """Ladder index of each local chain."""
        return torch.argmin((self.pt.local_parameters[:, None] - self.parameters.to(self.pt.local_parameters)[None, :])
                            .abs(), dim=1)

    @torch.no_grad()
    def _trajectories(self, n):
        for _ in range(int(n)):
            self.target.set_parameter(self.pt.local_parameters.to(torch.float64).cpu())
            self.x, _, _, _ = self.target.trajectory(self.x, self.generator)

    def thermalize(self, n=1):
        """n trajectories of every local chain at its current bc, no swaps."""
        if int(n) < 0:
            raise ValueError("thermalisation trajectories must be non-negative")
        self._trajectories(n)

    def _sweep(self):
        self._trajectories(self.updates_per_replica - 1)
        self.x, stats, hmc = distributed_rex_step(self.pt, self.target, self.x, self.generator,
                                                  periodic_translation=False, swap_rng=self.swap_rng)
        stats = dict(stats, hmc_accepted=hmc["accepted"])
        return stats

    def warmup(self, sweeps=1):
        sweeps = int(sweeps)
        if sweeps < 0:
            raise ValueError("warmup sweeps must be non-negative")
        for _ in range(sweeps):
            self._sweep()

    def _backward(self):
        """PG loss of the local chains at their labels, gradients summed over ranks."""
        t = self.pt.local_parameters.to(dtype=self.x.real.dtype)
        result = self._loss(self.x, t)
        (result["loss"].mean() / self.world_size).backward()
        for parameter in self.field.parameters():
            if parameter.requires_grad:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                dist.all_reduce(parameter.grad)
        return result

    def _gather(self, tensor):
        parts = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(parts, tensor.contiguous())
        return torch.cat(parts)

    def step(self):
        start = _clock(self.device)
        self.watch.interval()
        c0 = _cost(self.fermion)
        stats = self._sweep()
        c1 = _cost(self.fermion)
        traj_res, traj_bad = self.watch.interval()
        swept = _clock(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        result = self._backward()
        backward = _clock(self.device)
        loss, xi = result["loss"].detach(), result["xi"]
        flag = float(bool(torch.isfinite(loss).all()) and bool(torch.isfinite(xi).all())
                     and _finite_gradients([p for p in self.field.parameters() if p.requires_grad]))
        finite = torch.tensor(flag, device=self.device)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        grad_norm, updated = self._optimizer_step(bool(finite.item()))

        with torch.no_grad():
            plaq = mean_plaquette(plaquette_fields(self.x, self.geometry.D), self.geometry.N)
        f64 = lambda v: v.detach().to(device=self.device, dtype=torch.float64).reshape(-1)
        per_chain = torch.stack([f64(loss), f64(result["xi_gauge"]), f64(result["xi_fermion"]), f64(result["S_pf"]),
                                 f64(plaq), self._labels().to(torch.float64),
                                 f64(stats["hmc_accepted"].to(torch.float64))], dim=1)
        gathered = self._gather(per_chain).cpu()
        labels = gathered[:, 5].round().to(torch.long)
        out = _rung_statistics(self.volume, self.fermion.n_components, labels, self.world_size, gathered[:, 0],
                               gathered[:, 1], gathered[:, 2], gathered[:, 3], gathered[:, 4])
        n_links = self.world_size - 1
        links = (torch.tensor(stats["link_accepted"] + stats["link_proposed"], dtype=torch.float64, device=self.device)
                 if self.rank == 0 else torch.zeros(2 * n_links, dtype=torch.float64, device=self.device))
        dist.broadcast(links, src=0)
        acc, prop = links[:n_links].cpu(), links[n_links:].cpu()
        loss_cost = result["solves"]
        nan = float("nan")
        # sums over the ranks: solves, iterations, unconverged; maxima: times, residuals, memory
        sums = torch.tensor([c1[0] - c0[0], c1[1] - c0[1], c1[2] - c0[2], loss_cost["solves"],
                             loss_cost["cg_iterations"], loss_cost["cg_loop_iterations"], traj_bad,
                             loss_cost["unconverged_solves"]], dtype=torch.float64, device=self.device)
        dist.all_reduce(sums)
        residuals = torch.tensor([-1.0 if traj_res is None else traj_res,
                                  -1.0 if loss_cost["max_residual"] is None else loss_cost["max_residual"]],
                                 dtype=torch.float64, device=self.device)
        res_all = self._gather(residuals).view(self.world_size, 2).cpu()
        maxima = torch.tensor([swept - start, backward - swept, result["gauge_seconds"], loss_cost["seconds"],
                               torch.cuda.max_memory_allocated(self.device) / 2**30 if self.device.type == "cuda" else 0.0],
                              dtype=torch.float64, device=self.device)
        dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
        maxima = maxima.cpu()

        def nan_sticky(column):
            value = None
            for r in res_all[:, column].tolist():  # -1: no solve on that rank
                if r != r or r >= 0:
                    value = _nan_max(value, r)
            return value

        traj_max, loss_max = nan_sticky(0), nan_sticky(1)
        sums = sums.cpu().tolist()
        out.update({
            "swap_acceptance_by_edge": [float(v) for v in acc / prop.clamp_min(1.0)],
            "swap_acceptance": float(acc.sum() / prop.sum().clamp_min(1.0)) if n_links else nan,
            "hmc_acceptance": float(gathered[:, 6].mean()),
            "grad_norm": grad_norm, "optimizer_updated": updated, "updates": self.n_updates,
            "trajectory_solves": int(sums[0]), "trajectory_cg_iter": int(sums[1]),
            "trajectory_cg_applications": int(sums[2]),
            "loss_solves": int(sums[3]), "loss_cg_iter": int(sums[4]), "loss_cg_applications": int(sums[5]),
            "trajectory_max_residual": traj_max, "loss_max_residual": loss_max,
            "max_residual": _nan_max(traj_max, loss_max),
            "trajectory_unconverged_solves": int(sums[6]), "loss_unconverged_solves": int(sums[7]),
            "unconverged_solves": int(sums[6] + sums[7]),
            "sweep_s": round(float(maxima[0]), 4), "loss_s": round(float(maxima[1]), 4),
            "gauge_loss_s": round(float(maxima[2]), 4), "fermion_loss_s": round(float(maxima[3]), 4),
            "peak_memory_gb": round(float(maxima[4]), 2),
        })
        return out

    def state_dict(self):
        """State shared by all ranks, saved by rank 0 with the field."""
        return {"distributed": True, "world_size": self.world_size, "parameters": self.parameters.detach().cpu(),
                "parameter_matrix": self.pt.parameter_matrix.detach().cpu() if self.rank == 0 else None,
                "pt_step": int(self.pt._pt_step), "n_updates": self.n_updates,
                "optimizer": self.optimizer.state_dict()}

    def shard_state_dict(self):
        """State of this rank: its chains, their labels and its random streams."""
        shard = {"rank": self.rank, "configs": self.x.detach().cpu(),
                 "local_parameters": self.pt.local_parameters.detach().cpu(),
                 "generator": self.generator.get_state(), "swap_rng": self.swap_rng.get_state(),
                 "torch_rng_state": torch.get_rng_state()}
        if self.device.type == "cuda":
            shard["cuda_rng_state"] = torch.cuda.get_rng_state(self.device)
        return shard

    def load_state_dict(self, state, shard):
        if not state.get("distributed") or int(state["world_size"]) != self.world_size:
            raise ValueError("checkpoint was not written by a distributed run on this many ranks")
        if not torch.equal(state["parameters"].cpu(), self.parameters.detach().cpu()):
            raise ValueError("checkpoint replica ladder does not match")
        if int(shard["rank"]) != self.rank or tuple(shard["configs"].shape) != tuple(self.x.shape):
            raise ValueError("checkpoint shard does not belong to this rank")
        if self.rank == 0:
            self.pt.parameter_matrix.copy_(state["parameter_matrix"].to(self.pt.parameter_matrix))
        self.pt._scatter_parameters()
        if not torch.equal(self.pt.local_parameters.cpu(), shard["local_parameters"].to(self.pt.local_parameters.dtype)):
            raise ValueError("checkpoint chain labels disagree with the replica ladder state")
        self.pt._pt_step = int(state["pt_step"])
        self.target.set_parameter(self.pt.local_parameters.to(torch.float64).cpu())
        self.x = shard["configs"].to(device=self.device, dtype=self.x.dtype)
        self.generator.set_state(shard["generator"])
        self.swap_rng.set_state(shard["swap_rng"])
        self.n_updates = int(state["n_updates"])
        self.optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(shard["torch_rng_state"].cpu())
        if self.device.type == "cuda" and "cuda_rng_state" in shard:
            torch.cuda.set_rng_state(shard["cuda_rng_state"], self.device)
