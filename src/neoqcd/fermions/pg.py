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
`QCDAction` default). Anything else raises TypeError.
"""

import contextlib
import time
import warnings

import torch

from neoqcd.fermions.action import ParametricTwoFlavourWilson
from neoqcd.fermions.dirac import WilsonDirac
from neoqcd.fermions.pseudofermion import (
    ON_UNCONVERGED,
    TwoFlavourWilson,
    UnconvergedSolveWarning,
    convergence_limit,
)
from neoqcd.fermions.solver import cg

__all__ = ["LossSolveCounter", "pseudofermion_pg_term", "pg_fermion_kind", "hermitian_tangent"]


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


def pg_fermion_kind(fermion):
    """"wilson" for the reference even-odd terms (exact type); TypeError / ValueError otherwise."""
    if type(fermion) in _REFERENCE_TERMS:
        if not fermion.even_odd:
            raise ValueError("the PG pseudofermion term is even-odd only (TwoFlavourWilson(even_odd=True), the "
                             "QCDAction default)")
        return "wilson"
    raise TypeError(f"unsupported fermion term {type(fermion).__name__}: the PG pseudofermion term handles "
                    f"{', '.join(c.__name__ for c in _REFERENCE_TERMS)} (N_f = 2 Wilson, even-odd)")


# ---- operators --------------------------------------------------------------------------------
class _ReferenceOps:
    """D_hat, D_hat^dagger, gamma_5 and d_V D_hat on full-lattice even fields (dirac.py operators)."""

    def __init__(self, U, kappa, geometry, time_sign):
        self.geometry, self.kappa, self.time_sign = geometry, kappa, time_sign
        self.w = WilsonDirac(U, kappa, geometry, time_sign)
        self.eo = self.w.even_odd()

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
        return cg(self.eo.normal, b, tol=tol, max_iter=max_iter)


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
        fermion: `TwoFlavourWilson` / `ParametricTwoFlavourWilson` (even-odd) with phi set; its
            kappa, time_sign, geometry, solver_tol, max_iter and on_unconverged are the defaults.
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
    pg_fermion_kind(fermion)
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
        ops = _ReferenceOps(U0, kappa, geometry, fermion.time_sign)
        X_res = ops.solve(phi, tol, max_iter)
        counter.record(X_res, tol, "pg_X", policy, max_iter)
        X = X_res.x
        S_pf = _rdot(phi, X)
        with torch.enable_grad():
            Ug = U0.clone().requires_grad_(True)
            V = tangent_of(Ug)
            if V.shape != U.shape:
                raise ValueError(f"the velocity must return a tangent of shape {tuple(U.shape)}, got {tuple(V.shape)}")
            g_ops = _ReferenceOps(Ug, kappa, geometry, fermion.time_sign)
            dD = g_ops.tangent(V)
            DX = g_ops.eo.apply(X)
            dDX = dD(X)
            s = -2.0 * _rdot(DX, dDX)
            with torch.no_grad():  # (d_V M) X = gamma_5 (d_V D_hat) gamma_5 D_hat X + D_hat^dagger (d_V D_hat) X
                rhs = ops.eo.gamma5(dD(ops.eo.gamma5(DX))) + ops.eo.apply_dagger(dDX)
                rhs = rhs.detach()
            Y_res = ops.solve(rhs, tol, max_iter)
            counter.record(Y_res, tol, "pg_Y", policy, max_iter)
            t2 = 2.0 * _rdot(DX, g_ops.eo.apply(Y_res.x.detach()))
            (gradient,) = torch.autograd.grad((s + t2).sum(), Ug)
    return {"xi": s.detach() / volume, "gradient": gradient / volume, "S_pf": S_pf.detach(), "counter": counter}
