"""Batched Krylov solvers for the fermion operators.

All solvers work on fields of shape (B, ...) and treat the B samples as independent linear
systems: inner products and residual norms are per sample, and a sample whose relative residual
|r| / |b| has reached `tol` stops being updated (its step sizes are set to zero, never computed
from vanishing denominators). The operator is still applied to the whole batch at every
iteration, so the cost per iteration is that of the slowest sample.

The tolerance is relative to |b| (a sample with b = 0 is converged at x = 0). The convergence
target is never below roundoff: |r| <= max(tol |b|, eps |r_0|) with eps the machine epsilon of
the field and r_0 the initial residual, so tol = 0, or b = 0 with a nonzero x0, stop at the first
iteration where the residual is at roundoff instead of running into underflow. A sample whose
recursive residual becomes non-finite, or that meets a breakdown, is deactivated unconverged.
The returned `residual` is the true relative residual |b - A x| / |b| (|b - A x| when b = 0)
recomputed at the end, not the recursively updated one.
"""

from typing import NamedTuple

try:  # Python >= 3.8
    from typing import Protocol
except ImportError:  # pragma: no cover
    Protocol = object

import torch

__all__ = ["LinearOperator", "SolverResult", "cg", "bicgstab", "solve", "solve_normal", "MultishiftResult",
           "multishift_cg"]


class LinearOperator(Protocol):
    """Duck-typed interface of the fermion operators (WilsonDirac, EvenOddWilsonDirac).

    `apply(psi)`, `apply_dagger(psi)`, `normal(psi)` map fields of shape `field_shape` to fields of
    the same shape; `dtype` and `device` describe the fields. Even-odd operators also expose
    `prepare_rhs(b)`, `reconstruct(x_even, b)`, `gamma5(psi)` and the full operator `wilson`.
    """

    field_shape: tuple
    dtype: torch.dtype
    device: torch.device

    def apply(self, psi): ...

    def apply_dagger(self, psi): ...

    def normal(self, psi): ...


class SolverResult(NamedTuple):
    x: torch.Tensor
    iterations: torch.Tensor  # (B,) iterations performed by each sample
    residual: torch.Tensor  # (B,) true relative residual |b - A x| / |b|


def _dot(a, b):
    """Per-sample complex inner product <a, b> = sum conj(a) b, shape (B,)."""
    return (a.conj() * b).reshape(a.shape[0], -1).sum(dim=1)


def _norm2(a):
    """Per-sample squared norm, real, shape (B,)."""
    a = torch.view_as_real(a.resolve_conj()) if a.is_complex() else a
    return (a * a).reshape(a.shape[0], -1).sum(dim=1)


def _bview(c, field):
    return c.view((-1,) + (1,) * (field.dim() - 1))


def _safe(den, ok):
    """den where `ok`, 1 elsewhere: the value is discarded there, the division stays finite."""
    return torch.where(ok, den, torch.ones_like(den))


def _target(b, r, tol):
    """Squared per-sample convergence target max(tol^2 |b|^2, eps^2 |r_0|^2) (module docstring)."""
    eps = torch.finfo(b.real.dtype if b.is_complex() else b.dtype).eps
    return torch.maximum((tol ** 2) * _norm2(b), (eps ** 2) * _norm2(r))


def _finite(x):
    """Per-sample flag: every entry of the field is finite, shape (B,)."""
    return torch.isfinite(torch.view_as_real(x.resolve_conj()) if x.is_complex() else x).reshape(
        x.shape[0], -1).all(dim=1)


def _relative_residual(apply, b, x):
    r2 = _norm2(b - apply(x))
    b2 = _norm2(b)
    return torch.sqrt(r2 / torch.where(b2 > 0, b2, torch.ones_like(b2)))


def cg(apply_normal, b, x0=None, tol=1e-10, max_iter=1000):
    """Conjugate gradient for a hermitian positive-definite operator (e.g. `op.normal`).

    Args:
        apply_normal: callable A(psi), hermitian positive definite on each sample.
        b: right-hand side (B, ...).
        x0: initial guess (default zero).
        tol: relative residual target |r| / |b| per sample.
        max_iter: maximum number of iterations (one application of A each).
    Returns:
        SolverResult(x, iterations, residual).
    """
    B = b.shape[0]
    if x0 is None:
        x = torch.zeros_like(b)
        r = b.clone()
    else:
        x = x0.clone()
        r = b - apply_normal(x)
    p = r.clone()
    rr = _norm2(r)
    target = _target(b, r, tol)
    active = rr > target
    iterations = torch.zeros(B, dtype=torch.long, device=b.device)
    for _ in range(int(max_iter)):
        if not bool(active.any()):
            break
        Ap = apply_normal(p)
        pAp = _dot(p, Ap).real
        active = active & (pAp > 0) & torch.isfinite(pAp)  # breakdown: A not positive on p
        alpha = torch.where(active, rr / _safe(pAp, active), torch.zeros_like(rr))
        x = x + _bview(alpha, x).to(x.dtype) * p
        r = r - _bview(alpha, r).to(r.dtype) * Ap
        rr_new = _norm2(r)
        beta = torch.where(active, rr_new / _safe(rr, active), torch.zeros_like(rr))
        p = r + _bview(beta, p).to(p.dtype) * p
        rr = torch.where(active, rr_new, rr)
        iterations = iterations + active.long()
        active = active & torch.isfinite(rr) & (rr > target)
    return SolverResult(x, iterations, _relative_residual(apply_normal, b, x))


def bicgstab(apply, b, x0=None, tol=1e-10, max_iter=1000):
    """BiCGstab for a general (non-hermitian) operator, e.g. `op.apply` for the Wilson D.

    Each iteration applies the operator twice. Samples converge independently; a sample that
    converges at the half step (|s| below the target) takes omega = 0 and stops there. A
    breakdown (|<r_hat, v>| <= eps |r_hat| |v|, or |t|^2 = 0 with s above target) leaves the
    sample unconverged at its last finite iterate instead of producing NaN.
    Returns SolverResult(x, iterations, residual).
    """
    B = b.shape[0]
    if x0 is None:
        x = torch.zeros_like(b)
        r = b.clone()
    else:
        x = x0.clone()
        r = b - apply(x)
    r_hat = r.clone()
    p = torch.zeros_like(b)
    v = torch.zeros_like(b)
    cdtype = b.dtype
    rho = torch.ones(B, dtype=cdtype, device=b.device)
    alpha = torch.ones(B, dtype=cdtype, device=b.device)
    omega = torch.ones(B, dtype=cdtype, device=b.device)
    zero = torch.zeros(B, dtype=cdtype, device=b.device)
    eps = torch.finfo(b.real.dtype).eps if b.is_complex() else torch.finfo(b.dtype).eps
    target = _target(b, r, tol)
    rh2 = _norm2(r_hat)
    active = _norm2(r) > target
    iterations = torch.zeros(B, dtype=torch.long, device=b.device)
    for _ in range(int(max_iter)):
        if not bool(active.any()):
            break
        rho_new = _dot(r_hat, r)
        ok = active & (rho != 0) & (omega != 0)
        beta = torch.where(ok, (rho_new / _safe(rho, ok)) * (alpha / _safe(omega, ok)), zero)
        p = r + _bview(beta, p) * (p - _bview(omega, v) * v)
        v = apply(p)
        rv = _dot(r_hat, v)
        ok_a = active & (rv.abs() > eps * torch.sqrt(rh2 * _norm2(v))) & (rho_new != 0)
        alpha = torch.where(ok_a, rho_new / _safe(rv, ok_a), zero)
        s = r - _bview(alpha, v) * v
        half = _norm2(s) <= target
        t = apply(s)
        tt = _norm2(t)
        ok = ok_a & ~half & (tt > 0)
        omega = torch.where(ok, _dot(t, s) / _safe(tt.to(cdtype), ok), zero)
        x_new = x + _bview(alpha, x) * p + _bview(omega, x) * s
        r_new = s - _bview(omega, t) * t
        r2 = _norm2(r_new)
        # breakdown: alpha could not be formed, t vanished above target, or non-finite iterate
        good = ok_a & (half | (tt > 0)) & torch.isfinite(r2) & _finite(x_new)
        x = torch.where(_bview(good, x), x_new, x)
        r = torch.where(_bview(good, r), r_new, r)
        rho = torch.where(active, rho_new, rho)
        iterations = iterations + active.long()
        active = active & good & (r2 > target)
    return SolverResult(x, iterations, _relative_residual(apply, b, x))


def _full_operator(op):
    return getattr(op, "wilson", op)


def _square_solve(op, b, method, tol, max_iter, x0=None):
    if method == "bicgstab":
        return bicgstab(op.apply, b, x0=x0, tol=tol, max_iter=max_iter)
    if method == "cgne":
        return cg(op.normal, op.apply_dagger(b), x0=x0, tol=tol, max_iter=max_iter)
    raise ValueError(f"unknown method {method!r}: use 'cgne' or 'bicgstab'")


def solve(op, b, method="cgne", tol=1e-10, max_iter=1000, x0=None):
    """Solve the full-lattice system D x = b.

    For an even-odd operator (one with `prepare_rhs`) the Schur system D_hat x_e = b_e +
    kappa H_eo b_o is solved and the odd sites are reconstructed; otherwise D itself is inverted.
    `method` is 'cgne' (CG on D^dagger D with right-hand side D^dagger b, the default, always
    converges) or 'bicgstab'. `tol` is the tolerance of the inner solver; the returned residual
    is the true relative residual of the full system |b - D x| / |b|.
    """
    full = _full_operator(op)
    if hasattr(op, "prepare_rhs"):
        rhs = op.prepare_rhs(b)
        inner = _square_solve(op, rhs, method, tol, max_iter, x0=None if x0 is None else op.even(x0))
        x = op.reconstruct(inner.x, b)
    else:
        inner = _square_solve(op, b, method, tol, max_iter, x0=x0)
        x = inner.x
    return SolverResult(x, inner.iterations, _relative_residual(full.apply, b, x))


def solve_normal(op, b, tol=1e-10, max_iter=1000):
    """Solve the full-lattice normal equation D^dagger D x = b.

    With a WilsonDirac, CG on D^dagger D. With an even-odd operator, x = D^-1 gamma_5 D^-1 gamma_5 b
    (D^-dagger = gamma_5 D^-1 gamma_5) with two even-odd preconditioned solves; `iterations` is
    then the sum of the two. The returned residual is |b - D^dagger D x| / |b|.

    The even-site normal equation D_hat^dagger D_hat x_e = b_e of the even-odd pseudofermion
    action is `cg(eo.normal, b_e)` directly.
    """
    full = _full_operator(op)
    if hasattr(op, "prepare_rhs"):
        y = solve(op, op.gamma5(b), tol=tol, max_iter=max_iter)
        z = solve(op, op.gamma5(y.x), tol=tol, max_iter=max_iter)
        x, iterations = z.x, y.iterations + z.iterations
    else:
        res = cg(op.normal, b, tol=tol, max_iter=max_iter)
        x, iterations = res.x, res.iterations
    return SolverResult(x, iterations, _relative_residual(full.normal, b, x))


class MultishiftResult(NamedTuple):
    x: torch.Tensor  # (K,) + b.shape: x[k] solves (A + sigma_k) x = b
    iterations: torch.Tensor  # (K, B) iterations in which shift k was still updated, per sample
    residual: torch.Tensor  # (K, B) true relative residual |b - (A + sigma_k) x_k| / |b|


def multishift_cg(apply, b, shifts, tol=1e-10, max_iter=1000):
    """Multishift CG: (A + sigma_k) x_k = b for all shifts sigma_k >= 0 in one Krylov sequence.

    A (`apply`) must be hermitian positive definite on each sample; the shifts are a sequence of
    K floats shared by the batch, a (K,) tensor, or a (B, K) tensor of per-sample shifts. One
    application of A per iteration serves all shifts (Jegerlehner, hep-lat/9612014; Frommer et al.).

    Algorithm. CG runs on the base system A' = A + s_b, s_b = min_k sigma_k (per sample), whose
    solution converges last; with delta_k = sigma_k - s_b >= 0 the shifted residuals are collinear,
    r^k_i = zeta^k_i r_i, and (alpha_i, beta_i the base CG coefficients, beta_i = |r_{i+1}|^2/|r_i|^2,
    starting from zeta_{-1} = zeta_0 = 1, alpha_{-1} = 1, beta_{-1} = 0)
        zeta_{i+1} = zeta_i zeta_{i-1} alpha_{i-1}
                     / [alpha_i beta_{i-1} (zeta_{i-1} - zeta_i) + zeta_{i-1} alpha_{i-1} (1 + delta alpha_i)],
        alpha^k_i = alpha_i zeta_{i+1} / zeta_i,        x^k += alpha^k_i p^k,
        beta^k_i = beta_i (zeta_{i+1} / zeta_i)^2,       p^k <- zeta_{i+1} r_{i+1} + beta^k_i p^k.
    The base shift has delta = 0 and zeta = 1 identically, so it is one of the K shifts.

    Convergence is per sample and per shift: shift k of sample b stops being updated (its x, p and
    zeta frozen) once |r^k| = zeta^k |r| <= max(tol |b|, eps |b|) (the roundoff floor of `cg`);
    a sample stops when all its shifts have converged, or on a breakdown (p^dagger A' p <= 0 or a
    non-finite value), which leaves its unconverged shifts at their last iterate. The initial
    guess is zero. The returned residual is recomputed at the end (K extra applications of A).

    Returns MultishiftResult(x (K,) + b.shape, iterations (K, B), residual (K, B)).
    """
    B = b.shape[0]
    real = b.real.dtype if b.is_complex() else b.dtype
    s = torch.as_tensor(shifts, dtype=real, device=b.device)
    if s.dim() == 1:
        s = s.unsqueeze(0)
    if s.dim() != 2 or s.shape[0] not in (1, B) or s.shape[1] == 0:
        raise ValueError(f"shifts must have shape (K,) or ({B}, K), got {tuple(s.shape)}")
    s = s.expand(B, -1)
    if bool((s < 0).any()) or not bool(torch.isfinite(s).all()):
        raise ValueError("multishift_cg needs finite non-negative shifts")
    K = s.shape[1]
    base = s.min(dim=1).values  # (B,)
    delta = s - base[:, None]  # (B, K) >= 0

    def apply_base(v):
        return apply(v) + _bview(base, v).to(v.dtype) * v

    def kview(c):  # (B, K) -> (K, B, 1, ..., 1)
        return c.transpose(0, 1).reshape((K, B) + (1,) * (b.dim() - 1))

    x = torch.zeros((K,) + tuple(b.shape), dtype=b.dtype, device=b.device)
    r = b.clone()
    p = b.clone()
    ps = b.unsqueeze(0).repeat((K,) + (1,) * b.dim())
    rr = _norm2(r)
    target = _target(b, b, tol)  # (B,)
    ones = torch.ones((B, K), dtype=real, device=b.device)
    zeta, zeta_prev = ones.clone(), ones.clone()
    alpha_prev = torch.ones(B, dtype=real, device=b.device)
    beta_prev = torch.zeros(B, dtype=real, device=b.device)
    active_k = (zeta ** 2) * rr[:, None] > target[:, None]
    iterations = torch.zeros((B, K), dtype=torch.long, device=b.device)
    zero = torch.zeros(B, dtype=real, device=b.device)
    for _ in range(int(max_iter)):
        active = active_k.any(dim=1)
        if not bool(active.any()):
            break
        Ap = apply_base(p)
        pAp = _dot(p, Ap).real
        active = active & (pAp > 0) & torch.isfinite(pAp)  # breakdown: A' not positive on p
        active_k = active_k & active[:, None]
        alpha = torch.where(active, rr / _safe(pAp, active), zero)
        den = (alpha[:, None] * beta_prev[:, None] * (zeta_prev - zeta)
               + zeta_prev * alpha_prev[:, None] * (1.0 + delta * alpha[:, None]))
        ok = active_k & (den != 0) & torch.isfinite(den)
        zeta_next = torch.where(ok, zeta * zeta_prev * alpha_prev[:, None] / _safe(den, ok), zeta)
        ratio = torch.where(ok, zeta_next / _safe(zeta, ok), torch.zeros_like(zeta))
        alpha_k = alpha[:, None] * ratio
        x = x + kview(alpha_k).to(x.dtype) * ps
        r = r - _bview(alpha, r).to(r.dtype) * Ap
        rr_new = _norm2(r)
        beta = torch.where(active, rr_new / _safe(rr, active), zero)
        beta_k = beta[:, None] * ratio ** 2
        ps_new = kview(zeta_next).to(ps.dtype) * r.unsqueeze(0) + kview(beta_k).to(ps.dtype) * ps
        ps = torch.where(kview(ok), ps_new, ps)
        p = r + _bview(beta, p).to(p.dtype) * p
        iterations = iterations + ok.long()
        zeta_prev = torch.where(ok, zeta, zeta_prev)
        zeta = zeta_next
        alpha_prev = torch.where(active, alpha, alpha_prev)
        beta_prev = torch.where(active, beta, beta_prev)
        rr = torch.where(active, rr_new, rr)
        active_k = ok & torch.isfinite(rr)[:, None] & ((zeta ** 2) * rr[:, None] > target[:, None])
    b2 = _norm2(b)
    b2 = torch.where(b2 > 0, b2, torch.ones_like(b2))
    residual = torch.stack([
        torch.sqrt(_norm2(b - apply(x[k]) - _bview(s[:, k], b).to(b.dtype) * x[k]) / b2) for k in range(K)])
    return MultishiftResult(x, iterations.transpose(0, 1).contiguous(), residual)
