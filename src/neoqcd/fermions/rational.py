"""Rational approximations of x^alpha on a positive interval, in partial-fraction form (RHMC).

The RHMC (Clark and Kennedy, hep-lat/0608015) needs, on the spectral interval [lower, upper] of a
hermitian positive-definite operator M,

    r(x) = a_0 + sum_k a_k / (x + b_k)  ~  x^alpha,     b_k >= 0,

with a small *relative* error  delta = max_{x in [lower, upper]} |r(x) / x^alpha - 1|.  Then r(M) v
is a_0 v + sum_k a_k (M + b_k)^-1 v, one multishift CG (neoqcd.fermions.solver.multishift_cg).
Every approximation here is of type (n, n) (numerator and denominator of degree n = `degree`),
which is the partial-fraction form above with n poles.

Powers. alpha = -1 (exact) and 0 < |alpha| < 1. alpha = 0 (r = 1) and |alpha| >= 1 otherwise raise
ValueError: x^0 needs no approximation, and for |alpha| >= 1 a type-(n, n) approximation is either
degenerate (alpha = 1: p/q = x exactly, no partial-fraction form) or not of the Stieltjes class
with real negative poles that the RHMC relies on.

Methods (`rational_approximation(power, ...)` chooses with method="auto"):

- "exact"      alpha = -1: r(x) = 1 / x, a single pole at 0 (a_0 = 0, a_1 = 1, b_1 = 0), error 0.
- "zolotarev"  alpha = -1/2 and alpha = +1/2: Zolotarev's optimal (minimax relative error)
               approximation of 1/sqrt(x) (Zolotarev 1877; in the form of Chiu, Hsieh, Huang and
               Huang, hep-lat/0206007, eqs. (8)-(10)). With y = x / lower in [1, kappa],
                   R(y) = d_0 prod_{l=1}^n (y + c_{2l}) / (y + c_{2l-1}),
                   c_l = sn^2(u_l; k') / cn^2(u_l; k'),  u_l = l K' / (2n+1),  l = 1 .. 2n,
                   k'^2 = 1 - 1/kappa,  K' = K(k'),
               (sn, cn Jacobi elliptic functions and K the complete elliptic integral with
               parameter m = k'^2). The c_l are computed in multiprecision (mpmath ellipfun,
               ellipk, 40 digits, with k'^2 = 1 - 1/kappa formed in multiprecision so that the
               complementary parameter 1/kappa is exact) and used in multiprecision up to the final
               float64 partial fractions: in float64 the parameter m = 1 - 1/kappa and
               sn^2 / (1 - sn^2) lose digits (relative error of c_l ~1e-12 at kappa ~ 400, ~1e-10 at
               1.6e5, ~1e-8 at 1.6e7), which puts a floor on the achievable error. The scale d_0 is
               fixed so that the relative error sqrt(y) R(y) - 1 equioscillates symmetrically,
                   d_0 = 2 / (max g + min g),   delta = (max g - min g) / (max g + min g),
               g(y) = sqrt(y) prod(...), with max and min located numerically (log grid plus local
               refinement). x^{+1/2} is the reciprocal 1/R, which has the same relative error to
               first order (it is certified separately).
- "remez"      any other alpha (in particular x^{-1/4}, x^{+1/4}, x^{-gamma/n_pf}, ...): the
               rational Remez exchange algorithm for the minimax relative error of type (n, n),
               with a log-Chebyshev initial reference. At each step the nonlinear interpolation
               system (p, q polynomials in t = (2x - lower - upper)/(upper - lower), q(0) = 1)
                   p(x_i) - x_i^alpha (1 + (-1)^i E) q(x_i) = 0,   i = 0 .. 2n+1,
               is solved in multiprecision (mpmath; linear fixed-point iterations in E, then
               Newton). The 2n+1 zeros of the error between the reference points and the extremum
               of |error| in each of the 2n+2 sign intervals are then located in float64, with the
               rational function in product form (roots of p and q from mpmath): 20001-point log
               grid, brentq, bounded minimisation. The extrema become the new reference. Stops when
               they are levelled to a relative spread below `level_tol` (1e-3; the error is
               certified independently anyway), or to the float64 resolution of the error. The
               result is converted to partial fractions (roots of q in multiprecision; the poles
               must come out real and <= 0) and then rounded to float64.
               Working precision. The monomial coefficients of p and q cancel strongly near the
               lower end of the interval (the poles cluster at x ~ -lower): q(t) is a factor
               ~ kappa^(n/2) smaller there than sum_j |q_j t^j|, so the interpolation system needs
               about n log10(kappa) / 2 digits on top of the ~15 digits of the error itself. The
               default `dps` is `remez_dps(degree, kappa)` = max(30, 20 + 0.6 n log10(kappa)) (the
               measured failure threshold is ~15 + 0.5 n log10(kappa): degree 17 at kappa = 4e4
               fails at 50 digits and works at 55, degree 22 at kappa = 1.6e5 needs ~75). After
               every solve the cancellation is measured a posteriori, and a step whose interpolant
               is not accurate to 1e-3 of E, or whose error does not alternate on the reference,
               is a precision failure: the whole construction is repeated at the larger of 1.5 dps
               and the measured need, up to `max_dps` (400). An over-specified degree, whose
               minimax error is below the float64 resolution of the error evaluation (the error no
               longer alternates in float64 although the multiprecision solve is accurate), returns
               the current interpolant: its certified error is at the float64 floor.
               Cost. Newton stops when its step stops shrinking (the precision floor), and the
               roots of p and q are refined from those of the previous step (mpmath polyroots with
               roots_init): degree 17 at kappa = 4e4 takes ~3 s, degree 21 at 1.6e5 ~6 s.

Guarantee. The partial-fraction sum has cancellation for alpha > 0 (a_0 > 0, a_k < 0): with
    c(x) = (|a_0| + sum_k |a_k| / (x + b_k)) / |r(x)|
(c = 1 for alpha < 0, all terms positive; for alpha > 0 c is largest at x = lower, where
r ~ lower^alpha is small: on the default 4D interval max c = 37 / 7.8e2 for x^{1/4} / x^{1/2} at
m = 0.1 and 1.2e2 / 7.8e3 at m = 0.01), a plain float64 evaluation of r(x) has a rounding error up
to ~ (n + 2) c(x) eps relative (observed ~ c eps), which for x^{1/2} at small mass exceeds the
approximation error itself (4D m = 0.01, degree 22: 3.8e-12 against 6.7e-13). A worst-case
allowance of that size inside `error` would make tolerances of 1e-12 unreachable for the heat
bath at small mass although the rational function itself meets them, so two quantities are
stored:
- `error`: certified max relative error of the rational function with the float64 coefficients
  actually used, |r(x) / x^alpha - 1| on [lower, upper]. `relative_error(x)` evaluates it in
  compensated (double-double) arithmetic, error-free transformations for x + b_k, the divisions and
  the sum, so its own rounding is ~3 eps independently of c. It is evaluated on a log grid of 4001
  points, every local maximum in the upper half of the range is refined (vectorised zoom, 4 rounds
  of 65 points around each peak), and an allowance `allowance` = (8 (n + 2) + (n + 2)^2 c eps) eps
  is added, so that no evaluation of `relative_error` on any grid exceeds `error`. The error of a
  minimax approximation is a smooth equioscillating function, so the refined maxima are its
  supremum up to the resolution of the zoom (relative 1e-10 of the peak height).
- `rounding` = (2 (n + 2) max_x c(x) + 8) eps: the first-order bound on the extra rounding of a
  plain float64 evaluation (`__call__`, a_0 + sum_k a_k / (x + b_k) summed left to right, n + 1
  additions and two roundings per term, then the division by x^alpha; the bound is
  ((n + 2) c + 3) eps, doubled). So |plain float64 r(x) / x^alpha - 1| <= error + rounding on any
  grid. `cancellation` is max_x c(x) (times 1.01), on the 4001-point grid (c is smooth).
In the RHMC the same cancellation multiplies the multishift-solver error of r(M) phi: a heat bath
with x^{+g/2} is accurate to ~ cancellation x solver tolerance, not to `error`.
`max_relative_error(approx, grid)` measures the (compensated) error on any grid.

Degrees. `rational_approximation(power, lower, upper, tol=...)` returns the lowest degree whose
certified error is <= tol, or the requested `degree`. The search starts from the asymptotic
Zolotarev estimate delta_n ~ 4 exp(-2 pi^2 n / log(16 kappa)), predicts the next degree from the
error measured at the last one with the same rate, and bisects inside the bracket [largest degree
that fails, smallest that succeeds] until they are adjacent (typically 2-3 builds). A tolerance
below the float64 floor of the certified error raises RuntimeError. Results are cached per
(power, interval, degree, method); `RationalApproximation` objects are immutable (attributes
cannot be reassigned, the arrays are read-only), so the shared cached objects are safe.

Positivity. For alpha < 0 (x^alpha a Stieltjes function) the minimax approximation has a_0 >= 0,
a_k > 0 and b_k >= 0, so r(M) is positive definite. For 0 < alpha < 1 the residues are negative
and a_0 > 0. All b_k >= 0 is checked (multishift CG needs non-negative shifts).
"""

import functools
import math
from fractions import Fraction

import numpy as np

__all__ = [
    "RationalApproximation",
    "rational_approximation",
    "zolotarev_inverse_sqrt",
    "zolotarev_sqrt",
    "remez_approximation",
    "remez_dps",
    "max_relative_error",
    "log_grid",
]

EPS = float(np.finfo(np.float64).eps)
MAX_DEGREE = 200


def _read_only(values):
    arr = np.array(values, dtype=np.float64)  # always a fresh copy
    arr.flags.writeable = False
    return arr


def _two_sum(a, b):
    """s + e = a + b exactly (Knuth)."""
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _split(a):
    c = 134217729.0 * a  # 2^27 + 1 (Veltkamp)
    hi = c - (c - a)
    return hi, a - hi


def _two_prod(a, b):
    """p + e = a * b exactly (Dekker)."""
    p = a * b
    ah, al = _split(a)
    bh, bl = _split(b)
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


class RationalApproximation:
    """r(x) = constant + sum_k residues[k] / (x + shifts[k]) ~ x^power on [lower, upper].

    Attributes: power (Fraction), lower, upper, constant (a_0), residues (a_k), shifts (b_k >= 0)
    as read-only float64 numpy arrays, method ("exact", "zolotarev", "remez"), degree = number of
    poles, and (module docstring, Guarantee) error (certified max relative error on [lower, upper],
    including the allowance `allowance`), cancellation (max of c(x)), rounding (extra error bound
    of a plain float64 evaluation). Immutable: attributes cannot be reassigned after construction
    (instances are cached and shared).
    """

    def __init__(self, power, lower, upper, constant, residues, shifts, method, error=None):
        self.power = Fraction(power)
        self.lower = float(lower)
        self.upper = float(upper)
        self.constant = float(constant)
        self.residues = _read_only(residues)
        self.shifts = _read_only(shifts)
        self.method = str(method)
        if self.residues.shape != self.shifts.shape or self.residues.ndim != 1:
            raise ValueError("residues and shifts must be 1D arrays of equal length")
        if (self.shifts < 0).any():
            raise ValueError(f"negative shift in the rational approximation: {self.shifts}")
        if error is None:
            self.error, self.allowance, self.cancellation, self.rounding = _certify(self)
        else:
            self.error, self.allowance, self.cancellation, self.rounding = float(error), 0.0, 1.0, 0.0
        self._frozen = True

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError(f"RationalApproximation is immutable (cannot set {name!r})")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        raise AttributeError(f"RationalApproximation is immutable (cannot delete {name!r})")

    @property
    def degree(self):
        return int(self.shifts.shape[0])

    def __call__(self, x):
        """r(x) for a float, a numpy array or a torch tensor (elementwise), plain float64 sum."""
        out = self.constant + 0.0 * x
        for a, b in zip(self.residues, self.shifts):
            out = out + float(a) / (x + float(b))
        return out

    def cancellation_factor(self, x):
        """c(x) = (|a_0| + sum_k |a_k| / (x + b_k)) / |r(x)| on a numpy array (module docstring)."""
        x = np.asarray(x, dtype=np.float64)
        total = abs(self.constant) + 0.0 * x
        for a, b in zip(self.residues, self.shifts):
            total = total + abs(float(a)) / (x + float(b))
        return total / np.abs(self(x))

    def target(self, x):
        """x^power (numpy/float)."""
        return np.power(x, float(self.power))

    def relative_error(self, x):
        """r(x) / x^power - 1 on a numpy array, in compensated arithmetic (module docstring): each
        a_k / (x + b_k) and the sum in double-double, so the rounding is ~3 eps whatever the
        cancellation of the partial-fraction sum."""
        x = np.asarray(x, dtype=np.float64)
        hi = np.full(x.shape, self.constant)
        lo = np.zeros(x.shape)
        for a, b in zip(self.residues, self.shifts):
            a, b = float(a), float(b)
            d, de = _two_sum(x, b)  # x + b = d + de exactly
            q = a / d
            p, pe = _two_prod(q, d)  # q d = p + pe exactly
            ql = (((a - p) - pe) - q * de) / d  # a / (d + de) = q + ql + O(eps^2)
            hi, e = _two_sum(hi, q)
            lo = lo + (e + ql)
        t = self.target(x)
        return ((hi - t) + lo) / t

    def plain_relative_error(self, x):
        """r(x) / x^power - 1 with the plain float64 r(x) of `__call__` (rounding up to `rounding`)."""
        x = np.asarray(x, dtype=np.float64)
        return self(x) / self.target(x) - 1.0

    def __repr__(self):
        return (f"RationalApproximation(x^{self.power} on [{self.lower:.6g}, {self.upper:.6g}], "
                f"{self.method}, degree {self.degree}, error {self.error:.3e})")


def log_grid(lower, upper, n=4001):
    """n log-spaced points from lower to upper (both included), float64."""
    return np.exp(np.linspace(math.log(lower), math.log(upper), int(n)))


def max_relative_error(approx, grid, plain=False):
    """max |r(x) / x^power - 1| over the points of `grid` (numpy array, list or torch tensor);
    compensated evaluation (`relative_error`), or the plain float64 one with plain=True."""
    x = np.asarray(grid.cpu().numpy() if hasattr(grid, "cpu") else grid, dtype=np.float64)
    f = approx.plain_relative_error if plain else approx.relative_error
    return float(np.max(np.abs(f(x))))


def _refined_max(func, lower, upper, n_grid=4001, max_peaks=200, rounds=4, points=65):
    """max of func on [lower, upper]: log grid, then every local maximum in the upper half of the
    range refined by a vectorised zoom (`rounds` rounds of `points` points on the bracket of the
    current best point, which shrinks by (points - 1) / 2 per round)."""
    s = np.linspace(math.log(lower), math.log(upper), n_grid)
    v = func(np.exp(s))
    best = float(v.max())
    # local maxima of the grid values in the upper half of their range (the equioscillation peaks
    # of a minimax error all have the same height; smaller ones are roundoff noise)
    floor = best - 0.5 * (best - float(v.min()))
    left = np.concatenate(([True], v[1:] >= v[:-1]))
    right = np.concatenate((v[:-1] >= v[1:], [True]))
    peaks = np.nonzero((v >= floor) & left & right)[0]
    peaks = peaks[np.argsort(-v[peaks], kind="stable")][:max_peaks]
    lo = s[np.maximum(peaks - 1, 0)]
    hi = s[np.minimum(peaks + 1, n_grid - 1)]
    u = np.linspace(0.0, 1.0, points)
    rows = np.arange(peaks.size)
    for _ in range(rounds):
        pts = lo[:, None] + (hi - lo)[:, None] * u[None, :]
        vals = func(np.clip(np.exp(pts.ravel()), lower, upper)).reshape(pts.shape)
        best = max(best, float(vals.max()))
        k = vals.argmax(axis=1)
        h = (hi - lo) / (points - 1)
        centre = pts[rows, k]
        lo, hi = np.maximum(centre - h, lo), np.minimum(centre + h, hi)
    return best


def _certify(approx, n_grid=4001):
    """(error, allowance, cancellation, rounding) of `approx` (module docstring, Guarantee)."""
    peak = _refined_max(lambda x: np.abs(approx.relative_error(x)), approx.lower, approx.upper, n_grid)
    c = 1.01 * float(np.max(approx.cancellation_factor(log_grid(approx.lower, approx.upper, n_grid))))
    n2 = approx.degree + 2
    allowance = (8.0 * n2 + n2 ** 2 * c * EPS) * EPS
    rounding = (2.0 * n2 * c + 8.0) * EPS
    return peak + allowance, allowance, c, rounding


def certified_relative_error(approx, n_grid=4001):
    """Certified max relative error of `approx` on its interval (module docstring, Guarantee)."""
    return _certify(approx, n_grid)[0]


def _partial_fractions(scale, zeros, poles):
    """scale * prod(x - zeros) / prod(x - poles) = a_0 + sum a_k / (x - poles_k), equal degrees.

    In multiprecision (mpmath values or floats); returns (a_0, [a_k]) as mpmath numbers.
    """
    import mpmath as mp

    residues = []
    for k, pk in enumerate(poles):
        num = mp.mpf(1)
        for z in zeros:
            num *= (pk - z)
        den = mp.mpf(1)
        for j, pj in enumerate(poles):
            if j != k:
                den *= (pk - pj)
        residues.append(scale * num / den)
    return scale, residues


def _check_power(power):
    """Fraction(power) if supported (module docstring, Powers), else ValueError."""
    if isinstance(power, float):
        power = Fraction(power).limit_denominator(10 ** 6)
    power = Fraction(power)
    if power == 0:
        raise ValueError("power 0: x^0 = 1 needs no rational approximation")
    if power != -1 and not (-1 < power < 1):
        raise ValueError(f"power {power} not supported: need power = -1 (exact) or 0 < |power| < 1")
    return power


# ---- Zolotarev -----------------------------------------------------------------------------------

def _zolotarev_coefficients(kappa, degree, dps=40):
    """c_1 .. c_{2n} of the Zolotarev approximation of 1/sqrt(y) on [1, kappa] (module docstring),
    as mpmath numbers at `dps` digits."""
    import mpmath as mp

    n = int(degree)
    with mp.workdps(int(dps)):
        m = 1 - 1 / mp.mpf(kappa)  # k'^2, with the complementary parameter 1/kappa exact
        Kp = mp.ellipk(m)
        out = []
        for l in range(1, 2 * n + 1):
            u = l * Kp / (2 * n + 1)
            sn = mp.ellipfun("sn", u, m=m)
            cn = mp.ellipfun("cn", u, m=m)
            out.append((sn / cn) ** 2)
        return out


def _zolotarev_shape(lower, upper, degree, dps=50):
    """Zeros and poles (in x, multiprecision) and the scale d_0 / sqrt(lower) of the Zolotarev
    approximation of x^{-1/2} on [lower, upper]."""
    import mpmath as mp

    with mp.workdps(dps):
        lower_m = mp.mpf(lower)
        c = _zolotarev_coefficients(mp.mpf(upper) / lower_m, degree, dps=max(40, dps))
        zeros = [-lower_m * c[2 * l + 1] for l in range(degree)]  # x = -lower c_{2l}
        poles = [-lower_m * c[2 * l] for l in range(degree)]  # x = -lower c_{2l-1}
        zf = [float(z) for z in zeros]
        pf = [float(p) for p in poles]

        def g(x):  # sqrt(x) prod (x - zeros) / (x - poles), float64 evaluation for the search
            x = np.asarray(x, dtype=np.float64)
            out = np.sqrt(x)
            for z, p in zip(zf, pf):
                out = out * (x - z) / (x - p)
            return out

        # extrema of g on the interval: d_0 = 2 / (max + min)
        hi = _refined_max(g, lower, upper)
        lo = -_refined_max(lambda x: -g(x), lower, upper)
        scale = mp.mpf(2.0) / (mp.mpf(hi) + mp.mpf(lo))
        return zeros, poles, scale


@functools.lru_cache(maxsize=None)
def zolotarev_inverse_sqrt(lower, upper, degree):
    """Zolotarev optimal type-(n, n) approximation of x^{-1/2} on [lower, upper], n = degree."""
    import mpmath as mp

    with mp.workdps(50):
        zeros, poles, scale = _zolotarev_shape(lower, upper, int(degree))
        a0, res = _partial_fractions(scale, zeros, poles)
        return RationalApproximation(Fraction(-1, 2), lower, upper, float(a0), [float(a) for a in res],
                                     [-float(p) for p in poles], "zolotarev")


@functools.lru_cache(maxsize=None)
def zolotarev_sqrt(lower, upper, degree):
    """x^{+1/2} on [lower, upper] as the reciprocal of the Zolotarev approximation of x^{-1/2}."""
    import mpmath as mp

    with mp.workdps(50):
        zeros, poles, scale = _zolotarev_shape(lower, upper, int(degree))
        a0, res = _partial_fractions(1 / scale, poles, zeros)
        return RationalApproximation(Fraction(1, 2), lower, upper, float(a0), [float(a) for a in res],
                                     [-float(z) for z in zeros], "zolotarev")


# ---- Remez ---------------------------------------------------------------------------------------

class _PrecisionFailure(RuntimeError):
    """A Remez step that is not resolved at the working precision (retried at a higher dps).
    `need` is the estimated number of digits, if known."""

    def __init__(self, msg, need=None):
        super().__init__(msg)
        self.need = need


class _NoAlternation(RuntimeError):
    pass


def remez_dps(degree, kappa):
    """Default working precision (decimal digits) of the Remez construction of degree n on an
    interval of condition number kappa = upper / lower: 20 + 0.6 n log10(kappa), at least 30
    (module docstring, Working precision; the measured threshold is ~15 + 0.5 n log10(kappa))."""
    return int(max(30, 20 + math.ceil(0.6 * int(degree) * math.log10(max(float(kappa), 10.0)))))


def _gauss_solve(A, b):
    """Solve A x = b (lists of mpmath numbers) by Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [list(row) + [bi] for row, bi in zip(A, b)]
    for k in range(n):
        piv = max(range(k, n), key=lambda i: abs(M[i][k]))
        M[k], M[piv] = M[piv], M[k]
        pk = M[k][k]
        for i in range(k + 1, n):
            f = M[i][k] / pk
            if f:
                Mi, Mk = M[i], M[k]
                for j in range(k + 1, n + 1):
                    Mi[j] -= f * Mk[j]
    x = [None] * n
    for k in range(n - 1, -1, -1):
        acc = M[k][n]
        for j in range(k + 1, n):
            acc -= M[k][j] * x[j]
        x[k] = acc / M[k][k]
    return x


def _horner(coeffs, t):
    out = coeffs[-1]
    for c in reversed(coeffs[:-1]):
        out = out * t + c
    return out


class _RemezState:
    """Type-(n, n) rational p(t)/q(t) in the variable t = (2x - lower - upper)/(upper - lower),
    q(t) = 1 + q_1 t + ... (coefficients ascending), approximating f(x) = x^alpha."""

    def __init__(self, mp, alpha, lower, upper, n):
        self.mp = mp
        self.alpha = mp.mpf(alpha.numerator) / alpha.denominator
        self.lower = mp.mpf(lower)
        self.upper = mp.mpf(upper)
        self.n = n
        self.c1 = 2 / (self.upper - self.lower)
        self.c0 = -(self.upper + self.lower) / (self.upper - self.lower)

    def t(self, x):
        return self.c1 * x + self.c0

    def f(self, x):
        return self.mp.power(x, self.alpha)

    def error(self, p, q, x):
        t = self.t(x)
        return _horner(p, t) / (self.f(x) * _horner(q, t)) - 1

    def solve(self, ref, E0=0, q0=None):
        """Solve the interpolation system on the reference `ref` (2n+2 points); returns p, q, E.

        Unknowns p_0..p_n, q_1..q_n (q_0 = 1) and E. A few linear fixed-point steps (the E q term
        taken with the previous q), then Newton on the full nonlinear system, until the relative step
        is below 10^-(dps/2) or stops shrinking below 1e-10 (the precision floor of an ill-conditioned
        system, which the a-posteriori check_precision judges).
        """
        mp, n = self.mp, self.n
        m = 2 * n + 2
        ts = [self.t(x) for x in ref]
        fs = [self.f(x) for x in ref]
        signs = [(-1) ** i for i in range(m)]
        powers = []
        for t in ts:
            row = [mp.mpf(1)]
            for _ in range(n):
                row.append(row[-1] * t)
            powers.append(row)
        q = [mp.mpf(1)] + [mp.mpf(0)] * n if q0 is None else list(q0)
        E = mp.mpf(E0)
        p = None
        small = mp.mpf(10) ** (-(mp.mp.dps // 2))
        for _ in range(4):
            A, rhs = [], []
            for i in range(m):
                qi = sum(c * w for c, w in zip(q, powers[i]))
                A.append(list(powers[i]) + [-fs[i] * powers[i][j] for j in range(1, n + 1)]
                         + [-signs[i] * fs[i] * qi])
                rhs.append(fs[i])
            sol = _gauss_solve(A, rhs)
            p, q_new, E_new = sol[:n + 1], [mp.mpf(1)] + sol[n + 1:2 * n + 1], sol[2 * n + 1]
            done = abs(E_new - E) <= small * abs(E_new)
            q, E = q_new, E_new
            if done:
                break
        prev = None
        for _ in range(20):
            F, J = [], []
            for i in range(m):
                qi = sum(c * w for c, w in zip(q, powers[i]))
                g = fs[i] * (1 + signs[i] * E)
                F.append(-(sum(c * w for c, w in zip(p, powers[i])) - g * qi))
                J.append(list(powers[i]) + [-g * powers[i][j] for j in range(1, n + 1)]
                         + [-signs[i] * fs[i] * qi])
            d = _gauss_solve(J, F)
            p = [p[j] + d[j] for j in range(n + 1)]
            q = [q[0]] + [q[j] + d[n + j] for j in range(1, n + 1)]
            E = E + d[2 * n + 1]
            scale = max(abs(c) for c in p + q)
            step = max(max(abs(c) for c in d[:2 * n + 1]) / scale, abs(d[-1]) / abs(E))
            if step <= small:
                break
            # quadratic convergence has reached the precision floor: the step no longer shrinks
            if prev is not None and step > 0.25 * prev and step < mp.mpf(10) ** (-10):
                break
            prev = step
        return p, q, E

    def check_precision(self, p, q, E, ref):
        """A posteriori: the cancellation of the monomial sums of p and q at the reference points,
        max_i sum_j |c_j t_i^j| / |c(t_i)|, times 10^-dps and the size of the system, must be below
        1e-3 max(|E|, float64 resolution); else _PrecisionFailure with the needed digits."""
        mp, n = self.mp, self.n
        worst = mp.mpf(1)
        for x in ref:
            t = self.t(x)
            for c in (p, q):
                absval = _horner([abs(v) for v in c], abs(t))
                val = abs(_horner(c, t))
                worst = max(worst, absval / val if val else mp.inf)
        target = max(abs(E), mp.mpf(64 * (n + 1) * EPS))
        noise = worst * (2 * n + 2) ** 2 * mp.mpf(10) ** (-mp.mp.dps)
        if not noise <= mp.mpf("1e-3") * target:
            need = int(mp.ceil(mp.log10(worst * (2 * n + 2) ** 2 / target))) + 6 if worst != mp.inf else None
            raise _PrecisionFailure(f"cancellation {mp.nstr(worst, 3)} in the monomial sums not resolved "
                                    f"at {mp.mp.dps} digits (|E| = {mp.nstr(abs(E), 3)})", need)

    def roots(self, name, c):
        """Roots in t of the polynomial with ascending coefficients c (mpmath polyroots), started
        from the roots of the previous step's polynomial `name` when available (a few Durand-Kerner
        steps instead of a cold start)."""
        mp = self.mp
        init = getattr(self, "_roots", {}).get(name)
        try:
            r = mp.polyroots(list(reversed(c)), maxsteps=400, extraprec=mp.mp.prec, roots_init=init)
        except mp.NoConvergence:
            if init is None:
                raise
            r = mp.polyroots(list(reversed(c)), maxsteps=400, extraprec=mp.mp.prec)
        if not hasattr(self, "_roots"):
            self._roots = {}
        self._roots[name] = r
        return r

    def float_error(self, p, q):
        """The error p / (f q) - 1 as a vectorised float64 function of x, in product form
        C prod(x - zeros_p) / prod(x - zeros_q) x^-alpha - 1 (roots in multiprecision, possibly
        complex in early iterations), accurate to a few ulp relative to r(x)."""
        mp = self.mp
        n = self.n
        zp = [complex((r - self.c0) / self.c1) for r in self.roots("p", p)]
        zq = [complex((r - self.c0) / self.c1) for r in self.roots("q", q)]
        C = float(p[n] / q[n])
        alpha = float(self.alpha)

        def err(x):
            x = np.asarray(x, dtype=np.float64)
            out = np.full(x.shape, C, dtype=np.complex128) * np.power(x, -alpha)
            for z1, z2 in zip(zp, zq):
                out = out * (x - z1) / (x - z2)
            return out.real - 1.0

        return err

    def exchange(self, p, q, ref, n_grid=20001):
        """New reference: the extremum of |error| in each of the 2n+2 sign intervals (float64)."""
        from scipy.optimize import brentq, minimize_scalar

        err = self.float_error(p, q)
        ls = [math.log(float(x)) for x in ref]
        h = lambda s: err(np.exp(s))
        zeros = []
        for i in range(len(ls) - 1):
            ea, eb = float(h(ls[i])), float(h(ls[i + 1]))
            if ea * eb > 0:
                raise _NoAlternation("Remez: the error does not alternate on the reference")
            zeros.append(brentq(lambda s: float(h(s)), ls[i], ls[i + 1], xtol=1e-14, rtol=1e-14))
        edges = [math.log(float(self.lower))] + zeros + [math.log(float(self.upper))]
        grid = np.linspace(edges[0], edges[-1], n_grid)
        values_grid = np.abs(h(grid))
        new_ref, values = [], []
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            inside = np.nonzero((grid >= lo) & (grid <= hi))[0]
            cands = [lo, hi] if inside.size == 0 else [grid[inside[np.argmax(values_grid[inside])]]]
            if inside.size:
                k = inside[np.argmax(values_grid[inside])]
                a_, b_ = max(grid[max(k - 1, 0)], lo), min(grid[min(k + 1, n_grid - 1)], hi)
                if b_ > a_:
                    res = minimize_scalar(lambda s: -abs(float(h(s))), bounds=(a_, b_), method="bounded",
                                          options={"xatol": 1e-13})
                    cands.append(float(res.x))
            if i == 0:
                cands.append(lo)
            if i == len(edges) - 2:
                cands.append(hi)
            best = max(cands, key=lambda s: abs(float(h(s))))
            new_ref.append(self.mp.exp(self.mp.mpf(best)))
            values.append(abs(float(h(best))))
        return new_ref, values


def _remez_at(alpha, lower, upper, n, dps, level_tol, max_steps):
    """One Remez construction at `dps` digits; _PrecisionFailure if the precision is insufficient."""
    import mpmath as mp

    with mp.workdps(int(dps)):
        st = _RemezState(mp, alpha, lower, upper, n)
        L = mp.log(st.lower)
        R = mp.log(st.upper)
        ref = [mp.exp(L + (R - L) * (1 - mp.cos(mp.pi * i / (2 * n + 1))) / 2) for i in range(2 * n + 2)]
        E, q = 0, None
        roundoff = 64 * (n + 1) * EPS
        converged = False
        for step in range(int(max_steps)):
            p, q, E = st.solve(ref, E0=E, q0=q)
            st.check_precision(p, q, E, ref)
            try:
                ref_new, values = st.exchange(p, q, ref)
            except _NoAlternation:
                if abs(E) < roundoff:
                    # over-specified degree: the minimax error is below the float64 resolution of
                    # the error, so the float64 error is noise; keep the current interpolant
                    converged = True
                    break
                raise _PrecisionFailure(f"the error does not alternate on the reference "
                                        f"(step {step}, |E| = {mp.nstr(abs(E), 3)})")
            spread = (max(values) - min(values)) / max(values)
            ref = ref_new
            # levelled, or levelled down to the float64 resolution of the error (degrees whose
            # minimax error is near roundoff cannot be levelled further in float64)
            if spread < level_tol or max(values) - min(values) < 32 * (n + 1) * EPS:
                p, q, E = st.solve(ref, E0=E, q0=q)
                st.check_precision(p, q, E, ref)
                converged = True
                break
        if not converged:
            raise _PrecisionFailure(f"the exchange did not level in {max_steps} steps: spread {spread:.2e}")
        # partial fractions in x: q(t(x)) has roots t_k -> x_k
        q_roots = st.roots("q", q)
        poles = []
        for tr in q_roots:
            if abs(mp.im(tr)) > mp.mpf(10) ** (-int(dps) // 3) * (1 + abs(tr)):
                raise _PrecisionFailure(f"complex pole {mp.nstr(tr, 6)}")
            poles.append((mp.re(tr) - st.c0) / st.c1)
        if any(xk > 0 for xk in poles):
            raise _PrecisionFailure(f"positive pole {[mp.nstr(x, 6) for x in poles if x > 0]}")
        a0 = p[n] / q[n]
        residues = []
        dq = [j * q[j] for j in range(1, n + 1)]
        for xk in poles:
            tk = st.t(xk)
            residues.append(_horner(p, tk) / (st.c1 * _horner(dq, tk)))
        order = sorted(range(n), key=lambda k: -poles[k])
        return float(a0), [float(residues[k]) for k in order], [-float(poles[k]) for k in order]


def remez_approximation(power, lower, upper, degree, dps=None, level_tol=1e-3, max_steps=40, max_dps=400):
    """Minimax relative-error type-(n, n) approximation of x^power on [lower, upper] (Remez).

    `power` is a Fraction (or anything Fraction accepts), with 0 < |power| < 1. `dps`: initial
    working precision in decimal digits, default `remez_dps(degree, upper / lower)`; on a precision
    failure (module docstring) the construction is repeated at max(1.5 dps, estimated need) digits,
    up to `max_dps`. Raises RuntimeError if no precision up to `max_dps` works. Cached on the
    normalised arguments (the returned object is immutable and shared).
    """
    alpha = _check_power(power)
    if alpha == -1:
        raise ValueError("x^-1 is exact (method 'exact'); the Remez system is degenerate for it")
    n = int(degree)
    if n < 1:
        raise ValueError(f"degree must be >= 1, got {degree}")
    lower, upper = float(lower), float(upper)
    dps = remez_dps(n, upper / lower) if dps is None else int(dps)
    return _remez_cached(alpha, lower, upper, n, dps, float(level_tol), int(max_steps), int(max_dps))


@functools.lru_cache(maxsize=None)
def _remez_cached(alpha, lower, upper, n, dps, level_tol, max_steps, max_dps):
    failures = []
    while True:
        try:
            a0, res, shifts = _remez_at(alpha, lower, upper, n, dps, level_tol, max_steps)
            return RationalApproximation(alpha, lower, upper, a0, res, shifts, "remez")
        except _PrecisionFailure as exc:
            failures.append(f"dps {dps}: {exc}")
            if dps >= max_dps:
                raise RuntimeError(f"Remez for x^{alpha} degree {n} on [{lower:.6g}, {upper:.6g}] failed at "
                                   f"every working precision up to {max_dps} digits: " + "; ".join(failures))
            dps = int(min(max_dps, max(math.ceil(1.5 * dps), exc.need or 0)))


# ---- front end -----------------------------------------------------------------------------------

def _zolotarev_rate(lower, upper):
    """Asymptotic decrease of log(error) per pole of the minimax relative-error approximation of
    x^alpha on [lower, upper]: delta_n ~ 4 exp(-2 pi^2 n / log(16 kappa)) (Zolotarev)."""
    return 2.0 * math.pi ** 2 / math.log(16.0 * upper / lower)


def _lowest_degree(build, lower, upper, tol):
    """Lowest degree n with build(n).error <= tol (module docstring, Degrees)."""
    rate = _zolotarev_rate(lower, upper)
    built = {}

    def error(n):
        if n not in built:
            built[n] = build(n)
        return built[n].error

    fail, ok = 0, None  # largest degree known to fail (0: none), smallest known to succeed
    n = max(1, min(MAX_DEGREE, math.ceil(math.log(4.0 / tol) / rate)))
    while True:
        e = error(n)
        if e <= tol:
            ok = n if ok is None else min(ok, n)
        else:
            if built[n].allowance >= 0.5 * tol:
                raise RuntimeError(f"tolerance {tol:.1e} below the float64 floor of the certified error "
                                   f"(rounding allowance {built[n].allowance:.1e} at degree {n})")
            lower_fails = [k for k in built if k < n and built[k].error > tol]
            if lower_fails and e >= built[max(lower_fails)].error:
                k = max(lower_fails)
                raise RuntimeError(f"tolerance {tol:.1e} not reachable: the certified error stalls "
                                   f"({built[k].error:.2e} at degree {k}, {e:.2e} at degree {n})")
            fail = max(fail, n)
            if n >= MAX_DEGREE:
                raise RuntimeError(f"no degree up to {MAX_DEGREE} reaches the tolerance {tol:.1e}")
        if ok is not None and ok - fail <= 1:
            return built[ok]
        # predicted degree from the error at n, then kept strictly inside the bracket
        step = (math.log(e) - math.log(tol)) / rate
        guess = n + math.ceil(step) if e > tol else n - max(1, math.floor(-step))
        hi = MAX_DEGREE if ok is None else ok - 1
        guess = min(max(guess, fail + 1), hi)
        if guess in built:
            guess = (fail + ok) // 2 if ok is not None else min(n + 1, MAX_DEGREE)
        n = guess


def rational_approximation(power, lower, upper, degree=None, tol=None, method="auto", dps=None):
    """Rational approximation of x^power on [lower, upper] in partial-fraction form.

    power: a Fraction (e.g. Fraction(-1, 2) for the action of N_f = 2 in 4D, Fraction(1, 4) for its
        heat bath); floats are converted with limit_denominator(10^6). Supported: -1 and
        0 < |power| < 1 (module docstring, Powers).
    degree: number of poles; if None, the lowest degree with certified error <= tol.
    method: "auto" ("exact" for power = -1, "zolotarev" for power = +-1/2, "remez" otherwise),
        or "zolotarev" / "remez" explicitly.
    dps: initial Remez working precision (digits), default `remez_dps(degree, upper / lower)`.
    """
    power = _check_power(power)
    lower, upper = float(lower), float(upper)
    if not (0 < lower < upper):
        raise ValueError(f"need 0 < lower < upper, got [{lower}, {upper}]")
    if method == "auto":
        if power == -1:
            method = "exact"
        elif abs(power) == Fraction(1, 2):
            method = "zolotarev"
        else:
            method = "remez"
    if method == "exact":
        if power != -1:
            raise ValueError("method 'exact' is only available for power -1")
        return RationalApproximation(power, lower, upper, 0.0, [1.0], [0.0], "exact", error=0.0)
    if power == -1:
        raise ValueError("x^-1 is exact: use method 'exact' (or 'auto')")
    if method == "zolotarev":
        if power == Fraction(-1, 2):
            build = lambda n: zolotarev_inverse_sqrt(lower, upper, n)
        elif power == Fraction(1, 2):
            build = lambda n: zolotarev_sqrt(lower, upper, n)
        else:
            raise ValueError("Zolotarev is implemented for powers -1/2 and +1/2 only")
    elif method == "remez":
        build = lambda n: remez_approximation(power, lower, upper, n, dps=dps)
    else:
        raise ValueError(f"unknown method {method!r}")
    if degree is not None:
        if int(degree) < 1:
            raise ValueError(f"degree must be >= 1, got {degree}")
        return build(int(degree))
    if tol is None:
        raise ValueError("give a degree or a tolerance")
    if not tol > 0:
        raise ValueError(f"tol must be > 0, got {tol}")
    return _lowest_degree(build, lower, upper, float(tol))
