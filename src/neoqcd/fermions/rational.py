"""Rational approximations of x^alpha on a positive interval, in partial-fraction form (RHMC).

The RHMC (Clark and Kennedy, hep-lat/0608015) needs, on the spectral interval [lower, upper] of a
hermitian positive-definite operator M,

    r(x) = a_0 + sum_k a_k / (x + b_k)  ~  x^alpha,     b_k >= 0,

with a small *relative* error  delta = max_{x in [lower, upper]} |r(x) / x^alpha - 1|.  Then r(M) v
is a_0 v + sum_k a_k (M + b_k)^-1 v, one multishift CG (neoqcd.fermions.solver.multishift_cg).
Every approximation here is of type (n, n) (numerator and denominator of degree n = `degree`),
which is the partial-fraction form above with n poles.

Methods (`rational_approximation(power, ...)` chooses with method="auto"):

- "exact"      alpha = -1: r(x) = 1 / x, a single pole at 0 (a_0 = 0, a_1 = 1, b_1 = 0), error 0.
- "zolotarev"  alpha = -1/2 and alpha = +1/2: Zolotarev's optimal (minimax relative error)
               approximation of 1/sqrt(x) (Zolotarev 1877; in the form of Chiu, Hsieh, Huang and
               Huang, hep-lat/0206007, eqs. (8)-(10)). With y = x / lower in [1, kappa],
                   R(y) = d_0 prod_{l=1}^n (y + c_{2l}) / (y + c_{2l-1}),
                   c_l = sn^2(l K' / (2n+1); k') / (1 - sn^2(l K' / (2n+1); k')),  l = 1 .. 2n,
                   k'^2 = 1 - 1/kappa,  K' = K(k'),
               (sn the Jacobi elliptic function and K the complete elliptic integral with
               parameter m = k'^2, scipy.special.ellipj / ellipk). The scale d_0 is fixed so that
               the relative error sqrt(y) R(y) - 1 equioscillates symmetrically,
                   d_0 = 2 / (max g + min g),   delta = (max g - min g) / (max g + min g),
               g(y) = sqrt(y) prod(...), with max and min located numerically (log grid plus local
               refinement), which also certifies delta. x^{+1/2} is the reciprocal 1/R, which has
               the same relative error to first order (it is certified separately).
- "remez"      any other alpha (in particular x^{-1/4}, x^{+1/4}, x^{-gamma/n_pf}, ...): the
               rational Remez exchange algorithm for the minimax relative error of type (n, n),
               with a log-Chebyshev initial reference. At each step the nonlinear interpolation
               system (p, q polynomials in t = (2x - lower - upper)/(upper - lower), q(0) = 1)
                   p(x_i) - x_i^alpha (1 + (-1)^i E) q(x_i) = 0,   i = 0 .. 2n+1,
               is solved in multiprecision (mpmath, `dps` = 50 digits; linear fixed-point
               iterations in E, then Newton). The 2n+1 zeros of the error between the reference
               points and the extremum of |error| in each of the 2n+2 sign intervals are then
               located in float64, with the rational function in product form (roots of p and q
               from mpmath): 20001-point log grid, brentq, bounded minimisation. The extrema
               become the new reference. Stops when they are levelled to a relative spread below
               `level_tol` (1e-3; the error is certified independently anyway), or to the
               float64 resolution of the error. The result is converted to partial fractions
               (roots of q in multiprecision; the poles must come out real and <= 0) and then
               rounded to float64.

Guarantee. For every approximation the stored `error` is the maximum of |r(x) / x^alpha - 1| of
the *float64 partial-fraction* function actually used, evaluated on a log grid of 4001 points with
every local maximum in the upper half of the range refined by bounded scalar optimisation, plus an
allowance 8 (degree + 2) eps for the float64 rounding of the error evaluation itself (~3e-14 at
degree 13; without it a denser grid finds values above the refined maxima by ~1e-4 relative at
delta ~ 1e-11, which is pure rounding). The error of a minimax approximation is a smooth
equioscillating function, so the refined maxima are its supremum up to the accuracy of the
optimiser. `max_relative_error(approx, grid)` measures the error on any grid.

Degrees. `rational_approximation(power, lower, upper, tol=...)` returns the lowest degree whose
certified error is <= tol (searching upwards from the Zolotarev estimate), or the requested
`degree`. Results are cached per (power, interval, degree, method).

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
    "max_relative_error",
    "log_grid",
]


class RationalApproximation:
    """r(x) = constant + sum_k residues[k] / (x + shifts[k]) ~ x^power on [lower, upper].

    Attributes: power (Fraction), lower, upper, constant (a_0), residues (a_k), shifts (b_k >= 0)
    as float64 numpy arrays, error (certified max relative error on [lower, upper], see module
    docstring), method ("exact", "zolotarev", "remez"), degree = number of poles.
    """

    def __init__(self, power, lower, upper, constant, residues, shifts, method, error=None):
        self.power = Fraction(power)
        self.lower = float(lower)
        self.upper = float(upper)
        self.constant = float(constant)
        self.residues = np.asarray(residues, dtype=np.float64).copy()
        self.shifts = np.asarray(shifts, dtype=np.float64).copy()
        self.method = str(method)
        if self.residues.shape != self.shifts.shape or self.residues.ndim != 1:
            raise ValueError("residues and shifts must be 1D arrays of equal length")
        if (self.shifts < 0).any():
            raise ValueError(f"negative shift in the rational approximation: {self.shifts}")
        self.error = certified_relative_error(self) if error is None else float(error)

    @property
    def degree(self):
        return int(self.shifts.shape[0])

    def __call__(self, x):
        """r(x) for a float, a numpy array or a torch tensor (elementwise)."""
        out = self.constant + 0.0 * x
        for a, b in zip(self.residues, self.shifts):
            out = out + float(a) / (x + float(b))
        return out

    def target(self, x):
        """x^power (numpy/float)."""
        return np.power(x, float(self.power))

    def relative_error(self, x):
        """r(x) / x^power - 1 on a numpy array."""
        x = np.asarray(x, dtype=np.float64)
        return self(x) / self.target(x) - 1.0

    def __repr__(self):
        return (f"RationalApproximation(x^{self.power} on [{self.lower:.6g}, {self.upper:.6g}], "
                f"{self.method}, degree {self.degree}, error {self.error:.3e})")


def log_grid(lower, upper, n=4001):
    """n log-spaced points from lower to upper (both included), float64."""
    return np.exp(np.linspace(math.log(lower), math.log(upper), int(n)))


def max_relative_error(approx, grid):
    """max |r(x) / x^power - 1| over the points of `grid` (numpy array, list or torch tensor)."""
    x = np.asarray(grid.cpu().numpy() if hasattr(grid, "cpu") else grid, dtype=np.float64)
    return float(np.max(np.abs(approx.relative_error(x))))


def _refined_max(func, lower, upper, n_grid=4001, max_peaks=100):
    """max of func on [lower, upper]: log grid, then every local maximum refined by bounded search."""
    from scipy.optimize import minimize_scalar

    s = np.linspace(math.log(lower), math.log(upper), n_grid)
    v = func(np.exp(s))
    best = float(v.max())
    # local maxima of the grid values in the upper half of their range (the equioscillation peaks
    # of a minimax error all have the same height; smaller ones are roundoff noise)
    floor = best - 0.5 * (best - float(v.min()))
    peaks = [i for i in range(n_grid) if v[i] >= floor
             and (i == 0 or v[i] >= v[i - 1]) and (i == n_grid - 1 or v[i] >= v[i + 1])]
    peaks = sorted(peaks, key=lambda i: -v[i])[:max_peaks]
    for i in peaks:
        lo, hi = s[max(i - 1, 0)], s[min(i + 1, n_grid - 1)]
        res = minimize_scalar(lambda t: -float(func(np.exp(t))), bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-12})
        best = max(best, -float(res.fun))
    return best


def certified_relative_error(approx, n_grid=4001):
    """Certified max relative error of `approx` on its interval (module docstring): the refined
    maximum plus an allowance 8 (degree + 2) eps for the float64 rounding of r(x) / x^alpha - 1 itself,
    so that no float64 evaluation of the error on any grid exceeds it."""
    peak = _refined_max(lambda x: np.abs(approx.relative_error(x)), approx.lower, approx.upper, n_grid)
    return peak + 8 * (approx.degree + 2) * float(np.finfo(np.float64).eps)


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


# ---- Zolotarev -----------------------------------------------------------------------------------

def _zolotarev_coefficients(kappa, degree):
    """c_1 .. c_{2n} of the Zolotarev approximation of 1/sqrt(y) on [1, kappa] (module docstring)."""
    from scipy.special import ellipj, ellipk

    n = int(degree)
    m = 1.0 - 1.0 / float(kappa)  # parameter k'^2
    Kp = ellipk(m)
    u = np.arange(1, 2 * n + 1, dtype=np.float64) * Kp / (2 * n + 1)
    sn = ellipj(u, m)[0]
    return sn ** 2 / (1.0 - sn ** 2)


def _zolotarev_shape(lower, upper, degree, dps=50):
    """Zeros and poles (in x, multiprecision) and the scale d_0 / sqrt(lower) of the Zolotarev
    approximation of x^{-1/2} on [lower, upper]."""
    import mpmath as mp

    with mp.workdps(dps):
        lower_m = mp.mpf(lower)
        c = [mp.mpf(float(v)) for v in _zolotarev_coefficients(float(upper) / float(lower), degree)]
        zeros = [-lower_m * c[2 * l + 1] for l in range(degree)]  # x = -lower c_{2l}
        poles = [-lower_m * c[2 * l] for l in range(degree)]  # x = -lower c_{2l-1}

        def g(x):  # sqrt(x) prod (x - zeros) / (x - poles), float64 evaluation for the search
            x = np.asarray(x, dtype=np.float64)
            out = np.sqrt(x)
            for z, p in zip(zeros, poles):
                out = out * (x - float(z)) / (x - float(p))
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
        taken with the previous q), then Newton on the full nonlinear system.
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
            if max(abs(c) for c in d[:2 * n + 1]) <= small * scale and abs(d[-1]) <= small * abs(E):
                break
        return p, q, E

    def float_error(self, p, q):
        """The error p / (f q) - 1 as a vectorised float64 function of x, in product form
        C prod(x - zeros_p) / prod(x - zeros_q) x^-alpha - 1 (roots in multiprecision, possibly
        complex in early iterations), accurate to a few ulp relative to r(x)."""
        mp = self.mp
        n = self.n
        zp = [complex((r - self.c0) / self.c1) for r in mp.polyroots(list(reversed(p)), maxsteps=400,
                                                                       extraprec=mp.mp.prec)]
        zq = [complex((r - self.c0) / self.c1) for r in mp.polyroots(list(reversed(q)), maxsteps=400,
                                                                       extraprec=mp.mp.prec)]
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
                raise RuntimeError("Remez: the error does not alternate on the reference")
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


@functools.lru_cache(maxsize=None)
def remez_approximation(power, lower, upper, degree, dps=50, level_tol=1e-3, max_steps=40):
    """Minimax relative-error type-(n, n) approximation of x^power on [lower, upper] (Remez).

    `power` is a Fraction (or anything Fraction accepts). Raises RuntimeError if the exchange does
    not converge or the poles are not real and <= 0.
    """
    import mpmath as mp

    alpha = Fraction(power)
    n = int(degree)
    with mp.workdps(int(dps)):
        st = _RemezState(mp, alpha, lower, upper, n)
        L = mp.log(st.lower)
        R = mp.log(st.upper)
        ref = [mp.exp(L + (R - L) * (1 - mp.cos(mp.pi * i / (2 * n + 1))) / 2) for i in range(2 * n + 2)]
        E, q = 0, None
        for step in range(int(max_steps)):
            p, q, E = st.solve(ref, E0=E, q0=q)
            ref_new, values = st.exchange(p, q, ref)
            spread = (max(values) - min(values)) / max(values)
            ref = ref_new
            # levelled, or levelled down to the float64 resolution of the error (degrees whose
            # minimax error is near roundoff cannot be levelled further in float64)
            if spread < level_tol or max(values) - min(values) < 32 * (n + 1) * np.finfo(np.float64).eps:
                break
        else:
            raise RuntimeError(f"Remez for x^{alpha} degree {n} did not level: spread {spread}")
        p, q, E = st.solve(ref, E0=E, q0=q)
        # partial fractions in x: q(t(x)) has roots t_k -> x_k
        q_roots = mp.polyroots(list(reversed(q)), maxsteps=400, extraprec=4 * int(dps))
        poles = []
        for tr in q_roots:
            if abs(mp.im(tr)) > mp.mpf(10) ** (-int(dps) // 3) * (1 + abs(tr)):
                raise RuntimeError(f"Remez for x^{alpha} degree {n}: complex pole {tr}")
            poles.append((mp.re(tr) - st.c0) / st.c1)
        a0 = p[n] / q[n]
        residues = []
        dq = [j * q[j] for j in range(1, n + 1)]
        for xk in poles:
            tk = st.t(xk)
            residues.append(_horner(p, tk) / (st.c1 * _horner(dq, tk)))
        if any(xk > 0 for xk in poles):
            raise RuntimeError(f"Remez for x^{alpha} degree {n}: positive pole {poles}")
        order = sorted(range(n), key=lambda k: -poles[k])
        return RationalApproximation(alpha, lower, upper, float(a0), [float(residues[k]) for k in order],
                                     [-float(poles[k]) for k in order], "remez")


# ---- front end -----------------------------------------------------------------------------------

def _zolotarev_degree_estimate(lower, upper, tol):
    for n in range(1, 200):
        if zolotarev_inverse_sqrt(lower, upper, n).error <= tol:
            return n
    raise RuntimeError("no Zolotarev degree below 200 reaches the tolerance")


def rational_approximation(power, lower, upper, degree=None, tol=None, method="auto"):
    """Rational approximation of x^power on [lower, upper] in partial-fraction form.

    power: a Fraction (e.g. Fraction(-1, 2) for the action of N_f = 2 in 4D, Fraction(1, 4) for its
        heat bath); floats are converted with limit_denominator(10^6).
    degree: number of poles; if None, the lowest degree with certified error <= tol.
    method: "auto" ("exact" for power = -1, "zolotarev" for power = +-1/2, "remez" otherwise),
        or "zolotarev" / "remez" explicitly.
    """
    if isinstance(power, float):
        power = Fraction(power).limit_denominator(10 ** 6)
    power = Fraction(power)
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
    if method == "zolotarev":
        if power == Fraction(-1, 2):
            build = lambda n: zolotarev_inverse_sqrt(lower, upper, n)
        elif power == Fraction(1, 2):
            build = lambda n: zolotarev_sqrt(lower, upper, n)
        else:
            raise ValueError("Zolotarev is implemented for powers -1/2 and +1/2 only")
    elif method == "remez":
        build = lambda n: remez_approximation(power, lower, upper, n)
    else:
        raise ValueError(f"unknown method {method!r}")
    if degree is not None:
        return build(int(degree))
    if tol is None:
        raise ValueError("give a degree or a tolerance")
    n = max(1, _zolotarev_degree_estimate(lower, upper, tol) - 2) if method == "remez" else 1
    for n in range(n, 200):
        approx = build(n)
        if approx.error <= tol:
            return approx
    raise RuntimeError("no degree below 200 reaches the tolerance")
