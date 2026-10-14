import functools
import math
from fractions import Fraction
import torch
import neoqcd.sun_utils as sun

# Stout smearing of SU(3) links with its exact Jacobian. Adapted from
# `snf_code/smearing.py` in upstream merge commit a6640c9; the determinant identity is proved in
# that revision's `notes/su3_stout_jacobian.tex` (the 9-complex-coordinate determinant equals the
# 8-real-tangent Haar determinant).
# - the coefficients f_j of exp(iQ) = f0 + f1 Q + f2 Q^2 and their derivatives (exp_coefficients);
# - the single-link step U' = exp(iQ) U and log|det J| (smearing_step);
# - its explicit backward (SmearingStep).

# number of colours: only SU(3) is supported
N_COL = 3
LOGDET_VERSION = "su3_stout_full_haar_logdet_v1"

# Below c1 = Tr(Q^2)/2 = C1_SERIES the coefficients are summed from the Taylor series, truncated after
# N_SERIES terms: accurate to ~1e-15 for every coefficient up to c1 ~ 3.5, and more accurate there than
# the Morningstar-Peardon formulas, which divide by 9u^2 - w^2 ~ c1 up to three times (see
# notes/stout_smearing_backward.tex, paragraph "Small Q")
C1_SERIES = 3.
N_SERIES = 25

# ----------------------------------------------------------------------------------------------
# Cayley-Hamilton coefficients: exp(iQ) = f0 + f1 Q + f2 Q^2 for traceless hermitian 3x3 Q.
# Everything is a function of the invariants c0 = det Q = Tr(Q^3)/3 and c1 = Tr(Q^2)/2.
# The functions return a list of coefficients, each of shape (..., 3) with j = 0, 1, 2 in the last
# dimension:
#   order 1: f_j,  b1_j = df_j/dc1,  b2_j = df_j/dc0;
#   order 2: also d11_j = d^2 f_j/dc1^2,  d10_j = d^2 f_j/dc1 dc0,  d00_j = d^2 f_j/dc0^2.
# ----------------------------------------------------------------------------------------------

# Below |w| = XI_SERIES the xi functions are summed from their Taylor series: the closed forms of xi1
# and xi2 cancel, with relative rounding errors ~ eps/w^2 and ~ eps/w^4
XI_SERIES = 2.

def xi_taylor_coefficients():
    # Coefficients of xi0, xi1, xi2 in powers of w^2 (see notes/stout_smearing_backward.tex,
    # eq. (eq:xi-series)): xi0 = sum_n (-1)^n w^2n/(2n+1)!, xi1 = sum_n (-1)^(n+1) 2(n+1) w^2n/(2n+3)!,
    # xi2 = sum_n (-1)^(n+1) 4(n+1)(n+2) w^2n/(2n+5)!. Each series stops before the first term below
    # 1e-19 at |w| = XI_SERIES, where |xi_k| >= 0.04: the truncation error is below 1e-17, relative.
    terms = (lambda n: Fraction((-1)**n, math.factorial(2 * n + 1)),
             lambda n: Fraction((-1)**(n + 1) * 2 * (n + 1), math.factorial(2 * n + 3)),
             lambda n: Fraction((-1)**(n + 1) * 4 * (n + 1) * (n + 2), math.factorial(2 * n + 5)))
    w2 = Fraction(XI_SERIES)**2
    coefficients = []
    for term in terms:
        c = []
        while not c or abs(term(len(c))) * w2**len(c) >= Fraction(1, 10**19):
            c.append(float(term(len(c))))
        coefficients.append(c)
    return coefficients

XI_TAYLOR = xi_taylor_coefficients()

def xi_functions(w):
    # xi0 = sin(w)/w, xi1 = xi0'/w, xi2 = -xi1'/w: closed forms for |w| >= XI_SERIES, Taylor series
    # (xi_taylor_coefficients, by Horner's rule in w^2) below
    small = w.abs() < XI_SERIES
    ws = torch.where(small, torch.ones_like(w), w)
    w2, ws2 = w * w, ws * ws
    xi0 = torch.sin(ws) / ws
    xi1 = torch.cos(ws) / ws2 - torch.sin(ws) / (ws2 * ws)
    xi2 = (xi0 + 3. * xi1) / ws2
    series = []
    for c in XI_TAYLOR:
        s = torch.full_like(w2, c[-1])
        for cn in c[-2::-1]:
            s = s * w2 + cn
        series.append(s)
    return tuple(torch.where(small, s, x) for s, x in zip(series, (xi0, xi1, xi2)))

def mp_coefficients(c0, c1, order):
    # Coefficients (see the section banner) from the closed forms of Morningstar and Peardon,
    # hep-lat/0311018, accurate for c1 >= C1_SERIES: notes/su3_stout_jacobian.tex, eqs. (eq:fj)-(eq:rj),
    # and for the second derivatives notes/stout_smearing_backward.tex, eqs. (eq:r1u)-(eq:r2w).
    # The formulas are evaluated at |c0|; for c0 < 0 one uses f_j(-c0) = (-1)^j f_j(c0)^*, and each
    # derivative with respect to c0 adds one more sign.
    neg = c0 < 0
    c0max = 2. * (c1 / 3.)**1.5
    # At the degenerate spectrum |c0| = c0max, acos has an infinite derivative even though
    # the combined coefficient limits are smooth. Keep the argument one representable step
    # inside its domain; the analytic coefficient derivatives below retain the correct limit.
    ratio = (c0.abs() / c0max).clamp(max=1. - torch.finfo(c1.dtype).eps)
    theta = torch.arccos(ratio)
    u = torch.sqrt(c1 / 3.) * torch.cos(theta / 3.)
    w = torch.sqrt(c1) * torch.sin(theta / 3.)
    u2, w2 = u * u, w * w
    xi0, xi1, xi2 = xi_functions(w)
    cw = torch.cos(w)
    eu = torch.exp(1.j * u)
    e2u, emu = eu * eu, eu.conj()
    stack = lambda *x: torch.stack(x, dim=-1)

    # h_j, r1_j = dh_j/du, r2_j = (1/w) dh_j/dw
    h = stack((u2 - w2) * e2u + emu * (8. * u2 * cw + 2.j * u * (3. * u2 + w2) * xi0),
              2. * u * e2u - emu * (2. * u * cw - 1.j * (3. * u2 - w2) * xi0),
              e2u - emu * (cw + 3.j * u * xi0))
    r1 = stack(2. * (u + 1.j * (u2 - w2)) * e2u
               + 2. * emu * (4. * u * (2. - 1.j * u) * cw + 1.j * (9. * u2 + w2 - 1.j * u * (3. * u2 + w2)) * xi0),
               2. * (1. + 2.j * u) * e2u + emu * (-2. * (1. - 1.j * u) * cw + 1.j * (6. * u + 1.j * (w2 - 3. * u2)) * xi0),
               2.j * e2u + 1.j * emu * (cw - 3. * (1. - 1.j * u) * xi0))
    r2 = stack(-2. * e2u + 2.j * u * emu * (cw + (1. + 4.j * u) * xi0 + 3. * u2 * xi1),
               -1.j * emu * (cw + (1. + 2.j * u) * xi0 - 3. * u2 * xi1),
               emu * (xi0 - 3.j * u * xi1))
    if order == 2:
        # second derivatives of h; (1/w) dr1/dw coincides with dr2/du
        r1_u = stack(2. * e2u * (1. + 4.j * u - 2. * u2 + 2. * w2)
                     - 2. * emu * (4. * (-2. + 4.j * u + u2) * cw
                                   + 1.j * (18.j * u2 + 3. * u2 * u + 2.j * w2 + u * (-18. + w2)) * xi0),
                     8. * e2u * (1.j - u) + emu * (2. * (2.j + u) * cw + (12. * u - 3.j * u2 + 1.j * (6. + w2)) * xi0),
                     -4. * e2u + emu * (cw + 3.j * (2.j + u) * xi0))
        r1_w = stack(-4.j * e2u + 2. * emu * ((1.j + u) * cw + (1.j - 7. * u + 4.j * u2) * xi0 + 3. * u2 * (3.j + u) * xi1),
                     emu * (-cw + (1. - 2.j * u) * xi0 + 3. * u * (2.j + u) * xi1),
                     -emu * (1.j * xi0 + 3. * (1.j + u) * xi1))
        r2_w = stack(2.j * emu * u * ((1. + 4.j * u) * xi1 - xi0 - 3. * u2 * xi2),
                     1.j * emu * (-(1. + 2.j * u) * xi1 + xi0 - 3. * u2 * xi2),
                     emu * (xi1 + 3.j * u * xi2))

    u, u2, w2 = u.unsqueeze(-1), u2.unsqueeze(-1), w2.unsqueeze(-1)
    den = 9. * u2 - w2
    a = .5 * (3. * u2 - w2)
    # derivatives with respect to (c1, c0) from those with respect to (u, w): X_u = dX/du, X_w = (1/w) dX/dw
    d_dc1 = lambda X_u, X_w: (u * X_u + a * X_w) / den
    d_dc0 = lambda X_u, X_w: (.5 * X_u - 1.5 * u * X_w) / den

    f = h / den
    f_u = (r1 - 18. * u * f) / den
    f_w = (r2 + 2. * f) / den
    b1, b2 = d_dc1(f_u, f_w), d_dc0(f_u, f_w)
    coeffs = [f, b1, b2]
    if order == 2:
        f_uu = (r1_u - 18. * f - 36. * u * f_u) / den
        f_uw = (r1_w - 18. * u * f_w + 2. * f_u) / den
        f_ww = (r2_w + 4. * f_w) / den
        b1_u = (f_u + u * f_uu + 3. * u * f_w + a * f_uw - 18. * u * b1) / den
        b1_w = (u * f_uw - f_w + a * f_ww + 2. * b1) / den
        b2_u = (.5 * f_uu - 1.5 * f_w - 1.5 * u * f_uw - 18. * u * b2) / den
        b2_w = (.5 * f_uw - 1.5 * u * f_ww + 2. * b2) / den
        coeffs += [d_dc1(b1_u, b1_w), d_dc0(b1_u, b1_w), d_dc0(b2_u, b2_w)]

    # number of c0-derivatives of each returned coefficient
    n_c0 = (0, 0, 1, 0, 1, 2)
    j_sign = torch.tensor([1., -1., 1.], dtype=torch.double, device=c0.device)
    neg = neg.unsqueeze(-1)
    return [torch.where(neg, (-1)**n * j_sign * x.conj(), x) for n, x in zip(n_c0, coeffs)]

def series_polynomials():
    # The truncated Taylor series of f_j and of its derivatives as polynomials in (c0, c1), with exact
    # rational coefficients (see notes/stout_smearing_backward.tex, paragraph "Small Q"):
    #   exp(iQ) = sum_{n < N_SERIES} (iQ)^n/n!, with Q^n = a_n + b_n Q + c_n Q^2 reduced by Q^3 = c1 Q + c0,
    #   (a, b, c)_{n+1} = (c0 c_n, a_n + c1 c_n, b_n),  (a, b, c)_0 = (1, 0, 0),
    # so a_n, b_n, c_n are polynomials in (c0, c1) with integer coefficients, and
    # f_j = sum_n i^n/n! (a, b, c)_j. A polynomial is a dict {(m, k): coefficient of c0^m c1^k}.
    # Returns, for each derivative d = value, d/dc1, d/dc0, d2/dc1^2, d2/dc1dc0, d2/dc0^2 (the order of
    # the section banner) and each j, the pair (real part, imaginary part).
    def add(p, q, scale=1):
        for mk, v in q.items():
            p[mk] = p.get(mk, 0) + scale * v
        return p
    def times(p, dm, dk):
        return {(m + dm, k + dk): v for (m, k), v in p.items()}
    def d_c1(p):
        return {(m, k - 1): k * v for (m, k), v in p.items() if k > 0}
    def d_c0(p):
        return {(m - 1, k): m * v for (m, k), v in p.items() if m > 0}

    f = [({}, {}) for j in range(3)]
    abc = ({(0, 0): 1}, {}, {})
    for n in range(N_SERIES):
        # i^n/n!: real for even n, imaginary for odd n, sign + for n % 4 < 2
        part, sign = n % 2, (1 if n % 4 < 2 else -1)
        for j in range(3):
            add(f[j][part], abc[j], Fraction(sign, math.factorial(n)))
        a, b, c = abc
        abc = (times(c, 1, 0), add(dict(a), times(c, 0, 1)), b)

    derivs = [lambda p: p, d_c1, d_c0, lambda p: d_c1(d_c1(p)), lambda p: d_c0(d_c1(p)), lambda p: d_c0(d_c0(p))]
    return [[(d(re), d(im)) for re, im in f] for d in derivs]

@functools.lru_cache(maxsize=None)
def series_table(device):
    # series_polynomials as a table, built once per device in double precision. Returns the exponents
    # m, k of the monomials c0^m c1^k (one per row), the table of shape (2, monomials, 18) with the real
    # (0) and imaginary (1) parts of every coefficient in column 3 d + j, and the highest powers of c0
    # and c1 as Python ints, so that the evaluation needs no host-device sync.
    polys = series_polynomials()
    monomials = sorted({mk for per_d in polys for re_im in per_d for p in re_im for mk in p})
    row = {mk: i for i, mk in enumerate(monomials)}
    table = torch.zeros(2, len(monomials), 3 * len(polys), dtype=torch.double)
    for d, per_d in enumerate(polys):
        for j, re_im in enumerate(per_d):
            for part, p in enumerate(re_im):
                for mk, v in p.items():
                    table[part, row[mk], 3 * d + j] = float(v)
    m, k = torch.tensor(monomials).T
    return m.to(device), k.to(device), table.to(device), int(m.max()), int(k.max())

def series_coefficients(c0, c1, order):
    # Coefficients (see the section banner) from the truncated Taylor series, accurate for
    # c1 < C1_SERIES: the monomials c0^m c1^k times the fixed table of series_table, i.e. a few powers
    # and two matrix products.
    K = (1, 3, 6)[order]
    m, k, table, m_max, k_max = series_table(c0.device)
    # powers by repeated products, not torch.pow, whose gradient at c = 0 would be 0 * inf for c^0
    pow0, pow1 = [torch.ones_like(c0)], [torch.ones_like(c1)]
    for _ in range(m_max):
        pow0.append(pow0[-1] * c0)
    for _ in range(k_max):
        pow1.append(pow1[-1] * c1)
    monomials = torch.stack(pow0, dim=-1)[..., m] * torch.stack(pow1, dim=-1)[..., k]
    re = monomials @ table[0, :, :3 * K]
    im = monomials @ table[1, :, :3 * K]
    return list(torch.complex(re, im).unflatten(-1, (K, 3)).unbind(-2))

def exp_coefficients(c0, c1, order=1):
    # Coefficients (see the section banner) for any c1: the Taylor series below C1_SERIES, the closed
    # forms of Morningstar and Peardon above, evaluated only on the links that need them. Finding them
    # costs one host-device sync. Usually no link needs them: c1 <= 9/8 (sum_p |rho_p|)^2 (see
    # notes/stout_smearing_backward.tex, paragraph "Small Q"), below C1_SERIES for small rho.
    shape = c1.shape
    c0, c1 = c0.reshape(-1), c1.reshape(-1)
    small = c1 < C1_SERIES
    if bool(small.all()):
        return [x.reshape(shape + (3,)) for x in series_coefficients(c0, c1, order)]
    # the series gets harmless arguments on the other links, which then take the closed forms
    large = (~small).nonzero(as_tuple=True)
    se = series_coefficients(torch.where(small, c0, 0.), torch.where(small, c1, 0.), order)
    mp = mp_coefficients(c0[large], c1[large], order)
    return [s.index_put(large, m).reshape(shape + (3,)) for s, m in zip(se, mp)]


# ----------------------------------------------------------------------------------------------
# Single-link stout step U' = exp(iQ) U and its Jacobian, for a batch of links U (..., 3, 3) with
# their weighted staple sums C, Om = C U^dagger and Q = i/2 (Om^dagger - Om - Tr(...)/N):
# notes/su3_stout_jacobian.tex, eqs. (eq:omega)-(eq:update)
# ----------------------------------------------------------------------------------------------

# Rank-four tensors T_ijkl represent the linear map Z -> sum_jk T_ijkl Z_jk,
# see notes/su3_stout_jacobian.tex, Section "Tensor conventions" (sec:tensor)
def otimes(A, B):
    # the map Z -> A Z B
    return A[..., :, :, None, None] * B[..., None, None, :, :]

def oplus(A, B):
    # the map Z -> Tr(A Z) B
    return A.mT[..., None, :, :, None] * B[..., :, None, None, :]

def jacobian_matrix(J):
    # rank-four tensor -> 9x9 matrix J_(il),(jk) = dU'_il/dU_jk
    return torch.einsum('...ijkl->...iljk', J).reshape(J.shape[:-4] + (N_COL**2, N_COL**2))

def matrix_to_tensor(M):
    # 9x9 matrix -> rank-four tensor, the inverse of jacobian_matrix
    return torch.einsum('...iljk->...ijkl', M.reshape(M.shape[:-2] + (N_COL,) * 4))

def scal(x):
    # a batch of scalars (...,) broadcast against a batch of matrices (..., 3, 3)
    return x[..., None, None]

def inner(A, B):
    # complex Frobenius product Tr(A^dagger B)
    return (A.conj() * B).sum((-2, -1))

class StepQuantities():
    # All the intermediates of U' = exp(iQ) U and of its Jacobian, for a batch of links U with staple
    # sums C (smearing_step, SmearingStep). The constructor computes U' = X U, X = exp(iQ), and the
    # coefficients f_j, b1_j, b2_j (with order=2 also the second derivatives, for the backward);
    # jacobian() adds the products the Jacobian needs and returns it.
    def __init__(self, U, C, order=1):
        self.U, self.C = U, C
        self.Ud, self.Cd = sun.SUN_dagger(U), sun.SUN_dagger(C)
        # one 3x3 identity, broadcast against the links: no lattice-sized tensor (built on the host
        # and copied to the device) for a constant
        self.I = torch.eye(N_COL, dtype=U.dtype, device=U.device)
        self.Om = sun.SUN_mul(C, self.Ud)
        Z = sun.SUN_dagger(self.Om) - self.Om
        self.Q = .5j * (Z - scal(sun.SUN_trace(Z)) / N_COL * self.I)
        self.Q2 = sun.SUN_mul(self.Q, self.Q)
        self.c1 = .5 * sun.SUN_trace(self.Q2).real
        self.c0 = (self.Q * self.Q2.mT).sum((-2, -1)).real / 3.
        self.trQ2 = 2. * self.c1
        coeffs = exp_coefficients(self.c0, self.c1, order)
        self.f, self.b1, self.b2 = coeffs[:3]
        if order == 2:
            self.d11, self.d10, self.d00 = coeffs[3:]
        f0, f1, f2 = self.f.unbind(-1)
        self.X = scal(f0) * self.I + scal(f1) * self.Q + scal(f2) * self.Q2
        self.Up = sun.SUN_mul(self.X, U)

    def jacobian(self):
        # The Jacobian dU'/dU as a rank-four tensor (..., 3, 3, 3, 3):
        # J = e^{iQ} x 1 + sum_j A_j (+) Q^j U + i/2 f1 (...) + i/2 f2 (...),
        # see notes/su3_stout_jacobian.tex, eq. (eq:J-final)
        t = .5j
        self.QU = sun.SUN_mul(self.Q, self.U)
        self.Q2U = sun.SUN_mul(self.Q, self.QU)
        self.CdU = sun.SUN_mul(self.Cd, self.U)
        self.CdQU = sun.SUN_mul(self.Cd, self.QU)
        self.QOm = sun.SUN_mul(self.Q, self.Om)
        self.QQOm = sun.SUN_mul(self.Q, self.QOm)
        self.UdQU = sun.SUN_mul(self.Ud, self.QU)
        # M_1, M_2: eq. (eq:M); Theta: eq. (eq:Theta)
        self.M1 = sun.SUN_mul(self.Cd, self.Q) + sun.SUN_mul(self.Ud, self.QOm)
        self.M2 = sun.SUN_mul(self.Cd, self.Q2) + sun.SUN_mul(self.Ud, self.QQOm)
        self.Th = self.Cd + sun.SUN_mul(self.Ud, self.Om)
        f0, f1, f2 = self.f.unbind(-1)
        b2 = self.b2.unbind(-1)
        # A_j = i/2 (b1_j M1 + b2_j M2 - kappa_j Th / N): eq. (eq:A)
        self.kappa = (self.trQ2 * b2[0] + f1, self.trQ2 * b2[1] + 2. * f2, self.trQ2 * b2[2])
        self.A = [t * (scal(self.b1[..., j]) * self.M1 + scal(b2[j]) * self.M2
                       - scal(self.kappa[j]) / N_COL * self.Th) for j in range(3)]
        J = (otimes(self.X, self.I) + oplus(self.A[0], self.U) + oplus(self.A[1], self.QU)
             + oplus(self.A[2], self.Q2U))
        J = J + scal(scal(t * f1)) * (otimes(self.I, self.CdU) + otimes(self.Om, self.I))
        J = J + scal(scal(t * f2)) * (otimes(self.Q, self.CdU) + otimes(self.I, self.CdQU)
                                      + otimes(self.QOm, self.I) + otimes(self.Om, self.UdQU))
        return J

def smearing_step(U, C):
    # Stout step of a batch of links U (..., 3, 3) with staple sums C: returns U' = exp(iQ) U and
    # log|det dU'/dU| (...,). Differentiable by autograd; SmearingStep gives the same outputs with an
    # explicit backward.
    stepq = StepQuantities(U, C)
    logdet = torch.linalg.slogdet(jacobian_matrix(stepq.jacobian())).logabsdet
    return stepq.Up, logdet

# ----------------------------------------------------------------------------------------------
# Explicit backward, see notes/stout_smearing_backward.tex. Gradients follow the PyTorch convention
# for real losses L of complex Z: G_Z = dL/dRe Z + i dL/dIm Z, i.e. dL = Re Tr(G_Z^dagger dZ).
# Rules used throughout (eqs. (eq:rule-prod)-(eq:rule-dag)):
#   Y = A B        ->  G_A += G_Y B^dagger,  G_B += A^dagger G_Y
#   Y = s T        ->  G_s += Tr(T^dagger G_Y) (complex),  G_T += s^* G_Y
#   Y = A^dagger   ->  G_A += G_Y^dagger
# ----------------------------------------------------------------------------------------------

# Gradients of the factors of the rank-four tensors otimes and oplus, given the gradient G of the
# tensor (eq. (eq:contract))
def contract_left(G, B):
    # gradient of A in A x B: sum_kl G_ijkl B_kl^*
    return (G * B.conj()[..., None, None, :, :]).sum((-2, -1))

def contract_right(G, A):
    # gradient of B in A x B: sum_ij G_ijkl A_ij^*
    return (G * A.conj()[..., :, :, None, None]).sum((-4, -3))

def contract_oplus_left(G, B):
    # gradient of A in A (+) B: (A)_kj <- sum_il G_ijkl B_il^*
    return (G * B.conj()[..., :, None, None, :]).sum((-4, -1)).mT

def contract_oplus_right(G, A):
    # gradient of B in A (+) B: sum_jk G_ijkl A_kj^*
    return (G * A.conj().mT[..., None, :, :, None]).sum((-3, -2))

class SmearingStep(torch.autograd.Function):
    # smearing_step(U, C) with an explicit backward: the same outputs (U', log|det J|), and gradients
    # of U and C written out by hand instead of autograd through the Jacobian. Only U and C are
    # stored; everything else is recomputed in backward (StepQuantities with order=2).
    @staticmethod
    def forward(U, C):
        return smearing_step(U, C)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, G_Up, g_logdet):
        # (G_U, G_C) from the gradients of the outputs, G_Up of U' and g_logdet of log|det J|
        U, C = ctx.saved_tensors
        stepq = StepQuantities(U, C, order=2)
        J = stepq.jacobian()
        mul, dag = sun.SUN_mul, sun.SUN_dagger
        t, tc = .5j, -.5j
        f1, f2 = stepq.f[..., 1], stepq.f[..., 2]

        # log|det J|: seed G_J = g J^{-dagger}, written back as a rank-four tensor
        G = matrix_to_tensor(scal(g_logdet) * torch.linalg.inv(jacobian_matrix(J)).mH)
        L_I = torch.diagonal(G, dim1=-2, dim2=-1).sum(-1)   # sum_k G_ijkk
        R_I = torch.diagonal(G, dim1=-4, dim2=-3).sum(-1)   # sum_i G_iikl

        # U' = X U, and the term X x 1 of J
        G_X = mul(G_Up, stepq.Ud) + L_I
        G_U = mul(dag(stepq.X), G_Up)

        # A_0 (+) U + A_1 (+) Q U + A_2 (+) Q^2 U
        G_A = [contract_oplus_left(G, P) for P in (stepq.U, stepq.QU, stepq.Q2U)]
        G_U = G_U + contract_oplus_right(G, stepq.A[0])
        G_QU = contract_oplus_right(G, stepq.A[1])
        G_Q2U = contract_oplus_right(G, stepq.A[2])

        # i/2 f1 (1 x C^dag U + Om x 1)
        G_f1 = tc * (inner(stepq.CdU, R_I) + inner(stepq.Om, L_I))
        G_CdU = scal((t * f1).conj()) * R_I
        G_Om = scal((t * f1).conj()) * L_I

        # i/2 f2 (Q x C^dag U + 1 x C^dag Q U + Q Om x 1 + Om x U^dag Q U)
        L_CdU, R_Q = contract_left(G, stepq.CdU), contract_right(G, stepq.Q)
        L_UdQU, R_Om = contract_left(G, stepq.UdQU), contract_right(G, stepq.Om)
        G_f2 = tc * (inner(stepq.Q, L_CdU) + inner(stepq.CdQU, R_I) + inner(stepq.QOm, L_I)
                     + inner(stepq.Om, L_UdQU))
        tf2c = scal((t * f2).conj())
        G_Q = tf2c * L_CdU
        G_CdU = G_CdU + tf2c * R_Q
        G_CdQU = tf2c * R_I
        G_QOm = tf2c * L_I
        G_Om = G_Om + tf2c * L_UdQU
        G_UdQU = tf2c * R_Om

        # A_j = i/2 (b1_j M1 + b2_j M2 - kappa_j Th / N)
        G_M1 = sum(scal((t * stepq.b1[..., j]).conj()) * G_A[j] for j in range(3))
        G_M2 = sum(scal((t * stepq.b2[..., j]).conj()) * G_A[j] for j in range(3))
        G_Th = sum(scal((-t * stepq.kappa[j] / N_COL).conj()) * G_A[j] for j in range(3))
        G_b1 = torch.stack([tc * inner(stepq.M1, G_A[j]) for j in range(3)], dim=-1)
        G_b2 = torch.stack([tc * inner(stepq.M2, G_A[j]) for j in range(3)], dim=-1)
        # kappa = (trQ2 b2_0 + f1, trQ2 b2_1 + 2 f2, trQ2 b2_2), with trQ2 = 2 c1 real
        G_kappa = torch.stack([-tc / N_COL * inner(stepq.Th, G_A[j]) for j in range(3)], dim=-1)
        G_trQ2 = (G_kappa * stepq.b2.conj()).real.sum(-1)
        G_b2 = G_b2 + G_kappa * stepq.trQ2.unsqueeze(-1)
        G_f = torch.stack((torch.zeros_like(G_f1), G_f1 + G_kappa[..., 0], G_f2 + 2. * G_kappa[..., 1]), dim=-1)

        # M1 = C^dag Q + U^dag Q Om,  M2 = C^dag Q^2 + U^dag Q Q Om,  Th = C^dag + U^dag Om
        G_Cd = mul(G_M1, stepq.Q) + mul(G_M2, stepq.Q2) + G_Th
        G_Q = G_Q + mul(C, G_M1)
        G_Q2 = mul(C, G_M2)
        G_Ud = mul(G_M1, dag(stepq.QOm)) + mul(G_M2, dag(stepq.QQOm)) + mul(G_Th, dag(stepq.Om))
        G_QOm = G_QOm + mul(U, G_M1)
        G_QQOm = mul(U, G_M2)
        G_Om = G_Om + mul(U, G_Th)

        # intermediate products
        G_Q = G_Q + mul(G_QQOm, dag(stepq.QOm))
        G_QOm = G_QOm + mul(stepq.Q, G_QQOm)
        G_Ud = G_Ud + mul(G_UdQU, dag(stepq.QU))
        G_QU = G_QU + mul(U, G_UdQU) + mul(C, G_CdQU) + mul(stepq.Q, G_Q2U)
        G_Cd = G_Cd + mul(G_CdQU, dag(stepq.QU)) + mul(G_CdU, stepq.Ud)
        G_Q = G_Q + mul(G_Q2U, dag(stepq.QU)) + mul(G_QOm, dag(stepq.Om)) + mul(G_QU, stepq.Ud)
        G_Om = G_Om + mul(stepq.Q, G_QOm)
        G_U = G_U + mul(C, G_CdU) + mul(stepq.Q, G_QU)

        # X = f0 + f1 Q + f2 Q^2
        G_f = G_f + torch.stack((sun.SUN_trace(G_X), inner(stepq.Q, G_X), inner(stepq.Q2, G_X)), dim=-1)
        G_Q = G_Q + scal(f1.conj()) * G_X
        G_Q2 = G_Q2 + scal(f2.conj()) * G_X

        # Q^2 = Q Q, then the coefficients: for traceless hermitian dQ, dc1 = Tr(Q dQ) and dc0 = Tr(Q^2 dQ)
        G_Q = G_Q + mul(G_Q2, stepq.Q) + mul(stepq.Q, G_Q2)
        g1 = (G_f * stepq.b1.conj() + G_b1 * stepq.d11.conj()
              + G_b2 * stepq.d10.conj()).real.sum(-1) + 2. * G_trQ2
        g0 = (G_f * stepq.b2.conj() + G_b1 * stepq.d10.conj() + G_b2 * stepq.d00.conj()).real.sum(-1)
        G_Q = G_Q + scal(g1) * stepq.Q + scal(g0) * stepq.Q2

        # Q = i/2 (Z - Tr(Z)/N), Z = Om^dag - Om, Om = C U^dag
        G_Z = tc * (G_Q - scal(sun.SUN_trace(G_Q)) / N_COL * stepq.I)
        G_Om = G_Om + dag(G_Z) - G_Z
        G_C = mul(G_Om, U) + dag(G_Cd)
        G_Ud = G_Ud + mul(stepq.Cd, G_Om)
        G_U = G_U + dag(G_Ud)
        return G_U, G_C
