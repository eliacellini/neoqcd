"""su(N) algebra helpers for the molecular dynamics of the HMC.

Conventions (shared with neoqcd.fermions.hmc):
- algebra elements are *hermitian traceless* N x N matrices X; the group element they generate is
  exp(i X);
- the inner product on the algebra is <X, Y> = tr(X Y) (real for hermitian X, Y);
- `algebra_basis(N)` is orthonormal for it: tr(T^a T^b) = delta_ab, a = 1 .. N^2 - 1 (the
  generalised Gell-Mann matrices divided by sqrt(2)); components are x_a = tr(X T^a);
- the momenta have density proportional to exp(-(1/2) tr P^2) per link, i.e. P = sum_a p_a T^a with
  independent p_a ~ N(0, 1), so that <tr P^2> = N^2 - 1 per link.
"""

import math

import torch

from neoqcd.fermions._gauge import SU3_unitarize, SUN_dagger, SUN_mul, SUN_trace, exp_coefficients


def _scal(x):
    return x[..., None, None]


def trace_product(X, Y):
    """Re tr(X Y) over the last two (matrix) dimensions."""
    return (X * Y.transpose(-2, -1)).sum((-2, -1)).real


def project_hermitian_traceless(M):
    """Orthogonal projection of an arbitrary complex matrix onto the hermitian traceless matrices."""
    N = M.shape[-1]
    H = 0.5 * (M + SUN_dagger(M))
    eye = torch.eye(N, dtype=M.dtype, device=M.device)
    return H - _scal(SUN_trace(H)) / N * eye


def algebra_basis(N, dtype=torch.cdouble, device=None):
    """Orthonormal basis T^a of the hermitian traceless N x N matrices, tr(T^a T^b) = delta_ab.

    Shape (N^2 - 1, N, N). For N = 2 these are the Pauli matrices / sqrt(2), for N = 3 the
    Gell-Mann matrices / sqrt(2).
    """
    gens = []
    s = 1.0 / math.sqrt(2.0)
    for j in range(N):
        for k in range(j + 1, N):
            sym = torch.zeros((N, N), dtype=dtype, device=device)
            sym[j, k] = s
            sym[k, j] = s
            asym = torch.zeros((N, N), dtype=dtype, device=device)
            asym[j, k] = -1j * s
            asym[k, j] = 1j * s
            gens += [sym, asym]
    for l in range(1, N):
        diag = torch.zeros((N, N), dtype=dtype, device=device)
        norm = 1.0 / math.sqrt(l * (l + 1))
        for j in range(l):
            diag[j, j] = norm
        diag[l, l] = -l * norm
        gens.append(diag)
    return torch.stack(gens, dim=0)


def algebra_components(X):
    """Components x_a = tr(X T^a) of hermitian traceless X, shape (..., N^2 - 1)."""
    T = algebra_basis(X.shape[-1], dtype=X.dtype, device=X.device)
    return torch.einsum("...ij,aji->...a", X, T).real


def random_algebra(shape, generator=None, dtype=torch.cdouble, device=None):
    """Gaussian hermitian traceless matrices of shape `shape` = (..., N, N), density ~ exp(-tr X^2 / 2).

    A complex Gaussian matrix Z with E|Z_ij|^2 = 1 gives H = (Z + Z^dagger) / sqrt(2) with density
    ~ exp(-tr H^2 / 2) on all hermitian matrices; dropping the trace (an orthogonal projection, since
    the identity is orthogonal to the traceless subspace) leaves the same density on su(N).
    """
    Z = torch.randn(tuple(shape), generator=generator, dtype=dtype, device=device)
    H = (Z + SUN_dagger(Z)) / math.sqrt(2.0)
    N = shape[-1]
    eye = torch.eye(N, dtype=dtype, device=device)
    return H - _scal(SUN_trace(H)) / N * eye


def expi(Q, method="auto"):
    """exp(i Q) for hermitian traceless Q (..., N, N).

    method:
    - "auto": closed form for N = 2, Cayley-Hamilton for N = 3, `torch.linalg.matrix_exp` otherwise;
    - "closed": SU(2) closed form exp(iQ) = cos(q) + i sin(q)/q Q, q^2 = tr(Q^2)/2 (N = 2 only);
    - "cayley_hamilton": exp(iQ) = f0 + f1 Q + f2 Q^2 with the coefficients of
      neoqcd.smearing_jacobian.exp_coefficients (Taylor series for c1 = tr(Q^2)/2 < 3, the
      Morningstar-Peardon closed forms above) (N = 3 only);
    - "matrix_exp": torch.linalg.matrix_exp(i Q), any N.
    """
    N = Q.shape[-1]
    if method == "auto":
        method = {2: "closed", 3: "cayley_hamilton"}.get(N, "matrix_exp")
    if method == "matrix_exp":
        return torch.linalg.matrix_exp(1j * Q)
    eye = torch.eye(N, dtype=Q.dtype, device=Q.device)
    if method == "closed":
        if N != 2:
            raise ValueError("the closed-form exponential is for SU(2) only")
        # guarded sqrt: d sqrt(q2)/d q2 is infinite at q2 = 0, which made autograd return nan at
        # Q = 0; the masked branch has zero gradient there, which is exact to first order since
        # d q2 / dQ = 0 at Q = 0 (and torch.sinc has derivative 0 at 0). Values are unchanged.
        q2 = 0.5 * trace_product(Q, Q)
        positive = q2 > 0
        q = torch.where(positive, torch.sqrt(torch.where(positive, q2, torch.ones_like(q2))),
                        torch.zeros_like(q2))
        sinc = torch.sinc(q / math.pi)  # sin(q) / q, finite at q = 0
        return _scal(torch.cos(q)) * eye + 1j * _scal(sinc) * Q
    if method == "cayley_hamilton":
        if N != 3:
            raise ValueError("the Cayley-Hamilton exponential is for SU(3) only")
        Q2 = SUN_mul(Q, Q)
        c1 = 0.5 * SUN_trace(Q2).real
        c0 = trace_product(Q, Q2) / 3.0
        f = exp_coefficients(c0, c1)[0]
        return _scal(f[..., 0]) * eye + _scal(f[..., 1]) * Q + _scal(f[..., 2]) * Q2
    raise ValueError(f"unknown exponential method {method!r}")


def unitarize(U):
    """Project near-SU(N) matrices back onto SU(N).

    N = 3: `SU3_unitarize` (Gram-Schmidt on the first two rows, third row their cross product).
    N = 2: the SU(2) part (a, b) = ((U00 + U11^*)/2, (U01 - U10^*)/2), normalised.
    Other N: polar decomposition by SVD, then the determinant phase divided out.
    """
    N = U.shape[-1]
    if N == 3:
        return SU3_unitarize(U)
    if N == 2:
        a = 0.5 * (U[..., 0, 0] + U[..., 1, 1].conj())
        b = 0.5 * (U[..., 0, 1] - U[..., 1, 0].conj())
        norm = torch.sqrt(a.abs() ** 2 + b.abs() ** 2)
        a, b = a / norm, b / norm
        first = torch.stack((a, b), dim=-1)
        second = torch.stack((-b.conj(), a.conj()), dim=-1)
        return torch.stack((first, second), dim=-2)
    W, _, Vh = torch.linalg.svd(U)
    V = W @ Vh
    phase = torch.linalg.det(V) ** (-1.0 / N)
    return V * _scal(phase)
