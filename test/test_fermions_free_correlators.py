"""Free-field (U = 1) Wilson tests: pion correlator from the solver against the finite-lattice
momentum sum, pole masses, and log det D against the analytic eigenvalues.

Conventions are those of neoqcd.fermions.dirac: D = 1 - kappa H, kappa = 1 / (2 m0 + 2 D),
plane waves psi(n) = exp(-i p.n) u with D psi = exp(-i p.n) D(p) u, antiperiodic time
p_0 = (2 n_0 + 1) pi / T, periodic space p_k = 2 pi n_k / L.

Propagator from a point source at the origin. Since (1/V) sum_p exp(-i p.n) = delta_{n,0} for
the antiperiodic momentum set as well (the half-integer shift only gives exp(-i pi t / T) at
t = 0),
    S(n; 0) = (1/V) sum_p exp(-i p.n) D(p)^-1  (x 1_colour)
            = L^-(D-1) sum_{p_vec} exp(-i p_vec.x) f(t, p_vec),
    f(t, p_vec) = (1/T) sum_{p_0} exp(-i p_0 t) D(p_0, p_vec)^-1.
Pseudoscalar correlator, with gamma_5 hermiticity S(0; x) = gamma_5 S(x; 0)^dagger gamma_5:
    C(t) = sum_x tr[gamma_5 S(x; 0) gamma_5 S(0; x)] = sum_x tr[S(x; 0) S(x; 0)^dagger]
         = N L^-(D-1) sum_{p_vec} tr[f(t, p_vec) f(t, p_vec)^dagger]
(the spatial sum sets the two spatial momenta equal). This is evaluated exactly with S x S
matrix inverses and compared with the solver result.

Masses. At p_vec = 0 and r = 1 the time hops are the projectors (1 -+ gamma_0), so the
zero-momentum quark propagator is a single exponential exp(-E t) with E = log(1 + m0) for
1 <= t <= T - 1 (antiperiodic images included): its cosh effective mass equals E exactly. The
pion is two free quarks; its lowest state at zero total momentum is both quarks at rest, mass
2 log(1 + m0), approached from above by the cosh effective mass of C(t) as t -> T/2 (states
with p_vec != 0 contaminate at short t).
"""

import math
import unittest

import torch

from neoqcd.fermions._gauge import SUN_identity
from neoqcd.fermions.dirac import WilsonDirac, dense_matrix, free_momenta, free_wilson_eigenvalues, free_wilson_matrix
from neoqcd.fermions.lattice import Geometry
from neoqcd.fermions.solver import solve

C = torch.complex128


def unit_links(geometry, batch=1):
    return SUN_identity(geometry.link_shape(batch)[:-1], dtype=C)


def kappa_of(m0, D):
    return 1.0 / (2.0 * m0 + 2.0 * D)


def point_source_propagator(geometry, kappa, tol=1e-13):
    """S(n; 0) from the even-odd solver, shape lattice + (S, N) + (S, N) (sink, source)."""
    S, N = geometry.spin, geometry.N
    b = torch.zeros((S * N,) + geometry.lattice + (S, N), dtype=C)
    for s in range(S):
        for c in range(N):
            b[(s * N + c,) + (0,) * geometry.D + (s, c)] = 1.0
    D = WilsonDirac(unit_links(geometry), kappa, geometry, time_sign=-1.0)
    res = solve(D.even_odd(), b, method="cgne", tol=tol, max_iter=5000)
    x = res.x.reshape((S, N) + geometry.lattice + (S, N))
    perm = tuple(range(2, 2 + geometry.D)) + (2 + geometry.D, 3 + geometry.D, 0, 1)
    return x.permute(perm), res


def analytic_f(geometry, kappa):
    """f(t, p_vec) = (1/T) sum_{p_0} exp(-i p_0 t) D(p)^-1, shape (T, P, S, S), and p_vec (P, D-1)."""
    T, L, Dim = geometry.T, geometry.L, geometry.D
    p0 = (2 * torch.arange(T, dtype=torch.float64) + 1) * math.pi / T
    pvec = torch.stack([g.reshape(-1) for g in torch.meshgrid(
        *[2 * math.pi * torch.arange(L, dtype=torch.float64) / L] * (Dim - 1), indexing="ij")], -1)
    P = pvec.shape[0]
    p = torch.cat((p0[:, None, None].expand(T, P, 1), pvec[None].expand(T, P, Dim - 1)), dim=-1)
    Dinv = torch.linalg.inv(free_wilson_matrix(geometry, kappa, p))
    t = torch.arange(T, dtype=torch.float64)
    phase = torch.exp(-1j * t[:, None] * p0[None, :]).to(C) / T
    return torch.einsum("tq,qpab->tpab", phase, Dinv), pvec


def analytic_propagator(geometry, kappa):
    """S(n; 0) spin matrix (colour identity omitted), shape lattice + (S, S)."""
    f, pvec = analytic_f(geometry, kappa)
    L, Dim = geometry.L, geometry.D
    x = torch.stack([g.reshape(-1) for g in torch.meshgrid(
        *[torch.arange(L, dtype=torch.float64)] * (Dim - 1), indexing="ij")], -1)
    phase = torch.exp(-1j * (x.to(C) @ pvec.to(C).T)) / L ** (Dim - 1)  # (X, P)
    S = torch.einsum("xp,tpab->txab", phase, f)
    return S.reshape(geometry.lattice + S.shape[-2:])


def analytic_pion(geometry, kappa):
    f, _ = analytic_f(geometry, kappa)
    return geometry.N / geometry.L ** (geometry.D - 1) * f.abs().pow(2).sum(dim=(1, 2, 3))


def cosh_mass(c, t, T):
    """m solving c(t) / c(t+1) = cosh(m (t - T/2)) / cosh(m (t + 1 - T/2)), t + 1 <= T/2."""
    r = float(c[t] / c[t + 1])

    def g(m):
        return math.cosh(m * (t - T / 2)) / math.cosh(m * (t + 1 - T / 2)) - r

    lo, hi = 1e-10, 30.0
    if g(lo) * g(hi) > 0:
        raise ValueError("no cosh effective mass in the bracket")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(lo) * g(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# (geometry, m0, tolerance on |m_eff(T/2 - 1) / (2 log(1 + m0)) - 1|); the tolerance is the
# measured excited-state contamination of the exact correlator (5.5% and 0.6%) with margin.
CASES = (
    (Geometry(4, 8, 4, 3), 0.5, 0.07),
    (Geometry(2, 16, 8, 3), 0.5, 0.01),
)


class FreePionCorrelatorTest(unittest.TestCase):
    def test_propagator_and_pion_correlator(self):
        for geo, m0, mass_tol in CASES:
            with self.subTest(geometry=geo):
                kappa = kappa_of(m0, geo.D)
                prop, res = point_source_propagator(geo, kappa)
                self.assertLess(res.residual.max().item(), 1e-12)
                # full propagator against the momentum sum, colour diagonal
                ref = analytic_propagator(geo, kappa)
                ref_full = torch.einsum("...ab,cd->...acbd", ref, torch.eye(geo.N, dtype=C))
                self.assertLess((prop - ref_full).abs().max().item(), 1e-10 * ref.abs().max().item())
                # pseudoscalar correlator sum_x tr[S S^dagger] (gamma_5 hermiticity form)
                c_num = prop.abs().pow(2).reshape(geo.T, -1).sum(-1)
                c_an = analytic_pion(geo, kappa)
                self.assertLess(((c_num - c_an).abs() / c_an).max().item(), 1e-8)
                # time-reflection symmetry C(t) = C(T - t)
                refl = torch.cat((c_an[:1], c_an[1:].flip(0)))
                self.assertLess(((c_an - refl).abs() / c_an).max().item(), 1e-12)
                # effective mass: decreasing towards 2 log(1 + m0) from above
                T = geo.T
                m_pi = 2 * math.log(1 + m0)
                meff = [cosh_mass(c_num, t, T) for t in range(1, T // 2)]
                self.assertTrue(all(a > b for a, b in zip(meff, meff[1:])), meff)
                self.assertGreater(meff[-1], m_pi)
                self.assertLess(meff[-1] / m_pi - 1, mass_tol, (meff, m_pi))

    def test_zero_momentum_quark_pole_mass(self):
        # sum_x S(t, x; 0) is a single exponential in t: cosh mass log(1 + m0) for 1 <= t < T/2
        for geo, m0, _ in CASES:
            with self.subTest(geometry=geo):
                kappa = kappa_of(m0, geo.D)
                prop, _ = point_source_propagator(geo, kappa)
                G = prop.sum(dim=tuple(range(1, geo.D)))  # (T, S, N, S, N)
                g = G.abs().pow(2).reshape(geo.T, -1).sum(-1)
                E = math.log(1 + m0)
                for t in range(1, geo.T // 2):
                    self.assertLess(abs(cosh_mass(g, t, geo.T) / 2 - E), 1e-9)


class FreeLogDetTest(unittest.TestCase):
    def test_log_det_against_analytic_eigenvalues(self):
        # eigenvalues of D(p) = a + i b.gamma: a +- i |b| with a = 1 - 2 kappa sum cos p,
        # |b| = 2 kappa (sum sin^2 p)^(1/2), each with multiplicity S/2, times N for colour
        for geo, kappa in ((Geometry(4, 4, 2, 3), 0.11), (Geometry(2, 4, 4, 3), 0.2)):
            with self.subTest(geometry=geo):
                D = WilsonDirac(unit_links(geo), kappa, geo, time_sign=-1.0)
                sign, logabs = torch.linalg.slogdet(dense_matrix(D, geo))
                p = free_momenta(geo, -1.0)
                a = 1 - 2 * kappa * torch.cos(p).sum(-1)
                b = 2 * kappa * torch.sin(p).pow(2).sum(-1).sqrt()
                lam = torch.cat((torch.complex(a, b), torch.complex(a, -b)))
                lam = lam.repeat_interleave(geo.spin // 2 * geo.N)
                log_det = lam.log().sum()
                self.assertLess(abs(logabs.item() - log_det.real.item()), 1e-10 * abs(log_det.real.item()))
                # det D is real (conjugate pairs) and positive here
                self.assertLess(abs(math.remainder(log_det.imag.item(), 2 * math.pi)), 1e-10)
                self.assertLess(abs(sign.item() - 1.0), 1e-12)
                # same from the numerical eigenvalues of D(p) in neoqcd.fermions.dirac
                ref = free_wilson_eigenvalues(geo, kappa, -1.0).log().sum()
                self.assertLess(abs(ref.real.item() - log_det.real.item()), 1e-10 * abs(log_det.real.item()))
                # periodic time gives a different determinant
                Dp = WilsonDirac(unit_links(geo), kappa, geo, time_sign=1.0)
                _, logabs_p = torch.linalg.slogdet(dense_matrix(Dp, geo))
                self.assertGreater(abs(logabs_p.item() - logabs.item()), 1e-3)


if __name__ == "__main__":
    unittest.main()
