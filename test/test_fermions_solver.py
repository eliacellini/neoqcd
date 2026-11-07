"""Tests of the batched Krylov solvers (neoqcd.fermions.solver) on the Wilson-Dirac operator.

kappa choice: Haar-random links are the strong-coupling limit beta = 0, where the critical
hopping parameter is kappa_c ~ 0.25 in 4D; at U = 1 it is 1/8. kappa = 0.10 is below both, so
D is well conditioned and every solve converges in a few tens of iterations; 0.12 and 0.13
(closer to 1/8) are added to make the per-sample iteration counts differ. In 2D (free
kappa_c = 1/4) kappa = 0.20.
"""

import unittest

import torch

from neoqcd.fermions._gauge import SUN_haar
from neoqcd.fermions.dirac import WilsonDirac, dense_matrix
from neoqcd.fermions.lattice import Geometry
from neoqcd.fermions.solver import bicgstab, cg, solve, solve_normal

G4 = Geometry(4, 4, 4, 3)  # 4^4 SU(3)
G4s = Geometry(4, 4, 2, 3)  # 2^3 x 4 SU(3)
G2 = Geometry(2, 4, 4, 3)  # 2D 4 x 4 SU(3)
C = torch.complex128
TOL = 1e-10


def random_links(geometry, batch, seed):
    g = torch.Generator().manual_seed(seed)
    return SUN_haar(geometry.link_shape(batch)[:-1], generator=g)


def random_field(geometry, batch, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(geometry.wilson_shape(batch), dtype=C, generator=g)


def kappas(*values):
    return torch.tensor(values, dtype=torch.float64)


class SolverTest(unittest.TestCase):
    def test_cg_on_normal_operator(self):
        D = WilsonDirac(random_links(G4, 2, 1), 0.10, G4)
        b = random_field(G4, 2, 2)
        res = cg(D.normal, b, tol=TOL)
        self.assertLess(res.residual.max().item(), TOL)
        self.assertTrue((res.iterations > 0).all().item())
        # the recursive residual criterion agrees with an independent check of D^dag D x = b
        r = b - D.apply_dagger(D.apply(res.x))
        self.assertLess((r.norm() / b.norm()).item(), TOL)

    def test_bicgstab_on_wilson(self):
        for geo, kappa in ((G4, 0.10), (G2, 0.20)):
            with self.subTest(geometry=geo):
                D = WilsonDirac(random_links(geo, 2, 3), kappa, geo)
                b = random_field(geo, 2, 4)
                res = bicgstab(D.apply, b, tol=TOL)
                self.assertLess(res.residual.max().item(), TOL)
                self.assertTrue(torch.isfinite(res.x.abs()).all().item())

    def test_batched_convergence_with_per_sample_kappa(self):
        k = kappas(0.05, 0.10, 0.13)
        U = random_links(G4, 3, 5)
        b = random_field(G4, 3, 6)
        D = WilsonDirac(U, k, G4)
        for name, run in (("cg", lambda op, rhs: cg(op.normal, rhs, tol=TOL)),
                          ("bicgstab", lambda op, rhs: bicgstab(op.apply, rhs, tol=TOL))):
            with self.subTest(solver=name):
                res = run(D, b)
                self.assertLess(res.residual.max().item(), TOL)
                its = res.iterations.tolist()
                self.assertTrue(its[0] < its[1] < its[2], its)
                # each sample is exactly the solve of its own system: masked updates freeze the
                # converged samples instead of iterating them further
                for s in range(3):
                    single = run(WilsonDirac(U[s:s + 1], float(k[s]), G4), b[s:s + 1])
                    self.assertEqual(int(single.iterations[0]), its[s])
                    self.assertLess((single.x[0] - res.x[s]).abs().max().item(), 1e-13)

    def test_degenerate_samples(self):
        D = WilsonDirac(random_links(G4s, 3, 7), 0.10, G4s)
        b = random_field(G4s, 3, 8)
        b[1] = 0.0
        exact = solve(D, b, tol=1e-13).x
        x0 = exact.clone()
        x0[2] = 0.0  # sample 0 starts at the solution, sample 1 has b = 0, sample 2 is generic
        for name, solver, A in (("cg", cg, D.normal), ("bicgstab", bicgstab, D.apply)):
            with self.subTest(solver=name):
                rhs = D.apply_dagger(D.apply(exact)) if name == "cg" else b
                rhs = rhs.clone()
                rhs[1] = 0.0
                res = solver(A, rhs, x0=x0 * (torch.arange(3) != 1).view(-1, 1, 1, 1, 1, 1, 1), tol=1e-9)
                self.assertTrue(torch.isfinite(res.x.abs()).all().item())
                self.assertEqual(res.iterations.tolist()[:2], [0, 0])
                self.assertGreater(int(res.iterations[2]), 0)
                self.assertEqual(res.x[1].abs().max().item(), 0.0)
                self.assertLess(res.residual.max().item(), 1e-9)

    def test_solve_direct_and_even_odd(self):
        k = kappas(0.10, 0.12)
        D = WilsonDirac(random_links(G4s, 2, 9), k, G4s)
        b = random_field(G4s, 2, 10)
        ref = torch.stack([torch.linalg.solve(dense_matrix(D, batch_index=s), b[s].reshape(-1))
                           for s in range(2)])
        for op_name, op in (("full", D), ("even-odd", D.even_odd())):
            for method in ("cgne", "bicgstab"):
                with self.subTest(op=op_name, method=method):
                    res = solve(op, b, method=method, tol=1e-12)
                    self.assertLess(res.residual.max().item(), 1e-11)
                    err = (res.x.reshape(2, -1) - ref).norm(dim=1) / ref.norm(dim=1)
                    self.assertLess(err.max().item(), 1e-10)

    def test_even_odd_reduces_iterations(self):
        D = WilsonDirac(random_links(G4, 1, 11), 0.12, G4)
        b = random_field(G4, 1, 12)
        full = solve(D, b, tol=TOL)
        eo = solve(D.even_odd(), b, tol=TOL)
        self.assertLess(full.residual.item(), 1e-9)
        self.assertLess(eo.residual.item(), 1e-9)
        self.assertLess(int(eo.iterations), int(full.iterations))

    def test_solve_normal_full_and_even_odd(self):
        k = kappas(0.10, 0.12)
        D = WilsonDirac(random_links(G4s, 2, 13), k, G4s)
        b = random_field(G4s, 2, 14)
        a = solve_normal(D, b, tol=1e-12)
        e = solve_normal(D.even_odd(), b, tol=1e-12)
        self.assertLess(a.residual.max().item(), 1e-11)
        self.assertLess(e.residual.max().item(), 1e-10)
        for s in range(2):
            M = dense_matrix(D, batch_index=s)
            ref = torch.linalg.solve(M.conj().T @ M, b[s].reshape(-1))
            for x in (a.x, e.x):
                self.assertLess(((x[s].reshape(-1) - ref).norm() / ref.norm()).item(), 1e-10)

    def test_even_site_normal_equation(self):
        # The even-odd pseudofermion system: D_hat^dag D_hat x_e = b_e on even sites only.
        D = WilsonDirac(random_links(G4s, 1, 15), 0.12, G4s)
        eo = D.even_odd()
        b = eo.even(random_field(G4s, 1, 16))
        res = cg(eo.normal, b, tol=TOL)
        self.assertLess(res.residual.item(), TOL)
        self.assertEqual(eo.odd(res.x).abs().max().item(), 0.0)
        Mh = dense_matrix(eo)
        sites = eo.support.reshape(-1).repeat_interleave(G4s.spin * G4s.N)
        ref = torch.linalg.solve(Mh.conj().T @ Mh, b.reshape(-1)[sites])
        self.assertLess(((res.x.reshape(-1)[sites] - ref).norm() / ref.norm()).item(), 1e-9)


if __name__ == "__main__":
    unittest.main()
