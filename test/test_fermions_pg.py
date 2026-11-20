"""Tests of the PG pseudofermion term (neoqcd.fermions.pg.pseudofermion_pg_term).

xi_f = F . grad S_pf / V with phi held fixed, and its link gradient from two solves (X = M^-1 phi,
Y = M^-1 (d_F M) X) and autograd at fixed fields, against
  (i)  dense double autograd of phi^dagger (D_hat^dagger D_hat)^-1 phi with D_hat assembled from the
       links (`exact.assemble_even_odd`): value and the full Wirtinger gradient;
  (ii) central finite differences along the group, xi_f(exp(i eps T) U);
  (iii) the convergence policy (raise / warn / ignore) and non-finite inputs, as the package's terms;
plus the value against the MD force of `TwoFlavourWilson` (Re <i F U, V>: pins the algebra
normalisation) and the exact-type dispatch. 2D 4 x 4 SU(3) and 2^3 x 4 SU(3), batch 2, Haar links,
both time signs, per-sample kappa. CPU, complex128.
"""

import unittest
import warnings

import torch

from neoqcd.fermions._gauge import SUN_haar, SUN_mul
from neoqcd.fermions.action import QCDAction
from neoqcd.fermions.algebra import expi, project_hermitian_traceless, random_algebra
from neoqcd.fermions.exact import assemble_even_odd
from neoqcd.fermions.fast import FastTwoFlavourWilson
from neoqcd.fermions.lattice import Geometry, parity
from neoqcd.fermions.pg import LossSolveCounter, hermitian_tangent, pseudofermion_pg_term
from neoqcd.fermions.pseudofermion import TwoFlavourWilson, UnconvergedSolveWarning
from neoqcd.fermions.staggered_pseudofermion import StaggeredPseudofermion

G2 = Geometry(2, 4, 4, 3)  # 2D 4 x 4 SU(3), even-odd dimension 48
G4 = Geometry(4, 4, 2, 3)  # 2^3 x 4 SU(3), even-odd dimension 192
CASES = ((G2, (0.20, 0.17)), (G4, (0.12, 0.10)))


def haar(geometry, batch, seed):
    gen = torch.Generator().manual_seed(seed)
    return SUN_haar(geometry.link_shape(batch)[:-1], generator=gen), gen


def even_flat(psi, geometry):
    """Full-lattice Wilson field -> the dense even-site vector of `assemble_even_odd`."""
    B = psi.shape[0]
    even = (parity(geometry).reshape(-1) == 0).nonzero().squeeze(-1)
    return psi.reshape(B, geometry.volume, -1)[:, even].reshape(B, -1)


def dense_spf(U, phi, kappa, geometry, time_sign):
    """phi^dagger (D_hat^dagger D_hat)^-1 phi from the dense D_hat assembled from the links, (B,)."""
    Dh = assemble_even_odd(U, kappa, geometry, time_sign)
    p = even_flat(phi, geometry)
    M = Dh.mH @ Dh
    X = torch.linalg.solve(M, p.unsqueeze(-1)).squeeze(-1)
    return (p.conj() * X).real.sum(-1)


def dense_xi_and_gradient(U, phi, kappa, geometry, time_sign, tangent):
    """xi = Re <grad S_dense, V(U)> / V and its Wirtinger gradient by double autograd."""
    Ug = U.detach().clone().requires_grad_(True)
    S = dense_spf(Ug, phi, kappa, geometry, time_sign)
    (gS,) = torch.autograd.grad(S.sum(), Ug, create_graph=True)
    xi = (gS.conj() * tangent(Ug)).real.flatten(1).sum(1) / geometry.volume
    (g,) = torch.autograd.grad(xi.sum(), Ug)
    return xi.detach(), g


def u_dependent_tangent(geometry, batch, seed, scale=0.3):
    """A smooth algebra field of the links, F(U) = P_ht(i U A) (A fixed), and its tangent i F U."""
    gen = torch.Generator().manual_seed(seed)
    A = torch.randn(geometry.link_shape(batch), generator=gen, dtype=torch.complex128)
    return lambda U: 1j * torch.matmul(scale * project_hermitian_traceless(1j * torch.matmul(U, A)), U)


def make_term(geometry, kappa, time_sign, U, gen, **kw):
    term = TwoFlavourWilson(torch.tensor(kappa, dtype=torch.float64), geometry, time_sign=time_sign,
                            solver_tol=1e-13, **kw)
    term.refresh(U, gen)
    return term


class DenseReferenceTest(unittest.TestCase):
    """(i) value and full Wirtinger gradient against dense double autograd."""

    def test_against_dense_double_autograd(self):
        for geometry, kappa in CASES:
            for time_sign in (-1.0, 1.0):
                for kind in ("fixed", "u_dependent"):
                    with self.subTest(D=geometry.D, time_sign=time_sign, velocity=kind):
                        U, gen = haar(geometry, 2, 11)
                        term = make_term(geometry, kappa, time_sign, U, gen)
                        if kind == "fixed":
                            H = 0.3 * random_algebra(geometry.link_shape(2), generator=gen)
                            velocity, tangent = H, hermitian_tangent(H)
                        else:
                            velocity = tangent = u_dependent_tangent(geometry, 2, 5)
                        out = pseudofermion_pg_term(U, term, velocity)
                        xi, g = dense_xi_and_gradient(U, term.phi, term.kappa, geometry, time_sign, tangent)
                        self.assertGreater(float(xi.abs().max()), 1e-3)
                        torch.testing.assert_close(out["xi"], xi, rtol=1e-9, atol=1e-12)
                        scale = float(g.abs().max())
                        self.assertGreater(scale, 1e-4)
                        torch.testing.assert_close(out["gradient"], g, rtol=0, atol=1e-9 * scale)
                        self.assertEqual(out["counter"].solves, 2)
                        torch.testing.assert_close(out["S_pf"], dense_spf(U, term.phi, term.kappa, geometry,
                                                                          time_sign), rtol=1e-10, atol=0)

    def test_value_is_the_md_force_along_the_tangent(self):
        # Re <i F U, i H U> = tr(F H): the fermion force (basis lambda/sqrt 2) and the tangent of
        # any normalisation agree in the embedding; a factor 2 would show here.
        for geometry, kappa in CASES:
            with self.subTest(D=geometry.D):
                U, gen = haar(geometry, 2, 3)
                term = make_term(geometry, kappa, -1.0, U, gen)
                H = random_algebra(geometry.link_shape(2), generator=gen)
                out = pseudofermion_pg_term(U, term, H)
                F = term.force(U)
                expected = (torch.matmul(1j * F, U).conj() * hermitian_tangent(H)(U)).real.flatten(1).sum(1)
                torch.testing.assert_close(out["xi"] * geometry.volume, expected, rtol=1e-9, atol=1e-10)


class FiniteDifferenceTest(unittest.TestCase):
    """(ii) Re <G, i T U> = d/d eps xi_f(exp(i eps T) U) (4th-order central differences)."""

    def test_group_derivative(self):
        for geometry, kappa in CASES:
            for time_sign in (-1.0, 1.0):
                with self.subTest(D=geometry.D, time_sign=time_sign):
                    U, gen = haar(geometry, 2, 21)
                    term = make_term(geometry, kappa, time_sign, U, gen)
                    velocity = u_dependent_tangent(geometry, 2, 9)
                    T = random_algebra(geometry.link_shape(2), generator=gen)
                    out = pseudofermion_pg_term(U, term, velocity, tol=1e-14)
                    directional = (out["gradient"].conj() * torch.matmul(1j * T, U)).real.flatten(1).sum(1)
                    xi = lambda s: pseudofermion_pg_term(SUN_mul(expi(s * T), U), term, velocity, tol=1e-14)["xi"]
                    h = 1e-3
                    fd = (-xi(2 * h) + 8 * xi(h) - 8 * xi(-h) + xi(-2 * h)) / (12 * h)
                    self.assertGreater(float(directional.abs().min()), 1e-3)
                    torch.testing.assert_close(directional, fd, rtol=1e-7, atol=1e-10)


class PolicyTest(unittest.TestCase):
    """(iii) convergence policy and non-finite inputs."""

    def setUp(self):
        self.U, gen = haar(G2, 2, 31)
        self.H = 0.3 * random_algebra(G2.link_shape(2), generator=gen)
        self.gen = gen

    def test_unconverged_raise_warn_ignore(self):
        term = make_term(G2, (0.2, 0.2), -1.0, self.U, self.gen)
        with self.assertRaisesRegex(RuntimeError, "pg_X: CG not converged"):
            pseudofermion_pg_term(self.U, term, self.H, max_iter=2)
        counter = LossSolveCounter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = pseudofermion_pg_term(self.U, term, self.H, max_iter=2, on_unconverged="warn", counter=counter)
            pseudofermion_pg_term(self.U, term, self.H, max_iter=2, on_unconverged="warn", counter=counter)
        kinds = [str(w.message).split(":")[0] for w in caught if issubclass(w.category, UnconvergedSolveWarning)]
        self.assertEqual(kinds, ["pseudofermion_pg_term pg_X", "pseudofermion_pg_term pg_Y"])  # once per kind
        self.assertEqual(counter.unconverged_solves, 4)
        self.assertEqual(counter.solves, 4)
        self.assertGreater(counter.max_residual, 1e-6)
        self.assertTrue(bool(torch.isfinite(out["xi"]).all()))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ignored = pseudofermion_pg_term(self.U, term, self.H, max_iter=2, on_unconverged="ignore")
        self.assertEqual(ignored["counter"].unconverged_solves, 2)
        # the term's policy is the default
        term.on_unconverged = "ignore"
        self.assertEqual(pseudofermion_pg_term(self.U, term, self.H, max_iter=2)["counter"].unconverged_solves, 2)
        self.assertEqual(term.stats["solves"], 0)  # the loss solves are not the term's

    def test_non_finite_sample_is_named_and_isolated(self):
        term = make_term(G2, (0.2, 0.2), -1.0, self.U, self.gen)
        bad = self.U.clone()
        bad[1, 0, 0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, r"sample\(s\) \[1\]"):
            pseudofermion_pg_term(bad, term, self.H)
        out = pseudofermion_pg_term(bad, term, self.H, on_unconverged="ignore")
        self.assertTrue(out["counter"].max_residual != out["counter"].max_residual)  # NaN-sticky
        self.assertFalse(bool(torch.isfinite(out["xi"][1])))
        single = TwoFlavourWilson(0.2, G2, time_sign=-1.0, solver_tol=1e-13)
        single.phi = term.phi[:1]
        ref = pseudofermion_pg_term(self.U[:1], single, self.H[:1])
        torch.testing.assert_close(out["xi"][:1], ref["xi"], rtol=1e-12, atol=0)
        torch.testing.assert_close(out["gradient"][:1], ref["gradient"], rtol=0, atol=1e-12)
        # non-finite phi
        term.phi = term.phi.clone()
        term.phi[0].fill_(float("inf"))
        with self.assertRaisesRegex(RuntimeError, r"sample\(s\) \[0\]"):
            pseudofermion_pg_term(self.U, term, self.H)


class DispatchTest(unittest.TestCase):
    def test_exact_types(self):
        U, gen = haar(G2, 1, 41)
        H = random_algebra(G2.link_shape(1), generator=gen)
        qcd = QCDAction(G2, "kappa", beta=5.0, kappa=0.2)
        qcd.refresh(U, gen)
        self.assertTrue(torch.isfinite(pseudofermion_pg_term(U, qcd.fermion, H)["xi"]).all())  # parametric term
        with self.assertRaises(TypeError):
            pseudofermion_pg_term(U, StaggeredPseudofermion(0.3, G2), H)
        with self.assertRaisesRegex(ValueError, "even-odd"):
            pseudofermion_pg_term(U, TwoFlavourWilson(0.2, G2, even_odd=False), H)

        class Sub(TwoFlavourWilson):
            pass
        with self.assertRaises(TypeError):
            pseudofermion_pg_term(U, Sub(0.2, G2), H)
        fresh = TwoFlavourWilson(0.2, G2)
        with self.assertRaisesRegex(RuntimeError, "not set"):
            pseudofermion_pg_term(U, fresh, H)


if __name__ == "__main__":
    unittest.main()
