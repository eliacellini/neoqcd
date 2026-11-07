"""Tests of the Wilson-Dirac operator (neoqcd.fermions.dirac) against exact dense algebra."""

import unittest

import torch
from scipy.optimize import linear_sum_assignment

from neoqcd.fermions._gauge import SUN_haar, SUN_identity
from neoqcd.fermions.dirac import (
    WilsonDirac,
    dense_matrix,
    free_momenta,
    free_wilson_eigenvalues,
    free_wilson_matrix,
    gamma_matrices,
)
from neoqcd.fermions.lattice import Geometry, gauge_transform_field, gauge_transform_links
from neoqcd.fermions.solver import solve

G4 = Geometry(4, 4, 2, 3)  # 2^3 x 4, SU(3): dense dimension 384
G4_44 = Geometry(4, 4, 4, 3)  # 4^4, SU(3): dense dimension 3072
G2 = Geometry(2, 4, 4, 3)  # 2D 4 x 4, SU(3): dense dimension 96
C = torch.complex128


def random_links(geometry, batch, seed):
    g = torch.Generator().manual_seed(seed)
    return SUN_haar(geometry.link_shape(batch)[:-1], generator=g)


def random_field(geometry, batch, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(geometry.wilson_shape(batch), dtype=C, generator=g)


def unit_links(geometry, batch=1):
    return SUN_identity(geometry.link_shape(batch)[:-1], dtype=C)


def gamma5_dense(geometry):
    _, g5 = gamma_matrices(geometry.D)
    return torch.kron(torch.kron(torch.eye(geometry.volume, dtype=C), g5), torch.eye(geometry.N, dtype=C))


def closed_form_free_matrix(geometry, kappa, p):
    """D(p) = 1 - 2 kappa sum_mu cos p_mu + 2 i kappa sum_mu gamma_mu sin p_mu, written out here
    independently of free_wilson_matrix."""
    gammas, _ = gamma_matrices(geometry.D)
    one = torch.eye(geometry.spin, dtype=C)
    out = (1 - 2 * kappa * p.cos().sum().item()) * one
    for mu in range(geometry.D):
        out = out + 2j * kappa * p[mu].sin().item() * gammas[mu]
    return out


def max_multiset_distance(a, b):
    """Largest |a_i - b_pi(i)| for the optimal matching pi of two multisets of complex numbers."""
    cost = (a[:, None] - b[None, :]).abs().numpy()
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].max())


class _Adjoint:
    def __init__(self, op):
        self.op, self.geometry, self.dtype, self.device = op, op.geometry, op.dtype, op.device

    def apply(self, psi):
        return self.op.apply_dagger(psi)


class GammaTest(unittest.TestCase):
    def test_clifford_algebra(self):
        for D in (2, 4):
            with self.subTest(D=D):
                gammas, g5 = gamma_matrices(D)
                S = gammas.shape[-1]
                one = torch.eye(S, dtype=C)
                for mu in range(D):
                    self.assertTrue(torch.allclose(gammas[mu], gammas[mu].conj().T, atol=0))
                    self.assertLess((g5 @ gammas[mu] + gammas[mu] @ g5).abs().max().item(), 1e-15)
                    for nu in range(D):
                        anti = gammas[mu] @ gammas[nu] + gammas[nu] @ gammas[mu]
                        self.assertLess((anti - 2 * (mu == nu) * one).abs().max().item(), 1e-15)
                self.assertLess((g5 @ g5 - one).abs().max().item(), 1e-15)
                self.assertTrue(torch.allclose(g5, g5.conj().T, atol=0))

    def test_gamma5_convention(self):
        gammas, g5 = gamma_matrices(4)
        self.assertTrue(torch.equal(g5, gammas[0] @ gammas[1] @ gammas[2] @ gammas[3]))
        self.assertTrue(torch.equal(g5, torch.diag(torch.tensor([1, 1, -1, -1], dtype=C))))
        gammas, g5 = gamma_matrices(2)
        self.assertTrue(torch.equal(g5, -1j * gammas[0] @ gammas[1]))
        self.assertTrue(torch.equal(g5, torch.diag(torch.tensor([1, -1], dtype=C))))
        with self.assertRaises(NotImplementedError):
            gamma_matrices(3)

    def test_representation_literal(self):
        # openQCD chiral representation, written out entry by entry
        i = 1j
        g0 = [[0, 0, -1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, -1, 0, 0]]
        g1 = [[0, 0, 0, -i], [0, 0, -i, 0], [0, i, 0, 0], [i, 0, 0, 0]]  # [[0, -i s1], [i s1, 0]]
        g2 = [[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]]  # [[0, -i s2], [i s2, 0]]
        g3 = [[0, 0, -i, 0], [0, 0, 0, i], [i, 0, 0, 0], [0, -i, 0, 0]]  # [[0, -i s3], [i s3, 0]]
        g5 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]
        gammas, gamma5 = gamma_matrices(4)
        self.assertTrue(torch.equal(gammas, torch.tensor([g0, g1, g2, g3], dtype=C)))
        self.assertTrue(torch.equal(gamma5, torch.tensor(g5, dtype=C)))
        gammas, gamma5 = gamma_matrices(2)
        self.assertTrue(torch.equal(gammas, torch.tensor([[[0, 1], [1, 0]], [[0, -i], [i, 0]]], dtype=C)))
        self.assertTrue(torch.equal(gamma5, torch.tensor([[1, 0], [0, -1]], dtype=C)))


class WilsonDiracTest(unittest.TestCase):
    def test_gauge_covariance(self):
        for geo in (G4, G2):
            for time_sign in (-1.0, 1.0):
                with self.subTest(geometry=geo, time_sign=time_sign):
                    U = random_links(geo, 2, 11)
                    g = torch.Generator().manual_seed(12)
                    omega = SUN_haar((2,) + geo.lattice + (geo.N,), generator=g)
                    psi = random_field(geo, 2, 13)
                    kappa = torch.tensor([0.11, 0.13], dtype=torch.float64)
                    D = WilsonDirac(U, kappa, geo, time_sign)
                    Dg = WilsonDirac(gauge_transform_links(U, omega), kappa, geo, time_sign)
                    lhs = Dg.apply(gauge_transform_field(psi, omega))
                    rhs = gauge_transform_field(D.apply(psi), omega)
                    self.assertLess((lhs - rhs).abs().max().item(), 1e-12)
                    # the covariance is not trivial: without transforming the links it fails
                    wrong = D.apply(gauge_transform_field(psi, omega))
                    self.assertGreater((wrong - rhs).abs().max().item(), 1e-2)

    def test_gamma5_hermiticity_dense(self):
        for geo in (G4, G2):
            with self.subTest(geometry=geo):
                D = WilsonDirac(random_links(geo, 1, 21), 0.12, geo)
                M = dense_matrix(D, geo)
                G5 = gamma5_dense(geo)
                self.assertLess((M.conj().T - G5 @ M @ G5).abs().max().item(), 1e-13)
                Md = dense_matrix(_Adjoint(D), geo)
                self.assertLess((Md - M.conj().T).abs().max().item(), 1e-13)
                # D is not hermitian: the check above is not empty
                self.assertGreater((M - M.conj().T).abs().max().item(), 1e-2)

    def test_sparse_matches_dense(self):
        for geo in (G4, G2):
            with self.subTest(geometry=geo):
                U = random_links(geo, 2, 31)
                D = WilsonDirac(U, torch.tensor([0.10, 0.14], dtype=torch.float64), geo)
                psi = random_field(geo, 2, 32)
                out = D.apply(psi)
                for b in range(2):
                    M = dense_matrix(D, geo, batch_index=b)
                    ref = M @ psi[b].reshape(-1)
                    self.assertLess((out[b].reshape(-1) - ref).abs().max().item(), 1e-13)
                    Mn = dense_matrix(type("N", (), {"apply": D.batch_select(b).normal,
                                                      "dtype": C, "device": D.device})(), geo)
                    self.assertLess((Mn - M.conj().T @ M).abs().max().item(), 1e-12)

    def test_dense_flattening_order(self):
        # A source at (site, spin, colour) lands in column (site * S + s) * N + c.
        geo = G2
        D = WilsonDirac(random_links(geo, 1, 41), 0.2, geo)
        M = dense_matrix(D, geo)
        t, x, s, c = 2, 3, 1, 2
        psi = torch.zeros(geo.wilson_shape(1), dtype=C)
        psi[0, t, x, s, c] = 1.0
        col = ((t * geo.L + x) * geo.spin + s) * geo.N + c
        self.assertLess((M[:, col] - D.apply(psi).reshape(-1)).abs().max().item(), 1e-15)

    def test_sparse_matches_dense_4x4(self):
        # once on 4^4 SU(3) (dense dimension 3072), including apply_dagger vs the dense adjoint
        geo = G4_44
        D = WilsonDirac(random_links(geo, 1, 33), 0.13, geo)
        psi = random_field(geo, 1, 34)
        M = dense_matrix(D, geo)
        self.assertEqual(M.shape, (3072, 3072))
        self.assertLess((D.apply(psi).reshape(-1) - M @ psi.reshape(-1)).abs().max().item(), 1e-13)
        self.assertLess((D.apply_dagger(psi).reshape(-1) - M.conj().T @ psi.reshape(-1)).abs().max().item(),
                        1e-13)

    def test_float32_kappa_is_upcast_after_rounding(self):
        # Current behaviour, documented: a float32 kappa tensor is cast to float64 with its
        # float32-rounded value (no error, no warning); pass float64 to get the exact value.
        # A check belongs in dirac.py (_kappa_tensor).
        U = random_links(G2, 2, 35)
        k32 = torch.tensor([0.12, 0.13])
        self.assertEqual(k32.dtype, torch.float32)
        D32 = WilsonDirac(U, k32, G2)
        D64 = WilsonDirac(U, torch.tensor([0.12, 0.13], dtype=torch.float64), G2)
        self.assertEqual(D32.kappa.dtype, torch.float64)
        self.assertTrue(torch.equal(D32.kappa, k32.to(torch.float64)))
        self.assertTrue(torch.equal(D64.kappa, torch.tensor([0.12, 0.13], dtype=torch.float64)))
        self.assertGreater((D32.kappa - D64.kappa).abs().max().item(), 1e-10)
        self.assertTrue(torch.equal(WilsonDirac(U, k32.to(torch.float64), G2).kappa, D32.kappa))

    def test_free_spectrum(self):
        kappa = 0.11
        for geo in (G4, G2):
            spectra = {}
            for time_sign in (-1.0, 1.0):
                with self.subTest(geometry=geo, time_sign=time_sign):
                    D = WilsonDirac(unit_links(geo), kappa, geo, time_sign)
                    lam = torch.linalg.eigvals(dense_matrix(D, geo))
                    ref = free_wilson_eigenvalues(geo, kappa, time_sign)
                    self.assertEqual(lam.numel(), ref.numel())
                    self.assertLess(max_multiset_distance(lam, ref), 1e-10)
                    spectra[time_sign] = lam
            if len(spectra) < 2:
                continue  # a failure above is already reported for this geometry
            with self.subTest(geometry=geo, check="boundary conditions differ"):
                # the boundary condition changes the spectrum
                self.assertGreater(max_multiset_distance(spectra[-1.0], spectra[1.0]), 1e-2)
                # the two predictions are not interchangeable
                ref_p = free_wilson_eigenvalues(geo, kappa, 1.0)
                self.assertGreater(max_multiset_distance(spectra[-1.0], ref_p), 1e-2)

    def test_free_plane_waves_pin_momentum_sign(self):
        # psi(n) = exp(-i p.n) u is mapped to exp(-i p.n) D(p) u, n = array index. Only momenta
        # with sin p != 0 see the sign of the exponent and the assignment of the projectors
        # (1 -/+ gamma_mu), so the test uses three of those in every subtest (L = 4 in 4D).
        kappa = 0.11
        for geo in (G4_44, G2):
            for time_sign in (-1.0, 1.0):
                with self.subTest(geometry=geo, time_sign=time_sign):
                    D = WilsonDirac(unit_links(geo), kappa, geo, time_sign)
                    p_all = free_momenta(geo, time_sign)
                    n = torch.stack([g.reshape(-1) for g in torch.meshgrid(
                        *[torch.arange(k, dtype=torch.float64) for k in geo.lattice], indexing="ij")], -1)
                    g = torch.Generator().manual_seed(51)
                    u = torch.randn(geo.spin, geo.N, dtype=C, generator=g)
                    weight = p_all.sin().abs().sum(dim=1)
                    chosen = torch.nonzero(weight > 0.5).flatten()
                    self.assertGreaterEqual(chosen.numel(), 3)
                    for k in chosen[torch.linspace(0, chosen.numel() - 1, 3).long()].tolist():
                        p = p_all[k]
                        self.assertGreater(p.sin().abs().max().item(), 0.5)
                        wave = torch.exp(-1j * (n.to(C) @ p.to(C))).reshape(geo.lattice)
                        psi = (wave[..., None, None] * u)[None]
                        Dp = closed_form_free_matrix(geo, kappa, p)
                        self.assertLess((free_wilson_matrix(geo, kappa, p) - Dp).abs().max().item(), 1e-15)
                        ref = (wave[..., None, None] * (Dp @ u))[None]
                        self.assertLess((D.apply(psi) - ref).abs().max().item(), 1e-13)
                        # the opposite sign of the exponent fails since sin p != 0
                        psi_m = (wave.conj()[..., None, None] * u)[None]
                        ref_m = (wave.conj()[..., None, None] * (Dp @ u))[None]
                        self.assertGreater((D.apply(psi_m) - ref_m).abs().max().item(), 1e-3)

    def test_batch_independence(self):
        geo = G4
        U = random_links(geo, 2, 61)
        psi = random_field(geo, 2, 62)
        kappa = torch.tensor([0.09, 0.13], dtype=torch.float64)
        for time_sign in (-1.0, 1.0):
            D = WilsonDirac(U, kappa, geo, time_sign)
            eo = D.even_odd()
            outs = (D.apply(psi), D.apply_dagger(psi), D.normal(psi), D.hopping(psi),
                    eo.apply(psi), eo.normal(psi))
            for b in range(2):
                Db = WilsonDirac(U[b:b + 1], float(kappa[b]), geo, time_sign)
                eob = Db.even_odd()
                single = (Db.apply(psi[b:b + 1]), Db.apply_dagger(psi[b:b + 1]),
                          Db.normal(psi[b:b + 1]), Db.hopping(psi[b:b + 1]),
                          eob.apply(psi[b:b + 1]), eob.normal(psi[b:b + 1]))
                for o, s in zip(outs, single):
                    self.assertLess((o[b:b + 1] - s).abs().max().item(), 1e-14)


class EvenOddTest(unittest.TestCase):
    def test_schur_complement_and_determinant(self):
        for geo, seed in ((G4, 71), (G2, 72), (G4_44, 73)):
            with self.subTest(geometry=geo):
                D = WilsonDirac(random_links(geo, 1, seed), 0.12, geo)
                eo = D.even_odd()
                M = dense_matrix(D, geo)
                Mh = dense_matrix(eo, geo)
                n = Mh.shape[0]
                self.assertEqual(n, M.shape[0] // 2)
                # D_hat = D_ee - D_eo D_oo^-1 D_oe with D_oo = 1, from the dense blocks
                inner = geo.spin * geo.N
                even = eo.support.reshape(-1).repeat_interleave(inner)
                Dee, Deo = M[even][:, even], M[even][:, ~even]
                Doe, Doo = M[~even][:, even], M[~even][:, ~even]
                self.assertLess((Doo - torch.eye(n, dtype=C)).abs().max().item(), 1e-15)
                self.assertLess((Mh - (Dee - Deo @ Doe)).abs().max().item(), 1e-13)
                s_full, l_full = torch.linalg.slogdet(M)
                s_eo, l_eo = torch.linalg.slogdet(Mh)
                self.assertLess(abs(l_full.item() - l_eo.item()), 1e-10 * max(1.0, abs(l_full.item())))
                self.assertLess(abs((s_full - s_eo).item()), 1e-10)
                if geo is not G4_44:
                    Mhd = dense_matrix(type("A", (), {"apply": eo.apply_dagger, "support": eo.support,
                                                       "dtype": C, "device": eo.device})(), geo)
                    self.assertLess((Mhd - Mh.conj().T).abs().max().item(), 1e-13)

    def test_even_odd_solve_matches_dense_solve(self):
        for geo in (G4, G2):
            for method in ("cgne", "bicgstab"):
                with self.subTest(geometry=geo, method=method):
                    D = WilsonDirac(random_links(geo, 2, 81), torch.tensor([0.10, 0.12], dtype=torch.float64), geo)
                    b = random_field(geo, 2, 82)
                    res = solve(D.even_odd(), b, method=method, tol=1e-12)
                    for k in range(2):
                        x_ref = torch.linalg.solve(dense_matrix(D, geo, batch_index=k), b[k].reshape(-1))
                        err = (res.x[k].reshape(-1) - x_ref).norm() / x_ref.norm()
                        self.assertLess(err.item(), 1e-10)
                    self.assertLess(res.residual.max().item(), 1e-11)


if __name__ == "__main__":
    unittest.main()
