"""Tests of the staggered operator (neoqcd.fermions.staggered) against exact dense algebra.

Note on 2^3 x 4: with L = 2 the forward and backward spatial neighbours coincide, so at U = 1
the spatial hops cancel (sin p_k = 0 for p_k in {0, pi}) and the free spectrum does not see the
spatial phases. The free-field tests therefore use 4^4 and 2D 4 x 4; 2^3 x 4 is used with
random links, where it is not degenerate.
"""

import math
import unittest

import torch
from scipy.optimize import linear_sum_assignment

from neoqcd.fermions._gauge import SUN_haar, SUN_identity
from neoqcd.fermions.dirac import free_momenta
from neoqcd.fermions.lattice import Geometry, gauge_transform_field, gauge_transform_links, parity, shift
from neoqcd.fermions.staggered import (
    StaggeredDirac,
    free_staggered_eigenvalues,
    free_staggered_normal_eigenvalues,
    solve_staggered,
    staggered_dense_matrix,
    staggered_phases,
)

G4 = Geometry(4, 4, 2, 3)  # 2^3 x 4, SU(3): dense dimension 96
G4_44 = Geometry(4, 4, 4, 3)  # 4^4, SU(3): dense dimension 768
G2 = Geometry(2, 4, 4, 3)  # 2D 4 x 4, SU(3): dense dimension 48
G2_88 = Geometry(2, 8, 8, 3)  # 2D 8 x 8, SU(3): dense dimension 192
C = torch.complex128


def random_links(geometry, batch, seed):
    g = torch.Generator().manual_seed(seed)
    return SUN_haar(geometry.link_shape(batch)[:-1], generator=g)


def random_field(geometry, batch, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(geometry.staggered_shape(batch), dtype=C, generator=g)


def unit_links(geometry, batch=1):
    return SUN_identity(geometry.link_shape(batch)[:-1], dtype=C)


def max_multiset_distance(a, b):
    """Largest |a_i - b_pi(i)| for the optimal matching pi of two multisets of complex numbers."""
    cost = (a[:, None] - b[None, :]).abs().numpy()
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].max())


class _Op:
    """A bare operator for `staggered_dense_matrix`: `apply` only, no `batch_select`."""

    def __init__(self, apply, like, support=None):
        self.apply, self.geometry, self.dtype, self.device = apply, like.geometry, like.dtype, like.device
        if support is not None:
            self.support = support


def eps_dense(geometry):
    eps = (1 - 2 * parity(geometry)).reshape(-1).to(C)
    return torch.diag(eps.repeat_interleave(geometry.N))


class PhasesTest(unittest.TestCase):
    def test_values_and_algebra(self):
        for geo in (G4, G4_44, G2, G2_88):
            with self.subTest(geometry=geo):
                eta = staggered_phases(geo)
                n = torch.meshgrid(*[torch.arange(k) for k in geo.lattice], indexing="ij")
                self.assertTrue(torch.equal(eta[0], torch.ones(geo.lattice, dtype=torch.float64)))
                for mu in range(geo.D):
                    ref = (-1.0) ** sum(n[nu] for nu in range(mu)) if mu > 0 else torch.ones(geo.lattice)
                    self.assertTrue(torch.equal(eta[mu], torch.as_tensor(ref, dtype=torch.float64)))
                    self.assertTrue(torch.equal(eta[mu] ** 2, torch.ones_like(eta[mu])))
                    # eta_mu does not depend on x_mu: eta_mu(x +- mu) = eta_mu(x)
                    for s in (1, -1):
                        self.assertTrue(torch.equal(shift(eta[None, mu], mu, s)[0], eta[mu]))
                for mu in range(geo.D):
                    for nu in range(geo.D):
                        if mu == nu:
                            continue
                        # eta_mu(x) eta_nu(x + mu) = - eta_nu(x) eta_mu(x + nu), x + mu = roll(+1)
                        lhs = eta[mu] * shift(eta[None, nu], mu, 1)[0]
                        rhs = -eta[nu] * shift(eta[None, mu], nu, 1)[0]
                        self.assertTrue(torch.equal(lhs, rhs))
                        # and with the backward neighbours
                        lhs = eta[mu] * shift(eta[None, nu], mu, -1)[0]
                        rhs = -eta[nu] * shift(eta[None, mu], nu, -1)[0]
                        self.assertTrue(torch.equal(lhs, rhs))


class StaggeredDiracTest(unittest.TestCase):
    def test_gauge_covariance(self):
        for geo in (G4, G2):
            for time_sign in (-1.0, 1.0):
                with self.subTest(geometry=geo, time_sign=time_sign):
                    U = random_links(geo, 2, 11)
                    g = torch.Generator().manual_seed(12)
                    omega = SUN_haar((2,) + geo.lattice + (geo.N,), generator=g)
                    chi = random_field(geo, 2, 13)
                    mass = torch.tensor([0.05, 0.4], dtype=torch.float64)
                    D = StaggeredDirac(U, mass, geo, time_sign)
                    Dg = StaggeredDirac(gauge_transform_links(U, omega), mass, geo, time_sign)
                    for name in ("apply", "apply_dagger", "normal"):
                        lhs = getattr(Dg, name)(gauge_transform_field(chi, omega))
                        rhs = gauge_transform_field(getattr(D, name)(chi), omega)
                        self.assertLess((lhs - rhs).abs().max().item(), 1e-12)
                    rhs = gauge_transform_field(D.apply(chi), omega)
                    wrong = D.apply(gauge_transform_field(chi, omega))
                    self.assertGreater((wrong - rhs).abs().max().item(), 1e-2)

    def test_antihermitian_hopping_and_dagger_dense(self):
        mass = 0.25
        for geo in (G4, G2, G4_44):
            for time_sign in (-1.0, 1.0):
                with self.subTest(geometry=geo, time_sign=time_sign):
                    D = StaggeredDirac(random_links(geo, 1, 21), mass, geo, time_sign)
                    M = staggered_dense_matrix(D)
                    K = staggered_dense_matrix(_Op(D.hopping, D))
                    one = torch.eye(M.shape[0], dtype=C)
                    self.assertLess((M - mass * one - K).abs().max().item(), 1e-15)
                    self.assertLess((K + K.conj().T).abs().max().item(), 1e-14)
                    self.assertGreater(K.abs().max().item(), 0.1)
                    Md = staggered_dense_matrix(_Op(D.apply_dagger, D))
                    self.assertLess((Md - M.conj().T).abs().max().item(), 1e-14)
                    # eps D eps = D^dagger, and K only links opposite parities
                    E = eps_dense(geo)
                    self.assertLess((E @ M @ E - M.conj().T).abs().max().item(), 1e-14)
                    even = (parity(geo).reshape(-1) == 0).repeat_interleave(geo.N)
                    self.assertEqual(K[even][:, even].abs().max().item(), 0.0)
                    self.assertEqual(K[~even][:, ~even].abs().max().item(), 0.0)
                    # D is not hermitian: the checks above are not empty
                    self.assertGreater((M - M.conj().T).abs().max().item(), 1e-2)

    def test_sparse_matches_dense(self):
        for geo in (G4, G2, G4_44):
            with self.subTest(geometry=geo):
                U = random_links(geo, 2, 31)
                D = StaggeredDirac(U, torch.tensor([0.1, 0.7], dtype=torch.float64), geo)
                chi = random_field(geo, 2, 32)
                out, outd, outn = D.apply(chi), D.apply_dagger(chi), D.normal(chi)
                for b in range(2):
                    M = staggered_dense_matrix(D, batch_index=b)
                    v = chi[b].reshape(-1)
                    self.assertLess((out[b].reshape(-1) - M @ v).abs().max().item(), 1e-13)
                    self.assertLess((outd[b].reshape(-1) - M.conj().T @ v).abs().max().item(), 1e-13)
                    self.assertLess((outn[b].reshape(-1) - M.conj().T @ (M @ v)).abs().max().item(), 1e-12)

    def test_dense_flattening_order(self):
        geo = G2
        D = StaggeredDirac(random_links(geo, 1, 41), 0.2, geo)
        M = staggered_dense_matrix(D)
        t, x, c = 2, 3, 1
        chi = torch.zeros(geo.staggered_shape(1), dtype=C)
        chi[0, t, x, c] = 1.0
        col = (t * geo.L + x) * geo.N + c
        self.assertLess((M[:, col] - D.apply(chi).reshape(-1)).abs().max().item(), 1e-15)

    def test_batch_independence_and_shared_links(self):
        geo = G4
        U = random_links(geo, 2, 61)
        chi = random_field(geo, 2, 62)
        mass = torch.tensor([0.09, 0.5], dtype=torch.float64)
        for time_sign in (-1.0, 1.0):
            D = StaggeredDirac(U, mass, geo, time_sign)
            eo = D.even_odd()
            outs = (D.apply(chi), D.apply_dagger(chi), D.normal(chi), D.hopping(chi), eo.apply(chi))
            for b in range(2):
                Db = StaggeredDirac(U[b:b + 1], float(mass[b]), geo, time_sign)
                single = (Db.apply(chi[b:b + 1]), Db.apply_dagger(chi[b:b + 1]), Db.normal(chi[b:b + 1]),
                          Db.hopping(chi[b:b + 1]), Db.even_odd().apply(chi[b:b + 1]))
                for o, s in zip(outs, single):
                    self.assertLess((o[b:b + 1] - s).abs().max().item(), 1e-14)
        # one gauge field applied to a batch of vectors
        D1 = StaggeredDirac(U[:1], 0.3, geo)
        out = D1.apply(chi)
        for b in range(2):
            self.assertLess((out[b:b + 1] - D1.apply(chi[b:b + 1])).abs().max().item(), 1e-15)


class FreeSpectrumTest(unittest.TestCase):
    mass = 0.3

    def test_free_spectrum_of_D(self):
        for geo in (G4_44, G2, G2_88):
            spectra = {}
            for time_sign in (-1.0, 1.0):
                with self.subTest(geometry=geo, time_sign=time_sign):
                    D = StaggeredDirac(unit_links(geo), self.mass, geo, time_sign)
                    lam = torch.linalg.eigvals(staggered_dense_matrix(D))
                    ref = free_staggered_eigenvalues(geo, self.mass, time_sign)
                    self.assertEqual(lam.numel(), ref.numel())
                    self.assertLess(max_multiset_distance(lam, ref), 1e-10)
                    spectra[time_sign] = lam
                    # log det from the analytic eigenvalues (det D is real and positive)
                    sign, logabs = torch.linalg.slogdet(staggered_dense_matrix(D))
                    self.assertLess(abs(logabs.item() - ref.log().real.sum().item()), 1e-10)
                    self.assertLess(abs(sign.item() - 1.0), 1e-10)
            # the boundary condition changes the spectrum, and the predictions are distinct
            self.assertGreater(max_multiset_distance(spectra[-1.0], spectra[1.0]), 1e-2)
            ref_p = free_staggered_eigenvalues(geo, self.mass, 1.0)
            self.assertGreater(max_multiset_distance(spectra[-1.0], ref_p), 1e-2)

    def test_free_spectrum_detects_wrong_phases(self):
        # Without phases (naive one-component fermion) or with one phase sign flipped on a single
        # site, the spectrum differs; the reversed ordering eta_mu = (-1)^(sum_{nu > mu} n_nu)
        # is unitarily equivalent and must give the same spectrum.
        for geo in (G4_44, G2):
            with self.subTest(geometry=geo):
                ref = free_staggered_eigenvalues(geo, self.mass, -1.0)
                eta = staggered_phases(geo)
                n = torch.meshgrid(*[torch.arange(k) for k in geo.lattice], indexing="ij")
                reversed_eta = torch.stack([
                    (-1.0) ** sum(n[nu] for nu in range(mu + 1, geo.D)) if mu < geo.D - 1
                    else torch.ones(geo.lattice) for mu in range(geo.D)]).to(torch.float64)
                broken = eta.clone()
                broken[1][(0,) * geo.D] *= -1
                for name, phases, ok in (("none", torch.ones_like(eta), False),
                                         ("one flipped", broken, False),
                                         ("reversed", reversed_eta, True)):
                    D = StaggeredDirac(unit_links(geo), self.mass, geo)
                    D.eta = phases.unsqueeze(-1)
                    lam = torch.linalg.eigvals(staggered_dense_matrix(_Op(D.apply, D)))
                    dist = max_multiset_distance(lam, ref)
                    if ok:
                        self.assertLess(dist, 1e-10, name)
                    else:
                        self.assertGreater(dist, 1e-3, name)

    def test_normal_operator_on_plane_waves(self):
        # D^dagger D exp(-i p.n) u = (m^2 + sum sin^2 p_mu) exp(-i p.n) u for every momentum p.
        for geo in (G4_44, G2_88):
            for time_sign in (-1.0, 1.0):
                with self.subTest(geometry=geo, time_sign=time_sign):
                    D = StaggeredDirac(unit_links(geo), self.mass, geo, time_sign)
                    p_all = free_momenta(geo, time_sign)
                    n = torch.stack([g.reshape(-1) for g in torch.meshgrid(
                        *[torch.arange(k, dtype=torch.float64) for k in geo.lattice], indexing="ij")], -1)
                    waves = torch.exp(-1j * (n.to(C) @ p_all.to(C).T)).T.reshape((-1,) + geo.lattice)
                    u = torch.randn(geo.N, dtype=C, generator=torch.Generator().manual_seed(51))
                    chi = waves[..., None] * u
                    lam = self.mass ** 2 + torch.sin(p_all).pow(2).sum(-1)
                    ref = lam.view((-1,) + (1,) * (geo.D + 1)).to(C) * chi
                    self.assertLess((D.normal(chi) - ref).abs().max().item(), 1e-12)
                    # the full normal-operator spectrum, multiplicity N per momentum
                    Mn = staggered_dense_matrix(_Op(D.normal, D))
                    ev = torch.linalg.eigvalsh(Mn)
                    ref_ev = torch.sort(free_staggered_normal_eigenvalues(geo, self.mass, time_sign)).values
                    self.assertLess((ev - ref_ev).abs().max().item(), 1e-12)


class EvenOddTest(unittest.TestCase):
    def test_block_diagonal_and_determinants(self):
        mass = 0.2
        for geo, seed in ((G4, 71), (G2, 72), (G4_44, 73)):
            with self.subTest(geometry=geo):
                D = StaggeredDirac(random_links(geo, 1, seed), mass, geo)
                eo = D.even_odd()
                M = staggered_dense_matrix(D)
                Mn = staggered_dense_matrix(_Op(D.normal, D))
                Mee = staggered_dense_matrix(eo)
                even = eo.support.reshape(-1).repeat_interleave(geo.N)
                self.assertEqual(Mee.shape[0], M.shape[0] // 2)
                self.assertLess(Mn[even][:, ~even].abs().max().item(), 1e-14)
                self.assertLess(Mn[~even][:, even].abs().max().item(), 1e-14)
                self.assertLess((Mee - Mn[even][:, even]).abs().max().item(), 1e-13)
                self.assertLess((Mee - Mee.conj().T).abs().max().item(), 1e-14)
                self.assertGreater(torch.linalg.eigvalsh(Mee).min().item(), mass ** 2 - 1e-12)
                # det(D^dag D) = det(M_ee) det(M_oo) = det(M_ee)^2, and det D = det M_ee
                Moo = Mn[~even][:, ~even]
                _, l_n = torch.linalg.slogdet(Mn)
                _, l_ee = torch.linalg.slogdet(Mee)
                _, l_oo = torch.linalg.slogdet(Moo)
                s_d, l_d = torch.linalg.slogdet(M)
                scale = max(1.0, abs(l_n.item()))
                self.assertLess(abs(l_ee.item() - l_oo.item()), 1e-10 * scale)
                self.assertLess(abs(l_n.item() - 2 * l_ee.item()), 1e-10 * scale)
                self.assertLess(abs(l_d.item() - l_ee.item()), 1e-10 * scale)
                self.assertLess(abs(s_d.item() - 1.0), 1e-10)

    def test_cg_solves_D_with_per_sample_mass(self):
        for geo in (G4, G4_44, G2_88):
            with self.subTest(geometry=geo):
                mass = torch.tensor([0.05, 0.5], dtype=torch.float64)
                D = StaggeredDirac(random_links(geo, 2, 81), mass, geo)
                b = random_field(geo, 2, 82)
                res = solve_staggered(D, b, tol=1e-12, max_iter=5000)
                self.assertLess(res.residual.max().item(), 1e-10)
                r = b - D.apply(res.x)
                for k in range(2):
                    self.assertLess((r[k].norm() / b[k].norm()).item(), 1e-10)
                # the lighter sample needs more iterations
                self.assertGreater(res.iterations[0].item(), res.iterations[1].item())
                # x = D^dagger (D D^dagger)^-1 b, compared with the dense solve
                for k in range(2):
                    x_ref = torch.linalg.solve(staggered_dense_matrix(D, batch_index=k), b[k].reshape(-1))
                    err = (res.x[k].reshape(-1) - x_ref).norm() / x_ref.norm()
                    self.assertLess(err.item(), 1e-9)


if __name__ == "__main__":
    unittest.main()
