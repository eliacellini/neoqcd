"""Tests of neoqcd.fermions.lattice: shift convention, gauge transformations, boundary links."""

import unittest

import torch

from neoqcd.fermions._gauge import SUN_dagger, SUN_haar, SUN_mul, plaquette_fields
from neoqcd.fermions.lattice import (
    Geometry,
    even_odd_masks,
    fermion_boundary_links,
    gauge_transform_field,
    gauge_transform_links,
    parity,
    shift,
)

GEOMETRIES = (Geometry(4, 4, 2, 3), Geometry(2, 4, 4, 3), Geometry(4, 4, 2, 2))


def random_links(geometry, batch, seed):
    g = torch.Generator().manual_seed(seed)
    return SUN_haar(geometry.link_shape(batch)[:-1], generator=g)


def random_omega(geometry, batch, seed):
    g = torch.Generator().manual_seed(seed)
    return SUN_haar((batch,) + geometry.lattice + (geometry.N,), generator=g)


class LatticeTest(unittest.TestCase):
    def test_geometry_shapes(self):
        geo = Geometry(4, 4, 2, 3)
        self.assertEqual(geo.lattice, (4, 2, 2, 2))
        self.assertEqual(geo.volume, 32)
        self.assertEqual(geo.spin, 4)
        self.assertEqual(geo.wilson_shape(3), (3, 4, 2, 2, 2, 4, 3))
        self.assertEqual(Geometry(2, 4, 4, 3).spin, 2)

    def test_shift_reads_previous_array_index(self):
        f = torch.arange(4 * 3, dtype=torch.float64).reshape(1, 4, 3)
        self.assertTrue(torch.equal(shift(f, 0, 1)[0, 1], f[0, 0]))
        self.assertTrue(torch.equal(shift(f, 0, 1)[0, 0], f[0, 3]))
        self.assertTrue(torch.equal(shift(f, 1, -1)[0, :, 0], f[0, :, 1]))

    def test_plaquette_gauge_invariant_with_code_convention(self):
        for geo in GEOMETRIES:
            with self.subTest(geometry=geo):
                U = random_links(geo, 2, 1)
                omega = random_omega(geo, 2, 2)
                P = plaquette_fields(U, geo.D)
                P2 = plaquette_fields(gauge_transform_links(U, omega), geo.D)
                for a, b in zip(P, P2):
                    self.assertLess((a - b).abs().max().item(), 1e-12)

    def test_opposite_convention_is_not_invariant(self):
        # U_mu(x) -> Omega(x) U_mu(x) Omega(x - mu)^dagger must break invariance: the test above
        # is therefore sensitive to the direction of "x + mu".
        geo = Geometry(2, 4, 4, 3)
        U = random_links(geo, 1, 3)
        omega = random_omega(geo, 1, 4)
        wrong = torch.stack([SUN_mul(SUN_mul(omega, U[:, mu]), SUN_dagger(shift(omega, mu, -1)))
                             for mu in range(geo.D)], dim=1)
        P = plaquette_fields(U, geo.D)[0]
        P2 = plaquette_fields(wrong, geo.D)[0]
        self.assertGreater((P - P2).abs().max().item(), 1e-3)

    def test_gauge_transform_field_wilson_and_staggered(self):
        geo = Geometry(4, 4, 2, 3)
        omega = random_omega(geo, 2, 5)
        g = torch.Generator().manual_seed(6)
        psi = torch.randn(geo.wilson_shape(2), dtype=torch.complex128, generator=g)
        out = gauge_transform_field(psi, omega)
        ref = torch.einsum("ntxyzab,ntxyzsb->ntxyzsa", omega, psi)
        self.assertLess((out - ref).abs().max().item(), 1e-14)
        chi = torch.randn(geo.staggered_shape(2), dtype=torch.complex128, generator=g)
        out = gauge_transform_field(chi, omega)
        ref = torch.einsum("ntxyzab,ntxyzb->ntxyza", omega, chi)
        self.assertLess((out - ref).abs().max().item(), 1e-14)

    def test_boundary_links_flip_only_time_links_at_t0(self):
        geo = Geometry(4, 4, 2, 3)
        U = random_links(geo, 2, 7)
        U_copy = U.clone()
        Ub = fermion_boundary_links(U, -1.0)
        self.assertTrue(torch.equal(U, U_copy))  # gauge field untouched
        self.assertTrue(torch.equal(Ub[:, 0, 0], -U[:, 0, 0]))
        self.assertTrue(torch.equal(Ub[:, 0, 1:], U[:, 0, 1:]))
        self.assertTrue(torch.equal(Ub[:, 1:], U[:, 1:]))
        self.assertTrue(torch.equal(fermion_boundary_links(U, 1.0), U))

    def test_parity_masks(self):
        for geo in (Geometry(4, 4, 2, 3), Geometry(2, 4, 4, 3), Geometry(4, 4, 4, 3)):
            with self.subTest(geometry=geo):
                even, odd = even_odd_masks(geo)
                self.assertEqual(int(even.sum()), geo.volume // 2)
                self.assertTrue(torch.equal(even ^ odd, torch.ones_like(even)))
                p = parity(geo)
                for mu in range(geo.D):
                    self.assertTrue(torch.equal(torch.roll(p, 1, dims=mu), 1 - p))
        with self.assertRaises(ValueError):
            even_odd_masks(Geometry(2, 3, 4, 3))


if __name__ == "__main__":
    unittest.main()
