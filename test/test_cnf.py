import unittest
import math

import torch
from torch.func import functional_call

from neoqcd import sun_utils as sun
from neoqcd.cnf import (
    CloverChairLinkDrift,
    CNFProtocolLayer,
    apply_velocity,
    coeff_to_matrix,
    integrate_flow,
    lie_tangent,
    path_gradient_loss,
    path_gradient_residual,
)


def random_su3(batch=1, D=2, extents=(2, 2)):
    return sun.SUN_haar((batch, D, *extents, 3), dtype=torch.cdouble)


def gauge_transform(U, omega):
    D = U.shape[1]
    return torch.stack([
        sun.SUN_mul(
            sun.SUN_mul(omega, U[:, mu]),
            sun.SUN_dagger(torch.roll(omega, shifts=1, dims=1 + mu)),
        )
        for mu in range(D)
    ], dim=1)


def action(U, t):
    D = U.shape[1]
    volume = U.shape[2:-2]
    plaquettes = sun.plaquette_fields(U, D)
    trace_sum = sum(p.flatten(1).sum(1) for p in plaquettes)
    return t * (1.0 - trace_sum / (3.0 * math.prod(volume)))


class CloverChairCNFTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_gauge_covariance_su3(self):
        D, extents = 3, (4, 4, 4)
        U = random_su3(1, D, extents)
        omega = sun.SUN_haar((1, *extents, 3), dtype=U.dtype)
        field = CloverChairLinkDrift(D, hidden_dim=4, depth=1, loop_max=1, zero_init=False).to(dtype=U.real.dtype)
        F = field.drift(U, torch.tensor(0.4))
        Ug = gauge_transform(U, omega)
        Fg = field.drift(Ug, torch.tensor(0.4))
        expected = sun.SUN_mul(
            sun.SUN_mul(omega.unsqueeze(1), coeff_to_matrix(F)),
            sun.SUN_dagger(omega.unsqueeze(1)),
        )
        torch.testing.assert_close(coeff_to_matrix(Fg), expected, rtol=1e-9, atol=1e-9)

        self.assertAlmostEqual(action(U, torch.ones(1)).item(), action(Ug, torch.ones(1)).item(), places=11)

    def test_wilson_paths_match_su3_staples(self):
        D = 3
        U = random_su3(1, D, (4, 4, 4))
        field = CloverChairLinkDrift(D, hidden_dim=4, depth=1, loop_max=1)
        staples = field._staples(U)
        for mu in range(D):
            channel = 0
            for nu in range(D):
                if nu == mu:
                    continue
                torch.testing.assert_close(staples[:, mu, channel], sun.pstaple(U, mu, nu, D))
                torch.testing.assert_close(staples[:, mu, channel + 1], sun.nstaple(U, mu, nu, D))
                channel += 2

    def test_zero_drift_and_short_periodic_divergence(self):
        U = random_su3()
        identity = CloverChairLinkDrift(2, hidden_dim=4, depth=1, loop_max=1).to(dtype=U.real.dtype)
        F0, div0 = identity(U, 0.3)
        self.assertEqual(torch.count_nonzero(F0).item(), 0)
        self.assertEqual(torch.count_nonzero(div0).item(), 0)

        field = CloverChairLinkDrift(2, hidden_dim=4, depth=1, loop_max=2, zero_init=False).to(dtype=U.real.dtype)
        with torch.no_grad():
            field.output_layer.weight.mul_(0.02)
        F, div = field(U, 0.3)
        eps = 2e-5
        finite_difference = 0.0
        for mu in range(2):
            for x in range(2):
                for y in range(2):
                    site = (0, mu, x, y)
                    for a in range(8):
                        delta = torch.zeros_like(F)
                        delta[site + (a,)] = eps
                        plus = field.drift(apply_velocity(U, delta), 0.3)[site + (a,)]
                        minus = field.drift(apply_velocity(U, -delta), 0.3)[site + (a,)]
                        finite_difference = finite_difference + (plus - minus) / (2 * eps)
        torch.testing.assert_close(div[0], finite_difference, rtol=2e-4, atol=2e-5)

    def test_path_gradient_finite_difference_and_frozen_parameter_gradients(self):
        U = random_su3(batch=2).requires_grad_(True)
        field = CloverChairLinkDrift(2, hidden_dim=4, depth=1, loop_max=1, zero_init=False).to(dtype=U.real.dtype)
        with torch.no_grad():
            field.output_layer.weight.mul_(0.01)
        t = torch.tensor(0.35, dtype=U.real.dtype)
        stats = path_gradient_loss(field, action, U, t)
        self.assertEqual(stats["loss"].shape, (2,))
        self.assertTrue(torch.isfinite(stats["pinn"]))
        self.assertTrue(torch.isfinite(stats["F_norm"]).all())

        coeff = field.drift(U, t)
        eps = 1e-5
        xi_plus = path_gradient_residual(field, action, apply_velocity(U, coeff, eps), t)[0]
        xi_minus = path_gradient_residual(field, action, apply_velocity(U, coeff, -eps), t)[0]
        finite_difference = (xi_plus - xi_minus) / (2 * eps)
        torch.testing.assert_close(stats["loss"], finite_difference, rtol=3e-4, atol=3e-6)

        state = {name: value.detach() for name, value in field.named_parameters()}
        state.update(dict(field.named_buffers()))
        frozen = lambda value, time: functional_call(field, state, (value, time))
        xi = path_gradient_residual(frozen, action, U, t)[0]
        (xi_grad,) = torch.autograd.grad(xi.sum(), U)
        tangent = lie_tangent(U, field.drift(U, t))
        expected_loss = (xi_grad.conj() * tangent).real.flatten(1).sum(1)
        torch.testing.assert_close(stats["loss"], expected_loss, rtol=2e-8, atol=2e-8)

        parameters = list(field.parameters())
        actual_grads = torch.autograd.grad(stats["loss"].mean(), parameters, retain_graph=True,
                                           allow_unused=True)
        expected_grads = torch.autograd.grad(expected_loss.mean(), parameters, allow_unused=True)
        for actual, expected in zip(actual_grads, expected_grads):
            if actual is None or expected is None:
                self.assertIsNone(actual)
                self.assertIsNone(expected)
            else:
                torch.testing.assert_close(actual, expected, rtol=2e-7, atol=2e-9)
        grads = [g for g in actual_grads if g is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(g.abs().sum().item() for g in grads), 0.0)

    def test_cf4_round_trip_and_protocol_adapter(self):
        U = random_su3()
        field = CloverChairLinkDrift(2, hidden_dim=4, depth=1, loop_max=1, zero_init=False).to(dtype=U.real.dtype)
        with torch.no_grad():
            field.output_layer.weight.mul_(0.005)
        evolved, logdet = integrate_flow(U, 0.2, 0.5, field, n_steps=4)
        recovered, reverse_logdet = integrate_flow(evolved, 0.5, 0.2, field, n_steps=4)
        torch.testing.assert_close(recovered, U, rtol=2e-8, atol=2e-8)
        torch.testing.assert_close(logdet, -reverse_logdet, rtol=2e-7, atol=2e-8)
        torch.testing.assert_close(
            torch.matmul(evolved, sun.SUN_dagger(evolved)),
            torch.eye(3, dtype=U.dtype).expand_as(evolved),
            rtol=1e-10,
            atol=1e-10,
        )

        layer = CNFProtocolLayer(field, n_steps=4)
        out, layer_logdet = layer(U, 0.2, 0.3, rho_layer=None)
        torch.testing.assert_close(out, evolved)
        torch.testing.assert_close(layer_logdet, logdet)
        self.assertTrue(layer.full_lattice)
        self.assertEqual(layer.smearing_jacobian_version, None)
        self.assertEqual(layer.checkpoint_semantics["integrator"], "CF4")

    def test_time_range_feeds_unit_interval_to_encoder(self):
        U = random_su3()
        raw = CloverChairLinkDrift(2, hidden_dim=4, depth=1, loop_max=1, zero_init=False).to(dtype=U.real.dtype)
        scaled = CloverChairLinkDrift(
            2, hidden_dim=4, depth=1, loop_max=1, zero_init=False, time_range=(5.8, 6.2)
        ).to(dtype=U.real.dtype)
        scaled.load_state_dict(raw.state_dict())
        F, div = scaled(U, torch.tensor(6.1, dtype=torch.double))
        F_raw, div_raw = raw(U, torch.tensor(0.75, dtype=torch.double))
        torch.testing.assert_close(F, F_raw, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(div, div_raw, rtol=1e-12, atol=1e-12)
        self.assertEqual(scaled.checkpoint_semantics["time_range"], (5.8, 6.2))
        for bad in ((1.0, 1.0), (2.0, 1.0), (0.0, math.inf)):
            with self.subTest(time_range=bad), self.assertRaises(ValueError):
                CloverChairLinkDrift(2, hidden_dim=4, depth=1, time_range=bad)


class DefectCloverChairTests(unittest.TestCase):
    """The patch field equals the lattice CloverChair restricted to the links next to the defect."""

    def _fields(self, T=8, L=8):
        from neoqcd.cnf_io import build_cnf

        torch.manual_seed(12)
        config = dict(
            architecture="defect-clover-chair", domain="defect", D=4, T=T, L=L, beta=6.0, defect_size=1,
            time_slice=3, space_slice=3, parameter_min=0.0, parameter_max=1.0, batch_size=1, hidden_dim=4,
            depth=1, rho_init=1e-3, residual_n_pt=0, residual_mask_mod=2, orsteps=0,
            time_normalization="unit_interval",
        )
        field, pars, _ = build_cnf(config)
        with torch.no_grad():
            for parameter in field.parameters():
                parameter.add_(0.05 * torch.randn_like(parameter))
        return field, pars

    def test_matches_the_lattice_field_on_the_defect_links(self):
        # loop_max=1 (buffer 2) keeps the lattice reference affordable; the check does not
        # depend on loop_max. The patch [0, 4) x [0, 5)^3 sits inside a 6^4 torus, so a loop
        # leaving the patch would differ from the lattice one.
        from neoqcd.cnf import CloverChairLinkDrift, DefectCloverChairDrift
        from neoqcd.flow import FlowPars
        from neoqcd.theory import build_nf_changed_link_mask
        from neoqcd.utils import Defect, create_mask, defect_patch

        torch.manual_seed(12)
        defect = Defect(4, 6, 6, 1, 2, 2)
        pars = FlowPars(D=4, T=6, L=6, N=3, mask=create_mask(4, 6, 6), protocol=[1.0], batch_size=1,
                        device="cpu", defect=defect, **defect_patch(4, 6, 6, 1, 2, 2, buffer=2))
        full = CloverChairLinkDrift(4, hidden_dim=4, depth=1, loop_max=1, zero_init=False).to(dtype=torch.double)
        with self.assertRaisesRegex(ValueError, "patch buffer"):
            DefectCloverChairDrift(CloverChairLinkDrift(4, hidden_dim=4, depth=1, loop_max=2), pars)
        field = DefectCloverChairDrift(full, pars)
        U = sun.SUN_haar((1, 4, 6, 6, 6, 6, 3), dtype=torch.cdouble)
        t = torch.tensor([0.4], dtype=torch.double)
        changed = build_nf_changed_link_mask(pars, device=torch.device("cpu"))[None]
        self.assertEqual(int(changed.sum()), 75)
        with torch.no_grad():
            F, divergence = field(U, t)
            reference = full.drift(U, t)
            self.assertGreater(float(reference[changed].abs().max()), 1e-3)
            torch.testing.assert_close(F[changed], reference[changed], rtol=1e-12, atol=1e-13)
            self.assertEqual(float(F[~changed].abs().max()), 0.0)

            generators, structure = full._basis(U)
            S, C = full._staples(U), full._leaves(U)
            tt = full._time(t, U)
            f = lambda value: full._drift_given(value, S, C, tt, generators, structure)
            expected = sum(
                (torch.func.jvp(f, (U,), (1j * generators[a] @ U,))[1][..., a] * changed).sum()
                for a in range(8)
            )
        torch.testing.assert_close(divergence[0], expected, rtol=1e-11, atol=1e-12)

    def test_loop_max_sets_the_loops_and_the_patch_buffer(self):
        from neoqcd.cnf_io import build_cnf

        base = dict(
            architecture="defect-clover-chair", domain="defect", D=4, T=6, L=8, beta=6.0, defect_size=1,
            time_slice=3, space_slice=3, parameter_min=0.0, parameter_max=1.0, batch_size=1, hidden_dim=4,
            depth=1, rho_init=1e-3, residual_n_pt=0, residual_mask_mod=2, orsteps=0,
        )
        # Checkpoints without the key used loop_max = 2.
        for extra, loop_max in (({}, 2), ({"loop_max": 1}, 1), ({"loop_max": 2}, 2)):
            with self.subTest(**extra):
                field, pars, _ = build_cnf(dict(base, **extra))
                self.assertEqual((field.loop_max, field.buffer, pars.patch_buffer), (loop_max, loop_max + 1, loop_max + 1))
        lattice, _, _ = build_cnf(dict(base, architecture="clover-chair", domain="beta", loop_max=1))
        self.assertEqual(lattice.loop_max, 1)

    def test_path_gradient_reaches_the_network_with_bounded_memory(self):
        from neoqcd.cnf import path_gradient_loss

        field, _ = self._fields(T=6, L=8)
        self.assertEqual(field.buffer, 3)
        U = sun.SUN_haar((2, 4, 6, 8, 8, 8, 3), dtype=torch.cdouble)

        def action(value, t):
            return -2.0 * sum(p.flatten(1).sum(1).real for p in sun.plaquette_fields(value, 4))

        result = path_gradient_loss(field, action, U, torch.tensor([0.3, 0.7], dtype=torch.double), chunk_size=1)
        result["loss"].sum().backward()
        grads = [p.grad for p in field.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))


if __name__ == "__main__":
    unittest.main()
