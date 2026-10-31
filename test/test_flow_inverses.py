"""Inverses of every flow layer: each must undo its forward map and return minus its log-Jacobian."""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

import neoqcd.sun_utils as sun
from neoqcd.cnf import CloverChairLinkDrift, CNFProtocolLayer
from neoqcd.cnf_io import build_cnf
from neoqcd.smearing import CouplingLayer, DefectCouplingLayer, ResidualNormalizingFlows
from neoqcd.smearing_jacobian import inverse_smearing_step, smearing_step
from neoqcd.utils import Defect, create_mask
from test.test_cnf_adapters import _defect_config, _flow_pars

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("main_pt_obc", ROOT / "main" / "main_pt_obc.py")
PT_OBC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PT_OBC)


def _smooth_links(shape, spread):
    """SU(3) links exp(iH) with H traceless Hermitian of size ~spread, closer to a thermalized lattice than Haar."""
    A = torch.randn(*shape, 3, 3, dtype=torch.cdouble)
    H = 0.5 * (A + A.conj().transpose(-1, -2))
    H = H - torch.diagonal(H, dim1=-2, dim2=-1).mean(-1)[..., None, None] * torch.eye(3, dtype=torch.cdouble)
    return torch.linalg.matrix_exp(1j * spread * H)


class StoutInverseTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        _, self.pars, _ = build_cnf(_defect_config(batch_size=2))
        self.layer = DefectCouplingLayer(self.pars, torch.zeros(self.pars.rho_shape, dtype=torch.double), time=1.0)
        with torch.no_grad():
            # rho depends on the coupling and differs between links
            for parameter in self.layer.smearing.hyper_smearing.mlp.parameters():
                parameter.add_(0.05 * torch.randn_like(parameter))
        self.bc = torch.tensor([0.2, 0.7], dtype=torch.double)
        self.delta = torch.tensor([0.5, -0.5], dtype=torch.double)
        self.configs = {
            "haar": sun.SUN_haar((2, 4, 4, 6, 6, 6, 3), dtype=torch.cdouble),
            "smooth": _smooth_links((2, 4, 4, 6, 6, 6), 0.3),
        }

    def test_every_mask_is_inverted_link_by_link(self):
        smearing = self.layer.smearing
        rho_runtime, batched = smearing._runtime_rho(self.configs["haar"], None, self.bc, self.delta)
        links = smearing._links_for_smearing_mask(self.pars.small_mask[-1], self.pars.smeared_defect_mask[-1])
        for name, cfgs in self.configs.items():
            small = self.pars.defect.cut_defect(cfgs, buffer=2)
            flat = sun.flat_links(small, smearing.D)
            masks = [m for m in range(2 * smearing.D) if links.sites[m].numel() > 0]
            self.assertEqual(len(masks), 2 * smearing.D)
            for m in masks:
                with self.subTest(config=name, mask=m):
                    C = smearing._selected_stout_staples(
                        flat, tuple(small.shape), links, m, smearing._mask_rho(rho_runtime, batched, 0, m)
                    )
                    U = flat.index_select(1, links.links[m])
                    Up, logdet = smearing_step(U, C)
                    self.assertGreater((Up - U).abs().max().item(), 1e-4)
                    U_back, logdet_inverse, iterations, residual = inverse_smearing_step(Up, C, check_every=1)
                    self.assertLessEqual(residual.item(), 1e-12)
                    self.assertLess(iterations, 20)
                    torch.testing.assert_close(U_back, U, rtol=0, atol=1e-11)
                    torch.testing.assert_close(logdet_inverse, -logdet, rtol=0, atol=1e-11)

    def test_defect_layer_round_trips(self):
        for name, cfgs in self.configs.items():
            with self.subTest(config=name), torch.no_grad():
                forward, _, logdet = self.layer(cfgs, self.pars, None, beta=self.bc, delta_beta=self.delta)
                back, _, logdet_inverse = self.layer.inverse(forward, self.pars, None, beta=self.bc, delta_beta=self.delta)
                torch.testing.assert_close(back, cfgs, rtol=0, atol=1e-11)
                torch.testing.assert_close(logdet_inverse, -logdet, rtol=0, atol=1e-10)

                inverse, _, logdet_inverse = self.layer.inverse(cfgs, self.pars, None, beta=self.bc, delta_beta=self.delta)
                again, _, logdet = self.layer(inverse, self.pars, None, beta=self.bc, delta_beta=self.delta)
                torch.testing.assert_close(again, cfgs, rtol=0, atol=1e-11)
                torch.testing.assert_close(logdet, -logdet_inverse, rtol=0, atol=1e-10)

    def test_the_reverse_parameter_step_is_not_the_inverse(self):
        cfgs = self.configs["smooth"]
        with torch.no_grad():
            forward, _, logdet = self.layer(cfgs, self.pars, None, beta=self.bc, delta_beta=self.delta)
            reverse, _, logdet_reverse = self.layer(forward, self.pars, None, beta=self.bc + self.delta, delta_beta=-self.delta)
        self.assertGreater((reverse - cfgs).abs().max().item(), 1e-6)
        self.assertGreater((logdet + logdet_reverse).abs().max().item(), 1e-6)

    def test_unconverged_inverse_raises(self):
        with torch.no_grad():
            forward, _, _ = self.layer(self.configs["haar"], self.pars, None, beta=self.bc, delta_beta=self.delta)
            with self.assertRaisesRegex(RuntimeError, "did not converge"):
                self.layer.inverse(forward, self.pars, None, beta=self.bc, delta_beta=self.delta, max_iter=1)


def _perturb(module, scale=0.05):
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(scale * torch.randn_like(parameter))


class FlowInverseTests(unittest.TestCase):
    def assertRoundTrips(self, forward, inverse, cfgs, atol=1e-11, logdet_atol=1e-10):
        with torch.no_grad():
            out, logdet = forward(cfgs)
            self.assertGreater((out - cfgs).abs().max().item(), 1e-4)
            back, logdet_inverse = inverse(out)
            torch.testing.assert_close(back, cfgs, rtol=0, atol=atol)
            torch.testing.assert_close(logdet_inverse, -logdet, rtol=0, atol=logdet_atol)

            pre, logdet_inverse = inverse(cfgs)
            again, logdet = forward(pre)
            torch.testing.assert_close(again, cfgs, rtol=0, atol=atol)
            torch.testing.assert_close(logdet, -logdet_inverse, rtol=0, atol=logdet_atol)

    def test_full_lattice_smearing(self):
        torch.manual_seed(2)
        pars = _flow_pars("hyper-smearing")
        pars.mask = create_mask(2, 4, 4)
        layer = CouplingLayer(pars, torch.zeros(pars.rho_shape, dtype=torch.double), time=1.0)
        _perturb(layer.smearing.hyper_smearing)
        cfgs = sun.SUN_haar((2, 2, 4, 4, 3), dtype=torch.cdouble)
        beta = torch.tensor([0.3, 0.8], dtype=torch.double)
        self.assertRoundTrips(
            lambda x: layer(x, pars, None, beta=beta)[::2],
            lambda x: layer.inverse(x, pars, None, beta=beta)[::2],
            cfgs,
        )

    def test_full_lattice_smearing_with_a_given_rho(self):
        torch.manual_seed(7)
        cfgs = sun.SUN_haar((2, 2, 4, 4, 3), dtype=torch.cdouble)
        for functional in (False, True):
            with self.subTest(rho_functional=functional):
                pars = _flow_pars("smearing", use_rho_functional=functional)
                pars.mask = create_mask(2, 4, 4)
                rho = 0.05 * torch.rand(pars.rho_shape, dtype=torch.double)
                layer = CouplingLayer(pars, rho, time=1.0)
                self.assertRoundTrips(
                    lambda x: layer(x, pars, rho)[::2], lambda x: layer.inverse(x, pars, rho)[::2], cfgs
                )

    def test_full_lattice_residual_with_quadratic_terms_and_frozen_convolutions(self):
        torch.manual_seed(3)
        for architecture, n_pt, mask_mod in (("hyper-residual", 0, 2), ("residual", 1, 2), ("hyper-residual", 1, 4)):
            with self.subTest(architecture=architecture, n_pt=n_pt, mask_mod=mask_mod):
                pars = _flow_pars(architecture, n_pt=n_pt)
                pars.residual_quadratic = True
                pars.residual_mask_mod = mask_mod
                flow = ResidualNormalizingFlows(pars)
                _perturb(flow, 0.05)
                mask = create_mask(2, 4, 4)
                cfgs = sun.SUN_haar((2, 2, 4, 4, 3), dtype=torch.cdouble)
                beta = torch.tensor([0.3, 0.8], dtype=torch.double)
                delta = torch.tensor([0.5, -0.5], dtype=torch.double)
                self.assertRoundTrips(
                    lambda x: flow(x, mask, beta=beta, delta_beta=delta)[::2],
                    lambda x: flow.inverse(x, mask, beta=beta, delta_beta=delta)[::2],
                    cfgs,
                )

    def test_cnf_inverse_is_backward_integration_up_to_the_cf4_error(self):
        torch.manual_seed(4)
        field = CloverChairLinkDrift(3, hidden_dim=4, depth=1, loop_max=1).to(dtype=torch.double)
        _perturb(field)
        cfgs = sun.SUN_haar((2, 3, 4, 4, 4, 3), dtype=torch.cdouble)
        t = torch.tensor([0.2, 0.6], dtype=torch.double)
        delta = torch.tensor([0.3, -0.3], dtype=torch.double)
        errors = []
        with torch.no_grad():
            for n_steps in (2, 4):
                layer = CNFProtocolLayer(field, n_steps=n_steps)
                out, logdet = layer(cfgs, t, delta)
                back, logdet_inverse = layer.inverse(out, t, delta)
                errors.append(((back - cfgs).abs().max().item(), (logdet + logdet_inverse).abs().max().item()))
        # forward and back differ from the identity at fourth order in the step (fifth per step)
        for coarse, fine in zip(*errors):
            self.assertGreater(coarse / fine, 8.0)

    def test_neorex_protocol_layers(self):
        flags = ("--cpu --algorithm neorex --D 4 --T 4 --L 6 --N 3 --beta 6 --defect-size 1 --time-slice 2 "
                 "--space-slice 2 --parameter-min 0 --parameter-max 1 --batch-size 2").split()
        layers = {
            "smearing": ["--nf-layer-type", "smearing", "--use-hyper-smearing", "--hyper-smearing-mode", "per_link"],
            "residual": ["--nf-layer-type", "residual", "--residual-quadratic"],
        }
        cfgs = sun.SUN_haar((2, 4, 4, 6, 6, 6, 3), dtype=torch.cdouble)
        bc = torch.tensor([0.2, 0.7], dtype=torch.double)
        delta = torch.tensor([0.5, -0.5], dtype=torch.double)
        for name, extra in layers.items():
            with self.subTest(layer=name):
                torch.manual_seed(5)
                with mock.patch.object(sys, "argv", ["main_pt_obc", *flags, *extra]):
                    args = PT_OBC.parse_args()
                defect = Defect(args.D, args.T, args.L, args.defect_size, args.time_slice, args.space_slice)
                pars = PT_OBC._build_neorex_flow_pars(args, torch.device("cpu"), create_mask(args.D, args.T, args.L), defect)
                cls = PT_OBC.ResidualDefectSNFProtocolLayer if name == "residual" else PT_OBC.DefectSNFProtocolLayer
                layer = cls(pars)
                _perturb(layer)
                self.assertRoundTrips(lambda x: layer(x, bc, delta), lambda x: layer.inverse(x, bc, delta), cfgs)

        with self.subTest(layer="cnf"):
            torch.manual_seed(6)
            field, _, _ = build_cnf(_defect_config(batch_size=2))
            _perturb(field)
            layer = CNFProtocolLayer(field, n_steps=4)
            # backward integration: inverse up to the CF4 error, and the logdet is a quadrature
            self.assertRoundTrips(
                lambda x: layer(x, bc, delta), lambda x: layer.inverse(x, bc, delta), cfgs, atol=1e-6, logdet_atol=1e-4
            )


class ClassModeTests(unittest.TestCase):
    """Links with no classified staple are not transformed, so the map is the identity on them."""

    MODE = "class"

    def setUp(self):
        torch.manual_seed(8)
        self.field, self.pars, _ = build_cnf(_defect_config(batch_size=2, defect_size=2, L=7, hyper_smearing_mode=self.MODE))
        _perturb(self.field)
        self.layer = DefectCouplingLayer(self.pars, torch.zeros(self.pars.rho_shape, dtype=torch.double), time=1.0)
        self.layer.smearing.hyper_smearing.load_state_dict(self.field.smearing.hyper_smearing.state_dict())
        self.cfgs = _smooth_links((2, 4, 4, 7, 7, 7), 0.3)
        self.bc = torch.tensor([0.2, 0.7], dtype=torch.double)

    def test_only_classified_links_move_and_the_map_is_the_identity_at_zero_step(self):
        hyper = self.layer.smearing.hyper_smearing
        smeared = self.pars.smeared_defect_mask[-1].squeeze(0).squeeze(-1).squeeze(-1) > 0
        classified = hyper.classified_links() & smeared
        self.assertGreater(int(classified.sum()), 0)
        self.assertGreater(int(smeared.sum()), int(classified.sum()))
        with torch.no_grad():
            out, _, logdet = self.layer(self.cfgs, self.pars, None, beta=self.bc, delta_beta=torch.full((2,), 0.3, dtype=torch.double))
            small_out = self.pars.defect.cut_defect(out, buffer=2)
            small_in = self.pars.defect.cut_defect(self.cfgs, buffer=2)
            moved = ((small_out - small_in).abs().amax((-1, -2)) > 0).any(0)
            self.assertTrue(torch.equal(moved, classified))
            self.assertTrue(torch.isfinite(logdet).all())

            same, _, logdet = self.layer(self.cfgs, self.pars, None, beta=self.bc, delta_beta=torch.zeros(2, dtype=torch.double))
            self.assertTrue(torch.equal(same, self.cfgs))
            self.assertTrue(torch.all(logdet == 0))

    def test_field_and_inverse_follow_the_same_links(self):
        F = self.field.drift(self.cfgs, self.bc)
        velocity = F.abs().sum(-1).amax(0) > 0
        with torch.no_grad():
            out, _, logdet = self.layer(self.cfgs, self.pars, None, beta=self.bc, delta_beta=torch.full((2,), 0.3, dtype=torch.double))
            back, _, logdet_inverse = self.layer.inverse(out, self.pars, None, beta=self.bc, delta_beta=torch.full((2,), 0.3, dtype=torch.double))
        moved = ((out - self.cfgs).abs().amax((-1, -2)) > 0).any(0)
        self.assertTrue(torch.equal(velocity, moved))
        torch.testing.assert_close(back, self.cfgs, rtol=0, atol=1e-11)
        torch.testing.assert_close(logdet_inverse, -logdet, rtol=0, atol=1e-10)

    def test_path_gradient_is_finite(self):
        from neoqcd.cnf import path_gradient_loss

        def action(x, t):
            return -2.0 * sum(p.flatten(1).sum(1).real for p in sun.plaquette_fields(x, 4))

        result = path_gradient_loss(self.field, action, self.cfgs, self.bc)
        result["loss"].sum().backward()
        grads = [p.grad for p in self.field.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))


class PerSiteModeTests(ClassModeTests):
    """One rho per covered link and staple: the class-mode guarantees, with independent values."""

    MODE = "per_site"

    def test_every_covered_staple_has_its_own_rho(self):
        hyper = self.layer.smearing.hyper_smearing
        active = hyper.active_entries()
        self.assertEqual(hyper.coeffs_per_step, int(active.sum()))
        rho = hyper(beta=0.4, delta_beta=1.0)[0]
        self.assertTrue(torch.all(rho[~active] == 0))
        # After the perturbation, the covered entries take (almost) all different values.
        self.assertGreater(rho[active].unique().numel(), 0.9 * int(active.sum()))


if __name__ == "__main__":
    unittest.main()


class DeferredChecksTests(unittest.TestCase):
    """The series-only forward and the fixed-iteration inverse match the per-block checked paths."""

    def setUp(self):
        torch.manual_seed(5)
        _, self.pars, _ = build_cnf(_defect_config(batch_size=2))
        self.layer = DefectCouplingLayer(self.pars, torch.zeros(self.pars.rho_shape, dtype=torch.double), time=1.0)
        with torch.no_grad():
            for parameter in self.layer.smearing.hyper_smearing.mlp.parameters():
                parameter.add_(0.3 * torch.randn_like(parameter))
        self.cfgs = sun.SUN_haar((2, 4, self.pars.T, self.pars.L, self.pars.L, self.pars.L, 3), dtype=torch.cdouble)
        self.bc = torch.tensor([0.3, 0.6], dtype=torch.double)
        self.delta = torch.full((2,), 1 / 7, dtype=torch.double)

    def test_forward_takes_the_series_branch_once_checked_and_matches_the_plain_path(self):
        from neoqcd import smearing_jacobian as sj
        smearing = self.layer.smearing
        with torch.no_grad():
            fast, _, logdet_fast = self.layer(self.cfgs, self.pars, None, beta=self.bc, delta_beta=self.delta)
            # The plain path: per-block branch choice with its sync, which the scope removes.
            with mock.patch.object(sj.DeferredChecks, "resolve", return_value=(False, 0.0)):
                slow, _, logdet_slow = self.layer(self.cfgs, self.pars, None, beta=self.bc, delta_beta=self.delta)
        torch.testing.assert_close(fast, slow, rtol=0, atol=0)
        torch.testing.assert_close(logdet_fast, logdet_slow, rtol=0, atol=0)
        # Inside a scope no sync happens: resolve is the only place that reads the device.
        with torch.no_grad(), sj.DeferredChecks() as checks, mock.patch("torch.Tensor.all", side_effect=AssertionError("sync")):
            smearing._forward_blocks(self.pars.defect.cut_defect(self.cfgs, buffer=2), self.pars.small_mask[-1],
                                     None, dmasking=True, dmask=self.pars.smeared_defect_mask[-1], beta=self.bc, delta_beta=self.delta)
        self.assertIsNotNone(checks.c1_max)
        self.assertTrue(checks.resolve()[0])

    def test_fixed_iteration_inverse_matches_the_adaptive_one(self):
        with torch.no_grad():
            forward, _, logdet = self.layer(self.cfgs, self.pars, None, beta=self.bc, delta_beta=self.delta)
            # Haar links and a large rho need more than the production default of 10 iterations.
            fixed, _, logdet_fixed = self.layer.inverse(forward, self.pars, None, beta=self.bc, delta_beta=self.delta,
                                                        fixed_iterations=24)
            stats_fixed = dict(self.layer.smearing.last_inverse_stats)
            adaptive, _, logdet_adaptive = self.layer.inverse(forward, self.pars, None, beta=self.bc, delta_beta=self.delta,
                                                              fixed_iterations=0)
            stats_adaptive = dict(self.layer.smearing.last_inverse_stats)
        self.assertEqual(set(stats_fixed["iterations"]), {24})
        self.assertLessEqual(stats_fixed["residual"], 1e-12)
        self.assertNotEqual(set(stats_adaptive["iterations"]), {24})
        torch.testing.assert_close(fixed, adaptive, rtol=0, atol=1e-12)
        torch.testing.assert_close(fixed, self.cfgs, rtol=0, atol=1e-11)
        torch.testing.assert_close(logdet_fixed, logdet_adaptive, rtol=0, atol=1e-11)
        torch.testing.assert_close(logdet_fixed, -logdet, rtol=0, atol=1e-10)

    def test_a_failed_check_falls_back_to_the_adaptive_inverse(self):
        from neoqcd import smearing_jacobian as sj
        with torch.no_grad():
            forward, _, _ = self.layer(self.cfgs, self.pars, None, beta=self.bc, delta_beta=self.delta)
            with mock.patch.object(sj.DeferredChecks, "resolve", return_value=(True, 1.0)):
                back, _, _ = self.layer.inverse(forward, self.pars, None, beta=self.bc, delta_beta=self.delta)
        self.assertNotEqual(set(self.layer.smearing.last_inverse_stats["iterations"]), {10})
        torch.testing.assert_close(back, self.cfgs, rtol=0, atol=1e-11)
        # max_iter caps the fixed count too, so an impossible budget still raises through the fallback.
        with torch.no_grad(), self.assertRaisesRegex(RuntimeError, "did not converge"):
            self.layer.inverse(forward, self.pars, None, beta=self.bc, delta_beta=self.delta, max_iter=1)
