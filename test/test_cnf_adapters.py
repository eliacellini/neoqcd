from types import SimpleNamespace
import unittest

import torch

import neoqcd.sun_utils as sun
from neoqcd.cnf_adapters import make_existing_drift
from neoqcd.cnf import path_gradient_loss


def _flow_pars(architecture, *, use_rho_functional=False, n_pt=0):
    residual = architecture in {"residual", "hyper-residual"}
    hyper = architecture in {"hyper-smearing", "hyper-residual"}
    return SimpleNamespace(
        N=3,
        D=2,
        device=torch.device("cpu"),
        rho_shape_type=0,
        rho_shape=(1, 4, 1, 1, 1),
        jac_shape=(2, 2, 9, 9),
        use_nn=False,
        use_rho_functional=use_rho_functional,
        F=5,
        scale_fac=0.01,
        hidden_time_mlp=8,
        fit_train=0,
        use_hyper_smearing=(architecture == "hyper-smearing"),
        smearing_steps_per_layer=1,
        explicit_backward=False,
        hyper_smearing_mode="shared",
        hyper_normalize_by_nstep=False,
        hyper_rho_eps=0.0,
        hyper_scale_by_delta=False,
        hyper_rho_max=0.0,
        hyper_rho_init=0.01,
        hyper_time_embedding_dim=4,
        hyper_hidden_dim=8,
        hyper_depth=1,
        hyper_activation="silu",
        nstep=2,
        residual_parameterization="hyper" if architecture == "hyper-residual" else "static",
        residual_include_imag=True,
        residual_quadratic=False,
        residual_coeff_max=0.0,
        residual_n_pt=n_pt,
        residual_mask_mod=2,
        residual_coeff_init=0.01,
    )


class ExistingDriftTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.cfgs = sun.SUN_haar((1, 2, 2, 2, 3), dtype=torch.cdouble)

    def test_existing_architectures_produce_trainable_su3_fields(self):
        cfgs = sun.SUN_haar((2, 2, 2, 2, 3), dtype=torch.cdouble)
        for architecture in ("smearing", "hyper-smearing", "residual", "hyper-residual"):
            with self.subTest(architecture=architecture):
                field = make_existing_drift(architecture, _flow_pars(architecture), rho_init=0.01)
                coeffs = field.drift(cfgs, torch.tensor([0.4, 0.7], dtype=torch.float64))
                self.assertEqual(tuple(coeffs.shape), (2, 2, 2, 2, 8))
                self.assertTrue(torch.isfinite(coeffs).all())
                coeffs.square().mean().backward()
                self.assertTrue(any(p.grad is not None for p in field.parameters()))

    def test_residual_field_returns_finite_divergence(self):
        field = make_existing_drift("residual", _flow_pars("residual"))
        coeffs, divergence = field(self.cfgs, torch.tensor(0.7, dtype=torch.float64))
        self.assertEqual(tuple(coeffs.shape), (1, 2, 2, 2, 8))
        self.assertEqual(tuple(divergence.shape), (1,))
        self.assertTrue(torch.isfinite(coeffs).all())
        self.assertTrue(torch.isfinite(divergence).all())
        divergence.sum().backward()
        self.assertTrue(any(p.grad is not None for p in field.parameters()))

    def test_rho_functional_and_frozen_convolutions_keep_gradients(self):
        functional = make_existing_drift(
            "smearing", _flow_pars("smearing", use_rho_functional=True)
        )
        self.assertTrue((functional._smearing_rho(0.3, self.cfgs) >= 0).all())
        functional.drift(self.cfgs, torch.tensor([0.3], dtype=torch.float64)).square().mean().backward()
        self.assertTrue(any(p.grad is not None for p in functional.parameters()))

        cfgs = sun.SUN_haar((2, 2, 2, 2, 3), dtype=torch.cdouble)

        def action(value, t):
            plaquettes = sun.plaquette_fields(value, 2)
            return -t * sum(p.flatten(1).sum(1) for p in plaquettes)

        for t in (0.3, 0.7):
            functional.zero_grad(set_to_none=True)
            result = path_gradient_loss(
                functional, action, cfgs, torch.tensor(t, dtype=torch.float64)
            )
            self.assertTrue(result["loss"].requires_grad)
            result["loss"].mean().backward()
            self.assertTrue(all(p.requires_grad for p in functional.parameters()))
            self.assertTrue(any(p.grad is not None for p in functional.parameters()))

        residual = make_existing_drift(
            "residual", _flow_pars("residual", n_pt=1)
        )
        residual.drift(self.cfgs, torch.tensor(0.7, dtype=torch.float64)).square().mean().backward()
        self.assertIsNotNone(residual.residual.conv_eta.grad)

    def test_path_gradient_is_available_for_each_existing_architecture(self):
        def action(cfgs, t):
            plaquettes = sun.plaquette_fields(cfgs, 2)
            value = sum(p.flatten(1).sum(1) for p in plaquettes)
            return -t * value

        for architecture in ("smearing", "hyper-smearing", "residual", "hyper-residual"):
            with self.subTest(architecture=architecture):
                field = make_existing_drift(architecture, _flow_pars(architecture), rho_init=0.01)
                result = path_gradient_loss(
                    field, action, self.cfgs, torch.tensor([0.5], dtype=torch.float64)
                )
                self.assertEqual(result["loss"].shape, (1,))
                self.assertEqual(result["xi"].shape, (1,))
                self.assertTrue(torch.isfinite(result["loss"]).all())
                self.assertTrue(torch.isfinite(result["pinn"]).all())
                result["loss"].sum().backward()
                self.assertTrue(any(p.grad is not None for p in field.parameters()))


if __name__ == "__main__":
    unittest.main()


def _defect_config(**overrides):
    config = dict(
        architecture="defect-hyper-smearing", domain="defect", D=4, T=4, L=6, beta=6.0,
        defect_size=1, time_slice=2, space_slice=2, parameter_min=0.0, parameter_max=1.0,
        batch_size=1, hidden_dim=4, depth=1, rho_init=0.02, residual_n_pt=0,
        residual_mask_mod=2, orsteps=0, hyper_smearing_mode="per_link",
    )
    config.update(overrides)
    return config


class DefectHyperSmearingDriftTests(unittest.TestCase):
    def setUp(self):
        from neoqcd.cnf_io import build_cnf

        torch.manual_seed(5)
        self.field, self.pars, self.defect_mask = build_cnf(_defect_config())
        with torch.no_grad():
            # Make rho depend on the boundary coupling and differ between links.
            for parameter in self.field.smearing.hyper_smearing.mlp.parameters():
                parameter.add_(0.05 * torch.randn_like(parameter))
        self.cfgs = sun.SUN_haar((1, 4, 4, 6, 6, 6, 3), dtype=torch.cdouble)

    def test_velocity_lives_on_the_smeared_defect_links(self):
        F = self.field.drift(self.cfgs, 0.4)
        active = F.abs().sum(-1) > 0
        # 27 temporal links across the defect plane and 2 x 3 x 2^3 spatial ones.
        self.assertEqual(int(active.sum()), 75)
        t, s = 2, 2
        self.assertTrue(active[0, 0, t, s - 1:s + 2, s - 1:s + 2, s - 1:s + 2].all())
        for d in range(1, 4):
            self.assertTrue(active[0, d, t - 1:t + 1, s:s + 2, s:s + 2, s:s + 2].all())

    def test_divergence_matches_link_by_link_derivatives(self):
        t = torch.tensor(0.6, dtype=torch.double)
        F, div = self.field(self.cfgs, t)
        generators = self.field.generators
        expected = torch.zeros((), dtype=torch.double)
        for link in (F[0].abs().sum(-1) > 0).nonzero():
            index = (0, *link.tolist())
            for a in range(8):
                tangent = torch.zeros_like(self.cfgs)
                tangent[index] = 1j * generators[a] @ self.cfgs[index]
                _, dF = torch.func.jvp(lambda U: self.field.drift(U, t), (self.cfgs,), (tangent,))
                expected = expected + dF[index][a]
        torch.testing.assert_close(div[0], expected, rtol=1e-10, atol=1e-12)

    def test_gauge_covariance(self):
        from neoqcd.cnf import coeff_to_matrix

        omega = sun.SUN_haar((1, 4, 6, 6, 6, 3), dtype=torch.cdouble)
        transformed = torch.stack([
            sun.SUN_mul(sun.SUN_mul(omega, self.cfgs[:, mu]),
                        sun.SUN_dagger(torch.roll(omega, shifts=1, dims=1 + mu)))
            for mu in range(4)
        ], dim=1)
        F = coeff_to_matrix(self.field.drift(self.cfgs, 0.3))
        Fg = coeff_to_matrix(self.field.drift(transformed, 0.3))
        expected = sun.SUN_mul(sun.SUN_mul(omega.unsqueeze(1), F), sun.SUN_dagger(omega.unsqueeze(1)))
        torch.testing.assert_close(Fg, expected, rtol=1e-9, atol=1e-11)

    def test_first_order_agrees_with_the_discrete_defect_coupling_layer(self):
        from neoqcd.cnf import apply_velocity
        from neoqcd.smearing import DefectCouplingLayer

        layer = DefectCouplingLayer(self.pars, torch.zeros(self.pars.rho_shape, dtype=torch.double), time=1.0)
        layer.smearing.hyper_smearing.load_state_dict(self.field.smearing.hyper_smearing.state_dict())
        bc = torch.tensor([0.35], dtype=torch.double)
        F, div = self.field(self.cfgs, bc)
        errors = []
        for h in (1e-3, 2e-3):
            with torch.no_grad():
                discrete, _, logdet = layer(self.cfgs, self.pars, None, is_training=False,
                                            beta=bc, delta_beta=torch.tensor([h], dtype=torch.double))
            continuous = apply_velocity(self.cfgs, F, h)
            errors.append(((discrete - continuous).abs().max().item(), (logdet - h * div).abs().max().item()))
            self.assertGreater((continuous - self.cfgs).abs().max().item(), 100 * errors[-1][0])
        # Both differences are second order in the step.
        for coarse, fine in zip(errors[1], errors[0]):
            self.assertAlmostEqual(coarse / fine, 4.0, delta=0.5)

    def test_path_gradient_reaches_the_hypernetwork(self):
        def action(U, t):
            plaquettes = sun.plaquette_fields(U, 4)
            return t * (1.0 - sum(p.real.flatten(1).sum(1) for p in plaquettes) / 3.0)

        cfgs = sun.SUN_haar((2, 4, 4, 6, 6, 6, 3), dtype=torch.cdouble)
        result = path_gradient_loss(self.field, action, cfgs, torch.tensor([0.2, 0.8], dtype=torch.double))
        result["loss"].mean().backward()
        grads = [p.grad for p in self.field.parameters() if p.requires_grad]
        self.assertTrue(grads and all(g is not None and torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(g.abs().sum().item() for g in grads), 0.0)

    def test_patch_geometry_is_validated(self):
        from neoqcd.cnf_io import build_cnf

        for overrides in ({"L": 4}, {"time_slice": 1}, {"space_slice": 4}, {"domain": "beta"}):
            with self.subTest(**overrides), self.assertRaises(ValueError):
                build_cnf(_defect_config(**overrides))
