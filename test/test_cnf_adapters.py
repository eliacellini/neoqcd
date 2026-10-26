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
