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


class DefectResidualDriftTests(unittest.TestCase):
    """The defect residual field, with rectangles (one frozen convolution, mod 4) and without."""

    MODE = "shared"
    CASES = {"plaquettes": dict(residual_n_pt=0, residual_mask_mod=2, T=4, L=6, time_slice=2, space_slice=2),
             "rectangles": dict(residual_n_pt=1, residual_mask_mod=4, T=6, L=8, time_slice=3, space_slice=3)}

    def _field(self, case):
        from neoqcd.cnf_io import build_cnf

        torch.manual_seed(9)
        config = _defect_config(architecture="defect-hyper-residual", residual_mode=self.MODE, **self.CASES[case])
        field, pars, _ = build_cnf(config)
        with torch.no_grad():
            for parameter in field.residual.parameters():
                parameter.add_(0.05 * torch.randn_like(parameter))
        cfgs = sun.SUN_haar((1, 4, config["T"], config["L"], config["L"], config["L"], 3), dtype=torch.cdouble)
        return field, pars, cfgs

    def test_velocity_lives_on_the_transformed_defect_links(self):
        for case in self.CASES:
            with self.subTest(case=case):
                field, pars, cfgs = self._field(case)
                self.assertEqual(field.buffer, 2 + self.CASES[case]["residual_n_pt"])
                active = field.drift(cfgs, 0.4).abs().sum(-1) > 0
                self.assertEqual(int(active.sum()), self._moving_links(field, cfgs))

    def _moving_links(self, field, cfgs):
        return 75

    def test_a_link_never_moves_the_velocity_of_its_own_class(self):
        # The divergence perturbs a whole class at once; that is exact only if no link of a
        # class enters the velocity of another link of the same class.
        field, _, cfgs = self._field("rectangles")
        t = torch.tensor(0.6, dtype=torch.double)
        residual, buffer, defect = field.residual, field.buffer, field.defect
        small = defect.cut_defect(cfgs, buffer=buffer)
        # (classes, 1, D, *patch, 1, 1) -> the links of each class, on the full lattice
        mask = residual._runtime_mask(field.small_mask, small, periodic=False) * field.smeared_defect_mask
        mask = torch.stack([residual._active_mask(m) for m in mask])  # the links each class updates
        zero = torch.zeros((1, *cfgs.shape[1:-2], 1), dtype=torch.double)
        generators = field.generators
        with torch.no_grad():
            for m in range(mask.shape[0]):
                in_patch = mask[m, 0, ..., 0, 0].to(torch.double)[None, ..., None]
                members = defect.embedding(in_patch, zero, buffer=buffer)[0, ..., 0] > 0
                links = members.nonzero()
                if len(links) < 2:
                    continue
                index = (0, *links[0].tolist())
                for a in (0, 5):
                    tangent = torch.zeros_like(cfgs)
                    tangent[index] = 1j * generators[a] @ cfgs[index]
                    _, dF = torch.func.jvp(lambda U: field.drift(U, t), (cfgs,), (tangent,))
                    others = members.clone()
                    others[tuple(links[0].tolist())] = False
                    self.assertEqual(float(dF[0][others].abs().max()), 0.0, f"class {m}")

    def test_divergence_matches_link_by_link_derivatives_without_convolutions(self):
        field, _, cfgs = self._field("plaquettes")
        t = torch.tensor(0.6, dtype=torch.double)
        generators = field.generators
        expected = 0.0
        # No autograd graph: hundreds of derivatives kept alive through the weights exhaust memory.
        with torch.no_grad():
            F, div = field(cfgs, t)
            for link in (F[0].abs().sum(-1) > 0).nonzero():
                index = (0, *link.tolist())
                for a in range(8):
                    tangent = torch.zeros_like(cfgs)
                    tangent[index] = 1j * generators[a] @ cfgs[index]
                    _, dF = torch.func.jvp(lambda U: field.drift(U, t), (cfgs,), (tangent,))
                    expected += dF[index][a].item()
        self.assertAlmostEqual(div[0].item(), expected, delta=1e-10 * max(1.0, abs(expected)))

    def test_first_order_agreement_with_the_finite_defect_layer(self):
        from neoqcd.cnf import apply_velocity
        from neoqcd.smearing import ResidualNormalizingFlows

        for case in self.CASES:
            with self.subTest(case=case):
                field, pars, cfgs = self._field(case)
                layer = ResidualNormalizingFlows(pars)
                layer.load_state_dict(field.residual.state_dict())
                bc = torch.tensor([0.35], dtype=torch.double)
                F, div = field(cfgs, bc)
                buffer = field.buffer
                errors = []
                for h in (1e-3, 2e-3):
                    with torch.no_grad():
                        small, _, logdet = layer(
                            pars.defect.cut_defect(cfgs, buffer=buffer), pars.small_mask[-1], dmasking=True,
                            dmask=pars.smeared_defect_mask[-1], beta=bc, delta_beta=torch.tensor([h], dtype=torch.double),
                        )
                    finite = pars.defect.embedding(small, cfgs, buffer=buffer)
                    continuous = apply_velocity(cfgs, F, h)
                    errors.append(((finite - continuous).abs().max().item(), (logdet - h * div).abs().max().item()))
                    self.assertGreater((continuous - cfgs).abs().max().item(), 100 * errors[-1][0])
                for coarse, fine in zip(errors[1], errors[0]):
                    self.assertAlmostEqual(coarse / fine, 4.0, delta=0.5)

    def test_path_gradient_reaches_the_network(self):
        field, _, cfgs = self._field("rectangles")

        def action(U, t):
            return -2.0 * sum(p.flatten(1).sum(1).real for p in sun.plaquette_fields(U, 4))

        result = path_gradient_loss(field, action, cfgs.expand(2, *cfgs.shape[1:]).clone(), torch.tensor([0.3, 0.7], dtype=torch.double))
        result["loss"].sum().backward()
        grads = [p.grad for p in field.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertIsNotNone(field.residual.conv_eta.grad)


class DefectResidualClassModeTests(DefectResidualDriftTests):
    """Residual coefficients per defect class: the shared-mode guarantees, on the classified links."""

    MODE = "class"

    def _moving_links(self, field, cfgs):
        smeared = field.smeared_defect_mask[0, ..., 0, 0] > 0
        moving = int((field.residual.classified_links() & smeared).sum())
        self.assertGreater(moving, 0)
        self.assertLess(moving, 75)
        return moving

    def test_finite_layer_moves_only_classified_links_and_inverts(self):
        from neoqcd.smearing import ResidualNormalizingFlows

        for case in self.CASES:
            with self.subTest(case=case):
                field, pars, cfgs = self._field(case)
                layer = ResidualNormalizingFlows(pars)
                layer.load_state_dict(field.residual.state_dict())
                small = pars.defect.cut_defect(cfgs, buffer=field.buffer)
                kwargs = dict(dmasking=True, dmask=pars.smeared_defect_mask[-1],
                              beta=torch.tensor([0.35], dtype=torch.double), delta_beta=torch.tensor([0.2], dtype=torch.double))
                with torch.no_grad():
                    out, _, logdet = layer(small, pars.small_mask[-1], **kwargs)
                    back, _, logdet_inverse = layer.inverse(out, pars.small_mask[-1], **kwargs)
                moved = ((out - small).abs().amax((-1, -2)) > 0)[0]
                smeared = pars.smeared_defect_mask[-1][0, ..., 0, 0] > 0
                self.assertTrue(torch.equal(moved, layer.classified_links() & smeared))
                torch.testing.assert_close(back, small, rtol=0, atol=1e-11)
                torch.testing.assert_close(logdet_inverse, -logdet, rtol=0, atol=1e-10)

    def test_coefficients_differ_between_classes(self):
        field, _, _ = self._field("plaquettes")
        lin, quad, _ = field.residual._coefficients(torch.tensor([0.4]), torch.ones(1), 1, torch.double, "cpu")
        self.assertEqual(tuple(lin.shape), (1, 1, field.residual.n_classes, field.residual.n_features))
        self.assertGreater(float((lin[0, 0, 0] - lin[0, 0, 1]).abs().max()), 0.0)


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
