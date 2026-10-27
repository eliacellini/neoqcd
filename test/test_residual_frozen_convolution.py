import importlib.util
from pathlib import Path
import unittest

import torch

import neoqcd.sun_utils as sun
from neoqcd.flow import FlowPars
from neoqcd.smearing import ResidualNormalizingFlows
from neoqcd.utils import create_mask

from test.test_residual_normalizing_flows import exact_su3_exp_iQ, max_su3_error, random_su3


ROOT = Path(__file__).resolve().parents[1]
MAIN = None
try:
    spec = importlib.util.spec_from_file_location("main_nemcmc_snf_smoke", ROOT / "main" / "main_nemcmc_snf.py")
    MAIN = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(MAIN)
except ModuleNotFoundError:
    MAIN = None


def make_layer(D=2, T=4, L=4, n_pt=0, mask_mod=2, coeff_init=1e-3, quadratic=False):
    mask = create_mask(D, T, L)
    pars = FlowPars(
        D=D, T=T, L=L, N=3, mask=mask, protocol=[6.0], batch_size=1,
        device=torch.device("cpu"), smearing_steps_per_layer=1,
        residual_parameterization="static", residual_include_imag=True,
        residual_quadratic=quadratic, residual_coeff_init=coeff_init,
        residual_n_pt=n_pt, residual_mask_mod=mask_mod,
    )
    return ResidualNormalizingFlows(pars), pars


def local_gauge_transform(cfgs, gauge=None):
    D = cfgs.shape[1]
    lattice_shape = cfgs.shape[2:-2]
    g = random_su3((cfgs.shape[0],) + lattice_shape) if gauge is None else gauge
    out = cfgs.clone()
    for mu in range(D):
        g_forward = torch.roll(g, 1, dims=1 + mu)
        out[:, mu] = sun.SUN_mul(g, sun.SUN_mul(cfgs[:, mu], sun.SUN_dagger(g_forward)))
    return out, g


class ResidualFrozenConvolutionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12345)

    def test_default_state_dict_has_legacy_parameters_only(self):
        layer, legacy_pars = make_layer()
        delattr(legacy_pars, "residual_n_pt")
        delattr(legacy_pars, "residual_mask_mod")
        legacy = ResidualNormalizingFlows(legacy_pars)
        self.assertNotIn("conv_eta", layer.state_dict())
        self.assertEqual(list(layer.state_dict()), list(legacy.state_dict()))

    def test_all_extended_variants_are_finite_su3_and_have_eta_gradients(self):
        for n_pt, mask_mod in ((1, 2), (2, 2), (1, 4), (2, 4)):
            # In D=2 the first mod-2 convolution is degenerate; use D=4 once
            # to exercise its nonzero gradient while keeping the lattice tiny.
            dims = (4, 2, 2) if (n_pt, mask_mod) == (1, 2) else (2, 4, 4)
            layer, pars = make_layer(*dims, n_pt=n_pt, mask_mod=mask_mod)
            x = random_su3(pars.init_shape[:-1])
            y, _, logdet = layer(x, pars.mask, beta=torch.tensor([6.0]), delta_beta=torch.tensor([0.02]))
            self.assertTrue(torch.isfinite(y).all().item())
            self.assertTrue(torch.isfinite(logdet).all().item())
            self.assertLess(max_su3_error(y), 5e-12)
            loss = (y - x).abs().square().sum() + logdet.square().sum()
            loss.backward()
            self.assertTrue(torch.isfinite(layer.conv_eta.grad).all().item())
            self.assertGreater(layer.conv_eta.grad.abs().max().item(), 1e-12)

            y0, _, dlog0 = layer(
                x, pars.mask, beta=torch.tensor([6.0]), delta_beta=torch.tensor([0.0])
            )
            self.assertLess((y0 - x).abs().max().item(), 1e-12)
            self.assertLess(dlog0.abs().max().item(), 1e-12)

    def test_extended_loop_reaches_a_distant_frozen_link(self):
        base, pars0 = make_layer(T=4, L=4, n_pt=0, mask_mod=4)
        extended, pars1 = make_layer(T=4, L=4, n_pt=1, mask_mod=4)
        x = random_su3(pars0.init_shape[:-1])
        x_changed = x.clone()
        x_changed[0, 0, 0, 2] = sun.SUN_mul(random_su3(()), x_changed[0, 0, 0, 2])

        base_staple = base._oriented_staples(x, 0)
        base_changed = base._oriented_staples(x_changed, 0)
        self.assertLess((base_staple[0, :, 0, 0] - base_changed[0, :, 0, 0]).abs().max().item(), 1e-12)

        runtime_mask = extended._runtime_mask(pars1.mask, x)
        current_mask = runtime_mask[0, :]
        frozen = (1.0 - current_mask) * x
        frozen_changed = (1.0 - current_mask) * x_changed
        convolved = extended._frozen_convolution(frozen, 0, 0)
        convolved_changed = extended._frozen_convolution(frozen_changed, 0, 0)
        staple = extended._oriented_staples(convolved, 0)
        staple_changed = extended._oriented_staples(convolved_changed, 0)
        self.assertGreater((staple[0, :, 0, 0] - staple_changed[0, :, 0, 0]).abs().max().item(), 1e-10)

    def test_frozen_map_is_independent_of_other_active_links(self):
        layer, pars = make_layer(n_pt=2, mask_mod=4)
        x = random_su3(pars.init_shape[:-1])
        runtime_mask = layer._runtime_mask(pars.mask, x)
        current_mask = runtime_mask[0, :]
        self.assertTrue(bool(current_mask[0, 0, 0, 0].item()))
        self.assertTrue(bool(current_mask[0, 0, 1, 3].item()))
        x_changed = x.clone()
        x_changed[0, 0, 1, 3] = sun.SUN_mul(random_su3(()), x_changed[0, 0, 1, 3])
        kwargs = dict(beta=torch.tensor([6.0]), delta_beta=torch.tensor([0.02]), current_mask=current_mask)
        y, _, _ = layer.residual_update_mu(x, 0, **kwargs)
        y_changed, _, _ = layer.residual_update_mu(x_changed, 0, **kwargs)
        self.assertLess((y[0, 0, 0] - y_changed[0, 0, 0]).abs().max().item(), 1e-11)

    def test_local_gauge_equivariance_and_logdet_invariance(self):
        layer, pars = make_layer(n_pt=1, mask_mod=4)
        x = random_su3(pars.init_shape[:-1])
        x_g, gauge = local_gauge_transform(x)
        y, _, logdet = layer(x, pars.mask, beta=torch.tensor([6.0]), delta_beta=torch.tensor([0.02]))
        y_g, _, logdet_g = layer(x_g, pars.mask, beta=torch.tensor([6.0]), delta_beta=torch.tensor([0.02]))
        expected, _ = local_gauge_transform(y, gauge)
        self.assertLess((y_g - expected).abs().max().item(), 2e-10)
        self.assertLess((logdet_g - logdet).abs().max().item(), 2e-10)

    def test_complete_active_jacobian_and_small_fixed_point_roundtrip(self):
        layer, pars = make_layer(n_pt=1, mask_mod=4, coeff_init=2e-2, quadratic=True)
        with torch.no_grad():
            layer.static_quadratic.fill_(2e-2)
        x = random_su3(pars.init_shape[:-1])
        current_mask = layer._runtime_mask(pars.mask, x)[0, :]
        kwargs = dict(beta=torch.tensor([6.0]), delta_beta=torch.tensor([0.02]), current_mask=current_mask)
        y, detjac, analytic = layer.residual_update_mu(x, 0, **kwargs)
        active = current_mask[0, 0].squeeze().bool()
        sites = [tuple(site.tolist()) for site in torch.nonzero(active, as_tuple=False)]
        self.assertEqual(len(sites), 4)
        eps = 1e-5
        with torch.no_grad():
            finite = torch.empty((8 * len(sites), 8 * len(sites)), dtype=x.real.dtype)
            generators = layer.generators.to(dtype=x.dtype)
            for col_site, site_in in enumerate(sites):
                for generator_index, generator in enumerate(generators):
                    x_plus = x.clone()
                    x_minus = x.clone()
                    eplus = exact_su3_exp_iQ(eps * generator)
                    eminus = exact_su3_exp_iQ(-eps * generator)
                    x_plus[(0, 0) + site_in] = sun.SUN_mul(eplus, x_plus[(0, 0) + site_in])
                    x_minus[(0, 0) + site_in] = sun.SUN_mul(eminus, x_minus[(0, 0) + site_in])
                    yp, _, _ = layer.residual_update_mu(x_plus, 0, **kwargs)
                    ym, _, _ = layer.residual_update_mu(x_minus, 0, **kwargs)
                    for row_site, site_out in enumerate(sites):
                        delta_y = (yp[(0,) + site_out] - ym[(0,) + site_out]) / (2 * eps)
                        finite[8 * row_site:8 * row_site + 8, 8 * col_site + generator_index] = layer._tangent_coordinates(
                            delta_y, y[(0,) + site_out]
                        )
            analytic_block = torch.block_diag(*(analytic[(0,) + site] for site in sites))
            expected_logdet = torch.log(torch.clamp(detjac[0][active], min=1e-12)).sum()
            _, numeric_logdet = torch.linalg.slogdet(finite)
        self.assertGreater((finite - torch.eye(finite.shape[0], dtype=finite.dtype)).abs().max().item(), 1e-6)
        self.assertLess((analytic_block - finite).abs().max().item(), 1e-8)
        self.assertLess((expected_logdet - numeric_logdet).abs().item(), 1e-8)

        target = y[0, 0, 0]
        state = x.clone()
        state[0, 0, 0, 0] = target
        errors = []
        for _ in range(5):
            forward, _, _ = layer.residual_update_mu(state, 0, **kwargs)
            delta = sun.SUN_mul(forward[0, 0, 0], sun.SUN_dagger(state[0, 0, 0, 0]))
            state[0, 0, 0, 0] = sun.SUN_mul(sun.SUN_dagger(delta), target)
            errors.append((forward[0, 0, 0] - target).abs().max().item())
        self.assertLessEqual(errors[-1], errors[0])
        self.assertLess((state[0, 0, 0, 0] - x[0, 0, 0, 0]).abs().max().item(), 1e-10)

    def test_invalid_crop_and_periodic_masks_are_rejected(self):
        layer, pars = make_layer(n_pt=1, mask_mod=4)
        x = random_su3(pars.init_shape[:-1])
        # On the defect patch a convolution needs a three-site buffer; these masks carry the default two.
        with self.assertRaisesRegex(ValueError, "patch_buffer >= 3"):
            layer(x, pars.mask, dmasking=True, dmask=torch.ones_like(x[..., :1, :1]), delta_beta=torch.tensor([0.02]))

        bad_layer, bad_pars = make_layer(T=2, L=4, n_pt=0, mask_mod=4)
        bad_x = random_su3(bad_pars.init_shape[:-1])
        with self.assertRaisesRegex(ValueError, "divisible"):
            bad_layer(bad_x, bad_pars.mask, delta_beta=torch.tensor([0.02]))

    @unittest.skipUnless(MAIN is not None, "standalone main dependencies unavailable")
    def test_cli_builder_and_checkpoint_semantics(self):
        base = MAIN.parse_args(
            ["--algorithm", "snf", "--phase", "train", "--domain", "beta", "--cpu",
             "--D", "2", "--T", "4", "--L", "4", "--batch-size", "1",
             "--nf-architecture", "residual", "--cfg-cache", "off"]
        )
        base_semantics = MAIN._checkpoint_semantics(base, [5.8])
        self.assertNotIn("residual_n_pt", base_semantics)
        self.assertNotIn("residual_mask_mod", base_semantics)

        extended = MAIN.parse_args(
            ["--algorithm", "snf", "--phase", "train", "--domain", "beta", "--cpu",
             "--D", "2", "--T", "4", "--L", "4", "--batch-size", "1",
             "--nf-architecture", "residual", "--residual-n-pt", "2", "--residual-mask-mod", "4",
             "--cfg-cache", "off"]
        )
        flow, _ = MAIN._build_flow_pars(extended, torch.device("cpu"), [5.8], 1)
        self.assertEqual((flow.residual_n_pt, flow.residual_mask_mod), (2, 4))
        adapter = MAIN._make_snf_layer(extended, flow)
        self.assertEqual(adapter.layer.residual_flow.n_pt, 2)
        x = random_su3(flow.init_shape[:-1])
        raw_y, _, raw_logdet = adapter.layer.residual_flow(
            x, flow.mask, beta=torch.tensor([5.8]), delta_beta=torch.tensor([0.02])
        )
        adapter_y, adapter_logdet = adapter(x, torch.tensor([5.8]), torch.tensor([0.02]))
        self.assertTrue(torch.allclose(raw_y, adapter_y))
        self.assertTrue(torch.allclose(raw_logdet, adapter_logdet))
        clone = MAIN._make_snf_layer(extended, flow)
        clone.load_state_dict(adapter.state_dict())
        self.assertEqual(list(clone.state_dict()), list(adapter.state_dict()))
        semantics = MAIN._checkpoint_semantics(extended, [5.8])
        self.assertEqual((semantics["residual_n_pt"], semantics["residual_mask_mod"]), (2, 4))

        for extra, message in (
            (("--residual-n-pt", "-1"), "non-negative"),
            (("--nf-architecture", "smearing", "--residual-n-pt", "1"), "residual"),
            (("--residual-mask-mod", "4", "--T", "2"), "multiples"),
        ):
            argv = ["--algorithm", "snf", "--phase", "train", "--domain", "beta", "--cpu",
                    "--D", "2", "--T", "4", "--L", "4", "--batch-size", "1",
                    "--nf-architecture", "residual", "--cfg-cache", "off"] + list(extra)
            with self.assertRaisesRegex(ValueError, message):
                MAIN.parse_args(argv)

        # On the defect a convolution widens the patch buffer to three sites.
        defect = ["--algorithm", "snf", "--phase", "train", "--domain", "defect", "--cpu", "--D", "4",
                  "--batch-size", "1", "--nf-architecture", "residual", "--defect-size", "2",
                  "--residual-n-pt", "1", "--cfg-cache", "off"]
        with self.assertRaisesRegex(ValueError, "T >= 6"):
            MAIN.parse_args(defect + ["--T", "4", "--L", "6", "--time-slice", "2", "--space-slice", "2"])
        args = MAIN.parse_args(defect + ["--T", "6", "--L", "8", "--time-slice", "3", "--space-slice", "3"])
        flow, _ = MAIN._build_flow_pars(args, torch.device("cpu"), [0.5], 1)
        self.assertEqual(flow.patch_buffer, 3)
        self.assertEqual(tuple(flow.small_mask[-1].shape[3:7]), (6, 8, 8, 8))

    @unittest.skipUnless(MAIN is not None, "standalone main dependencies unavailable")
    def test_defect_patch_matches_the_periodic_lattice_restricted_to_the_same_links(self):
        from neoqcd.theory import build_nf_changed_link_mask

        # Patch origin at (0, 0, 0, 0): the patch colouring equals the lattice one.
        for mask_mod in (2, 4):
            with self.subTest(mask_mod=mask_mod):
                torch.manual_seed(31 + mask_mod)
                args = MAIN.parse_args(
                    ["--algorithm", "snf", "--phase", "train", "--domain", "defect", "--cpu",
                     "--D", "4", "--T", "8", "--L", "8", "--batch-size", "2",
                     "--nf-architecture", "hyper-residual", "--defect-size", "1", "--time-slice", "3",
                     "--space-slice", "3", "--residual-n-pt", "1", "--residual-mask-mod", str(mask_mod),
                     "--cfg-cache", "off"]
                )
                flow, _ = MAIN._build_flow_pars(args, torch.device("cpu"), [0.5], 2)
                self.assertEqual(flow.patch_buffer, 3)
                patch = MAIN._make_snf_layer(args, flow)
                with torch.no_grad():
                    for parameter in patch.parameters():
                        parameter.add_(0.05 * torch.randn_like(parameter))
                full = ResidualNormalizingFlows(flow)
                full.load_state_dict(patch.layer.state_dict())

                x = random_su3((2, 4, 8, 8, 8, 8))
                bc = torch.tensor([0.3, 0.6], dtype=torch.float64)
                delta = torch.tensor([0.25, -0.25], dtype=torch.float64)
                changed = build_nf_changed_link_mask(flow, device=torch.device("cpu"))
                dmask = changed.to(torch.float64).view(1, *changed.shape, 1, 1)
                with torch.no_grad():
                    y_patch, logdet_patch = patch(x, bc, delta)
                    y_full, _, logdet_full = full(x, flow.mask, dmasking=True, dmask=dmask, beta=bc, delta_beta=delta)
                    back, logdet_back = patch.layer.inverse(
                        flow.defect.cut_defect(y_patch, buffer=3), flow.small_mask[-1], dmasking=True,
                        dmask=flow.smeared_defect_mask[-1], beta=bc, delta_beta=delta,
                    )[::2]
                self.assertGreater((y_patch - x).abs().max().item(), 1e-3)
                torch.testing.assert_close(y_patch, y_full, rtol=0, atol=1e-12)
                torch.testing.assert_close(logdet_patch, logdet_full, rtol=0, atol=1e-11)
                torch.testing.assert_close(back, flow.defect.cut_defect(x, buffer=3), rtol=0, atol=1e-11)
                torch.testing.assert_close(logdet_back, -logdet_patch, rtol=0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
