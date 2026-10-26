import unittest
from itertools import product
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import warnings

import torch

import neoqcd.sun_utils as sun
from neoqcd.distributedPT import ParallelTempering, _neorex_checkpoint_state
from neoqcd.mcmc import HBOR, NEMCMC_update
from neoqcd.smearing import Smearing, _su3_generators
from neoqcd.smearing_jacobian import LOGDET_VERSION, SmearingStep, StepQuantities, smearing_step
from neoqcd.theory import Wilson_action
from neoqcd.utils import Defect, create_mask, load_optimizer, load_weights, save, write


class ExactSmearingTest(unittest.TestCase):
    def test_large_unbatched_step_and_empty_batch(self):
        U = torch.eye(3, dtype=torch.cdouble)
        C = 4.0j * torch.diag(torch.tensor([2.0, -1.0, -1.0], dtype=torch.double)).to(torch.cdouble)
        updated, logdet = smearing_step(U, C)
        Q = 0.5j * ((C @ U.mH).mH - C @ U.mH)
        self.assertTrue(torch.isfinite(updated).all().item())
        self.assertTrue(torch.isfinite(logdet).item())
        self.assertLess((updated - torch.matrix_exp(1j * Q) @ U).abs().max().item(), 1e-12)

        empty_u = U.expand(0, 3, 3).clone()
        empty_c = C.expand(0, 3, 3).clone()
        empty_up, empty_logdet = smearing_step(empty_u, empty_c)
        self.assertEqual(tuple(empty_up.shape), (0, 3, 3))
        self.assertEqual(tuple(empty_logdet.shape), (0,))

    def test_degenerate_spectrum_autograd_matches_explicit_backward(self):
        U0 = torch.eye(3, dtype=torch.cdouble).unsqueeze(0)
        for scale in (1.0, 2.0, 8.0):
            C0 = (1j * torch.diag(torch.tensor([2.0, -1.0, -1.0], dtype=torch.double)) * scale)
            C0 = C0.to(torch.cdouble).unsqueeze(0)
            grads = []
            for step in (smearing_step, SmearingStep.apply):
                U = U0.clone().requires_grad_()
                C = C0.clone().requires_grad_()
                updated, logdet = step(U, C)
                grad = torch.autograd.grad(updated.real.sum() + 0.1 * logdet.sum(), (U, C))
                self.assertTrue(all(torch.isfinite(g).all().item() for g in grad))
                grads.append(grad)
            for actual, expected in zip(grads[0], grads[1]):
                self.assertLess((actual - expected).abs().max().item(), 1e-11)

    def test_logdet_matches_finite_difference_haar_tangent_jacobian(self):
        torch.manual_seed(13)
        U = sun.SUN_haar((3,), dtype=torch.cdouble)
        C0 = torch.randn((3, 3), dtype=torch.cdouble)
        generators = _su3_generators(dtype=U.dtype, device=U.device)
        eye = torch.eye(3, dtype=U.dtype)
        eps = 2e-6
        with torch.no_grad():
            for scale in (0.05, 1.5):
                C = scale * C0
                updated, logdet = smearing_step(U, C)
                columns = []
                for generator in generators:
                    plus = torch.matrix_exp(1j * eps * generator) @ U
                    minus = torch.matrix_exp(-1j * eps * generator) @ U
                    up_plus = StepQuantities(plus, C).Up
                    up_minus = StepQuantities(minus, C).Up
                    delta = (up_plus - up_minus) / (2.0 * eps)
                    hermitian = -1j * delta @ updated.mH
                    hermitian = 0.5 * (hermitian + hermitian.mH)
                    hermitian = hermitian - torch.trace(hermitian) / 3.0 * eye
                    columns.append(torch.stack([2.0 * torch.trace(hermitian @ g).real for g in generators]))
                tangent_jacobian = torch.stack(columns, dim=-1)
                tangent_logdet = torch.linalg.slogdet(tangent_jacobian).logabsdet
                self.assertLess(abs(float(logdet.item() - tangent_logdet.item())), 1e-7)


class ParallelTemperingAcceptanceTest(unittest.TestCase):
    def test_exact_and_linear_accept_favorable_and_thresholded_swaps(self):
        pt = object.__new__(ParallelTempering)
        pt.rank, pt.world_size, pt.B = 0, 2, 4
        pt.device, pt.param_dtype = torch.device("cpu"), torch.double
        parameters = torch.tensor([0.0, 2.0], dtype=torch.double)
        energies = (
            torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.double),
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.double),
        )
        actions = tuple(energy[:, None] * parameters[None, :] for energy in energies)
        # Delta S is (-2, 0, 2, 2); the last two swaps straddle exp(-2).
        uniforms = torch.tensor([0.999, 0.999, 0.5, 0.1], dtype=torch.double)
        expected = torch.tensor([[2.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 0.0]], dtype=torch.double)

        for rule in ("exact", "linear"):
            with self.subTest(rule=rule):
                pt.parameter_matrix = parameters.expand(pt.B, -1).clone()
                payload = actions if rule == "exact" else energies

                def gather(local, gather_list, dst):
                    for output, source in zip(gather_list, payload):
                        output.copy_(source)

                attempt = pt._attempt_swaps_exact_actions if rule == "exact" else pt._attempt_swaps_linear_observable
                with patch("neoqcd.distributedPT.dist.gather", side_effect=gather), patch(
                    "neoqcd.distributedPT.torch.rand", return_value=uniforms
                ):
                    stats = attempt(payload[0], "even")

                self.assertTrue(torch.equal(pt.parameter_matrix, expected))
                self.assertEqual(stats["n_accepted"], 3)
                self.assertEqual(stats["n_proposed"], 4)
                self.assertEqual(stats["link_accepted"], [3])


class GroupAndActionTest(unittest.TestCase):
    def test_haar_and_unitarity_for_su2_su3(self):
        for N in (2, 3):
            links = sun.SUN_haar((64, N), dtype=torch.cdouble)
            eye = sun.SUN_identity(links.shape[:-1], dtype=links.dtype)
            self.assertLess((links.mH @ links - eye).abs().max().item(), 1e-12)
            self.assertLess((sun.SUN_determinant(links) - 1.0).abs().max().item(), 1e-12)

    def test_action_and_mean_plaquette_include_every_plane(self):
        for D in (2, 4):
            lattice = (2,) * D
            identity = sun.SUN_identity((1, D) + lattice + (3,), dtype=torch.cdouble)
            pars = SimpleNamespace(D=D, N=3, device=torch.device("cpu"))
            action = Wilson_action(pars, beta=5.5)
            self.assertLess(action(identity, defect=None).abs().max().item(), 1e-12)
            plaquettes = sun.plaquette_fields(identity, D)
            self.assertAlmostEqual(float(sun.mean_plaquette(plaquettes, 3).item()), 1.0, places=12)

        planes = [torch.full((2, 3), v, dtype=torch.double) for v in (1.0, 2.0, 3.0)]
        self.assertTrue(torch.allclose(sun.mean_plaquette(planes, 3), torch.full((2,), 2.0 / 3.0, dtype=torch.double)))

    def test_action_tracks_runtime_defect_parameter(self):
        pars = SimpleNamespace(D=2, N=2, device=torch.device("cpu"))
        action = Wilson_action(pars, beta=3.0, defect_par=1.0)
        cfg = sun.SUN_haar((1, 2, 2, 2, 2), dtype=torch.cdouble)
        defect = torch.ones((2, 2, 2, 2), dtype=torch.double)
        defect[0].zero_()
        full = action(cfg, defect)
        action.defect_par = 0.0
        weighted = action(cfg, defect)
        expected = Wilson_action(pars, beta=3.0, defect_par=0.0)(cfg, defect)
        self.assertTrue(torch.allclose(weighted, expected))
        self.assertFalse(torch.allclose(full, weighted))


class SelectedHBORTest(unittest.TestCase):
    def _flow_pars(self, N, D, T, L):
        return SimpleNamespace(N=N, D=D, orsteps=1, device=torch.device("cpu"))

    def test_selected_staples_and_updates_match_full_lattice_reference(self):
        torch.manual_seed(7)
        D, T, L = 2, 2, 3
        defect = torch.ones((2, D, T, L))
        defect[1].zero_()
        defect[1, 0, 1, 1] = 1.0
        for N in (2, 3):
            hbor = HBOR(self._flow_pars(N, D, T, L), beta=5.0, defect_par=0.6)
            cfgs = sun.SUN_haar((1, D, T, L, N), dtype=torch.cdouble)
            mask = create_mask(D, T, L)
            links = hbor._links_for_mask(mask)
            flat = sun.flat_links(cfgs, D)
            rand = torch.rand((1, 18 if N == 3 else 6, D, T, L), dtype=torch.double)
            for m in range(2 * D):
                mu = m // 2
                selected = links.sites[m]
                gathered = hbor._gathered_staples(cfgs, flat, links, m, defect)
                full_staples = hbor.sum_of_staples_defect(cfgs, mu, defect).flatten(1, 2)
                self.assertLess((gathered - full_staples.index_select(1, selected)).abs().max().item(), 1e-11)

                rand_m = rand[:, :, mu].flatten(2).index_select(2, selected)
                if N == 2:
                    selected_hb = hbor._su2_selected_hb(flat, links, m, rand_m, defect, cfgs)
                    full_hb = hbor.SU2_link_update_hb(cfgs, mu, rand, defect)[:, 0].flatten(1, 2)
                    selected_over = hbor._su2_selected_over(flat, links, m, defect, cfgs)
                    full_over = hbor.SU2_link_update_over(cfgs, mu, defect)[:, 0].flatten(1, 2)
                else:
                    selected_hb = hbor._su3_selected_hb(flat, links, m, rand_m, defect, cfgs)
                    full_hb = hbor.SU3_link_update_hb(cfgs, mu, rand, defect)[:, 0].flatten(1, 2)
                    selected_over = hbor._su3_selected_over(flat, links, m, defect, cfgs)
                    full_over = hbor.SU3_link_update_over(cfgs, mu, defect)[:, 0].flatten(1, 2)
                self.assertLess((selected_hb - full_hb.index_select(1, selected)).abs().max().item(), 1e-10)
                self.assertLess((selected_over - full_over.index_select(1, selected)).abs().max().item(), 1e-10)

    def test_mask_cache_refreshes_after_in_place_mutation(self):
        D, T, L = 2, 2, 2
        hbor = HBOR(self._flow_pars(2, D, T, L), beta=4.0)
        mask = create_mask(D, T, L)
        before = hbor._links_for_mask(mask)
        m = 0
        selected_site = int(before.sites[m][0].item())
        mask[m, 0, 0].flatten()[selected_site] = False
        after = hbor._links_for_mask(mask)
        self.assertIsNot(before, after)
        self.assertNotIn(selected_site, after.sites[m].tolist())

    def test_flow_layers_share_bounded_indices_and_invalidate_all_source_masks(self):
        D, T, L = 2, 4, 4
        mask = create_mask(D, T, L)
        flow_pars = SimpleNamespace(
            N=3,
            D=D,
            T=T,
            L=L,
            orsteps=0,
            device=torch.device("cpu"),
            mask=mask,
            rho_shape=(1, 2 * D, 1, 1, 1),
            jac_shape=None,
            rho_shape_type=0,
            use_nn=False,
            smearing_steps_per_layer=1,
            fit_train=0,
            use_hyper_smearing=False,
            explicit_backward=False,
        )
        hbor0 = HBOR(flow_pars, beta=4.0)
        hbor1 = HBOR(flow_pars, beta=4.0)
        smear0 = Smearing(flow_pars, time=0.25)
        smear1 = Smearing(flow_pars, time=0.75)
        shared = hbor0._links_for_mask(mask)
        self.assertIs(hbor1._links_for_mask(mask), shared)
        self.assertIs(smear0._links_for_smearing_mask(mask), shared)
        self.assertIs(smear1._links_for_smearing_mask(mask), shared)

        dmask = torch.ones((1, D, T, L, 1, 1), dtype=torch.bool)
        defect_links = smear0._links_for_smearing_mask(mask, dmask)
        self.assertIs(smear1._links_for_smearing_mask(mask, dmask), defect_links)
        dmask[:, :, 0] = False
        self.assertIsNot(smear1._links_for_smearing_mask(mask, dmask), defect_links)

        mask[0, 0, 0, 0, 0, 0, 0] = 0
        changed = hbor1._links_for_mask(mask)
        self.assertIsNot(changed, shared)
        self.assertIs(smear0._links_for_smearing_mask(mask), changed)
        self.assertLessEqual(len(flow_pars._masked_links_cache_owner._entries), 8)

    def test_su2_and_su3_heatbath_beta_normalization(self):
        rand = torch.tensor([[0.1, 0.5, 0.1, 0.5, 0.3, 0.7]], dtype=torch.double)
        beta, k = 6.0, 4.0
        for N, xi in ((2, 1.0 / (beta * k)), (3, 1.5 / (beta * k))):
            hbor = HBOR(self._flow_pars(N, 2, 2, 2), beta=beta)
            staple = k * sun.SUN_identity((1, 2), dtype=torch.cdouble)
            proposal, accepted = hbor.heatbath_standard(staple, rand, N / 2.0)
            lam2 = -0.5 * xi * (
                torch.log(1.0 - rand[:, 0])
                + torch.cos(2.0 * torch.pi * (1.0 - rand[:, 1])).square()
                * torch.log(1.0 - rand[:, 2])
            )
            self.assertTrue(accepted.all().item())
            self.assertAlmostEqual(float(proposal[0, 0, 0].real.item()), float(1.0 - 2.0 * lam2.item()), places=12)


class SelectedSmearingTest(unittest.TestCase):
    def test_selected_staples_cover_all_rho_shapes_and_defect_mask(self):
        torch.manual_seed(11)
        D, T, L, batch = 4, 4, 4, 2
        cfgs = sun.SUN_haar((batch, D, T, L, L, L, 3), dtype=torch.cdouble)
        mask = create_mask(D, T, L)
        dmask = torch.ones((1, D, T, L, L, L, 1, 1), dtype=torch.int64)
        dmask[:, :, ::2, ::2] = 0
        links = None
        flat = sun.flat_links(cfgs, D)
        cfg_shape = tuple(cfgs.shape)

        for rho_type in range(4):
            rho_shape = {
                0: (1, 2 * D, 1, 1, 1, 1, 1),
                1: (1, 2 * D, 6, 1, 1, 1, 1),
                2: (1, 2 * D, 6, T, L, L, L),
                3: (1, 2 * D, 6, T, L, L, L),
            }[rho_type]
            smearing = Smearing(
                SimpleNamespace(
                    N=3,
                    D=D,
                    rho_shape_type=rho_type,
                    rho_shape=rho_shape,
                    jac_shape=(batch, T, L, L, L, 9, 9),
                    use_nn=False,
                    device=torch.device("cpu"),
                    smearing_steps_per_layer=1,
                    fit_train=0,
                    use_hyper_smearing=False,
                    explicit_backward=False,
                ),
                time=1.0,
            )
            links = smearing._links_for_smearing_mask(mask, dmask)
            nslots = 6
            if rho_type == 0:
                rho_cases = [torch.tensor(0.04), torch.tensor([0.04, 0.07])]
            elif rho_type == 1:
                rho_cases = [torch.full((nslots,), 0.04), torch.full((batch, nslots), 0.04)]
            else:
                rho_cases = [
                    torch.full((nslots, T, L, L, L), 0.04),
                    torch.full((batch, nslots, T, L, L, L), 0.04),
                ]
            for rho in rho_cases:
                for m in range(2 * D):
                    sites = links.sites[m]
                    if sites.numel() == 0:
                        continue
                    selected = smearing._selected_stout_staples(flat, cfg_shape, links, m, rho)
                    full = smearing.stout_staples(cfgs, m // 2, rho)
                    full_selected = full.flatten(1, D).index_select(1, sites)
                    self.assertLess((selected - full_selected).abs().max().item(), 1e-11)

            if rho_type == 0:
                rho_layer = torch.full(rho_shape, 0.04, dtype=torch.double)
                selected_cfgs, _, selected_logdet = smearing(
                    cfgs.clone(), mask, rho_layer, dmasking=True, dmask=dmask
                )
                reference_cfgs = cfgs.clone()
                reference_logdet = torch.zeros(batch, dtype=torch.double)
                dims = tuple(range(1, D + 2))
                for mu in range(D):
                    for eo in range(2):
                        m = eo + 2 * mu
                        current_mask = mask[m, :] * dmask
                        rho = torch.abs(rho_layer[0, m, :])
                        cfgs_new, jac = smearing.stout_smearing(reference_cfgs, mu, rho)
                        reference_cfgs = current_mask * cfgs_new + (1.0 - current_mask) * reference_cfgs
                        reference_logdet = reference_logdet + torch.sum(
                            current_mask.squeeze(-1).squeeze(-1) * jac, dims
                        )
                self.assertLess((selected_cfgs - reference_cfgs).abs().max().item(), 1e-11)
                self.assertLess((selected_logdet - reference_logdet).abs().max().item(), 1e-10)


class StapleRollTest(unittest.TestCase):
    def test_single_direction_rolls_match_full_tensor_reference(self):
        torch.manual_seed(4)
        for D, extents in ((2, (4, 6)), (4, (4, 4, 4, 4))):
            cfgs = torch.randn((1, D) + extents + (3, 3), dtype=torch.cdouble)
            for mu in range(D):
                for nu in range(D):
                    if mu == nu:
                        continue
                    old_upper = sun.SUN_mul(
                        torch.roll(cfgs, 1, dims=(-D + mu - 2))[:, nu],
                        sun.SUN_dagger(torch.roll(cfgs, 1, dims=(-D + nu - 2))[:, mu]),
                    )
                    old_upper = sun.SUN_mul(old_upper, sun.SUN_dagger(cfgs[:, nu]))
                    old_lower = sun.SUN_mul(
                        sun.SUN_dagger(
                            torch.roll(cfgs, (1, -1), dims=(-D + mu - 2, -D + nu - 2))[:, nu]
                        ),
                        sun.SUN_dagger(torch.roll(cfgs, -1, dims=(-D + nu - 2))[:, mu]),
                    )
                    old_lower = sun.SUN_mul(
                        old_lower, torch.roll(cfgs, -1, dims=(-D + nu - 2))[:, nu]
                    )
                    self.assertLess((sun.pstaple(cfgs, mu, nu, D) - old_upper).abs().max().item(), 1e-12)
                    self.assertLess((sun.nstaple(cfgs, mu, nu, D) - old_lower).abs().max().item(), 1e-12)


class NeuralSmearingCompatibilityTest(unittest.TestCase):
    def test_su3_neural_staples_still_run_through_full_path(self):
        D, T, L = 2, 4, 4
        flow_pars = SimpleNamespace(
            N=3,
            D=D,
            rho_shape=(1, 2 * D, 1, 1, 1),
            jac_shape=None,
            rho_shape_type=0,
            use_nn=True,
            device=torch.device("cpu"),
            smearing_steps_per_layer=1,
            fit_train=0,
            use_hyper_smearing=False,
            explicit_backward=False,
            net=torch.nn.Identity(),
            residual=False,
        )
        layer = Smearing(flow_pars, time=0.5)
        calls = []

        def neural_staples(cfgs, mu, rho):
            calls.append(mu)
            return torch.zeros_like(cfgs[:, mu])

        layer.compute_staples = neural_staples
        cfgs = sun.SUN_haar((1, D, T, L, 3), dtype=torch.cdouble)
        mask = create_mask(D, T, L)
        rho = torch.ones((1, 2 * D, 1, 1, 1), dtype=torch.double)
        updated, _, logdet = layer(cfgs, mask, rho)
        self.assertEqual(calls, [0, 0, 1, 1])
        self.assertLess((updated - cfgs).abs().max().item(), 1e-12)
        self.assertLess(logdet.abs().max().item(), 1e-12)


class CheckpointUtilityTest(unittest.TestCase):
    def test_atomic_checkpoint_saves_scheduler_and_smearing_semantics(self):
        class Smearing(torch.nn.Linear):
            N = 3

        model = torch.nn.Sequential(Smearing(2, 2))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "flow.pt")
            save(model, [optimizer], path, schedulers=[scheduler])
            checkpoint = torch.load(path, map_location="cpu")
            self.assertEqual(checkpoint["smearing_jacobian_version"], "su3_stout_full_haar_logdet_v1")
            self.assertEqual(len(checkpoint["schedulers_state_dict"]), 1)
            self.assertFalse(any(name.endswith(".tmp") for name in os.listdir(directory)))

            restored = torch.nn.Sequential(Smearing(2, 2))
            restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.5)
            restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=5)
            load_weights(restored, path)
            load_optimizer(
                restored,
                [restored_optimizer],
                path,
                schedulers=[restored_scheduler],
            )
            self.assertEqual(restored_scheduler.state_dict(), scheduler.state_dict())

            legacy_path = os.path.join(directory, "legacy.pt")
            torch.save({"model_state_dict": model.state_dict()}, legacy_path)
            with self.assertRaisesRegex(ValueError, "smearing_jacobian_version"):
                load_weights(restored, legacy_path)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                load_weights(restored, legacy_path, allow_legacy_smearing=True)
            self.assertTrue(caught)

    def test_neorex_checkpoint_version_accepts_current_and_residual_legacy_only(self):
        state = {"weight": torch.ones(2)}
        current = {
            "checkpoint_format": "neoqcd_neorex_flow",
            "format_version": 1,
            "smearing_jacobian_version": LOGDET_VERSION,
            "state_dict": state,
        }
        self.assertIs(_neorex_checkpoint_state(current, LOGDET_VERSION), state)
        with self.assertRaisesRegex(ValueError, "unversioned state_dict"):
            _neorex_checkpoint_state(state, LOGDET_VERSION)
        self.assertIs(_neorex_checkpoint_state(state, None), state)
        self.assertIs(_neorex_checkpoint_state(state), state)
        with self.assertRaisesRegex(ValueError, "semantics do not match"):
            _neorex_checkpoint_state(
                {**current, "smearing_jacobian_version": "older"}, LOGDET_VERSION
            )

    def test_history_write_preserves_double_precision(self):
        value = 0.12345678901234566
        with tempfile.TemporaryDirectory() as directory:
            root = os.path.join(directory, "train")
            write({"ESS": [value], "var_loss": [value], "loss": [value]}, root)
            with open(root + "_loss.dat") as stream:
                self.assertEqual(float(stream.readline()), value)


class DefectPatchTest(unittest.TestCase):
    def test_generic_two_dimensional_crop_and_embed(self):
        defect = Defect(D=2, T=8, L=8, dsize=2, time_slice=4, space_slice=3)
        cfgs = torch.arange(1 * 2 * 8 * 8 * 2 * 2, dtype=torch.double).reshape(1, 2, 8, 8, 2, 2)
        patch = defect.cut_defect(cfgs, buffer=1)
        self.assertEqual(tuple(patch.shape), (1, 2, 2, 4, 2, 2))
        embedded = defect.embedding(torch.zeros_like(patch), cfgs, buffer=1)
        self.assertTrue((embedded[:, 0, 4, 2:6] == 0).all().item())
        self.assertTrue((embedded[:, 1, 4, 3:6] == 0).all().item())

    def test_hierarchical_updates_are_nested_inside_full_box_support(self):
        defect = Defect(D=2, T=8, L=8, dsize=2, time_slice=4, space_slice=3)
        outer_mask = create_mask(2, 4, 6)
        inner_mask = create_mask(2, 2, 4)
        fp = SimpleNamespace(
            D=2,
            T=8,
            L=8,
            N=2,
            mask=create_mask(2, 8, 8),
            device=torch.device("cpu"),
            orsteps=0,
            updates_per_layer=1,
            hierarchical_levels=2,
            hierarchical_rectangles=[2, 1],
            hierarchical_updates=[1, 1],
            defect=defect,
            smeared_defect_mask=[outer_mask, inner_mask],
            small_mask=[torch.ones_like(outer_mask), torch.ones_like(inner_mask)],
            small_defect_mask=[None, None],
        )
        update = NEMCMC_update(fp, beta=4.0)

        class AddOne:
            def __init__(self):
                self.inputs = []

            def __call__(self, cfgs, mask, defect_mask):
                self.inputs.append(cfgs.clone())
                return cfgs + 1.0

        fake_update = AddOne()
        update.update = fake_update
        cfgs = torch.zeros((1, 2, 8, 8, 2, 2), dtype=torch.cdouble)
        changed = update.hierarchical(cfgs, 0, fp)
        self.assertEqual(tuple(fake_update.inputs[0].shape[2:4]), (4, 6))
        self.assertEqual(tuple(fake_update.inputs[1].shape[2:4]), (2, 4))
        self.assertGreater(fake_update.inputs[1].abs().sum().item(), 0.0)

        def expected_box_links(buffer):
            extents = (2 * buffer,) + (defect.dsize + 2 * buffer,) * (defect.D - 1)
            starts = (defect.time_slice - buffer,) + (defect.space_slice - buffer,) * (defect.D - 1)
            expected = torch.zeros((defect.D, defect.T) + (defect.L,) * (defect.D - 1), dtype=torch.bool)
            for mu in range(defect.D):
                for coords in product(*(range(size) for size in extents)):
                    if all(
                        (coords[axis] > 0 if axis == mu else 0 < coords[axis] < extents[axis] - 1)
                        for axis in range(defect.D)
                    ):
                        global_coords = tuple(start + local for start, local in zip(starts, coords))
                        expected[(mu,) + global_coords] = True
            return expected

        outer_expected = expected_box_links(2)
        inner_expected = expected_box_links(1)
        actual_values = changed[0].real.diagonal(dim1=-2, dim2=-1).mean(-1)
        self.assertTrue(torch.equal((actual_values != 0), outer_expected))
        self.assertTrue(torch.equal(actual_values == 2, inner_expected))
        self.assertTrue(torch.equal(actual_values == 1, outer_expected & ~inner_expected))

        boxes = fp._hierarchical_boxes_cache[4]
        second_update = NEMCMC_update(fp, beta=4.0)
        self.assertIs(second_update._hierarchical_boxes_for(fp, (2, 1))[0].links, boxes[0].links)
        fp.mask[0, 0, 0, 0, 0, 0, 0] = 0
        refreshed = second_update._hierarchical_boxes_for(fp, (2, 1))
        self.assertIsNot(refreshed[0].links, boxes[0].links)


if __name__ == "__main__":
    unittest.main()
