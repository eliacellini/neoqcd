import unittest
from types import SimpleNamespace

import torch

import neoqcd.sun_utils as sun
from neoqcd.mcmc import HBOR, SU2_left_multiply_SU3
from neoqcd.utils import create_mask


class HBORSU2Test(unittest.TestCase):
    def test_heatbath_and_overrelaxation_preserve_su2_field(self):
        torch.manual_seed(0)
        D, T, L = 2, 2, 2
        flow_pars = SimpleNamespace(N=2, D=D, orsteps=1, device=torch.device("cpu"))
        hbor = HBOR(flow_pars, beta=6.0)
        cfgs = sun.hot_SU2_start_nonuniform((1, D, T, L, 2), dtype=torch.cdouble)
        rand = torch.rand((1, 6, D, T, L), dtype=cfgs.real.dtype)

        updated = hbor(cfgs, create_mask(D, T, L), defect=None, rand=rand)

        self.assertEqual(updated.shape, cfgs.shape)
        self.assertTrue(torch.isfinite(updated.real).all().item())
        self.assertTrue(torch.isfinite(updated.imag).all().item())
        identity = sun.SUN_identity(updated.shape[:-1], dtype=updated.dtype)
        self.assertLess((sun.SUN_mul(sun.SUN_dagger(updated), updated) - identity).abs().max().item(), 1e-12)
        self.assertLess((sun.SUN_determinant(updated) - 1.0).abs().max().item(), 1e-12)


class SU2SubgroupProductTest(unittest.TestCase):
    def test_two_row_update_equals_matmul_and_embedded_product(self):
        torch.manual_seed(0)
        flow_pars = SimpleNamespace(N=3, D=4, orsteps=1, device=torch.device("cpu"))
        hbor = HBOR(flow_pars, beta=6.0)
        links = sun.SUN_haar((5, 7, 3), dtype=torch.cdouble)
        subgroup = sun.SUN_haar((5, 7, 2), dtype=torch.cdouble)
        for phb in range(3):
            with self.subTest(phb=phb):
                updated = SU2_left_multiply_SU3(links, subgroup, phb)
                pair = [c for c in range(3) if c != phb]
                # Reference 1: the former torch.matmul on the two affected rows.
                rows_matmul = links.clone()
                rows_matmul[..., pair, :] = torch.matmul(subgroup, links[..., pair, :])
                # Reference 2: the SU(2) element embedded in SU(3), times the whole link.
                embedded = torch.matmul(hbor.SU2toSU3(subgroup, phb), links)
                self.assertLess((updated - rows_matmul).abs().max().item(), 1e-14)
                self.assertLess((updated - embedded).abs().max().item(), 1e-14)
                self.assertTrue(torch.equal(updated[..., phb, :], links[..., phb, :]))


class HierarchicalDefectTest(unittest.TestCase):
    def test_box_updates_with_a_defect_move_only_the_interior_links_of_the_largest_box(self):
        # NEMCMC_update.hierarchical with a defect: DefectBox used to cut the (2, D, T, L, L, L)
        # defect mask with the indexing of a configuration (IndexError before any sweep)
        from neoqcd.mcmc import NEMCMC_update
        from neoqcd.utils import Defect, DefectBox
        torch.manual_seed(0)
        T, L = 8, 8
        defect = Defect(4, T, L, dsize=2, time_slice=4, space_slice=3)
        pars = SimpleNamespace(N=3, D=4, orsteps=1, device=torch.device("cpu"), mask=create_mask(4, T, L),
                               defect=defect, updates_per_layer=1, hierarchical_levels=2,
                               hierarchical_rectangles=[2, 1], hierarchical_updates=[1, 2])
        update = NEMCMC_update(pars, 6.0, defect_mask=defect.create_defect_mask(), defect_par=0.3)
        cfgs = sun.SUN_haar((2, 4, T, L, L, L, 3), dtype=torch.cdouble)
        with torch.no_grad():
            out = update.hierarchical(cfgs, 0, pars)
            full = update(cfgs, pars)[0]
        largest = DefectBox(defect, pars.mask, None, 2)
        moved = torch.zeros(4 * T * L ** 3, dtype=torch.bool)
        moved[largest.lattice_links] = True
        moved = moved.reshape(1, 4, T, L, L, L, 1, 1).expand_as(out)
        self.assertEqual((out - cfgs)[~moved].abs().max().item(), 0.0)
        self.assertGreater((out - cfgs)[moved].abs().max().item(), 1e-3)
        identity = sun.SUN_identity(full.shape[:-1], dtype=full.dtype)
        self.assertLess((sun.SUN_mul(sun.SUN_dagger(full), full) - identity).abs().max().item(), 1e-12)


class LocalActionChangeTest(unittest.TestCase):
    def test_halo_action_change_equals_the_whole_lattice_difference_without_host_reads(self):
        # S_halo with the bounding boxes of HaloPlaquetteMasks (no host sync) against the Wilson
        # action of the whole lattice, for a halo inside the lattice and one that wraps around it
        from unittest import mock
        import neoqcd.theory as theory
        from neoqcd.mcmc import DefectMCMCAdapter
        from neoqcd.utils import Defect
        T, L = 6, 8
        for space_slice in (3, 1):
            with self.subTest(space_slice=space_slice):
                torch.manual_seed(space_slice)
                defect = Defect(4, T, L, dsize=2, time_slice=3, space_slice=space_slice)
                pars = SimpleNamespace(N=3, D=4, T=T, L=L, orsteps=1, device=torch.device("cpu"),
                                       mask=create_mask(4, T, L), defect=defect)
                adapter = DefectMCMCAdapter(pars, 6.0, defect.create_defect_mask(), batch_size=3)
                self.assertIsInstance(adapter._nf_halo_plaquette_masks, theory.HaloPlaquetteMasks)
                x_old = sun.SUN_haar((3, 4, T, L, L, L, 3), dtype=torch.cdouble)
                changed = theory.build_nf_changed_link_mask(pars).reshape(1, 4, T, L, L, L, 1, 1)
                x_new = torch.where(changed, sun.SUN_haar(x_old.shape[:-1], dtype=torch.cdouble), x_old)
                p_old = torch.tensor([0.0, 0.3, 0.9], dtype=torch.float64)
                p_new = torch.tensor([0.2, 0.5, 1.0], dtype=torch.float64)
                expected = adapter.action(x_new, p_new) - adapter.action(x_old, p_old)
                with mock.patch.object(theory, "_bbox_slices", side_effect=AssertionError("bounding box read")):
                    got = adapter.local_nf_action_change(x_old, x_new, p_old, p_new)
                self.assertGreater(expected.abs().min().item(), 1.0)
                self.assertLess((got - expected).abs().max().item(), 1e-9)
                plain = dict(adapter._nf_halo_plaquette_masks)  # bounding boxes found at the call
                ref = theory.local_nf_action_change(x_old, x_new, adapter.defect_mask, 6.0, p_old, p_new,
                                                    plain, 4, 3)
                self.assertTrue(torch.equal(got, ref))


class WilsonActionPrecisionTest(unittest.TestCase):
    def test_link_factors_of_an_integer_defect_mask_are_double(self):
        # Defect.create_defect_mask is an int64 tensor: int64 * Python float would give float32 link
        # factors (relative error ~3e-8 on bc = 0.3)
        from neoqcd.theory import Wilson_action
        from neoqcd.utils import Defect
        T, L = 4, 4
        torch.manual_seed(0)
        defect = Defect(4, T, L, dsize=2, time_slice=1, space_slice=1).create_defect_mask()
        self.assertEqual(defect.dtype, torch.int64)
        pars = SimpleNamespace(N=3, D=4, device=torch.device("cpu"))
        cfgs = sun.SUN_haar((2, 4, T, L, L, L, 3), dtype=torch.cdouble)
        action = Wilson_action(pars, 6.0, 0.3)
        self.assertTrue(torch.equal(action(cfgs, defect), action(cfgs, defect.double())))


if __name__ == "__main__":
    unittest.main()
