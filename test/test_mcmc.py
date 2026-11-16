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


if __name__ == "__main__":
    unittest.main()
