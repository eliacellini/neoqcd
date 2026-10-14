import unittest
from types import SimpleNamespace

import torch

import neoqcd.sun_utils as sun
from neoqcd.mcmc import HBOR
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


if __name__ == "__main__":
    unittest.main()
