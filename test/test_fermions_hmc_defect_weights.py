"""`hmc.defect_plaquette_weights` and `hmc.batched_defect_plaquette_weights` with a non-floating
defect mask (review of 20 November): an integer mask used to give float32 link factors (scalar
version) and to truncate the coupling to an integer (batched version). Both now take the mask as
float64 and agree with the float64 mask to rounding."""

import unittest

import torch

from neoqcd.fermions.hmc import batched_defect_plaquette_weights, defect_plaquette_weights


def mask(D=2, T=4, L=4, dtype=torch.float64):
    d = torch.zeros((D, T) + (L,) * (D - 1), dtype=dtype)
    d[0, 0, 0:2] = 1
    return torch.stack((1 - d, d), dim=0)


class DefectWeightsDtypeTest(unittest.TestCase):

    def test_integer_masks(self):
        for D in (2, 4):
            ref_mask = mask(D=D)
            for dtype in (torch.int64, torch.int32, torch.uint8):
                m = mask(D=D, dtype=dtype)
                with self.subTest(D=D, dtype=dtype):
                    for bc in (0.0, 0.37, 1.0):
                        w = defect_plaquette_weights(m, bc, D)
                        ref = defect_plaquette_weights(ref_mask, bc, D)
                        for key in ref:
                            self.assertEqual(w[key].dtype, torch.float64)
                            self.assertTrue(torch.equal(w[key], ref[key]))
                    bcs = torch.tensor([0.0, 0.37, 0.5, 1.0], dtype=torch.float64)
                    wb = batched_defect_plaquette_weights(m, bcs, D)
                    refb = batched_defect_plaquette_weights(ref_mask, bcs, D)
                    for key in refb:
                        self.assertEqual(wb[key].dtype, torch.float64)
                        self.assertTrue(torch.equal(wb[key], refb[key]))
                    # entry b is the scalar version at bc_b (0.37 not truncated to 0)
                    for b, bc in enumerate(bcs.tolist()):
                        scalar = defect_plaquette_weights(m, bc, D)
                        for key in scalar:
                            self.assertTrue(torch.equal(wb[key][b], scalar[key]))
            self.assertLess(min(v.min().item() for v in batched_defect_plaquette_weights(
                mask(D=D, dtype=torch.int64), torch.tensor([0.37]), D).values()), 0.5)

    def test_float_mask_unchanged(self):
        m32 = mask(dtype=torch.float32)
        w = defect_plaquette_weights(m32, 0.5, 2)
        self.assertEqual(next(iter(w.values())).dtype, torch.float32)  # a floating mask keeps its dtype


if __name__ == "__main__":
    unittest.main()
