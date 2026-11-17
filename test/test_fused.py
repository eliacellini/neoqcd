"""The fused HBOR kernel (neoqcd.fused) against the torch sweep of mcmc.HBOR.

On the CPU the kernel runs as its emulation (the same source built with the host C++ compiler); on
CUDA the same comparisons run on the GPU. Both paths get the same random numbers, so the outputs
agree to float64 rounding, amplified by the update (~1e-13 per sweep on thermalized links).
"""

import unittest
import warnings
from types import SimpleNamespace

import torch

import neoqcd.fused as fused
import neoqcd.sun_utils as sun
from neoqcd.mcmc import HBOR
from neoqcd.utils import Defect, create_mask

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
TOL = 1e-11


def _cpu_backend():
    ok, why = fused.backend_status("cpu")
    return ok, why


def _pars(T, L, device, orsteps=1):
    return SimpleNamespace(N=3, D=4, orsteps=orsteps, device=torch.device(device),
                           mask=create_mask(4, T, L).to(device), defect=None)


def _thermalized(hbor, pars, batch, T, L, sweeps=4, seed=0):
    # A few torch sweeps from a Haar-random start: away from the badly conditioned random links
    g = torch.Generator().manual_seed(seed)
    cfgs = sun.SUN_haar((batch, 4, T, L, L, L, 3), dtype=torch.cdouble, generator=g).to(pars.device)
    with torch.no_grad():
        for _ in range(sweeps):
            cfgs = hbor(cfgs, pars.mask, None)
    return cfgs


class FusedHBORTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ok, why = _cpu_backend()
        if not ok:
            raise unittest.SkipTest(f"no CPU build of the kernel: {why}")

    def _compare(self, device, T=4, L=4, batch=2, orsteps=1, defect=None, defect_par=1.0, mask=None,
                 reverse=False, beta=6.0, seed=0):
        pars = _pars(T, L, device, orsteps)
        torch.manual_seed(seed)
        reference = HBOR(pars, beta=beta, defect_par=defect_par)
        cfgs = _thermalized(HBOR(pars, beta=6.0), pars, batch, T, L, seed=seed)
        update_mask = pars.mask if mask is None else mask.to(device)
        rand = torch.rand((batch, 18, 4, T, L, L, L), dtype=torch.float64, device=device)
        defect = None if defect is None else defect.to(device)
        with torch.no_grad():
            expected = reference(cfgs, update_mask, defect, rand=rand, reverse=reverse)
            kernel = HBOR(pars, beta=beta, defect_par=defect_par)
            kernel.use_fused_kernels = True
            kernel.fused_emulate = True
            kernel.fused_strict = True
            got = kernel(cfgs, update_mask, defect, rand=rand, reverse=reverse)
        self.assertEqual(kernel.fused_used, device if device == "cuda" else "cpu")
        self.assertEqual(reference.fused_used, None)
        self.assertGreater((expected - cfgs).abs().max().item(), 1e-3)  # the sweep moved the links
        self.assertLess((got - expected).abs().max().item(), TOL)
        return cfgs, got

    def test_whole_lattice_without_defect(self):
        for device in DEVICES:
            for reverse in (False, True):
                with self.subTest(device=device, reverse=reverse):
                    self._compare(device, reverse=reverse)

    def test_several_overrelaxation_sweeps_and_a_non_cubic_lattice(self):
        for device in DEVICES:
            with self.subTest(device=device):
                self._compare(device, T=6, L=4, orsteps=3, batch=3)

    def test_defect_with_one_parameter_including_bc_zero(self):
        # bc = 0: the links of the defect have a vanishing staple sum (the Haar branch)
        defect = Defect(4, 4, 4, dsize=1, time_slice=1, space_slice=1).create_defect_mask()
        for device in DEVICES:
            for bc in (0.0, 0.3):
                for reverse in (False, True):
                    with self.subTest(device=device, bc=bc, reverse=reverse):
                        self._compare(device, defect=defect, defect_par=bc, reverse=reverse)

    def test_defect_with_a_parameter_per_chain(self):
        defect = Defect(4, 6, 4, dsize=2, time_slice=2, space_slice=1).create_defect_mask()
        bc = torch.tensor([0.0, 0.4, 1.0, 0.7], dtype=torch.float64)
        for device in DEVICES:
            for reverse in (False, True):
                with self.subTest(device=device, reverse=reverse):
                    self._compare(device, T=6, L=4, batch=4, defect=defect, defect_par=bc, reverse=reverse)

    def test_protocol_patch(self):
        d = Defect(4, 6, 6, dsize=1, time_slice=2, space_slice=2)
        defect, patch = d.create_defect_mask(), d.protocol_patch_mask(1)
        mask = (create_mask(4, 6, 6) * patch).contiguous()
        bc = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        for device in DEVICES:
            for reverse in (False, True):
                with self.subTest(device=device, reverse=reverse):
                    cfgs, got = self._compare(device, T=6, L=6, batch=3, defect=defect, defect_par=bc,
                                              mask=mask, reverse=reverse)
                    inside = patch[0, ..., 0, 0].to(torch.bool).to(device)
                    inside = inside.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(got)
                    self.assertEqual((got - cfgs)[~inside].abs().max().item(), 0.0)

    def test_beta_per_chain(self):
        for device in DEVICES:
            with self.subTest(device=device):
                beta = torch.tensor([5.7, 6.0, 6.3], dtype=torch.float64)
                self._compare(device, batch=3, beta=beta)

    def test_kernel_unless_an_input_needs_gradients_and_torch_on_cpu_without_emulation(self):
        pars = _pars(4, 4, "cpu")
        hbor = HBOR(pars, beta=6.0)
        hbor.use_fused_kernels = True
        cfgs = sun.SUN_haar((1, 4, 4, 4, 4, 4, 3), dtype=torch.cdouble)
        with torch.no_grad():
            hbor(cfgs, pars.mask, None)
        self.assertEqual(hbor.fused_used, "torch")
        hbor.fused_emulate = True
        # grad mode alone (main_pt_obc.py sweeps in grad mode): no input requires gradients, the kernel runs
        with torch.enable_grad():
            hbor(cfgs, pars.mask, None)
        self.assertEqual(hbor.fused_used, "cpu")
        # an input that requires gradients: the torch sweep, which builds the graph
        with torch.enable_grad():
            out = hbor(cfgs.clone().requires_grad_(True), pars.mask, None)
        self.assertEqual(hbor.fused_used, "torch")
        self.assertTrue(out.requires_grad)
        with torch.no_grad():
            hbor(cfgs.clone().requires_grad_(True), pars.mask, None)
        self.assertEqual(hbor.fused_used, "cpu")

    def test_other_dtype_falls_back_with_a_warning(self):
        pars = _pars(4, 4, "cpu")
        hbor = HBOR(pars, beta=6.0)
        hbor.use_fused_kernels = True
        hbor.fused_emulate = True
        cfgs = sun.SUN_haar((1, 4, 4, 4, 4, 4, 3), dtype=torch.cdouble).to(torch.complex64)
        fused._WARNED.clear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with torch.no_grad():
                hbor(cfgs, pars.mask, None)
        self.assertEqual(hbor.fused_used, "torch")
        self.assertTrue(any("complex128" in str(w.message) for w in caught))
        hbor.fused_strict = True
        with self.assertRaises(fused.BackendUnavailable), torch.no_grad():
            hbor(cfgs, pars.mask, None)


if __name__ == "__main__":
    unittest.main()
