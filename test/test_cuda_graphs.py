"""CUDA graph replays of the HBOR sweep and of the defect smearing layer match eager execution.

Needs a GPU (skipped elsewhere): run on Leonardo with `sbatch job/debug_gpu_tests.sh test.test_cuda_graphs`.
The timings are printed for the record.
"""

import time
import unittest
from types import SimpleNamespace

import torch

import neoqcd.sun_utils as sun
from neoqcd.cnf_io import build_cnf
from neoqcd.mcmc import HBOR, DefectMCMCAdapter
from neoqcd.utils import Defect, create_mask

CUDA = torch.cuda.is_available()


def _timed(fn, repeats=10):
    fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats * 1e3


@unittest.skipUnless(CUDA, "needs a GPU")
class HBORGraphTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.device = torch.device("cuda")
        D, T, L = 4, 8, 8
        self.pars = SimpleNamespace(N=3, D=D, orsteps=4, device=self.device, mask=create_mask(D, T, L).to(self.device),
                                    defect=None)
        self.defect = Defect(D, T, L, dsize=2, time_slice=4, space_slice=3)
        self.defect_mask = self.defect.create_defect_mask().to(self.device)
        self.patch = (self.pars.mask * self.defect.protocol_patch_mask(1, device=self.device)).contiguous()

    def _compare(self, mask, batch, par, label):
        cfgs = sun.SUN_haar((batch, 4, 8, 8, 8, 8, 3), dtype=torch.cdouble, device=self.device)
        eager = HBOR(self.pars, beta=6.0, defect_par=par)
        graphed = HBOR(self.pars, beta=6.0, defect_par=par)
        graphed.use_cuda_graphs = True
        for reverse in (False, True):
            for trial in range(2):  # the second replay must see the new inputs
                rand = torch.rand(eager._rand_shape(cfgs), device=self.device, dtype=torch.float64)
                x = cfgs if trial == 0 else sun.SUN_haar(cfgs.shape[:-1], dtype=torch.cdouble, device=self.device)
                expected = eager(x, mask, self.defect_mask, rand=rand, reverse=reverse)
                got = graphed(x, mask, self.defect_mask, rand=rand, reverse=reverse)
                self.assertTrue(graphed.use_cuda_graphs, "capture failed and fell back to eager")
                self.assertGreater((expected - x).abs().max().item(), 1e-3)
                self.assertLess((got - expected).abs().max().item(), 1e-14, (label, reverse, trial))
        t_eager = _timed(lambda: eager(cfgs, mask, self.defect_mask))
        t_graph = _timed(lambda: graphed(cfgs, mask, self.defect_mask))
        print(f"[{label}] batch {batch}: eager {t_eager:.1f} ms  graph {t_graph:.1f} ms", flush=True)
        return t_eager, t_graph

    def test_patch_sweep_with_per_chain_coupling(self):
        par = torch.linspace(0.0, 1.0, 64, dtype=torch.float64, device=self.device)
        # The sweep is GPU-latency bound: the replay matches eager time, it does not beat it.
        self._compare(self.patch, 64, par, "patch h=1, batched bc")

    def test_patch_sweep_without_defect(self):
        # Every chain at bc = 1 takes the standard-staple branch, a different graph.
        par = torch.ones(64, dtype=torch.float64, device=self.device)
        self._compare(self.patch, 64, par, "patch h=1, bc=1")

    def test_full_lattice_sweep(self):
        par = torch.linspace(0.0, 1.0, 16, dtype=torch.float64, device=self.device)
        self._compare(self.pars.mask, 16, par, "whole lattice, batched bc")

    def test_adapter_replays_with_the_coin_and_new_couplings(self):
        adapter = DefectMCMCAdapter(self.pars, beta=6.0, defect_mask=self.defect_mask, batch_size=8,
                                    patch_mask=self.defect.protocol_patch_mask(1, device=self.device),
                                    symmetric_updates=True)
        adapter._batch_hbor.use_cuda_graphs = True
        x = sun.SUN_haar((8, 4, 8, 8, 8, 8, 3), dtype=torch.cdouble, device=self.device)
        for p in (torch.linspace(0.0, 0.9, 8), torch.full((8,), 0.5), torch.ones(8)):
            adapter.set_parameter(p.to(dtype=torch.float64))
            out = adapter(x, nsteps=2)
            self.assertTrue(torch.isfinite(out.real).all())
            identity = sun.SUN_identity(out.shape[:-1], dtype=out.dtype, device=self.device)
            self.assertLess((sun.SUN_mul(sun.SUN_dagger(out), out) - identity).abs().max().item(), 1e-12)
        self.assertTrue(adapter._batch_hbor.use_cuda_graphs)
        self.assertGreaterEqual(len(adapter._batch_hbor._graphs), 2)


@unittest.skipUnless(CUDA, "needs a GPU")
class SmearingLayerGraphTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.device = torch.device("cuda")
        from test.test_cnf_adapters import _defect_config
        import importlib.util
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("main_pt_obc_graphs", root / "main" / "main_pt_obc.py")
        self.main = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.main)
        _, self.pars, _ = build_cnf(
            _defect_config(batch_size=16, D=4, T=8, L=8, defect_size=2, time_slice=4, space_slice=3), device="cuda"
        )

    def _layer(self, graphs):
        layer = self.main.DefectSNFProtocolLayer(self.pars).to(self.device)
        with torch.no_grad():
            for parameter in layer.layer.smearing.hyper_smearing.mlp.parameters():
                parameter.add_(0.1 * torch.randn_like(parameter))
        layer.use_cuda_graphs = graphs
        return layer

    def test_forward_and_inverse_replays_match_eager(self):
        torch.manual_seed(2)
        eager, graphed = self._layer(False), self._layer(True)
        graphed.load_state_dict(eager.state_dict())
        x = sun.SUN_haar((16, 4, 8, 8, 8, 8, 3), dtype=torch.cdouble, device=self.device)
        bc = torch.linspace(0.1, 0.8, 16, dtype=torch.float64, device=self.device)
        delta = torch.full_like(bc, 1 / 28)
        with torch.no_grad():
            for trial in range(2):
                if trial:
                    x = sun.SUN_haar(x.shape[:-1], dtype=torch.cdouble, device=self.device)
                    bc = bc.flip(0)
                y_e, ld_e = eager(x, bc, delta)
                y_g, ld_g = graphed(x, bc, delta)
                self.assertTrue(graphed.use_cuda_graphs, "capture failed and fell back to eager")
                self.assertLess((y_g - y_e).abs().max().item(), 1e-14)
                self.assertLess((ld_g - ld_e).abs().max().item(), 1e-12)
                xi_e, ldi_e = eager.inverse(y_e, bc, delta)
                xi_g, ldi_g = graphed.inverse(y_e, bc, delta)
                self.assertLess((xi_g - xi_e).abs().max().item(), 1e-14)
                self.assertLess((ldi_g - ldi_e).abs().max().item(), 1e-12)
                self.assertLess((xi_g - x).abs().max().item(), 1e-10)
            t_fe = _timed(lambda: eager(x, bc, delta))
            t_fg = _timed(lambda: graphed(x, bc, delta))
            t_ie = _timed(lambda: eager.inverse(y_e, bc, delta))
            t_ig = _timed(lambda: graphed.inverse(y_e, bc, delta))
        print(f"[smearing layer] batch 16: forward eager {t_fe:.1f} ms graph {t_fg:.1f} ms  "
              f"inverse eager {t_ie:.1f} ms graph {t_ig:.1f} ms", flush=True)


if __name__ == "__main__":
    unittest.main()
