"""The O-REX/NEO-REX swap protocol on a box around the defect (--protocol-box, utils.ProtocolBox).

The swap step on the box is compared with the swap step on the whole lattice with the same random
numbers: the HB+OR random numbers are drawn on the whole lattice, recorded, and replayed cut by the
box. The two must agree to rounding (the action changes sum the same plaquettes in another order).
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.multiprocessing as mp
import torch.nn as nn

import neoqcd.fused as fused
import neoqcd.sun_utils as sun
from neoqcd.distributedPT import NEOREX, NEOREXProtocolWrapper, OutOfEquilibriumParallelTempering
from neoqcd.mcmc import HBOR, DefectMCMCAdapter
from neoqcd.utils import Defect, ProtocolBox, create_mask

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("main_pt_obc_box", ROOT / "main" / "main_pt_obc.py")
PT_OBC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PT_OBC)

TOL_LINKS = 1e-12
TOL_WORK = 1e-10


class _Tape:
    """Stands in for DefectMCMCAdapter._batch_hbor. On the whole lattice (box None) it draws the random
    numbers of every sweep from its own generator and records them; on a box it replays the records,
    cut by the box. The global generator (the coin of the symmetric updates) is not touched."""

    def __init__(self, hbor, records, box=None, seed=0):
        self.hbor, self.records, self.box = hbor, records, box
        self.generator = torch.Generator().manual_seed(seed)
        self.position = 0

    def __getattr__(self, name):
        return getattr(self.hbor, name)

    def __call__(self, cfgs, mask, defect, reverse=False):
        if self.box is None:
            rand = torch.rand(self.hbor._rand_shape(cfgs), generator=self.generator, dtype=torch.float64)
            self.records.append(rand)
        else:
            rand = self.box.cut(self.records[self.position], lattice_dim=3)
            self.position += 1
        return self.hbor(cfgs, mask, defect, rand=rand, reverse=reverse)


class _Local(nn.Module):
    """Stands in for DDP on a single process."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _thermalized(T, L, batch, seed, sweeps=3):
    torch.manual_seed(seed)
    pars = SimpleNamespace(N=3, D=4, orsteps=1, device=torch.device("cpu"))
    mask = create_mask(4, T, L)
    hbor = HBOR(pars, beta=6.0)
    cfgs = sun.SUN_haar((batch, 4, T, L, L, L, 3), dtype=torch.cdouble)
    with torch.no_grad():
        for _ in range(sweeps):
            cfgs = hbor(cfgs, mask, None)
    return cfgs


def _adapter(defect, T, L, batch, fused_kernel=False):
    pars = SimpleNamespace(N=3, D=4, T=T, L=L, orsteps=1, device=torch.device("cpu"), mask=create_mask(4, T, L),
                           defect=defect)
    adapter = DefectMCMCAdapter(pars, 6.0, defect.create_defect_mask(), batch, patch_mask=defect.protocol_patch_mask(1),
                                symmetric_updates=True)
    if fused_kernel:
        adapter._batch_hbor.use_fused_kernels = True
        adapter._batch_hbor.fused_emulate = True
        adapter._batch_hbor.fused_strict = True
    return adapter


def _swap_step(pt, x, start, target, active, accept, run):
    """Run pt's swap step of one parity with the pairing and the accept decision fixed."""
    pt.local_parameters = start.clone()
    pt._prepare_orex_targets = lambda parity: (target.clone(), active.clone())
    pt._accept_orex_work = lambda work: (accept.clone(), {"work": work.clone()})
    pt._scatter_parameters = lambda: None
    return run(x)


# Swap of three chains: the first two move (one up, one down), the third is not paired this parity
START = torch.tensor([0.0, 0.7, 1.0], dtype=torch.float64)
TARGET = torch.tensor([0.35, 0.3, 1.0], dtype=torch.float64)
ACTIVE = torch.tensor([True, True, False])
ACCEPT = torch.tensor([True, False, True])


class ProtocolBoxGeometryTest(unittest.TestCase):
    def test_box_of_the_production_patch_and_its_masks(self):
        # 16^4, defect of size 3, --protocol-patch-halfsize 1, smearing layer (buffer 2): 6 x 9^3 sites
        defect = Defect(4, 16, 16, 3, time_slice=8, space_slice=6)
        box = ProtocolBox(defect, 1, flow_buffer=2)
        self.assertEqual(box.shape, (6, 9, 9, 9))
        self.assertEqual((box.defect.time_slice, box.defect.space_slice), (2, 3))
        self.assertFalse(box.covers_lattice)
        # the cut masks are the masks of the defect at its box coordinates
        self.assertTrue(torch.equal(box.cut(defect.protocol_patch_mask(1)), box.defect.protocol_patch_mask(1)))
        self.assertTrue(torch.equal(box.cut(defect.create_defect_mask()), box.defect.create_defect_mask()))
        self.assertEqual(int(box.cut(defect.protocol_patch_mask(1)).sum()), int(defect.protocol_patch_mask(1).sum()))
        # every patch link lies at least one site inside the box (its staples are in the box)
        patch = box.defect.protocol_patch_mask(1)[0, 0, ..., 0, 0].to(torch.bool)
        for axis in range(4):
            sites = patch.sum(dim=tuple(a for a in range(4) if a != axis)) > 0
            self.assertFalse(bool(sites[0]) or bool(sites[-1]), axis)

    def test_wrapped_box_cut_and_paste(self):
        # a defect at the lattice edge: the patch and the box wrap around
        defect = Defect(4, 8, 8, 1, time_slice=0, space_slice=7)
        box = ProtocolBox(defect, 1)
        self.assertEqual(box.shape, (6, 7, 7, 7))
        self.assertTrue(torch.equal(box.cut(defect.protocol_patch_mask(1)), box.defect.protocol_patch_mask(1)))
        x = torch.randn((2, 4, 8, 8, 8, 8, 3, 3), dtype=torch.cdouble)
        self.assertTrue(torch.equal(box.paste(x, box.cut(x)), x))
        changed = box.paste(x, box.cut(x) + 1.0)
        moved = (changed != x).flatten(2, 5).any(-1).any(-1).any(1).any(0)  # lattice sites that changed
        self.assertEqual(int(moved.sum()), box.volume)
        self.assertTrue(torch.equal(moved.nonzero().squeeze(-1), box.sites.sort().values))

    def test_a_box_as_large_as_the_lattice_takes_the_whole_extent(self):
        # 8^4 with a defect of size 2: the patch with its frame needs 2 + 6 = 8 sites in space
        box = ProtocolBox(Defect(4, 8, 8, 2, time_slice=4, space_slice=3), 1, flow_buffer=2)
        self.assertEqual(box.shape, (6, 8, 8, 8))
        self.assertEqual(box.defect.space_slice, 3)
        self.assertTrue(ProtocolBox(Defect(4, 4, 4, 1, time_slice=2, space_slice=1), 1).covers_lattice)


class OREXProtocolBoxTest(unittest.TestCase):
    def _run(self, defect, T, L, box, records, fused_kernel=False):
        x = _thermalized(T, L, 3, seed=11)
        adapter = _adapter(defect, T, L, 3, fused_kernel)
        if box is not None:
            adapter = adapter.on_box(box)
        adapter._batch_hbor = _Tape(adapter._batch_hbor, records, box)
        pt = object.__new__(OutOfEquilibriumParallelTempering)
        pt.config = SimpleNamespace(orex_protocol_steps=2, orex_mcmc_steps_per_step=1)
        pt.B, pt.device, pt.param_dtype, pt.parameter_name = 3, torch.device("cpu"), torch.float64, "bc"
        pt.protocol_box = box
        torch.manual_seed(3)  # the coins of the symmetric updates
        with torch.no_grad():
            out, stats = _swap_step(pt, x, START, TARGET, ACTIVE, ACCEPT,
                                    lambda x: pt._step_parity(x, adapter, "even"))
        return x, out, stats["work"], adapter

    def assertSameSwap(self, defect, T, L, fused_kernel=False):
        box = ProtocolBox(defect, 1)
        self.assertFalse(box.covers_lattice)
        records = []
        x, full, work_full, _ = self._run(defect, T, L, None, records, fused_kernel)
        _, boxed, work_box, adapter = self._run(defect, T, L, box, records, fused_kernel)
        self.assertEqual(adapter._batch_hbor.position, len(records))  # every sweep replayed
        self.assertGreater(len(records), 0)
        if fused_kernel:
            self.assertEqual(adapter._batch_hbor.fused_used, "cpu")
        self.assertGreater((full[0] - x[0]).abs().max().item(), 1e-3)  # the accepted chain moved
        self.assertEqual((full[1] - x[1]).abs().max().item(), 0.0)  # the rejected one did not
        self.assertEqual((full[2] - x[2]).abs().max().item(), 0.0)  # nor the one out of the swap
        self.assertGreater(work_full[:2].abs().min().item(), 1e-3)
        self.assertEqual(work_full[2].item(), 0.0)
        self.assertLess((boxed - full).abs().max().item(), TOL_LINKS)
        self.assertLess((work_box - work_full).abs().max().item(), TOL_WORK)

    def test_box_equals_whole_lattice(self):
        self.assertSameSwap(Defect(4, 8, 8, 1, time_slice=4, space_slice=3), 8, 8)

    def test_box_equals_whole_lattice_with_the_patch_wrapped_around(self):
        self.assertSameSwap(Defect(4, 8, 8, 1, time_slice=0, space_slice=7), 8, 8)

    def test_box_equals_whole_lattice_with_the_fused_kernel(self):
        ok, why = fused.backend_status("cpu")
        if not ok:
            self.skipTest(f"no CPU build of the kernel: {why}")
        self.assertSameSwap(Defect(4, 8, 8, 1, time_slice=4, space_slice=3), 8, 8, fused_kernel=True)


def _perturb(module, scale, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(scale * torch.randn(parameter.shape, generator=g, dtype=parameter.dtype))


def _smearing_layer(T, L, defect, kind="smearing"):
    flags = (f"--cpu --algorithm neorex --D 4 --T {T} --L {L} --N 3 --beta 6 --defect-size {defect.dsize} "
             f"--time-slice {defect.time_slice} --space-slice {defect.space_slice} --parameter-min 0 "
             f"--parameter-max 1 --batch-size 3 --nf-layer-type {kind}").split()
    flags += (["--use-hyper-smearing", "--hyper-smearing-mode", "class", "--hyper-rho-init", "0.05",
               "--hyper-no-normalize-by-nstep"] if kind == "smearing" else ["--residual-quadratic"])
    with mock.patch.object(sys, "argv", ["main_pt_obc", *flags]):
        args = PT_OBC.parse_args()
    pars = PT_OBC._build_neorex_flow_pars(args, torch.device("cpu"), create_mask(4, T, L), defect)
    torch.manual_seed(5)
    cls = PT_OBC.ResidualDefectSNFProtocolLayer if kind == "residual" else PT_OBC.DefectSNFProtocolLayer
    layer = cls(pars)
    _perturb(layer, 0.05, 5)
    return layer, pars.patch_buffer


class NEOREXProtocolBoxTest(unittest.TestCase):
    def _run(self, layer, defect, T, L, box, records):
        x = _thermalized(T, L, 3, seed=13)
        adapter = _adapter(defect, T, L, 3)
        if box is not None:
            adapter = adapter.on_box(box)
        layer.set_frame(None if box is None else box.defect)
        adapter._batch_hbor = _Tape(adapter._batch_hbor, records, box)
        wrapper = NEOREXProtocolWrapper(layer, adapter, 2, stochastic_update=adapter, stochastic_steps_per_step=1,
                                        parameter_name="bc")
        pt = object.__new__(NEOREX)
        pt.ddp_flow = _Local(wrapper)
        pt.B, pt.rank, pt.world_size = 3, 0, 2
        pt.device, pt.param_dtype, pt.parameter_name = torch.device("cpu"), torch.float64, "bc"
        pt.protocol_box = box
        torch.manual_seed(3)
        with torch.no_grad():
            out, stats = _swap_step(pt, x, START, TARGET, ACTIVE, ACCEPT,
                                    lambda x: pt._neorex_step_parity(x, "even", train_flow=False))
        return x, out, stats["work"]

    def assertSameSwap(self, kind):
        T, L = 8, 8
        defect = Defect(4, T, L, 1, time_slice=4, space_slice=3)
        layer, buffer = _smearing_layer(T, L, defect, kind)
        box = ProtocolBox(defect, 1, flow_buffer=buffer)
        self.assertFalse(box.covers_lattice)
        records = []
        x, full, work_full = self._run(layer, defect, T, L, None, records)
        _, boxed, work_box = self._run(layer, defect, T, L, box, records)
        self.assertGreater((full[0] - x[0]).abs().max().item(), 1e-3)
        self.assertEqual((full[1] - x[1]).abs().max().item(), 0.0)
        self.assertEqual((full[2] - x[2]).abs().max().item(), 0.0)
        self.assertGreater(work_full[:2].abs().min().item(), 1e-3)
        self.assertLess((boxed - full).abs().max().item(), TOL_LINKS)
        self.assertLess((work_box - work_full).abs().max().item(), TOL_WORK)

    def test_smearing_layer_box_equals_whole_lattice(self):
        self.assertSameSwap("smearing")

    def test_residual_layer_box_equals_whole_lattice(self):
        self.assertSameSwap("residual")


class MainProtocolBoxTest(unittest.TestCase):
    """main_pt_obc.py with --protocol-box on two ranks: O-REX, and NEO-REX training its smearing flow."""

    def _run(self, extra):
        from test.test_cnf_neorex import _free_port, _multirank_worker
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = ["--cpu", "--D", "4", "--T", "8", "--L", "8", "--N", "3", "--defect-size", "1",
                    "--time-slice", "4", "--space-slice", "3", "--beta", "6", "--parameter-min", "0",
                    "--parameter-max", "1", "--batch-size", "2", "--thermal-steps", "1", "--steps", "2",
                    "--orsteps", "1", "--no-cfg-cache", "--output-dir", str(root / "run"), "--main-dir", str(root),
                    "--protocol-patch-halfsize", "1", "--protocol-symmetric-updates", "--protocol-box", *extra]
            mp.spawn(_multirank_worker, args=(2, _free_port(), argv), nprocs=2, join=True)
            meta = json.loads((root / "run" / "run_meta.json").read_text(encoding="utf-8"))
        swap = meta["swap"]
        self.assertEqual(swap["total_proposed"], sum(swap["link_proposed"]))
        self.assertGreater(swap["total_proposed"], 0)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in meta["work"]["link_mean"]))

    def test_orex(self):
        self._run(["--algorithm", "orex", "--orex-protocol-steps", "2"])

    def test_neorex_training_the_smearing_flow(self):
        self._run(["--algorithm", "neorex", "--nf-layer-type", "smearing", "--use-hyper-smearing",
                   "--hyper-smearing-mode", "class", "--neorex-protocol-steps", "2",
                   "--neorex-heatbath-steps-per-step", "1"])


if __name__ == "__main__":
    unittest.main()
