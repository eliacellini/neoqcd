"""Wilson cooling and the clover topological charge (neoqcd.topology), torch and kernels (neoqcd.fused).

On the CPU the kernels run as their emulation (the same source built with the host C++ compiler); on
CUDA the same comparisons run on the GPU.
"""

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch.multiprocessing as mp

import torch

import neoqcd.fused as fused
import neoqcd.sun_utils as sun
import neoqcd.topology as top
from neoqcd.mcmc import HBOR
from neoqcd.utils import create_mask

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
H = torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64)  # the U(1) direction diag(1, -1, 0) in SU(3)


def _thermalized(T, L, batch, seed, sweeps=5):
    g = torch.Generator().manual_seed(seed)
    cfgs = sun.SUN_haar((batch, 4, T, L, L, L, 3), dtype=torch.cdouble, generator=g)
    torch.manual_seed(seed)
    hbor = HBOR(SimpleNamespace(N=3, D=4, orsteps=1, device=torch.device("cpu")), beta=6.0)
    mask = create_mask(4, T, L)
    with torch.no_grad():
        for _ in range(sweeps):
            cfgs = hbor(cfgs, mask, None)
    return cfgs


def _constant_flux(extent, fluxes):
    """Abelian links exp(i theta_mu(x) H) with a constant plaquette angle -s_{mu nu} = -2 pi n / (L_mu L_nu)
    in the planes of `fluxes` {(mu, nu): n}, with "x + mu" = torch.roll(., 1, mu) as in sun.pstaple:
    theta_nu(x) += s c_mu(x), and the transition theta_mu(x) -= s L_mu c_nu(x) on the slice c_mu = 0."""
    coords = torch.meshgrid(*[torch.arange(n, dtype=torch.float64) for n in extent], indexing="ij")
    theta = [torch.zeros(extent, dtype=torch.float64) for _ in range(4)]
    for (mu, nu), n in fluxes.items():
        s = 2.0 * math.pi * n / (extent[mu] * extent[nu])
        theta[nu] = theta[nu] + s * coords[mu]
        theta[mu] = theta[mu] - s * extent[mu] * coords[nu] * (coords[mu] == 0)
    phases = torch.stack(theta)[..., None] * H  # (4, T, L, L, L, 3)
    return torch.diag_embed(torch.exp(1j * phases)).unsqueeze(0)  # (1, 4, T, L, L, L, 3, 3)


def _random_gauge(cfgs, seed, diagonal=False):
    # U_mu(x) -> g(x) U_mu(x) g(x + mu)^dagger, "x + mu" = roll(+1); diagonal: g(x) in the torus of SU(3)
    gen = torch.Generator().manual_seed(seed)
    if diagonal:
        a = 2 * math.pi * torch.rand(cfgs[:, 0].shape[:-2] + (2,), dtype=torch.float64, generator=gen)
        g = torch.diag_embed(torch.exp(1j * torch.cat((a, -a.sum(-1, keepdim=True)), dim=-1)))
    else:
        g = sun.SUN_haar(cfgs[:, 0].shape[:-1], dtype=cfgs.dtype, generator=gen)
    out = []
    for mu in range(4):
        g_next = torch.roll(g, 1, dims=1 + mu)
        out.append(sun.SUN_mul(sun.SUN_mul(g, cfgs[:, mu]), sun.SUN_dagger(g_next)))
    return torch.stack(out, dim=1)


class CloverChargeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ok, why = fused.backend_status("cpu")
        if not ok:
            raise unittest.SkipTest(f"no CPU build of the kernels: {why}")

    def test_constant_flux_gives_the_exact_lattice_charge(self):
        # F_{mu nu} = H sin(phi) on every site, so Q = 2 V sin(phi01) sin(phi23) / (4 pi^2) ~ 2 n01 n23
        extent = (8, 6, 6, 6)
        volume = 8 * 6 ** 3
        for n01, n23 in ((1, 1), (-1, 1), (2, -1), (0, 3)):
            cfgs = _constant_flux(extent, {(0, 1): n01, (2, 3): n23})
            # precondition: every (0, 1) plaquette is exp(-i s01 H)
            s01 = 2 * math.pi * n01 / (8 * 6)
            s23 = 2 * math.pi * n23 / 36
            plaq = sun.SUN_mul(sun.SUN_mul(cfgs[:, 0], torch.roll(cfgs[:, 1], 1, 1)),
                               sun.SUN_dagger(sun.SUN_mul(cfgs[:, 1], torch.roll(cfgs[:, 0], 1, 2))))
            expected_plaq = torch.diag_embed(torch.exp(-1j * s01 * H)).expand_as(plaq)
            self.assertLess((plaq - expected_plaq).abs().max().item(), 1e-12)
            exact = 2.0 * volume * math.sin(s01) * math.sin(s23) / (4 * math.pi ** 2)
            with self.subTest(n01=n01, n23=n23):
                self.assertLess(abs(top.clover_charge_torch(cfgs).item() - exact), 1e-11)
                for device in DEVICES:
                    got = fused.clover_charge(cfgs.to(device), emulate=True)
                    self.assertLess(abs(got.item() - exact), 1e-11, device)
                if n01 * n23:
                    self.assertAlmostEqual(exact / (2 * n01 * n23), 1.0, delta=0.02)  # near the integer 2 n01 n23

    def test_kernel_equals_torch_and_the_charge_is_gauge_invariant(self):
        cfgs = _thermalized(6, 4, 2, seed=1)
        reference = top.clover_charge_torch(cfgs)
        self.assertGreater(reference.abs().max().item(), 1e-3)
        for device in DEVICES:
            with self.subTest(device=device):
                got = fused.clover_charge(cfgs.to(device), emulate=True)
                self.assertLess((got.cpu() - reference).abs().max().item(), 1e-12)
        gauged = _random_gauge(cfgs, seed=2)
        self.assertGreater((gauged - cfgs).abs().max().item(), 0.1)
        self.assertLess((top.clover_charge_torch(gauged) - reference).abs().max().item(), 1e-12)
        self.assertLess((fused.clover_charge(gauged, emulate=True) - reference).abs().max().item(), 1e-12)

    def test_unequal_extents(self):
        # the charge kernel takes any four extents (cooling needs T x L^3)
        g = torch.Generator().manual_seed(4)
        A = torch.randn(2, 4, 4, 5, 6, 3, 3, 3, dtype=torch.cdouble, generator=g)
        Hm = 0.5 * (A + A.conj().transpose(-1, -2))
        Hm = Hm - torch.diagonal(Hm, dim1=-2, dim2=-1).mean(-1)[..., None, None] * torch.eye(3, dtype=torch.cdouble)
        cfgs = torch.linalg.matrix_exp(0.7j * Hm)
        reference = top.clover_charge_torch(cfgs)
        self.assertLess((fused.clover_charge(cfgs, emulate=True) - reference).abs().max().item(), 1e-12)


class CoolingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ok, why = fused.backend_status("cpu")
        if not ok:
            raise unittest.SkipTest(f"no CPU build of the kernels: {why}")

    def test_kernel_equals_torch(self):
        cfgs = _thermalized(6, 4, 2, seed=3)
        reference = top.cool_torch(cfgs, 4)
        for device in DEVICES:
            with self.subTest(device=device):
                got = fused.cooling(cfgs.to(device), 4, emulate=True)
                self.assertLess((got.cpu() - reference).abs().max().item(), 1e-12)
        self.assertGreater((reference - cfgs).abs().max().item(), 0.1)
        # no re-unitarization: 60 sweeps keep the links in SU(3) to rounding
        cooled = fused.cooling(cfgs, 60, emulate=True)
        identity = sun.SUN_identity(cooled.shape[:-1], dtype=cooled.dtype)
        self.assertLess((sun.SUN_mul(sun.SUN_dagger(cooled), cooled) - identity).abs().max().item(), 1e-12)
        self.assertLess((sun.SUN_determinant(cooled) - 1).abs().max().item(), 1e-12)

    def test_every_sweep_lowers_the_action_and_the_input_is_not_changed(self):
        cfgs = _thermalized(4, 4, 2, seed=5)
        before = cfgs.clone()
        plaquettes = [sun.plaquette_SUN(cfgs, 4, 3)]
        x = cfgs
        for _ in range(6):
            x = top.cool(x, 1, emulate=True)
            plaquettes.append(sun.plaquette_SUN(x, 4, 3))
        self.assertTrue(torch.equal(cfgs, before))
        for a, b in zip(plaquettes, plaquettes[1:]):
            self.assertTrue(bool((b > a).all()), (a, b))
        self.assertGreater(plaquettes[-1].min().item(), 0.95)

    def test_cooling_is_covariant_under_diagonal_gauge_transformations(self):
        # The SU(2) subgroups are fixed colour blocks, so cooling commutes with a gauge transformation only if
        # it maps each block to itself: g(x) diagonal. Then the cooled charges agree to rounding (a wrong
        # neighbour in the cooling would break this); a generic gauge transformation changes them slightly.
        cfgs = _thermalized(6, 4, 2, seed=6)
        q = top.topological_charge(cfgs, cooling_steps=10, emulate=True)
        q_diag = top.topological_charge(_random_gauge(cfgs, seed=7, diagonal=True), cooling_steps=10, emulate=True)
        self.assertLess((q - q_diag).abs().max().item(), 1e-9)
        cooled_torch = top.topological_charge(_random_gauge(cfgs, seed=7, diagonal=True), 10, use_kernels=False)
        self.assertLess((q - cooled_torch).abs().max().item(), 1e-9)

    def test_topological_charge_with_and_without_kernels(self):
        cfgs = _thermalized(4, 4, 3, seed=8)
        q_torch = top.topological_charge(cfgs, cooling_steps=8, use_kernels=False)
        q_kernel = top.topological_charge(cfgs, cooling_steps=8, emulate=True)
        self.assertLess((q_kernel - q_torch).abs().max().item(), 1e-11)
        self.assertEqual(tuple(top.topological_charge(cfgs[:0], 8, emulate=True).shape), (0,))


class MainTopologyTest(unittest.TestCase):
    """main_pt_obc.py --topology-every on two ranks (O-REX, torch cooling on the CPU)."""

    def _run(self, replicas):
        from test.test_cnf_neorex import _free_port, _multirank_worker
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = ["--cpu", "--algorithm", "orex", "--orex-protocol-steps", "1", "--D", "4", "--T", "8", "--L", "8",
                    "--N", "3", "--defect-size", "1", "--time-slice", "4", "--space-slice", "3", "--beta", "6",
                    "--parameter-min", "0", "--parameter-max", "1", "--batch-size", "2", "--thermal-steps", "1",
                    "--steps", "3", "--orsteps", "1", "--no-cfg-cache", "--output-dir", str(root / "run"),
                    "--main-dir", str(root), "--topology-every", "1", "--cooling-steps", "2",
                    "--topology-replicas", replicas]
            mp.spawn(_multirank_worker, args=(2, _free_port(), argv), nprocs=2, join=True)
            history = np.load(root / "run" / "replica_history.npz")
            meta = json.loads((root / "run" / "run_meta.json").read_text(encoding="utf-8"))
            return {key: history[key] for key in history.files}, meta["replica_history"]

    def test_charge_of_every_chain_and_of_the_top_rung(self):
        everything, summary_all = self._run("all")
        charge = everything["topological_charge"]
        self.assertEqual(charge.shape, (3, 2, 2))  # (measurements, ranks, chains)
        self.assertTrue(np.all(np.isfinite(charge)))
        self.assertEqual(everything["topology_steps"].tolist(), [0, 1, 2])
        self.assertEqual(int(everything["cooling_steps"]), 2)
        self.assertEqual(summary_all["topology_top"]["measurements"], 3)

        top, summary_top = self._run("top")
        charge = top["topological_charge"]
        labels = top["labels"][top["topology_steps"]]
        measured = np.isfinite(charge)
        # exactly the chains of the top rung, one per chain slot and measurement
        self.assertTrue(np.array_equal(measured, labels == 1))
        self.assertEqual(summary_top["topology_top"]["measurements"], 3)
        self.assertTrue(math.isfinite(summary_top["topology_top"]["Q"]["value"]) or
                        summary_top["topology_top"]["Q"]["status"] == "insufficient_statistics")


if __name__ == "__main__":
    unittest.main()
