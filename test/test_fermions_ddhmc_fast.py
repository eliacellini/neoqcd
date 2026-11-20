"""Fast path of the DD-HMC (neoqcd.fermions.fast_ddhmc) against the reference terms of
neoqcd.fermions.ddhmc: Haar links, batch > 1, both time signs, double and mixed precision, the
fused hop (pure-torch "loop" and the host build "cpu" of the CUDA source, skipped without a C++
compiler). CPU, complex128, seconds per test; statistical checks behind NEOQCD_SLOW=1."""

import os
import unittest

import torch

from neoqcd.fermions._gauge import SUN_haar
from neoqcd.fermions.ddhmc import DDHMC, BlockDecomposition, BlockTwoFlavour, DDHMCHamiltonian
from neoqcd.fermions.fast import HalfLatticeWilson
from neoqcd.fermions.fast_ddhmc import FastBlockTwoFlavour, FastDDHMCHamiltonian, FusedBlockTwoFlavour
from neoqcd.fermions.fused import backend_status
from neoqcd.fermions.lattice import Geometry

VERBOSE = bool(os.environ.get("NEOQCD_TEST_VERBOSE"))
SLOW = os.environ.get("NEOQCD_SLOW") == "1"
CPU_KERNEL, CPU_WHY = backend_status("cpu")

G2 = Geometry(2, 8, 8, 3)  # 2D 8x8 SU(3), 4x4 blocks
G4 = Geometry(4, 8, 8, 2)  # 4D 8^4 SU(2), 4^4 blocks
TIGHT = dict(solver_tol=1e-12, action_tol=1e-12, max_iter=5000)


def say(msg):
    if VERBOSE:
        print("\n" + msg)


def haar_links(geometry, batch, seed):
    gen = torch.Generator().manual_seed(seed)
    return SUN_haar(geometry.link_shape(batch)[:-1], generator=gen), gen


def variants():
    """(label, class, kwargs) of the fast block terms under test."""
    out = [("fast double", FastBlockTwoFlavour, {}),
           ("fast mixed", FastBlockTwoFlavour, {"precision": "mixed", "inner_tol": 1e-5}),
           ("fused loop", FusedBlockTwoFlavour, {"kernel": "loop"}),
           ("fused loop mixed", FusedBlockTwoFlavour, {"kernel": "loop", "precision": "mixed"})]
    if CPU_KERNEL:
        out += [("fused cpu", FusedBlockTwoFlavour, {"kernel": "cpu", "strict": True}),
                ("fused cpu site mixed", FusedBlockTwoFlavour,
                 {"kernel": "cpu", "strict": True, "link_layout": "site", "precision": "mixed"})]
    return out


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


class FastBlockTermTest(unittest.TestCase):
    def compare(self, geometry, kappa, time_sign, seed, label=""):
        bd = BlockDecomposition(geometry, 4)
        U, _ = haar_links(geometry, 2, seed)
        inactive = (bd.link_mask(~bd.active_links, U) > 0).expand_as(U)
        ref = BlockTwoFlavour(kappa, bd, time_sign=time_sign, reuse_guess=False, **TIGHT)
        state = torch.Generator().manual_seed(seed + 1).get_state()
        g = torch.Generator()
        g.set_state(state)
        ref.refresh(U, g)
        S_ref, F_ref = ref.action(U), ref.force(U)
        V = U.clone()
        V[:, 0] = U[:, 0] @ U[:, 1]  # another configuration, links still SU(N)
        dS_ref = ref.action_difference(U, V)
        worst = {}
        for name, cls, kw in variants():
            with self.subTest(variant=name, label=label):
                term = cls(kappa, bd, time_sign=time_sign, reuse_guess=False, **TIGHT, **kw)
                g.set_state(state)
                term.refresh(U, g)
                self.assertLess(rel(term.phi, ref.phi), 1e-14)  # same draws, no solve
                self.assertLess(rel(term._refresh_action, ref._refresh_action), 1e-14)  # eta^dag eta, summed in another order
                S, F = term.action(U), term.force(U)
                self.assertEqual(F[inactive].abs().max().item(), 0.0)
                self.assertLess(rel(S, S_ref), 1e-10)
                self.assertLess(rel(S, term._refresh_action), 1e-10)  # S at the refresh links = eta^dag eta
                self.assertLess(rel(F, F_ref), 1e-9)
                dS = term.action_difference(U, V)
                self.assertLess((dS - dS_ref).abs().max().item(), 1e-9 * S_ref.abs().max().item())
                self.assertEqual(term.stats["unconverged_solves"], 0)
                worst[name] = (rel(S, S_ref), rel(F, F_ref))
        say(f"{label}: fast block term vs BlockTwoFlavour, (action, force) rel. err.: "
            + ", ".join(f"{k} ({a:.1e}, {b:.1e})" for k, (a, b) in worst.items())
            + ("" if CPU_KERNEL else f"; kernel cpu not tested: {CPU_WHY}"))

    def test_2d_both_time_signs(self):
        for time_sign in (-1.0, 1.0):
            self.compare(G2, 0.2, time_sign, 10 + int(time_sign), f"2D 8x8 SU(3) time_sign {time_sign:+g}")

    def test_4d_su2(self):
        self.compare(G4, 0.12, -1.0, 20, "4D 8^4 SU(2)")

    def test_per_chain_kappa(self):
        bd = BlockDecomposition(G2, 4)
        U, gen = haar_links(G2, 2, 30)
        kappa = torch.tensor([0.15, 0.21], dtype=torch.float64)
        ref = BlockTwoFlavour(kappa, bd, reuse_guess=False, **TIGHT)
        fast = FastBlockTwoFlavour(kappa, bd, reuse_guess=False, **TIGHT)
        state = gen.get_state()
        ref.refresh(U, gen)
        gen.set_state(state)
        fast.refresh(U, gen)
        self.assertLess(rel(fast.action(U), ref.action(U)), 1e-10)
        self.assertLess(rel(fast.force(U), ref.force(U)), 1e-9)

    def test_operator_cache_on_the_unmasked_links(self):
        bd = BlockDecomposition(G2, 4)
        U, gen = haar_links(G2, 2, 31)
        term = FastBlockTwoFlavour(0.2, bd)
        op = term.operator(U)
        self.assertIsInstance(op, HalfLatticeWilson)
        self.assertIs(term.operator(U), op)  # cached on U itself, not on the masked copy
        self.assertTrue(torch.equal(op.U, U * bd.link_mask(bd.inblock_links, U)))
        self.assertEqual(op.U[(bd.link_mask(bd.cut_links, U) > 0).expand_as(U)].abs().max().item(), 0.0)
        U.mul_(1.0)  # an in-place write bumps _version: a new operator
        self.assertIsNot(term.operator(U), op)
        op2 = term.operator(U)
        term.kappa = 0.19  # a new kappa: a new operator
        self.assertIsNot(term.operator(U), op2)
        fused = FusedBlockTwoFlavour(0.2, bd, kernel="loop")
        self.assertEqual(fused.operator(U).kernel, "loop")
        self.assertTrue(torch.equal(fused.operator(U).U, op.U))

    def test_dd_trajectory_fast_block_vs_reference(self):
        """One DD trajectory (Schur > block > gauge) with the fast block term against the
        reference Hamiltonian from the same state: dH and the links to the solver tolerance."""
        bd = BlockDecomposition(G2, 4)
        U, gen = haar_links(G2, 2, 40)
        kw = dict(solver_tol=1e-13, action_tol=1e-13)
        out = {}
        for name, ham in (("reference", DDHMCHamiltonian(5.0, 0.2, bd, 2, 2, block_kwargs=dict(kw, reuse_guess=False),
                                                         schur_kwargs=kw)),
                          ("fast", FastDDHMCHamiltonian(5.0, 0.2, bd, 2, 2, block_kwargs=dict(kw, reuse_guess=False),
                                                        schur_kwargs=kw)),
                          ("fused", FastDDHMCHamiltonian(5.0, 0.2, bd, 2, 2, block_kwargs=dict(kw, reuse_guess=False),
                                                         schur_kwargs=kw, kernel="loop"))):
            g = torch.Generator().manual_seed(41)
            hmc = DDHMC(G2, ham, tau=1.0, n_steps=2, translation="luscher")
            U1, acc, dH, st = hmc.cycle(U, g)
            out[name] = (U1, dH)
        self.assertIsInstance(FastDDHMCHamiltonian(5.0, 0.2, bd).block, FastBlockTwoFlavour)
        for name in ("fast", "fused"):
            self.assertLess((out[name][1] - out["reference"][1]).abs().max().item(), 1e-8)
            self.assertLess((out[name][0] - out["reference"][0]).abs().max().item(), 1e-10)
        say(f"DD trajectory, fast block term: dH reference {out['reference'][1].tolist()}, "
            f"|fast - ref| {(out['fast'][1] - out['reference'][1]).abs().max().item():.1e}")

    def test_hamiltonian_options(self):
        bd = BlockDecomposition(G2, 4)
        ham = FastDDHMCHamiltonian(5.0, 0.2, bd, fast_kwargs={"precision": "mixed", "check_every": 5},
                                   kernel="loop", link_layout="site")
        self.assertIsInstance(ham.block, FusedBlockTwoFlavour)
        self.assertEqual((ham.block.precision, ham.block.check_every, ham.block.kernel, ham.block.link_layout),
                         ("mixed", 5, "loop", "site"))
        with self.assertRaises(ValueError):
            FastDDHMCHamiltonian(5.0, 0.2, bd, fast_kwargs={"kernel": "loop"})


if __name__ == "__main__":
    unittest.main()
