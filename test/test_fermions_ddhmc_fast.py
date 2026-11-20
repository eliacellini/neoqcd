"""Fast path of the DD-HMC (neoqcd.fermions.fast_ddhmc) against the reference terms of
neoqcd.fermions.ddhmc: Haar links, batch > 1, both time signs, double and mixed precision, the
fused hop (pure-torch "loop" and the host build "cpu" of the CUDA source, skipped without a C++
compiler). CPU, complex128, seconds per test; statistical checks behind NEOQCD_SLOW=1."""

import os
import unittest

import torch

from neoqcd.fermions._gauge import SUN_haar
from neoqcd.fermions.ddhmc import DDHMC, BlockDecomposition, BlockTwoFlavour, DDHMCHamiltonian, SchurTwoFlavour
from neoqcd.fermions.fast import HalfLatticeWilson
from neoqcd.fermions.fast_ddhmc import (
    FastBlockTwoFlavour,
    FastDDHMCHamiltonian,
    FastSchurTwoFlavour,
    FusedBlockTwoFlavour,
)
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


# 4D: the double-precision fast term and one fused mixed-precision variant (2D runs all of them)
FOUR_D_VARIANTS = ("fast double", "fused cpu site mixed" if CPU_KERNEL else "fused loop mixed")


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


class FastBlockTermTest(unittest.TestCase):
    def compare(self, geometry, kappa, time_sign, seed, label="", names=None):
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
            if names is not None and name not in names:
                continue
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
        self.compare(G4, 0.12, -1.0, 20, "4D 8^4 SU(2)", FOUR_D_VARIANTS)

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

    def test_hamiltonian_options(self):
        bd = BlockDecomposition(G2, 4)
        ham = FastDDHMCHamiltonian(5.0, 0.2, bd, fast_kwargs={"precision": "mixed", "check_every": 5},
                                   kernel="loop", link_layout="site")
        self.assertIsInstance(ham.block, FusedBlockTwoFlavour)
        self.assertIsInstance(ham.schur, FastSchurTwoFlavour)
        for term in (ham.block, ham.schur):
            self.assertEqual((term.precision, term.check_every, term.kernel, term.link_layout),
                             ("mixed", 5, "loop", "site"))
        self.assertIsInstance(FastDDHMCHamiltonian(5.0, 0.2, bd).block, FastBlockTwoFlavour)
        self.assertIs(type(FastDDHMCHamiltonian(5.0, 0.2, bd, fast_schur=False).schur), SchurTwoFlavour)
        with self.assertRaises(ValueError):
            FastDDHMCHamiltonian(5.0, 0.2, bd, fast_kwargs={"kernel": "loop"})
        for bad in ({"precision": "single"}, {"kernel": "triton"}, {"link_layout": "aos"}, {"colour_product": "x"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                FastSchurTwoFlavour(0.2, bd, **bad)


def schur_variants():
    """(label, kwargs) of FastSchurTwoFlavour under test."""
    out = [("fast double", {}), ("fast mixed", {"precision": "mixed", "inner_tol": 1e-5}),
           ("fused loop", {"kernel": "loop"}), ("fused loop mixed", {"kernel": "loop", "precision": "mixed"})]
    if CPU_KERNEL:
        out += [("fused cpu", {"kernel": "cpu", "strict": True}),
                ("fused cpu site mixed", {"kernel": "cpu", "strict": True, "link_layout": "site", "precision": "mixed"})]
    return out


class FastSchurTermTest(unittest.TestCase):
    def compare(self, geometry, kappa, time_sign, seed, label="", names=None):
        bd = BlockDecomposition(geometry, 4)
        U, _ = haar_links(geometry, 2, seed)
        inactive = (bd.link_mask(~bd.active_links, U) > 0).expand_as(U)
        ref = SchurTwoFlavour(kappa, bd, time_sign=time_sign, **TIGHT)
        state = torch.Generator().manual_seed(seed + 1).get_state()
        g = torch.Generator()
        g.set_state(state)
        ref.refresh(U, g)
        S_ref, F_ref = ref.action(U), ref.force(U)
        V = U.clone()
        V[:, 0] = U[:, 0] @ U[:, 1]
        dS_ref = ref.action_difference(U, V)
        worst = {}
        for name, kw in schur_variants():
            if names is not None and name not in names:
                continue
            with self.subTest(variant=name, label=label):
                term = FastSchurTwoFlavour(kappa, bd, time_sign=time_sign, **TIGHT, **kw)
                g.set_state(state)
                term.refresh(U, g)
                self.assertEqual((term.stats["solves"], term.stats["refresh_calls"]), (2, 1))
                self.assertLess(rel(term.phi, ref.phi), 1e-11)  # two block solves at 1e-12
                self.assertLess(rel(term._refresh_action, ref._refresh_action), 1e-14)
                outside = ~bd.exterior_boundary.reshape((1,) + geometry.lattice + (1, 1)).expand_as(term.phi)
                self.assertEqual(term.phi[outside].abs().max().item(), 0.0)
                S, F = term.action(U), term.force(U)
                self.assertEqual((term.stats["solves"], term.stats["action_calls"], term.stats["force_calls"]), (5, 1, 1))
                self.assertEqual(F[inactive].abs().max().item(), 0.0)
                self.assertLess(rel(S, S_ref), 1e-10)
                self.assertLess(rel(S, term._refresh_action), 1e-10)  # S_R at the refresh links = eta^dag eta
                self.assertLess(rel(F, F_ref), 1e-9)
                dS = term.action_difference(U, V)
                self.assertLess((dS - dS_ref).abs().max().item(), 1e-9 * S_ref.abs().max().item())
                self.assertEqual(term.stats["unconverged_solves"], 0)
                self.assertLess(term.stats["last_residual"].max().item(), 1e-11)
                self.assertGreater(term.stats["cg_loop_iterations"], 0)
                worst[name] = (rel(S, S_ref), rel(F, F_ref))
        say(f"{label}: FastSchurTwoFlavour vs SchurTwoFlavour, (action, force) rel. err.: "
            + ", ".join(f"{k} ({a:.1e}, {b:.1e})" for k, (a, b) in worst.items())
            + ("" if CPU_KERNEL else f"; kernel cpu not tested: {CPU_WHY}"))

    def test_2d_both_time_signs(self):
        for time_sign in (-1.0, 1.0):
            self.compare(G2, 0.2, time_sign, 50 + int(time_sign), f"2D 8x8 SU(3) time_sign {time_sign:+g}")

    def test_4d_su2(self):
        self.compare(G4, 0.12, -1.0, 60, "4D 8^4 SU(2)", FOUR_D_VARIANTS)

    def test_per_chain_kappa(self):
        bd = BlockDecomposition(G2, 4)
        U, gen = haar_links(G2, 2, 61)
        kappa = torch.tensor([0.15, 0.21], dtype=torch.float64)
        ref, fast = SchurTwoFlavour(kappa, bd, **TIGHT), FastSchurTwoFlavour(kappa, bd, **TIGHT)
        state = gen.get_state()
        ref.refresh(U, gen)
        gen.set_state(state)
        fast.refresh(U, gen)
        self.assertLess(rel(fast.action(U), ref.action(U)), 1e-10)
        self.assertLess(rel(fast.force(U), ref.force(U)), 1e-9)

    def test_against_dense_matrices(self):
        """2D 8x8 SU(3), 4x4 blocks, both time signs: the refresh inverts R (the dense
        P(phi - D^-1 D_dOmega phi) of the fast phi_R is eta), the action is the dense |R^-1 phi|^2,
        the hops D_dOmega / D_dOmegaS equal the dense cut-link hops, and the force equals autograd of
        the dense action on every active link."""
        from neoqcd.fermions.algebra import algebra_basis
        from neoqcd.fermions.exact import assemble_hopping
        from neoqcd.fermions.fast import to_full
        G, kappa, bd = G2, 0.2, BlockDecomposition(G2, 4)
        N = G.N
        basis = algebra_basis(N)
        n = G.volume * G.spin * G.N
        eye = torch.eye(n, dtype=torch.cdouble)

        def site(m):
            return m.reshape(-1).repeat_interleave(G.spin * G.N).to(torch.cdouble)

        black, white, P = site(bd.black), site(bd.white), site(bd.exterior_boundary)
        cut = bd.link_mask(bd.cut_links, torch.zeros(1, dtype=torch.cdouble))
        for time_sign in (-1.0, 1.0):
            with self.subTest(time_sign=time_sign):
                U, gen = haar_links(G, 1, 70 + int(time_sign))
                term = FastSchurTwoFlavour(kappa, bd, time_sign=time_sign, **TIGHT)
                state = gen.get_state()
                term.refresh(U, gen)
                gen.set_state(state)
                eta = term._draw_eta(U, gen).reshape(-1)
                phi = term.phi.reshape(-1)

                def dense(theta):
                    X = torch.einsum("...a,aij->...ij", theta.to(U.dtype), basis)
                    V = torch.matmul(torch.linalg.matrix_exp(1j * X), U)
                    D = eye - kappa * assemble_hopping(V, G, time_sign=time_sign)[0]
                    Hcut = assemble_hopping(V * cut, G, time_sign=time_sign)[0]
                    return D, Hcut

                theta = torch.zeros(U.shape[:-2] + (N * N - 1,), dtype=torch.float64, requires_grad=True)
                D, Hcut = dense(theta)
                Ddo = black[:, None] * (-kappa * Hcut) * white[None, :]
                Ddos = white[:, None] * (-kappa * Hcut) * black[None, :]
                chi = P * (phi - torch.linalg.solve(D, Ddo @ phi))
                S = (chi.conj() * chi).real.sum()
                self.assertLess((chi - eta).abs().max().item(), 1e-10)  # R^-1 R eta = eta
                self.assertLess(abs(S.item() - term.action(U).item()), 1e-9)
                # the hops through the full operator with site masks = the dense cut-link hops
                op = term.full_operator(U)
                psi = torch.randn(G.wilson_shape(1), dtype=torch.cdouble, generator=gen)
                for fn, M in ((term.D_dOmega, Ddo), (term.D_dOmegaS, Ddos)):
                    h = fn(op, term._split(psi))
                    got = to_full(h[0], G, 0, odd=h[1]).reshape(-1)
                    self.assertLess((got - M.detach() @ psi.reshape(-1)).abs().max().item(), 1e-13)
                g, = torch.autograd.grad(S, theta)
                active = bd.link_mask(bd.active_links, U)[..., 0, 0]
                F_auto = torch.einsum("...a,aij->...ij", g.to(U.dtype), basis) * active[..., None, None]
                F = term.force(U)
                err = (F - F_auto).abs().max().item() / F_auto.abs().max().item()
                say(f"FastSchurTwoFlavour vs dense (time_sign {time_sign:+g}): |chi - eta| "
                    f"{(chi - eta).abs().max().item():.1e}, force vs autograd rel. err. {err:.1e}")
                self.assertLess(err, 1e-9)

    def test_stats_timer_hook_and_policy(self):
        from neoqcd.fermions.pseudofermion import TwoFlavourWilson
        bd = BlockDecomposition(G2, 4)
        U, gen = haar_links(G2, 2, 80)
        term = FastSchurTwoFlavour(0.2, bd, precision="mixed")
        self.assertTrue(set(TwoFlavourWilson(0.2, G2).stats) <= set(term.stats))
        self.assertTrue({"cg_loop_iterations", "outer_iterations", "double_applications"} <= set(term.stats))
        calls = []
        original = term._solve

        def wrapped(*a, **k):  # main_hmc's SolveTimer: an instance attribute
            calls.append(1)
            return original(*a, **k)

        term._solve = wrapped
        term.refresh(U, gen)
        term.action(U)
        term.force(U)
        self.assertEqual(len(calls), 5)
        self.assertEqual(term.stats["solves"], 5)
        self.assertGreater(term.stats["double_applications"], 0)
        self.assertEqual(tuple(term.stats["cg_iterations_per_sample"].shape), (2,))
        with self.assertRaisesRegex(RuntimeError, "FastSchurTwoFlavour refresh: CG not converged"):
            FastSchurTwoFlavour(0.2, bd, max_iter=2).refresh(U, gen)
        lax = FastSchurTwoFlavour(0.2, bd, max_iter=2, on_unconverged="ignore")
        lax.refresh(U, gen)
        lax.force(U)
        self.assertEqual(lax.stats["unconverged_solves"], 4)
        self.assertGreater(lax.stats["last_residual"].max().item(), 1e-6)
        # the operator cache: the same links tensor reuses it, an in-place write or a new kappa do not
        op = term.full_operator(U)
        self.assertIs(term.full_operator(U), op)
        U.mul_(1.0)
        self.assertIsNot(term.full_operator(U), op)
        op = term.full_operator(U)
        term.kappa = 0.19
        self.assertIsNot(term.full_operator(U), op)


def dd_hamiltonians(bd, kw, beta=5.0, kappa=0.2, substeps=2):
    """Reference and fast DD Hamiltonians with the same settings (Schur > block x substeps >
    gauge x substeps)."""
    common = dict(n_block_substeps=substeps, n_gauge_substeps=substeps, block_kwargs=dict(kw, reuse_guess=False),
                  schur_kwargs=kw)
    out = {"reference": DDHMCHamiltonian(beta, kappa, bd, **common),
           "fast": FastDDHMCHamiltonian(beta, kappa, bd, **common),
           "fast mixed": FastDDHMCHamiltonian(beta, kappa, bd, fast_kwargs={"precision": "mixed"}, **common),
           "fast block only": FastDDHMCHamiltonian(beta, kappa, bd, fast_schur=False, **common),
           "fused loop": FastDDHMCHamiltonian(beta, kappa, bd, kernel="loop", **common)}
    if CPU_KERNEL:
        out["fused cpu"] = FastDDHMCHamiltonian(beta, kappa, bd, kernel="cpu", strict=True, **common)
    return out


class FastDDTrajectoryTest(unittest.TestCase):
    def test_trajectory_fast_vs_reference(self):
        """Whole DD cycles (translation, refresh, three-level MD, Metropolis) from the same state
        and generator: dH and the links of every fast variant equal the reference to the solver
        tolerance, the same accept decisions (2D 8x8 SU(3) at both time signs, every variant)."""
        self.check_cycles([(G2, 0.2, -1.0, 2), (G2, 0.2, 1.0, 2)], tau=1.0, n_steps=2, substeps=2)

    def test_trajectory_fast_vs_reference_4d(self):
        """The same in 4D, 8^4 SU(2), one short cycle (tau 0.2, one step per level), the reference
        against one fast operator (the cpu fused kernel when it builds, else the --fast one)."""
        self.check_cycles([(G4, 0.12, -1.0, 1)], tau=0.2, n_steps=1, substeps=1,
                          names=("reference", "fused cpu" if CPU_KERNEL else "fast"))

    def check_cycles(self, cases, tau, n_steps, substeps, names=None):
        kw = dict(solver_tol=1e-13, action_tol=1e-13)
        for G, kappa, time_sign, n_cycles in cases:
            bd = BlockDecomposition(G, 4)
            U0, _ = haar_links(G, 2, 90)
            out = {}
            hams = dd_hamiltonians(bd, kw, kappa=kappa, substeps=substeps)
            if names is not None:
                hams = {k: v for k, v in hams.items() if k in names}
            for name, ham in hams.items():
                for term in (ham.schur, ham.block):
                    term.time_sign = time_sign
                g = torch.Generator().manual_seed(91)
                hmc = DDHMC(G, ham, tau=tau, n_steps=n_steps, translation="luscher")
                U, dHs, accs = U0, [], []
                for _ in range(n_cycles):
                    U, acc, dH, _ = hmc.cycle(U, g)
                    dHs.append(dH)
                    accs.append(acc)
                out[name] = (U, torch.cat(dHs), torch.cat(accs))
            ref = out.pop("reference")
            for name, (U, dH, acc) in out.items():
                with self.subTest(lattice=G.lattice, time_sign=time_sign, variant=name):
                    self.assertLess((dH - ref[1]).abs().max().item(), 1e-8)
                    self.assertTrue(torch.equal(acc, ref[2]))
                    self.assertLess((U - ref[0]).abs().max().item(), 1e-10)
            say(f"DD cycles {G.lattice} time_sign {time_sign:+g}: dH reference {ref[1].tolist()}, max |fast - ref| "
                + ", ".join(f"{k} {(v[1] - ref[1]).abs().max().item():.1e}" for k, v in out.items()))

    @unittest.skipUnless(SLOW, "~2-3 min on one CPU thread: NEOQCD_SLOW=1")
    def test_statistics_fast_vs_reference(self):
        """2D 8x8 SU(3), 4x4 blocks, beta = 5.0, kappa = 0.22, B = 32 chains from a cold start,
        luscher translations: <exp(-dH)> = 1 for the fast DD-HMC, and its plaquette equals the
        reference DD-HMC's (independent chains: other seeds) within 4 sigma."""
        import math
        from neoqcd.fermions._gauge import SUN_identity, mean_plaquette, plaquette_fields
        bd = BlockDecomposition(G2, 4)
        kw = dict(solver_tol=1e-10)
        res = {}
        for name, seed in (("reference", 700), ("fast", 710)):
            ham = dd_hamiltonians(bd, kw, kappa=0.22)[name]
            hmc = DDHMC(G2, ham, tau=1.0, n_steps=2, translation="luscher")
            gen = torch.Generator().manual_seed(seed)
            U = SUN_identity(G2.link_shape(32)[:-1], dtype=torch.cdouble)
            plaq, x = [], []
            for i in range(30 + 90):
                U, _, _, st = hmc.cycle(U, gen)
                if i >= 30:
                    plaq.append(mean_plaquette(plaquette_fields(U, 2), 3))
                    x.append(st["exp_minus_dH"])
            p = torch.stack(plaq).mean(0)
            e = torch.stack(x).mean(0)
            res[name] = (p.mean().item(), (p.std() / math.sqrt(32)).item(), e.mean().item(), (e.std() / math.sqrt(32)).item())
        (p_r, e_pr, x_r, e_xr), (p_f, e_pf, x_f, e_xf) = res["reference"], res["fast"]
        msg = (f"DD-HMC 2D 8x8 kappa 0.22: <P> reference {p_r:.4f}({e_pr:.4f}) fast {p_f:.4f}({e_pf:.4f}); "
               f"<exp(-dH)> reference {x_r:.4f}({e_xr:.4f}) fast {x_f:.4f}({e_xf:.4f})")
        say(msg)
        self.assertLess(abs(x_f - 1.0), 4 * e_xf, msg=msg)
        self.assertLess(abs(p_f - p_r), 4 * math.hypot(e_pr, e_pf), msg=msg)


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device (run on the cluster)")
class CudaDDTest(unittest.TestCase):
    """On a GPU: the masks built on the device, and DD cycles of the fast (broadcast_sum) and fused
    CUDA-kernel Hamiltonians (both link layouts, double and mixed) against the reference
    Hamiltonian on the same device and generator: dH to 1e-8, the same accept decisions."""

    def test_cycles_cuda_vs_reference(self):
        device = torch.device("cuda")
        kw = dict(solver_tol=1e-13, action_tol=1e-13)
        for G, kappa in ((G2, 0.2), (Geometry(4, 8, 8, 3), 0.153)):
            bd = BlockDecomposition(G, 4, device=device)
            self.assertEqual(bd.active_links.device.type, "cuda")
            U0 = haar_links(G, 2, 95)[0].to(device)
            common = dict(n_block_substeps=2, n_gauge_substeps=2, block_kwargs=dict(kw, reuse_guess=False),
                          schur_kwargs=kw)
            hams = {"reference": DDHMCHamiltonian(5.6, kappa, bd, **common),
                    "fast": FastDDHMCHamiltonian(5.6, kappa, bd, **common),
                    "cuda soa": FastDDHMCHamiltonian(5.6, kappa, bd, kernel="cuda", strict=True, **common),
                    "cuda site mixed": FastDDHMCHamiltonian(5.6, kappa, bd, kernel="cuda", strict=True, link_layout="site",
                                                            fast_kwargs={"precision": "mixed"}, **common)}
            out = {}
            for name, ham in hams.items():
                g = torch.Generator(device=device).manual_seed(96)
                hmc = DDHMC(G, ham, tau=0.5, n_steps=2, translation="luscher")
                U, dHs, accs = U0, [], []
                for _ in range(2):
                    U, acc, dH, _ = hmc.cycle(U, g)
                    dHs.append(dH)
                    accs.append(acc)
                out[name] = (U, torch.cat(dHs), torch.cat(accs))
            self.assertTrue(all(k[1].startswith("cuda") for k in bd._mask_cache))
            ref = out.pop("reference")
            for name, (U, dH, acc) in out.items():
                with self.subTest(lattice=G.lattice, variant=name):
                    self.assertLess((dH - ref[1]).abs().max().item(), 1e-8)
                    self.assertTrue(torch.equal(acc, ref[2]))
                    self.assertLess((U - ref[0]).abs().max().item(), 1e-10)
            say(f"CUDA DD cycles {G.lattice}: dH reference {ref[1].tolist()}, max |fast - ref| "
                + ", ".join(f"{k} {(v[1] - ref[1]).abs().max().item():.1e}" for k, v in out.items()))


if __name__ == "__main__":
    unittest.main()
