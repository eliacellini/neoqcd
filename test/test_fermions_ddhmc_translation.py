"""Tests of the DD-HMC translations (neoqcd.fermions.ddhmc): M. Luescher, hep-lat/0409106, eq. (3.9)
and appendix C, eqs. (C.1) and (C.2).

CPU, complex128, fixed seeds; prints behind NEOQCD_TEST_VERBOSE.
(a) A translation is an exact symmetry: covariance of the Wilson operator with the boundary sign
    moved by psi -> time_sign psi on the wrapped time slices; dense det D (modulus and phase) and
    the gauge action on 4D lattices, for time shifts across the antiperiodic boundary.
(b) The paper's translation vectors (translation="luscher"): n and v of eqs. (C.1), (C.2), the
    sequence of `DDHMC.next_shifts` and `cycle`, the visiting distribution of the links per period
    (exact, against an analytic count) and in a long run against random vectors; the active-link
    fractions quoted in sect. 3.5; decompositions without active links are refused.
(c) <exp(-dH)> = 1 over "luscher" cycles, inactive links bitwise unchanged in every trajectory.
The other DD-HMC checks (reversibility, frozen links, forces, factorisation, the slow comparison
with the exact-determinant Metropolis) are in test_fermions_ddhmc.py.
"""

import math
import os
import unittest

import torch

from neoqcd.fermions._gauge import SUN_haar, SUN_identity, mean_plaquette, plaquette_fields
from neoqcd.fermions.ddhmc import (
    TRANSLATIONS,
    BlockDecomposition,
    DDHMC,
    DDHMCHamiltonian,
    luscher_translation_step,
    translate,
)
from neoqcd.fermions.dirac import WilsonDirac
from neoqcd.fermions.exact import assemble_even_odd
from neoqcd.fermions.hmc import WilsonGaugeAction
from neoqcd.fermions.lattice import Geometry

VERBOSE = bool(os.environ.get("NEOQCD_TEST_VERBOSE"))

G2_8 = Geometry(2, 8, 8, 3)  # 2D 8x8 SU(3), 4x4 blocks: the smallest 2D case with active links


def say(msg):
    if VERBOSE:
        print("\n" + msg)


def haar_links(geometry, batch, seed):
    gen = torch.Generator().manual_seed(seed)
    return SUN_haar(geometry.link_shape(batch)[:-1], generator=gen), gen


def translate_spinor(psi, shifts, time_sign):
    """psi'(n) = sigma(n) psi(n - s) per chain, the companion of `translate(U, s)`: sigma(n) =
    time_sign on the time slices 0 <= n_0 < s_0 (those that came across the time boundary) and 1
    elsewhere, so that the boundary twist stays on the time links at t = 0."""
    B, D = shifts.shape
    T = psi.shape[1]
    out = []
    for b in range(B):
        s = tuple(int(v) % psi.shape[1 + mu] for mu, v in enumerate(shifts[b].tolist()))
        p = torch.roll(psi[b:b + 1], shifts=s, dims=tuple(range(1, 1 + D)))
        sigma = torch.ones(T, dtype=psi.real.dtype)
        sigma[:s[0]] = time_sign
        out.append(p * sigma.view((1, T) + (1,) * (psi.dim() - 2)))
    return torch.cat(out)


def period_counts(bd, n, step):
    """Visits of every link of the field in one period: random vector r = 0, then k * step,
    k = 1, ..., n - 1 (after U'(n) = U(n - s) the link at x sits at x + s). For a uniform random
    vector the visits of ONE link are distributed as these counts over all links."""
    D = bd.active_links.shape[0]
    c = torch.zeros(bd.active_links.shape, dtype=torch.long)
    for k in range(n):
        c += torch.roll(bd.active_links, shifts=tuple(-k * v for v in step), dims=tuple(range(1, D + 1))).long()
    return c


def dd_hamiltonian(bd, beta, kappa, tol=1e-10, n_block=2, n_gauge=2):
    return DDHMCHamiltonian(beta, kappa, bd, n_block_substeps=n_block, n_gauge_substeps=n_gauge,
                            block_kwargs={"solver_tol": tol, "reuse_guess": False},
                            schur_kwargs={"solver_tol": tol})


# ---------------------------------------------------------------------------------------------
# (a) symmetry
# ---------------------------------------------------------------------------------------------
class TranslationSymmetryTest(unittest.TestCase):

    def test_dirac_covariance_time_sign(self):
        """D[T_s U] (sigma T_s psi) = sigma T_s (D[U] psi) to rounding for time_sign = -1 and +1,
        shifts across the time boundary and in all directions (4^4 SU(3), 8 x 4^3 SU(2)). Without
        sigma it fails exactly on the chains shifted in time with time_sign = -1: the sign
        bookkeeping is what makes det D translation invariant."""
        kappa = 0.13
        for G, seed in ((Geometry(4, 4, 4, 3), 101), (Geometry(4, 8, 4, 2), 102)):
            U, gen = haar_links(G, 5, seed)
            shifts = torch.tensor([[1, 0, 0, 0], [G.T - 1, 0, 0, 0], [0, 1, 2, 3], [3, 1, 3, 2],
                                   [G.T // 2, 2, 0, 1]])
            Ut = translate(U, shifts)
            psi = torch.randn(G.wilson_shape(5), generator=gen, dtype=torch.cdouble)
            for ts in (-1.0, 1.0):
                lhs = WilsonDirac(Ut, kappa, G, ts).apply(translate_spinor(psi, shifts, ts))
                rhs = translate_spinor(WilsonDirac(U, kappa, G, ts).apply(psi), shifts, ts)
                err = (lhs - rhs).abs().max().item()
                naive = (WilsonDirac(Ut, kappa, G, ts).apply(translate_spinor(psi, shifts, 1.0))
                         - translate_spinor(WilsonDirac(U, kappa, G, ts).apply(psi), shifts, 1.0))
                naive = naive.reshape(5, -1).abs().max(1).values
                say(f"{G.lattice} SU({G.N}) time_sign {ts:+.0f}: covariance error {err:.1e}; "
                    f"without the sign {[f'{x:.1e}' for x in naive.tolist()]}")
                self.assertLess(err, 1e-13)
                for b in range(5):
                    if ts == -1.0 and int(shifts[b, 0]) != 0:
                        self.assertGreater(naive[b].item(), 1e-2)
                    else:
                        self.assertLess(naive[b].item(), 1e-13)

    def test_dense_determinant_and_gauge_action(self):
        """det D (modulus and phase of the dense even-odd matrix of exact.py, built from the links
        independently of WilsonDirac) and the Wilson gauge action are unchanged by translations
        across the time boundary and in all directions, for time_sign = -1 and +1; the twist is
        felt (log|det| differs between -1 and +1), so the check is not empty. 4^4 and 8 x 4^3
        SU(2), Haar links."""
        kappa, beta = 0.15, 2.0
        for G, seed in ((Geometry(4, 4, 4, 2), 111), (Geometry(4, 8, 4, 2), 112)):
            U, gen = haar_links(G, 1, seed)
            shifts = torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0], [G.T - 1, 0, 0, 0], [G.T // 2 + 1, 1, 2, 3],
                                   [0, 3, 0, 1]])
            U5 = U.repeat((5,) + (1,) * (U.dim() - 1))
            Ut = translate(U5, shifts)
            self.assertTrue(torch.equal(Ut[0], U[0]))
            self.assertTrue(torch.equal(translate(Ut, -shifts), U5))
            S_g = WilsonGaugeAction(beta, G).action(Ut)
            self.assertLess((S_g - S_g[0]).abs().max().item(), 1e-12 * S_g[0].abs().item())
            logabs = {}
            for ts in (-1.0, 1.0):
                sign, la = zip(*[torch.linalg.slogdet(assemble_even_odd(Ut[b:b + 1], kappa, G, ts)[0])
                                 for b in range(5)])
                sign, la = torch.stack(sign), torch.stack(la)
                logabs[ts] = la[0].item()
                say(f"{G.lattice} time_sign {ts:+.0f}: log|det D| {la.tolist()}, phase {sign.tolist()}")
                self.assertLess((la - la[0]).abs().max().item(), 1e-10)
                self.assertLess((sign - sign[0]).abs().max().item(), 1e-10)
            self.assertGreater(abs(logabs[-1.0] - logabs[1.0]), 1e-3)


# ---------------------------------------------------------------------------------------------
# (b) the translation vectors of appendix C
# ---------------------------------------------------------------------------------------------
class LuscherSchemeTest(unittest.TestCase):

    def test_step_eqs_C1_C2(self):
        """n = min{10, b_0, ..., b_3} / 2 and v = b / n (eq. C.2), nearest integer with halves up when
        b_mu / n is not an integer; for 8^4 blocks n = 4 and v = (2, 2, 2, 2) (eq. C.1)."""
        cases = {
            (8, 8, 8, 8): (4, (2, 2, 2, 2)),  # eq. (C.1): a random translation, then three times v
            (4, 4, 4, 4): (2, (2, 2, 2, 2)),
            (6, 6, 6, 6): (3, (2, 2, 2, 2)),
            (10, 10, 10, 10): (5, (2, 2, 2, 2)),
            (12, 12, 12, 12): (5, (2, 2, 2, 2)),  # 12 / 5 = 2.4
            (16, 16, 16, 16): (5, (3, 3, 3, 3)),  # 3.2
            (20, 20, 20, 20): (5, (4, 4, 4, 4)),
            (8, 6, 6, 12): (3, (3, 2, 2, 4)),  # the 32 x 24^3 blocks of table 1: 8 / 3 -> 3
            (10, 8): (4, (3, 2)),  # 2.5 -> 3
            (6, 6): (3, (2, 2)),  # fig. 9: three copies of the grid
            (4, 4): (2, (2, 2)),
            (2, 4): (1, (2, 4)),  # n = 1: no regular step
        }
        for b, expect in cases.items():
            self.assertEqual(luscher_translation_step(b), expect, msg=str(b))
        self.assertEqual(luscher_translation_step((8, 8, 8, 8), n=2), (2, (4, 4, 4, 4)))
        for bad in ((3, 4), (0, 4), (), (4, 5)):
            with self.assertRaises(ValueError):
                luscher_translation_step(bad)
        with self.assertRaises(ValueError):
            luscher_translation_step((4, 4), n=0)

    def test_active_fraction_sect_3_5(self):
        """Sect. 3.5: with 6^4 blocks the active links are "a minority of 25%", and the fraction
        exceeds 50% from 12^4 blocks on (so 10^4 stays below). Masks on 12^4 and 24^4 lattices
        against the analytic count."""
        b6 = BlockDecomposition(Geometry(4, 12, 12, 2), 6)
        b12 = BlockDecomposition(Geometry(4, 24, 24, 2), 12)
        for bd in (b6, b12):
            self.assertAlmostEqual(bd.active_fraction, bd.expected_active_fraction(), places=14)
        self.assertAlmostEqual(b6.active_fraction, 4 * 5 * 4 ** 3 / (4 * 6 ** 4), places=14)
        self.assertEqual(round(100 * b6.active_fraction), 25)
        self.assertGreater(b12.active_fraction, 0.5)
        b10 = BlockDecomposition(Geometry(4, 20, 20, 2), 10)
        self.assertLess(b10.active_fraction, 0.5)
        say(f"active fractions: 6^4 {b6.active_fraction:.4f}, 10^4 {b10.active_fraction:.4f}, "
            f"12^4 {b12.active_fraction:.4f}")

    def test_no_active_links_refused(self):
        """A block size 2 in any direction leaves no active link: DDHMC raises for every scheme."""
        for G, b in ((G2_8, (2, 4)), (Geometry(4, 4, 4, 2), 2)):
            bd = BlockDecomposition(G, b)
            self.assertEqual(bd.active_fraction, 0.0)
            for tr in TRANSLATIONS:
                with self.assertRaises(ValueError):
                    DDHMC(G, DDHMCHamiltonian(2.0, 0.12, bd), tau=0.5, n_steps=1, translation=tr)

    def test_sequence(self):
        """next_shifts with translation="luscher", 16^4 with 8^4 blocks: per period one uniform
        random vector (its own per chain), then three shifts (2, 2, 2, 2), then a new random vector;
        a new batch size restarts the period; n_regular = 1 gives n = 2 and v = (4, 4, 4, 4)."""
        G = Geometry(4, 16, 16, 2)
        bd = BlockDecomposition(G, 8)
        ham = DDHMCHamiltonian(2.0, 0.12, bd)
        hmc = DDHMC(G, ham, tau=0.5, n_steps=1, translation="luscher")
        self.assertEqual((hmc.n_regular, hmc.luscher_step), (3, (2, 2, 2, 2)))
        gen = torch.Generator().manual_seed(121)
        B = 64
        step = torch.tensor([2, 2, 2, 2]).expand(B, -1)
        randoms = []
        for k in range(12):
            s = hmc.next_shifts(B, gen)
            self.assertEqual(tuple(s.shape), (B, 4))
            if k % 4 == 0:
                randoms.append(s)
                self.assertTrue(bool(((s >= 0) & (s < 16)).all()))
            else:
                self.assertTrue(torch.equal(s, step), msg=f"cycle {k}: {s[:2].tolist()}")
        self.assertFalse(torch.equal(randoms[0], randoms[1]))
        self.assertGreater(len({tuple(r) for r in randoms[0].tolist()}), B // 2)  # chains differ
        allr = torch.cat(randoms)
        for mu in range(4):  # the random vectors move the grid relative to the field by any residue
            self.assertEqual(set((allr[:, mu] % 8).tolist()), set(range(8)))
        hmc.next_shifts(B, gen)  # random vector of a new period
        s = hmc.next_shifts(B + 1, gen)  # new batch size: a new period, not the regular step
        self.assertFalse(bool((s == 2).all()))
        self.assertTrue(torch.equal(hmc.next_shifts(B + 1, gen), torch.tensor([2, 2, 2, 2]).expand(B + 1, -1)))
        h2 = DDHMC(G, ham, tau=0.5, n_steps=1, translation="luscher", n_regular=1)
        self.assertEqual(h2.luscher_step, (4, 4, 4, 4))
        seq = [tuple(h2.next_shifts(1, gen)[0].tolist()) for _ in range(4)]
        self.assertEqual((seq[1], seq[3]), ((4, 4, 4, 4), (4, 4, 4, 4)))
        self.assertNotEqual(seq[0], seq[2])  # two random vectors
        self.assertEqual(h2._k, 0)  # period n = 2

    def test_period_coverage_exact(self):
        """The visiting distribution of one period (exact over the uniform random vector). The mean
        is n f in every case (each cycle activates the fraction f). Fig. 9 (2D, 6x6 blocks, three
        grids): every link is active at least once per period. In 4D the paper's scheme does not
        cover every link in a period; analytic count for cubic blocks, b = 2n, v = (2, ..., 2): a link
        along mu is cut at one of the n offsets iff its local y_mu is even, and every transverse
        coordinate excludes exactly one offset (uniformly), so P(never) = P(the excluded offsets
        exhaust all n). 8^4: (1/2) 4!/4^4 = 3/64. 6^4: (1/2)(3!/3^3) + (1/2)(36/3^4) = 1/3
        (36 surjections of 4 onto 3). None for 10^4 and 12^4. In all cases the variance of the
        visits is below n f (1 - f), that of n independent random vectors (appendix C: "more
        balanced visiting frequencies")."""
        cases = ((G2_8, 4, None), (Geometry(2, 12, 12, 3), 6, 0.0), (Geometry(2, 16, 16, 3), 8, 0.0),
                 (Geometry(4, 8, 8, 2), 4, None), (Geometry(4, 12, 12, 2), 6, 1 / 3),
                 (Geometry(4, 16, 16, 2), 8, 3 / 64), (Geometry(4, 20, 20, 2), 10, 0.0),
                 (Geometry(4, 16, 8, 2), (8, 4, 4, 4), None))
        for G, b, never in cases:
            bd = BlockDecomposition(G, b)
            n, v = luscher_translation_step(bd.block_size)
            c = period_counts(bd, n, v).double()
            f = bd.expected_active_fraction()
            p0 = (c == 0).double().mean().item()
            var = ((c - c.mean()) ** 2).mean().item()
            say(f"{G.lattice} blocks {bd.block_size}: n = {n}, v = {v}, visits per period mean {c.mean().item():.4f} "
                f"(n f = {n * f:.4f}), variance {var:.4f} (random {n * f * (1 - f):.4f}), never visited {p0:.4f}")
            self.assertAlmostEqual(c.mean().item(), n * f, places=12)
            self.assertLess(var, n * f * (1 - f))
            if never is not None:
                self.assertAlmostEqual(p0, never, places=12)

    def test_long_run_balance(self):
        """100 cycles of DDHMC.next_shifts on 16^4 with 8^4 blocks: every link is visited
        100 f times on average for both schemes; the spread of the visit counts across links is
        clearly smaller with "luscher" than with "random", and so is the shortfall of the least
        visited links (appendix C)."""
        G = Geometry(4, 16, 16, 2)
        bd = BlockDecomposition(G, 8)
        ham = DDHMCHamiltonian(2.0, 0.12, bd)
        n_cycles = 100
        out = {}
        for tr in ("random", "luscher"):
            hmc = DDHMC(G, ham, tau=0.5, n_steps=1, translation=tr)
            gen = torch.Generator().manual_seed(131)
            pos = torch.zeros(4, dtype=torch.long)
            c = torch.zeros(bd.active_links.shape, dtype=torch.long)
            for _ in range(n_cycles):
                pos = pos + hmc.next_shifts(1, gen)[0]
                c += torch.roll(bd.active_links, shifts=tuple(-int(x) for x in pos), dims=(1, 2, 3, 4)).long()
            out[tr] = c.double()
        f = bd.expected_active_fraction()
        for tr, c in out.items():
            self.assertAlmostEqual(c.mean().item(), n_cycles * f, places=9)
        sd = {tr: c.std().item() for tr, c in out.items()}
        lo = {tr: c.min().item() for tr, c in out.items()}
        say(f"100 cycles, 8^4 blocks: visits mean {n_cycles * f:.2f}, std random {sd['random']:.2f} / "
            f"luscher {sd['luscher']:.2f}, min random {lo['random']:.0f} / luscher {lo['luscher']:.0f}")
        self.assertLess(sd["luscher"], 0.8 * sd["random"])
        self.assertGreater(lo["luscher"], lo["random"])


# ---------------------------------------------------------------------------------------------
# (c) the update with the "luscher" translations
# ---------------------------------------------------------------------------------------------
class LuscherCycleTest(unittest.TestCase):
    """2D 8x8 SU(3), 4x4 blocks (n = 2, v = (2, 2)), beta = 5.0, kappa = 0.2."""

    BETA, KAPPA = 5.0, 0.2

    def test_cycles(self):
        """Cold start, 12 cycles of 16 chains with translation="luscher": the applied shifts follow
        the sequence (random, (2, 2), random, ...), the inactive links of every output equal the
        TRANSLATED input bitwise, rejected chains equal it entirely, links stay in SU(3),
        <exp(-dH)> = 1 within 4 sigma after 4 cycles, good acceptance."""
        bd = BlockDecomposition(G2_8, 4)
        B = 16
        gen = torch.Generator().manual_seed(141)
        U = SUN_identity(G2_8.link_shape(B)[:-1], dtype=torch.cdouble)
        ham = dd_hamiltonian(bd, self.BETA, self.KAPPA)
        hmc = DDHMC(G2_8, ham, tau=1.0, n_steps=2, translation="luscher")
        self.assertEqual((hmc.n_regular, hmc.luscher_step), (1, (2, 2)))
        inactive = (bd.link_mask(~bd.active_links, U) > 0).expand_as(U)
        eye = torch.eye(3, dtype=U.dtype)
        x = []
        for i in range(12):
            U_new, acc, dH, st = hmc.cycle(U, gen)
            s = st["shifts"]
            if i % 2 == 1:
                self.assertTrue(torch.equal(s, torch.tensor([2, 2]).expand(B, -1)))
            Ut = translate(U, s)
            self.assertTrue(torch.equal(U_new[inactive], Ut[inactive]))
            for b in range(B):
                if not acc[b]:
                    self.assertTrue(torch.equal(U_new[b], Ut[b]))
            U = U_new
            if i >= 4:
                x.append(st["exp_minus_dH"])
        self.assertLess((U @ U.conj().transpose(-1, -2) - eye).abs().max().item(), 1e-12)
        self.assertLess((torch.linalg.det(U) - 1).abs().max().item(), 1e-12)
        x = torch.cat(x)
        m, e = x.mean().item(), (x.std() / math.sqrt(x.numel())).item()
        acc = x.clamp(max=1.0).mean().item()
        say(f"luscher cycles: <exp(-dH)> = {m:.4f}({e:.4f}), <min(1, exp(-dH))> = {acc:.3f}, "
            f"plaquette {mean_plaquette(plaquette_fields(U, 2), 3).mean().item():.4f}")
        self.assertLess(abs(m - 1.0), 4 * e)
        self.assertGreater(acc, 0.7)


if __name__ == "__main__":
    unittest.main()
