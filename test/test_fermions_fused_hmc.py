"""Tests of neoqcd.fermions.fused_hmc (fused molecular-dynamics kernels) against the torch code.

Targets: "cpu" (the kernel source compiled for the host and run as a loop over the CUDA thread
index: same index arithmetic; skipped without a C++ compiler) and "cuda" (the twins, skipped without
a CUDA device or nvcc; they run in job/debug_bench_fermions.sh). Equality is to rounding (the
kernels sum in another order), on Haar-random links with batch > 1 and unequal extents.
"""

import copy
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from neoqcd.fermions import fused as F  # noqa: E402
from neoqcd.fermions import fused_hmc as M  # noqa: E402
from neoqcd.fermions._gauge import SUN_haar, SUN_mul  # noqa: E402
from neoqcd.fermions.algebra import algebra_basis, expi, random_algebra, unitarize  # noqa: E402
from neoqcd.fermions.fast import (  # noqa: E402
    FastTwoFlavourWilson,
    HalfLatticeWilson,
    lean_cg,
    mixed_precision_cg,
    to_half,
)
from neoqcd.fermions.hmc import (  # noqa: E402
    HMC,
    Hamiltonian,
    WilsonGaugeAction,
    batched_defect_plaquette_weights,
    defect_plaquette_weights,
)
from neoqcd.fermions.lattice import Geometry  # noqa: E402

G46 = Geometry(4, 6, 4, 3)  # 4^3 x 6, SU(3)
G42 = Geometry(4, 4, 2, 2)  # 2^3 x 4, SU(2)
G2 = Geometry(2, 6, 4, 3)  # 2D 6 x 4 (T = 6, L = 4), SU(3)
G22 = Geometry(2, 4, 6, 2)  # 2D 4 x 6, SU(2)
GEOMETRIES = (G46, G42, G2, G22)
C, C64 = torch.complex128, torch.complex64

_SAVED_CACHE = None
_TMP = None


def setUpModule():
    """Build the kernel libraries in a temporary cache unless NEOQCD_KERNEL_CACHE is set."""
    global _SAVED_CACHE, _TMP
    _SAVED_CACHE = os.environ.get("NEOQCD_KERNEL_CACHE")
    if _SAVED_CACHE is None:
        _TMP = tempfile.TemporaryDirectory(prefix="neoqcd_kernels_")
        os.environ["NEOQCD_KERNEL_CACHE"] = _TMP.name


def tearDownModule():
    if _SAVED_CACHE is None:
        os.environ.pop("NEOQCD_KERNEL_CACHE", None)
        _TMP.cleanup()


def links(geometry, batch, seed):
    g = torch.Generator().manual_seed(seed)
    return SUN_haar(geometry.link_shape(batch)[:-1], generator=g)


def momenta(geometry, batch, seed):
    return random_algebra(geometry.link_shape(batch), generator=torch.Generator().manual_seed(seed))


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def defect_mask(geometry, seed):
    """A random gauge-only defect: about a quarter of the links get the factor bc."""
    g = torch.Generator().manual_seed(seed)
    on = (torch.rand((geometry.D,) + tuple(geometry.lattice), generator=g) < 0.25).to(torch.float64)
    return torch.stack([1.0 - on, on])


def md_reason(target):
    if target == "cuda" and not torch.cuda.is_available():
        return "no CUDA device"
    ok, why = M.md_backend_status(target)
    return None if ok else why


class _Target:
    target = "cpu"
    device = "cpu"

    @classmethod
    def setUpClass(cls):
        reason = md_reason(cls.target)
        if reason is not None:
            raise unittest.SkipTest(f"{cls.target} MD kernels unavailable: {reason}")

    def dev(self, t):
        return t.to(self.device)


class _GaugeForceEquality(_Target):
    def test_force_equals_torch(self):
        """Force and kick vs WilsonGaugeAction: SU(2) and SU(3), 4D and 2D, unequal extents, batch 3,
        scalar and per-sample beta; relative 1e-14."""
        for geo in GEOMETRIES:
            U = self.dev(links(geo, 3, 1))
            P = self.dev(momenta(geo, 3, 2))
            for beta in (5.6, torch.tensor([5.6, 5.0, 6.1], dtype=torch.float64)):
                with self.subTest(geo=geo, beta=type(beta).__name__):
                    ref = WilsonGaugeAction(beta, geo).force(U)
                    act = M.FusedWilsonGaugeAction(beta, geo, target=self.target, strict=True)
                    got = act.force(U)
                    self.assertEqual(act.kernel_used, self.target)
                    self.assertEqual(got.dtype, C)
                    self.assertLess(rel(got, ref), 1e-14)
                    kick = act.force_kick(U, P, 0.37)
                    self.assertLess((kick - (P - 0.37 * ref)).abs().max().item(), 1e-14)

    def test_defect_weights(self):
        """Plaquette weights of a gauge-only defect: per-sample (B, lattice) fields, lattice fields,
        and a mixture of both layouts; the weights change the force (non-trivial test)."""
        for geo in GEOMETRIES:
            U = self.dev(links(geo, 3, 3))
            mask = self.dev(defect_mask(geo, 4))
            bcs = torch.tensor([0.0, 0.4, 0.9], dtype=torch.float64)
            layouts = {
                "batched": batched_defect_plaquette_weights(mask, self.dev(bcs), geo.D),
                "lattice": defect_plaquette_weights(mask, 0.3, geo.D),
            }
            mixed = dict(layouts["lattice"])
            first = min(mixed)
            mixed[first] = layouts["batched"][first]
            layouts["mixed"] = mixed
            plain = WilsonGaugeAction(5.6, geo).force(U)
            for name, w in layouts.items():
                with self.subTest(geo=geo, layout=name):
                    ref = WilsonGaugeAction(5.6, geo, w).force(U)
                    self.assertGreater(rel(ref, plain), 1e-3)
                    act = M.FusedWilsonGaugeAction(5.6, geo, w, target=self.target, strict=True)
                    self.assertLess(rel(act.force(U), ref), 1e-14)
                    P = self.dev(momenta(geo, 3, 5))
                    self.assertLess((act.force_kick(U, P, -0.2) - (P + 0.2 * ref)).abs().max().item(), 1e-14)

    def test_weights_cache_follows_the_weights(self):
        """The packed weights follow an in-place change of a weight field and a new weights dict."""
        geo = G46
        U = self.dev(links(geo, 2, 6))
        mask = self.dev(defect_mask(geo, 7))
        w = batched_defect_plaquette_weights(mask, self.dev(torch.tensor([0.2, 0.5], dtype=torch.float64)), geo.D)
        act = M.FusedWilsonGaugeAction(5.6, geo, w, target=self.target, strict=True)
        act.force(U)
        w[(0, 1)].mul_(0.5)
        self.assertLess(rel(act.force(U), WilsonGaugeAction(5.6, geo, w).force(U)), 1e-14)
        w2 = batched_defect_plaquette_weights(mask, self.dev(torch.tensor([0.9, 0.1], dtype=torch.float64)), geo.D)
        act.plaquette_weights = w2
        self.assertLess(rel(act.force(U), WilsonGaugeAction(5.6, geo, w2).force(U)), 1e-14)

    def test_strided_conj_and_neg_views(self):
        """Non-contiguous, offset, lazily conjugated and negated links and momenta are read as their values."""
        geo = G46
        U = self.dev(links(geo, 2, 8))
        P = self.dev(momenta(geo, 2, 9))
        ref = WilsonGaugeAction(5.6, geo).force(U)
        act = M.FusedWilsonGaugeAction(5.6, geo, target=self.target, strict=True)
        big = torch.zeros((3,) + tuple(U.shape), dtype=C, device=U.device)
        big[1] = U
        views = {
            "offset": big[1],
            "transposed": U.transpose(-1, -2).contiguous().transpose(-1, -2),
            "conj": U.conj().resolve_conj().conj(),
            "neg": torch._neg_view(-U),
            "batch_strided": torch.stack([U[0], U[0], U[1]])[::2][[0, 1]],
        }
        self.assertTrue(views["conj"].is_conj())
        self.assertTrue(views["neg"].is_neg())
        self.assertFalse(views["transposed"].is_contiguous())
        for name, V in views.items():
            with self.subTest(view=name):
                want = ref if name != "batch_strided" else WilsonGaugeAction(5.6, geo).force(V.contiguous())
                self.assertLess(rel(act.force(V), want), 1e-14)
        Pv = P.conj().resolve_conj().conj()  # P hermitian: the same values through a lazy conj view
        self.assertLess((act.force_kick(U, Pv, 0.1) - (P - 0.1 * ref)).abs().max().item(), 1e-14)

    def test_autograd_inputs_refused(self):
        geo = G46
        U = self.dev(links(geo, 2, 10)).requires_grad_()
        act = M.FusedWilsonGaugeAction(5.6, geo, target=self.target, strict=True)
        with self.assertRaisesRegex(RuntimeError, "not differentiable"):
            act.force(U)
        with torch.no_grad():
            self.assertLess(rel(act.force(U), WilsonGaugeAction(5.6, geo).force(U.detach())), 1e-14)


def special_algebra(N, dtype=C):
    """Hermitian traceless matrices that stress the exponential: zero, tiny, a degenerate spectrum of
    either sign of c0 (|c0| = c0max, the clamp of the closed forms), c1 just below and above the
    series threshold 3, and a large one (c1 ~ 50)."""
    T = algebra_basis(N, dtype=dtype)
    out = [torch.zeros(N, N, dtype=dtype), 1e-9 * T[0]]
    if N == 3:
        d = torch.diag(torch.tensor([1.0, 1.0, -2.0], dtype=torch.float64)).to(dtype)
        for scale in (0.3, 1.5, -1.5, 4.0):
            out.append(scale * d)
        base = T[0] + 0.5 * T[3] - 0.7 * T[7]
        c1 = 0.5 * torch.trace(base @ base).real
        for target in (3.0 - 1e-12, 3.0 + 1e-12, 50.0):
            out.append(base * (target / c1) ** 0.5)
    else:
        for scale in (0.5, 2.0, 7.0):
            out.append(scale * T[2])
    return torch.stack(out)


class _UpdateEquality(_Target):
    def test_update_modes_equal_torch(self):
        """Kick, move and kick + move vs P - c F and SUN_mul(expi(h P'), U): SU(2), SU(3), h from 0.01
        (series) to 3 (closed forms on most links); U' within 1e-14 (series) / 2e-14 (closed forms)."""
        for N in (3, 2):
            g = torch.Generator().manual_seed(20 + N)
            shape = (2, 4, 3, 5, N, N)
            U = self.dev(SUN_haar(shape[:-1], generator=g))
            P = self.dev(random_algebra(shape, generator=g))
            F = self.dev(random_algebra(shape, generator=g))
            for h in (0.01, 0.1, 1.0, 3.0):
                with self.subTest(N=N, h=h):
                    Pk = P - 0.3 * F
                    want = SUN_mul(expi(h * Pk), U)
                    Pn, Un = M.md_update(P, U=U, F=F, coef=0.3, h=h)
                    self.assertLess((Pn - Pk).abs().max().item(), 1e-15)
                    self.assertLess((Un - want).abs().max().item(), 2e-14)
                    self.assertIsNone(M.md_update(P, U=U, h=h)[0])
                    self.assertLess((M.md_update(P, U=U, h=h)[1] - SUN_mul(expi(h * P), U)).abs().max().item(), 2e-14)
                    Pk2, none = M.md_update(P, F=F, coef=-h)
                    self.assertIsNone(none)
                    self.assertLess((Pk2 - (P + h * F)).abs().max().item(), 1e-15)

    def test_special_exponentials(self):
        """exp(iQ) of special_algebra (degenerate spectra, the series threshold, zero) vs algebra.expi."""
        for N in (3, 2):
            Q = self.dev(special_algebra(N))
            eye = torch.eye(N, dtype=C, device=Q.device).expand_as(Q).contiguous()
            got = M.md_update(Q, U=eye, h=1.0)[1]
            want = expi(Q)
            for i in range(Q.shape[0]):
                with self.subTest(N=N, i=i):
                    self.assertLess((got[i] - want[i]).abs().max().item(), 5e-14)
                    self.assertLess((got[i] @ got[i].conj().T - torch.eye(N, dtype=C, device=Q.device)).abs().max().item(), 1e-13)

    def test_views_and_autograd(self):
        geo = G46
        U = self.dev(links(geo, 2, 30))
        P = self.dev(momenta(geo, 2, 31))
        F = self.dev(momenta(geo, 2, 32))
        want_P, want_U = P - 0.2 * F, SUN_mul(expi(0.1 * (P - 0.2 * F)), U)
        views = {"conj": (P.conj().resolve_conj().conj(), U, F), "neg": (P, torch._neg_view(-U), F),
                 "strided": (P.transpose(-1, -2).contiguous().transpose(-1, -2), U, torch._neg_view(-F))}
        for name, (p, u, f) in views.items():
            with self.subTest(view=name):
                Pn, Un = M.md_update(p, U=u, F=f, coef=0.2, h=0.1)
                self.assertLess((Pn - want_P).abs().max().item(), 1e-15)
                self.assertLess((Un - want_U).abs().max().item(), 1e-14)
        with self.assertRaisesRegex(RuntimeError, "not differentiable"):
            M.md_update(P.clone().requires_grad_(), U=U, h=0.1)
        with self.assertRaises(ValueError):
            M.md_update(P.to(C64), U=U.to(C64), h=0.1)
        with self.assertRaises(ValueError):
            M.md_update(P, U=U)  # a move needs h


def _gauge_hmc(geo, target, beta=5.6, weights=None, tau=0.5, n_steps=5, integrator="omelyan", **kw):
    act = M.FusedWilsonGaugeAction(beta, geo, weights, target=target, strict=target is not None)
    return M.FusedHMC(geo, Hamiltonian([([act], 1)]), tau=tau, n_steps=n_steps, integrator=integrator,
                      target=target, strict=target is not None, **kw)


def _fermion_hmc(geo, target, kappa=0.13, n_gauge=3, tau=0.5, n_steps=4, weights=None, md_kernels="all", **kw):
    """Fused-hop Hamiltonian (host build of the hop, or "loop") with the MD kernels of `md_kernels` on
    `target` (None: all off, the torch molecular dynamics)."""
    ham = M.FusedMDHamiltonian(5.6, kappa, geo, n_gauge_substeps=n_gauge, plaquette_weights=weights,
                               md_kernels=md_kernels if target is not None else "none", md_target=target,
                               md_strict=target is not None,
                               kernel="cpu" if md_reason("cpu") is None else "loop",
                               solver_tol=1e-12, action_tol=1e-13, **kw)
    return M.FusedHMC(geo, ham, tau=tau, n_steps=n_steps, target=ham.md_hmc_target(), strict=target is not None)


class _IntegratorTest(_Target):
    def _compare(self, make, geo, B=3, seed=40, n_traj=2):
        """n_traj trajectories with the kernels and with the torch path from the same seed: dH equal to
        1e-9 (absolute), identical accept decisions, links within 1e-11."""
        out = {}
        for target in (None, self.target):
            hmc = make(target)
            gen = torch.Generator(device=self.device).manual_seed(seed)
            U = self.dev(links(geo, B, seed + 1))
            dHs, accs = [], []
            for _ in range(n_traj):
                U, acc, dH, _ = hmc.trajectory(U, gen)
                dHs.append(dH)
                accs.append(acc)
            out[target] = (U, torch.stack(dHs), torch.stack(accs), hmc)
        (U0, dH0, acc0, h0), (U1, dH1, acc1, h1) = out[None], out[self.target]
        self.assertEqual(h1.kernel_used, self.target if h1._choice.target is not None else "torch")
        self.assertEqual(h0.kernel_used, "torch")
        self.assertLess((dH1 - dH0).abs().max().item(), 1e-9)
        self.assertTrue(torch.equal(acc1, acc0))
        self.assertLess((U1 - U0).abs().max().item(), 1e-11)
        return h1

    def test_pure_gauge_trajectories(self):
        for geo, integ in ((G46, "omelyan"), (G22, "leapfrog"), (G42, "omelyan")):
            with self.subTest(geo=geo, integrator=integ):
                hmc = self._compare(lambda t: _gauge_hmc(geo, t, integrator=integ), geo)
                self.assertEqual(hmc.last_counts["move_fused"], 5 if integ == "leapfrog" else 10)
                self.assertEqual(hmc.last_counts["kick_torch"] + hmc.last_counts["move_torch"], 0)

    def test_defect_weights_trajectory(self):
        geo = G46
        mask = self.dev(defect_mask(geo, 41))
        w = batched_defect_plaquette_weights(mask, self.dev(torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)), geo.D)
        self._compare(lambda t: _gauge_hmc(geo, t, weights=w), geo)

    def test_fermion_trajectories(self):
        """Two-level Hamiltonian (fused hop fermion term outside, fused gauge force inside), both time
        signs: dH and accept decisions as the torch integrator."""
        for ts in (-1.0, 1.0):
            with self.subTest(time_sign=ts):
                hmc = self._compare(lambda t: _fermion_hmc(G46, t, time_sign=ts), G46, B=2, n_traj=1)
                counts = hmc.last_counts
                self.assertGreater(counts["move_fused"], 0)
                self.assertEqual(counts["kick_torch"] + counts["move_torch"], 0)

    def test_each_component_alone(self):
        """Every MD component alone (and none) gives the torch trajectory to rounding."""
        for comp in M.MD_COMPONENTS:
            with self.subTest(component=comp):
                hmc = self._compare(lambda t: _fermion_hmc(G46, t, md_kernels=comp), G46, B=2, n_traj=1)
                ham = hmc.hamiltonian
                used = {"gauge_force": ham.gauge.kernel_used, "fermion_force": ham.fermion.force_kernel_used,
                        "update": hmc.kernel_used, "cg": ham.fermion.cg_kernel_used}
                for name, path in used.items():
                    self.assertEqual(path, self.target if name == comp else "torch", name)
                if comp == "update":  # forces from torch, kicks and moves through the update kernel
                    self.assertEqual(hmc.last_counts["kick_fused"] + hmc.last_counts["move_fused"], 0)
                    self.assertGreater(hmc.last_counts["move_kernel"], 0)

    def test_unfused_force_kicks_use_the_update_kernel(self):
        hmc = self._compare(lambda t: _gauge_hmc(G46, t, fuse_force_kicks=False), G46)
        self.assertEqual(hmc.last_counts["kick_fused"], 0)
        self.assertEqual(hmc.last_counts["move_kernel"], 10)

    def test_reversibility(self):
        """Forward then backward (tau -> -tau): U and P return to the start within 1e-12 (pure gauge)
        and 1e-10 (fermions, CG from zero at tolerance 1e-13)."""
        cases = [(_gauge_hmc(G46, self.target), 1e-12, G46),
                 (_fermion_hmc(G46, self.target, reuse_guess=False), 1e-10, G46)]
        for hmc, tol, geo in cases:
            with self.subTest(hamiltonian=type(hmc.hamiltonian).__name__):
                U = self.dev(links(geo, 2, 42))
                P = self.dev(momenta(geo, 2, 43))
                hmc.hamiltonian.refresh(U, torch.Generator(device=self.device).manual_seed(44))
                U1, P1, _ = hmc.integrate(U, P)
                U2, P2, _ = hmc.integrate(U1, -P1)
                self.assertLess((U2 - U).abs().max().item(), tol)
                self.assertLess((P2 + P).abs().max().item(), tol)

    def test_energy_conservation_scaling(self):
        """dH of one trajectory from fixed (U, P) falls as the step squared: |dH(2n)| / |dH(n)| ~ 1/4
        (pure gauge, omelyan, 4D 4^3 x 6)."""
        geo = G46
        U = self.dev(unitarize(SUN_mul(expi(0.3 * momenta(geo, 2, 45)), links(geo, 2, 46) * 0 + torch.eye(3, dtype=C))))
        P = self.dev(momenta(geo, 2, 47))
        dHs = []
        for n in (8, 16, 32):
            hmc = _gauge_hmc(geo, self.target, tau=1.0, n_steps=n)
            U1, P1, _ = hmc.integrate(U, P)
            dHs.append(hmc.hamiltonian.energy_difference(U, P, unitarize(U1), P1).abs())
        r1, r2 = (dHs[1] / dHs[0]).tolist(), (dHs[2] / dHs[1]).tolist()
        for r in r1 + r2:
            self.assertGreater(r, 0.18)
            self.assertLess(r, 0.32)


def half_fields(geo, B, seed):
    g = torch.Generator().manual_seed(seed)
    return [to_half(torch.randn(geo.wilson_shape(B), dtype=C, generator=g), geo, p) for p in (0, 1, 0, 1)]


def torch_fermion_force(op, fields, scale):
    G = op.hopping_derivative(*fields)
    return scale.view((-1,) + (1,) * (G.dim() - 1)).to(G.dtype) * G


class _FermionForceEquality(_Target):
    def test_force_equals_hopping_derivative(self):
        """fermion_force = 2 kappa^2 HalfLatticeWilson.hopping_derivative (and the kick) to 1e-14:
        SU(2), SU(3), 2D, 4D, unequal extents, both time signs, per-sample kappa, batch 3."""
        k = torch.tensor([0.11, 0.12, 0.13], dtype=torch.float64)
        for geo in GEOMETRIES:
            U = self.dev(links(geo, 3, 50))
            fields = [self.dev(f) for f in half_fields(geo, 3, 51)]
            for ts in (-1.0, 1.0):
                with self.subTest(geo=geo, time_sign=ts):
                    op = HalfLatticeWilson(U, k, geo, time_sign=ts)
                    sc = 2.0 * op.kappa ** 2
                    want = torch_fermion_force(op, fields, sc)
                    self.assertLess(rel(M.fermion_force(op, *fields, sc), want), 1e-14)
                    P = self.dev(momenta(geo, 3, 52))
                    got = M.fermion_force(op, *fields, sc, mom=P, coef=0.3)
                    self.assertLess((got - (P - 0.3 * want)).abs().max().item(), 1e-14)

    def test_shared_links_views_and_autograd(self):
        geo = G46
        U = self.dev(links(geo, 1, 53))
        op = HalfLatticeWilson(U, 0.12, geo)
        fields = [self.dev(f) for f in half_fields(geo, 3, 54)]
        sc = 2.0 * op.kappa ** 2
        want = torch_fermion_force(op, fields, sc)
        self.assertEqual(want.shape[0], 3)
        self.assertLess(rel(M.fermion_force(op, *fields, sc), want), 1e-14)
        views = [fields[0].conj().resolve_conj().conj(), torch._neg_view(-fields[1]),
                 fields[2].transpose(-1, -2).contiguous().transpose(-1, -2), fields[3]]
        self.assertLess(rel(M.fermion_force(op, *views, sc), want), 1e-14)
        with self.assertRaisesRegex(RuntimeError, "not differentiable"):
            M.fermion_force(op, fields[0].clone().requires_grad_(), *fields[1:], sc)
        with self.assertRaises(ValueError):
            M.fermion_force(op.to(C64), *[f.to(C64) for f in fields], sc)

    def test_term_force_and_kick_equal_the_fast_term(self):
        """FusedForceTwoFlavourWilson: action and force equal FastTwoFlavourWilson (same draws), the
        force kick equals P - c F, one force solve each, double and mixed precision."""
        geo = G46
        k = torch.tensor([0.12, 0.13], dtype=torch.float64)
        U = self.dev(links(geo, 2, 55))
        hop = "cpu" if self.target == "cpu" else "cuda"
        for precision in ("double", "mixed"):
            with self.subTest(precision=precision):
                gen = torch.Generator(device=self.device)
                fast = FastTwoFlavourWilson(k, geo, solver_tol=1e-12, action_tol=1e-13, precision=precision)
                fast.refresh(U, gen.manual_seed(56))
                F_ref = fast.force(U)
                term = M.FusedForceTwoFlavourWilson(k, geo, md_target=self.target, md_strict=True, kernel=hop,
                                                    strict=True, solver_tol=1e-12, action_tol=1e-13,
                                                    precision=precision)
                term.refresh(U, gen.manual_seed(56))
                self.assertLess(rel(term.force(U), F_ref), 1e-9)
                self.assertEqual(term.force_kernel_used, self.target)
                P = self.dev(momenta(geo, 2, 57))
                solves = term.stats["solves"]
                kicked = term.force_kick(U, P, 0.25)
                self.assertEqual(term.stats["solves"], solves + 1)
                self.assertLess((kicked - (P - 0.25 * F_ref)).abs().max().item(), 1e-9 * F_ref.abs().max().item())


class _CGEquality(_Target):
    def _system(self, geo=G46, B=3, seed=60, dtype=C):
        U = self.dev(links(geo, B, seed))
        op = HalfLatticeWilson(U, torch.tensor([0.11, 0.12, 0.13][:B], dtype=torch.float64), geo).to(dtype)
        b = self.dev(to_half(torch.randn(geo.wilson_shape(B), dtype=C, generator=torch.Generator().manual_seed(seed + 1)),
                             geo)).to(dtype)
        return op, b

    def step(self):
        return M.FusedCGStep(M.load_md_backend(self.target, self.device), self.device)

    def test_lean_cg_equal(self):
        """lean_cg with the fused iteration: the same per-sample iteration counts and loop count, x and
        the true residual equal to rounding (complex128 at 1e-12, complex64 at 1e-5), check intervals
        1 and 10."""
        for dtype, tol, xtol in ((C, 1e-12, 1e-12), (C64, 1e-5, 1e-4)):
            op, b = self._system(dtype=dtype)
            for every in (1, 10):
                with self.subTest(dtype=dtype, check_every=every):
                    i0, i1 = {}, {}
                    r0 = lean_cg(op.normal, b, tol=tol, max_iter=400, check_every=every, info=i0)
                    step = self.step()
                    r1 = lean_cg(op.normal, b, tol=tol, max_iter=400, check_every=every, info=i1, step=step)
                    self.assertGreater(step.calls, 10)
                    # the same counts (the CPU emulation is deterministic here; on the GPU a rounding
                    # difference at the convergence boundary may move a count by one)
                    slack = 0 if self.device == "cpu" else 1
                    self.assertLessEqual((r0.iterations - r1.iterations).abs().max().item(), slack)
                    self.assertLessEqual(abs(i0["loop_iterations"] - i1["loop_iterations"]), slack * every)
                    self.assertLess(rel(r1.x, r0.x), xtol)
                    self.assertTrue(bool(i1["converged"].all()))
                    self.assertLess(r1.residual.max().item(), 10 * max(tol, 1e3 * torch.finfo(r1.residual.dtype).eps))

    def test_mixed_precision_equal(self):
        op, b = self._system()
        low = op.to(C64)
        i0, i1 = {}, {}
        r0 = mixed_precision_cg(op, low, b, tol=1e-12, max_iter=2000, info=i0)
        r1 = mixed_precision_cg(op, low, b, tol=1e-12, max_iter=2000, info=i1, step=self.step())
        self.assertLess(rel(r1.x, r0.x), 1e-10)
        self.assertTrue(bool(i1["converged"].all()))
        self.assertLess(r1.residual.max().item(), 1e-11)

    def test_zero_and_nonfinite_samples(self):
        """A zero right-hand side stays zero with 0 iterations; a NaN sample is isolated: the healthy
        samples end bitwise as in a batch without it."""
        op, b = self._system(B=3)
        zero = b.clone()
        zero[1] = 0
        r = lean_cg(op.normal, zero, tol=1e-12, max_iter=400, step=self.step())
        self.assertEqual(int(r.iterations[1]), 0)
        self.assertEqual(r.x[1].abs().max().item(), 0.0)
        op1, b1 = self._system(B=1)  # sample 0 of the same system (kappa 0.11, links of sample 0)
        nan = b.clone()
        nan[1, 0, 0, 0] = float("nan")
        U2 = torch.stack([op.U[0], op.U[0], op.U[0]])
        op2 = HalfLatticeWilson(U2, 0.11, G46)
        nan[0] = b1[0]
        nan[2] = b1[0]
        r2 = lean_cg(op2.normal, nan, tol=1e-12, max_iter=400, step=self.step())
        alone = lean_cg(HalfLatticeWilson(op.U[:1], 0.11, G46).normal, b1, tol=1e-12, max_iter=400, step=self.step())
        if self.device == "cpu":
            self.assertTrue(torch.equal(r2.x[0], alone.x[0]))
            self.assertTrue(torch.equal(r2.x[2], alone.x[0]))
        else:  # torch's CUDA sum of the partials may reduce a (3, n) and a (1, n) tensor in another order
            self.assertLess(rel(r2.x[0], alone.x[0]), 1e-12)
            self.assertLess(rel(r2.x[2], alone.x[0]), 1e-12)
        self.assertFalse(bool(torch.isfinite(r2.residual[1])))

    def test_unwritable_inputs_return_none(self):
        op, b = self._system()
        step = self.step()
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rr = (r.abs() ** 2).reshape(3, -1).sum(1)
        conv = torch.zeros(3, dtype=torch.bool, device=b.device)
        its = torch.zeros(3, dtype=torch.long, device=b.device)
        strided = torch.zeros(b.shape[:-1] + (2 * b.shape[-1],), dtype=C, device=b.device)[..., ::2]
        self.assertIsNone(step(op.normal, strided, r, p, rr, rr, conv, its))
        self.assertIsNone(step(op.normal, x, torch._neg_view(-r), p, rr, rr, conv, its))
        self.assertIsNone(step(op.normal, x.to(C64), r.to(C64), p.to(C64), rr, rr, conv, its))  # rr of the wrong dtype
        self.assertEqual(step.calls, 0)
        self.assertIsNotNone(step(op.normal, x, r, p, rr, rr * 1e-24, conv, its))
        self.assertEqual(its.tolist(), [1, 1, 1])

    def test_term_with_cg_kernels(self):
        """FusedForceTwoFlavourWilson with cg_target: action and force as FastTwoFlavourWilson (same
        draws), double and mixed precision; the solves report the kernel path."""
        geo = G46
        k = torch.tensor([0.12, 0.13], dtype=torch.float64)
        U = self.dev(links(geo, 2, 61))
        hop = "cpu" if self.target == "cpu" else "cuda"
        for precision in ("double", "mixed"):
            with self.subTest(precision=precision):
                gen = torch.Generator(device=self.device)
                fast = FastTwoFlavourWilson(k, geo, solver_tol=1e-12, action_tol=1e-13, precision=precision)
                fast.refresh(U, gen.manual_seed(62))
                term = M.FusedForceTwoFlavourWilson(k, geo, cg_target=self.target, md_strict=True, kernel=hop,
                                                    strict=True, solver_tol=1e-12, action_tol=1e-13,
                                                    precision=precision)
                term.refresh(U, gen.manual_seed(62))
                self.assertLess((term.action(U) - fast.action(U)).abs().max().item(), 1e-8)
                self.assertEqual(term.cg_kernel_used, self.target)
                self.assertLess(rel(term.force(U), fast.force(U)), 1e-9)
                slack = 0 if self.device == "cpu" else term.stats["solves"]
                self.assertLessEqual(abs(term.stats["cg_iterations"] - fast.stats["cg_iterations"]), slack)


class CpuCGTest(_CGEquality, unittest.TestCase):
    target, device = "cpu", "cpu"


class CudaCGTest(_CGEquality, unittest.TestCase):
    target, device = "cuda", "cuda"


class CpuFermionForceTest(_FermionForceEquality, unittest.TestCase):
    target, device = "cpu", "cpu"


class CudaFermionForceTest(_FermionForceEquality, unittest.TestCase):
    target, device = "cuda", "cuda"


class CpuUpdateTest(_UpdateEquality, unittest.TestCase):
    target, device = "cpu", "cpu"


class CudaUpdateTest(_UpdateEquality, unittest.TestCase):
    target, device = "cuda", "cuda"


class CpuIntegratorTest(_IntegratorTest, unittest.TestCase):
    target, device = "cpu", "cpu"


class CudaIntegratorTest(_IntegratorTest, unittest.TestCase):
    """CUDA twin (the fermion trajectories use the host build of the hop: skipped on the GPU, where
    main_hmc runs them with --kernel cuda)."""

    target, device = "cuda", "cuda"

    def test_fermion_trajectories(self):
        self.skipTest("the fermion trajectory test uses the host build of the hop")

    def test_each_component_alone(self):
        self.skipTest("the fermion trajectory test uses the host build of the hop")

    def test_reversibility(self):
        self.skipTest("covered on the CPU emulation; the CUDA build is checked by the equality tests")


class CpuGaugeForceTest(_GaugeForceEquality, unittest.TestCase):
    target, device = "cpu", "cpu"


class CudaGaugeForceTest(_GaugeForceEquality, unittest.TestCase):
    target, device = "cuda", "cuda"


class FallbackTest(unittest.TestCase):
    def test_torch_target_is_the_parent(self):
        geo = G46
        U = links(geo, 2, 11)
        act = M.FusedWilsonGaugeAction(5.6, geo, target=None)
        self.assertTrue(torch.equal(act.force(U), WilsonGaugeAction(5.6, geo).force(U)))
        self.assertEqual(act.kernel_used, "torch")

    def test_unsupported_inputs_fall_back_with_one_warning_or_raise(self):
        """complex64 links, D = 3 and a CUDA target on CPU tensors run the torch force (one warning per
        reason); strict=True raises BackendUnavailable."""
        U64 = links(G46, 2, 12).to(C64)
        G3 = Geometry(3, 4, 4, 3)
        U3 = links(G3, 2, 13)
        cases = [("cpu", G46, U64), ("cpu", G3, U3), ("cuda", G46, links(G46, 2, 14))]
        if md_reason("cpu") is not None:
            cases = cases[2:]
        for target, geo, U in cases:
            with self.subTest(target=target, geo=geo, dtype=U.dtype):
                F._WARNED.clear()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    act = M.FusedWilsonGaugeAction(5.6, geo, target=target)
                    got = act.force(U)
                    act.force(U)
                self.assertTrue(torch.equal(got, WilsonGaugeAction(5.6, geo).force(U)))
                self.assertEqual(act.kernel_used, "torch")
                self.assertEqual(len([w for w in caught if "fused MD kernel" in str(w.message)]), 1)
                with self.assertRaises(F.BackendUnavailable):
                    M.FusedWilsonGaugeAction(5.6, geo, target=target, strict=True).force(U)

    def test_torch_integrator_without_target(self):
        """target None is the parent's integrator exactly (bitwise); with a non-default exp method the
        moves are the torch ones and only the kicks run through the update kernel (rounding)."""
        geo = G46
        U = links(geo, 2, 16)
        for method in ("auto", "matrix_exp"):
            act = WilsonGaugeAction(5.6, geo)
            ref = HMC(geo, Hamiltonian([([act], 1)]), tau=0.5, n_steps=3, exp_method=method)
            want = ref.trajectory(U.clone(), torch.Generator().manual_seed(17))
            targets = [None] + (["cpu"] if md_reason("cpu") is None and method != "auto" else [])
            for target in targets:
                with self.subTest(method=method, target=target):
                    hmc = M.FusedHMC(geo, Hamiltonian([([act], 1)]), tau=0.5, n_steps=3, exp_method=method,
                                     target=target)
                    got = hmc.trajectory(U.clone(), torch.Generator().manual_seed(17))
                    if target is None:
                        self.assertTrue(torch.equal(got[0], want[0]))
                        self.assertTrue(torch.equal(got[2], want[2]))
                    else:
                        self.assertEqual(hmc.last_counts["move_torch"], 6)
                        self.assertLess((got[0] - want[0]).abs().max().item(), 1e-13)
                        self.assertLess((got[2] - want[2]).abs().max().item(), 1e-9)

    def test_copy_and_pickle(self):
        if md_reason("cpu") is not None:
            self.skipTest(md_reason("cpu"))
        geo = G46
        U = links(geo, 2, 15)
        act = M.FusedWilsonGaugeAction(5.6, geo, target="cpu", strict=True)
        want = act.force(U)
        for clone in (copy.deepcopy(act), pickle.loads(pickle.dumps(act))):
            self.assertTrue(torch.equal(clone.force(U), want))
            self.assertEqual(clone.kernel_used, "cpu")

    def test_library_name_and_cache_key(self):
        """The MD library is cached apart from the hop library (its own name and source hash)."""
        if md_reason("cpu") is not None:
            self.skipTest(md_reason("cpu"))
        path = Path(M.load_md_backend("cpu").path)
        self.assertTrue(path.name.startswith("neoqcd_md_cpu_"), path.name)
        self.assertNotEqual(F._build_spec("cpu")[3], F._build_spec("cpu", backend_class=M._MDBackend)[3])
        self.assertEqual(F._build_spec("cpu")[3].name.split("_")[:3], ["neoqcd", "hop", "cpu"])


class CudaSourceTest(unittest.TestCase):
    def test_clang_cuda_front_end(self):
        """Host and device halves of the CUDA build of MD_KERNEL_SOURCE parse and type-check (clang -x
        cuda with the stub runtime header of test_fermions_fused), warnings included."""
        from test.test_fermions_fused import _CUDA_STUB
        clang = shutil.which("clang++")
        if clang is None:
            self.skipTest("no clang++")
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "probe.cu"
            probe.write_text("__attribute__((global)) void k(int* x) { x[0] = 1; }\n")
            res = subprocess.run([clang, "-x", "cuda", "--cuda-host-only", "--cuda-gpu-arch=sm_80", "-nocudainc",
                                  "-nocudalib", "-fsyntax-only", str(probe)], capture_output=True, text=True)
            if res.returncode != 0:
                self.skipTest(f"clang++ has no CUDA front end: {res.stderr.strip()[:200]}")
            stub = Path(tmp) / "stub"
            stub.mkdir()
            (stub / "cuda_runtime.h").write_text(_CUDA_STUB + _MATH_STUB)
            src = Path(tmp) / "md.cu"
            src.write_text(M.MD_KERNEL_SOURCE)
            for side in ("--cuda-host-only", "--cuda-device-only"):
                cmd = [clang, "-x", "cuda", side, "--cuda-gpu-arch=sm_80", "-nocudainc", "-nocudalib", "-D__CUDACC__",
                       "-D__CUDACC_VER_MAJOR__=12", "-D__CUDACC_VER_MINOR__=1", f"-I{stub}", "-std=c++14", "-Wall",
                       "-fsyntax-only", str(src)]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                self.assertEqual(res.returncode, 0, f"{side}: {res.stderr}")
                self.assertNotIn("warning", res.stderr, f"{side}: {res.stderr}")


class HostSourceTest(unittest.TestCase):
    def test_host_compiler_has_no_warnings(self):
        """The host build of MD_KERNEL_SOURCE (the CPU emulation) compiles with -Wall -Wextra without a
        warning (a duplicate 'static' is an error for some GCC versions)."""
        cxx = F._find_cxx()
        if cxx is None:
            self.skipTest("no host C++ compiler")
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "md.cpp"
            src.write_text(M.MD_KERNEL_SOURCE)
            res = subprocess.run([cxx, "-std=c++14", "-Wall", "-Wextra", "-fsyntax-only", str(src)],
                                 capture_output=True, text=True, timeout=300)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertNotIn("warning", res.stderr, res.stderr)


# device declarations of the libm functions the kernels call (cuda_runtime.h provides them under nvcc)
_MATH_STUB = r"""
__attribute__((device)) double sqrt(double); __attribute__((device)) double sin(double);
__attribute__((device)) double cos(double); __attribute__((device)) double acos(double);
__attribute__((device)) double pow(double, double); __attribute__((device)) double fabs(double);
__attribute__((device)) double exp(double); __attribute__((device)) double log(double);
"""


if __name__ == "__main__":
    unittest.main()
