"""Tests of the fermionic NE-MCMC (neoqcd.fermions.snf, phase 3 of fermions_plan.md): Jarzynski
equality on the joint (U, phi) distribution against exact reweighting, and the work/heat
bookkeeping, for protocols in kappa, beta and bc.

CPU, complex128, fixed seeds; 2D 4 x 4 SU(3), beta = 5, kappa = 0.18, B = 64 chains; HMC blocks of
one trajectory (tau = 1, 2 Omelyan steps, 3 gauge substeps; acceptance ~ 0.95).

Exact references (Z(lambda) = int dU dphi exp(-S(U, phi; lambda)), constants cancel):
- kappa: Z(k2)/Z(k1) = < det(D_hat^dagger D_hat)(k2) / det(D_hat^dagger D_hat)(k1) >_{k1}, with the
  dense determinant of `exact.log_det_two_flavour` on every configuration of an independent
  equilibrium sample at k1;
- beta:  Z(b2)/Z(b1) = < exp(-(b2 - b1) S_g(U; beta = 1)) >_{b1}.
Power: in kappa the 8-step run tests the fermionic work against zero (13 sigma). In beta the work
is gauge-only, and with one step it is the reweighting exponent itself, so the beta test gets its
power from a quenched reference instead: the same 8-step run with pure-gauge HMC blocks (blocks
that ignore the fermions, same random numbers) must give a dF that differs by more than 2 sigma of
the paired jackknife. The effect is small here (dF_quenched - dF ~ 0.04-0.08 for beta 5 -> 5.6,
kappa = 0.18): 3 sigma would need about 4x the statistics (> 2 min for this file).
Errors: reference from the spread of the 64 chain means; NE-MCMC from the delete-one jackknife of
`jarzynski_free_energy` over the 256 work values (4 starting snapshots x 64 chains, snapshots 3
trajectories apart on the same chains, so the error may be slightly underestimated); 768 in beta
(8 more snapshots, 3 trajectories apart, from an independent set of 64 chains).
"""

import math
import unittest

import torch

from neoqcd.fermions import hmc as H
from neoqcd.fermions._gauge import SUN_identity, Wilson_action, mean_plaquette, plaquette_fields
from neoqcd.fermions.action import QCDAction
from neoqcd.fermions.exact import log_det_two_flavour
from neoqcd.fermions.lattice import Geometry
from neoqcd.fermions.pseudofermion import TwoFlavourHamiltonian
from neoqcd.fermions.snf import (
    OBSERVABLES,
    FermionNEMCMC,
    FieldState,
    HMCBlock,
    effective_sample_size,
    jarzynski_free_energy,
)

G2 = Geometry(2, 4, 4, 3)
B = 64
BETA, KAPPA = 5.0, 0.18
KAPPA2, BETA2 = 0.21, 5.6
HMC_KW = dict(tau=1.0, n_steps=2, n_traj=1, integrator="omelyan")
N_SUB = 3
N_THERM = 20


def cold_links(batch=B):
    return SUN_identity(G2.link_shape(batch)[:-1], dtype=torch.cdouble)


def plain_hmc(beta, kappa, seed, n_therm, n_meas, every=1, **kw):
    """Plain fermionic HMC from a cold start; returns the list of configurations after
    n_therm + k * every trajectories, k = 1 .. n_meas."""
    gen = torch.Generator().manual_seed(seed)
    ham = TwoFlavourHamiltonian(beta, kappa, G2, n_gauge_substeps=N_SUB, **kw)
    hmc = H.HMC(G2, ham, tau=HMC_KW["tau"], n_steps=HMC_KW["n_steps"], integrator=HMC_KW["integrator"])
    U = cold_links()
    out = []
    for i in range(n_therm + n_meas * every):
        U, _, _, _ = hmc.trajectory(U, gen)
        if i >= n_therm and (i - n_therm + 1) % every == 0:
            out.append(U)
    return out


def gauge_slope(U):
    """dS_g/dbeta = S_g(U; beta = 1)."""
    return H.WilsonGaugeAction(1.0, G2).action(U)


def defect_mask():
    d = torch.zeros((G2.D,) + G2.lattice, dtype=torch.double)
    d[0, 0, 0:2] = 1.0  # two time links at t = 0
    return torch.stack((1.0 - d, d), dim=0)


class Ensembles:
    """Equilibrium samples at (BETA, KAPPA), built once for the whole module."""

    ready = False

    @classmethod
    def build(cls):
        if cls.ready:
            return
        # reference sample: 64 chains x 20 measurements, 1 trajectory apart
        ref = plain_hmc(BETA, KAPPA, 1001, N_THERM, 20)
        cls.ref_plaquette = torch.stack([mean_plaquette(plaquette_fields(U, G2.D), G2.N) for U in ref])  # (20, B)
        cls.ref_det_ratio = torch.stack([
            torch.exp(log_det_two_flavour(U, KAPPA2, G2, method="assembled")
                      - log_det_two_flavour(U, KAPPA, G2, method="assembled")) for U in ref])
        cls.ref_beta_ratio = torch.stack([torch.exp(-(BETA2 - BETA) * gauge_slope(U)) for U in ref])
        # independent starting points for the NE-MCMC: 4 snapshots, 3 trajectories apart
        cls.starts = plain_hmc(BETA, KAPPA, 2002, N_THERM, 4, every=3)
        # 4 more on independent chains, for the beta test (quenched-reference resolution)
        cls.beta_starts = cls.starts + plain_hmc(BETA, KAPPA, 2003, N_THERM, 4, every=3)
        cls.ready = True


def chain_mean_error(x):
    """Mean over (measurements, chains) and error from the spread of the chain means."""
    m = x.mean(0)
    return m.mean().item(), (m.std() / math.sqrt(m.numel())).item()


def make_action(name, **kw):
    return QCDAction(G2, name, beta=BETA, kappa=KAPPA, defect_mask=defect_mask(), n_gauge_substeps=N_SUB, **kw)


def run_ne(name, protocol, seed, starts, block_class=None):
    """NE-MCMC from every starting snapshot; returns the concatenated W, Q and the last driver."""
    gen = torch.Generator().manual_seed(seed)
    act = make_action(name)
    block = None if block_class is None else block_class(act, **HMC_KW)
    ne = FermionNEMCMC(act, protocol, block=block, **({} if block is not None else HMC_KW))
    Ws, Qs = [], []
    for U in starts:
        _, W, Q, _ = ne.forward(FieldState(U), gen)
        Ws.append(W)
        Qs.append(Q)
    return torch.cat(Ws), torch.cat(Qs), ne


def linear_protocol(a, b, n):
    return [a + (b - a) * (i + 1) / n for i in range(n)]


class QuenchedBlock(HMCBlock):
    """Pure-gauge HMC block (`hmc.WilsonGaugeHamiltonian` at beta = parameter): a block that ignores
    the fermions, the wrong dynamics the beta test must detect. Beta protocol only. Each trajectory
    first draws the pseudofermions as the fermionic one does (phi enters only the S bookkeeping), so
    with the same seed both runs use the same momenta and Metropolis uniforms (common random
    numbers: correlated work, see `paired_difference`)."""

    @torch.no_grad()
    def __call__(self, U, parameter, generator=None, S_old=None):
        hmc = H.HMC(G2, H.WilsonGaugeHamiltonian(float(parameter), G2), self.tau, self.n_steps,
                    integrator=self.integrator, exp_method=self.exp_method)
        info = {"acceptance": [], "exp_minus_dH": []}
        for _ in range(self.n_traj):
            self.action.refresh(U, generator)
            U, _, _, stats = hmc.trajectory(U, generator)
            info["acceptance"].append(stats["acceptance"])
            info["exp_minus_dH"].append(stats["exp_minus_dH"])
        S_new = self.action.action(U, parameter)
        return U, S_new - S_old, S_new, info


def paired_difference(W_a, W_b):
    """dF(W_a) - dF(W_b) (Jarzynski) with the delete-one jackknife error of the difference, deleting
    the same configuration from both: the error of correlated work samples (same starts and random
    numbers), which hypot(e_a, e_b) would overestimate."""
    def loo_dF(W):
        shift = W.min()
        e = torch.exp(-(W - shift))
        return -(torch.log((e.sum() - e) / (W.numel() - 1)) - shift)
    n = W_a.numel()
    jk = loo_dF(W_a) - loo_dF(W_b)
    d = jarzynski_free_energy(W_a)[0] - jarzynski_free_energy(W_b)[0]
    return d, math.sqrt((n - 1) / n * ((jk - jk.mean()) ** 2).sum().item())


class JarzynskiTest(unittest.TestCase):
    """3, 4. <exp(-W)> = Z(lambda_n)/Z(lambda_0) for kappa and beta protocols, one jump and 8 steps."""

    @classmethod
    def setUpClass(cls):
        Ensembles.build()

    def check(self, name, start, end, ref_ratio, starts):
        r, e_r = chain_mean_error(ref_ratio)
        dF_ref, e_ref = -math.log(r), e_r / r
        lines = [f"{name}: {start} -> {end}, exact reweighting Z_n/Z_0 = {r:.4f}({e_r:.4f}), "
                 f"dF = {dF_ref:.4f}({e_ref:.4f})"]
        results = {}
        for n_step in (1, 8):
            W, Q, ne = run_ne(name, linear_protocol(start, end, n_step), 3000 + n_step, starts)
            dF, e = jarzynski_free_energy(W)
            ess = effective_sample_size(W)
            acc = sum(a for info in ne.last_run["block_info"] for a in info["acceptance"]) / n_step
            results[n_step] = (dF, e, W.mean().item(), (W.std() / math.sqrt(W.numel())).item())
            W8 = W
            lines.append(f"  n_step = {n_step}: <exp(-W)> = {math.exp(-dF):.4f}({math.exp(-dF) * e:.4f}), "
                         f"dF = {dF:.4f}({e:.4f}), <W> = {W.mean().item():.4f}, ESS = {ess:.3f}, "
                         f"(dF - dF_ref)/sigma = {(dF - dF_ref) / math.hypot(e, e_ref):+.2f}, "
                         f"last-run acceptance {acc:.3f}")
        if name == "beta":
            W_q, _, _ = run_ne(name, linear_protocol(start, end, 8), 3008, starts, QuenchedBlock)
            dF_q, e_q = jarzynski_free_energy(W_q)
            dF8, e8 = results[8][:2]
            d, e_d = paired_difference(W_q, W8)
            sep = abs(d) / e_d
            lines.append(f"  quenched blocks, n_step = 8: dF = {dF_q:.4f}({e_q:.4f}), dF_quenched - dF8 = "
                         f"{d:.4f}({e_d:.4f}) paired = {sep:.2f} sigma ({abs(d) / math.hypot(e8, e_q):.2f} "
                         f"with hypot(e8, e_q))")
        msg = "\n".join(lines)
        print("\n" + msg)
        if name == "beta":
            self.assertGreater(sep, 2.0, msg=msg)
        else:
            self.assertGreater(abs(r - 1.0), 5 * e_r, msg=msg)
        for n_step, (dF, e, W_mean, e_W) in results.items():
            self.assertLess(abs(dF - dF_ref), 3 * math.hypot(e, e_ref), msg=msg)
            self.assertGreater(W_mean, dF_ref - 3 * math.hypot(e_W, e_ref), msg=msg)  # second law <W> >= dF
        (dF1, e1, _, _), (dF8, e8, _, _) = results[1], results[8]
        self.assertLess(abs(dF1 - dF8), 3 * math.hypot(e1, e8), msg=msg)
        self.assertLess(e8, e1 * 1.5, msg=msg)

    def test_kappa_protocol(self):
        self.check("kappa", KAPPA, KAPPA2, Ensembles.ref_det_ratio, Ensembles.starts)

    def test_beta_protocol(self):
        self.check("beta", BETA, BETA2, Ensembles.ref_beta_ratio, Ensembles.beta_starts)


class RecordingNEMCMC(FermionNEMCMC):
    """Records every jump: fields, parameters, dW and the fermion solves it spent."""

    def forward(self, *args, **kwargs):
        self.jumps = []
        return super().forward(*args, **kwargs)

    def _jump(self, U, p_old, p_new, S_cur):
        solves = self.action.fermion.stats["solves"]
        dW, S_new = super()._jump(U, p_old, p_new, S_cur)
        self.jumps.append((U, self.action.phi, p_old, p_new, dW, self.action.fermion.stats["solves"] - solves))
        return dW, S_new


class RecordingBlock(HMCBlock):
    """Records, after every block, the fields (U, phi), the parameter and the end-of-block S."""

    def __call__(self, U, parameter, generator=None, S_old=None):
        U, dQ, S_new, info = super().__call__(U, parameter, generator, S_old)
        self.records.append((U, self.action.phi.clone(), parameter, S_new, info))
        return U, dQ, S_new, info


class BookkeepingTest(unittest.TestCase):
    """5. W + Q + S_0 - S_final = 0 per chain for the three protocols (S_final recomputed from
    scratch), each jump equal to an independent action difference, and in bc the fermion part of
    the work identically zero (no solve, gauge-only Wilson_action differences)."""

    PROTOCOLS = {"beta": (BETA, [5.1, 5.2, 5.25]), "kappa": (KAPPA, [0.19, 0.2, 0.205]),
                 "bc": (1.0, [0.6, 0.3, 0.0])}

    def test_accounting(self):
        Ensembles.build()
        U0 = Ensembles.starts[0][:16]
        for name, (start, protocol) in self.PROTOCOLS.items():
            with self.subTest(parameter=name):
                gen = torch.Generator().manual_seed(77)
                act = make_action(name, bc=1.0)
                act.set_parameter(start)
                ne = RecordingNEMCMC(act, protocol, **HMC_KW)
                state, W, Q, obs = ne.forward(FieldState(U0), gen)
                S0 = ne.last_run["S0"]
                # S_final from scratch, by an independent action object at the final parameter
                check = make_action(name, bc=1.0)
                check.set_parameter(protocol[-1])
                check.phi = state.phi
                S_final = check.action(state.U)
                scale = S_final.abs().max().item()
                self.assertLess((W + Q + S0 - S_final).abs().max().item(), 1e-10 * scale)
                self.assertTrue(torch.equal(state.work, W) and torch.equal(state.heat, Q))
                self.assertEqual(obs.shape, (16, len(protocol), len(OBSERVABLES)))
                self.assertLess((obs[:, :, 0] + obs[:, :, 1] - (obs[:, :, 3] - S0[:, None])).abs().max().item(),
                                1e-10 * scale)
                self.assertTrue(torch.equal(obs[:, -1, 0], W))
                self.assertTrue(torch.allclose(obs[0, :, 4], torch.tensor(protocol, dtype=torch.double)))
                # each jump is the action difference at fixed (U, phi) by an independent object
                self.assertEqual(len(ne.jumps), len(protocol))
                for U, phi, p_old, p_new, dW, solves in ne.jumps:
                    ref_act = make_action(name, bc=1.0)
                    ref_act.phi = phi
                    ref = ref_act.action(U, p_new) - ref_act.action(U, p_old)
                    self.assertLess((dW - ref).abs().max().item(), 1e-10 * ref.abs().max().item())
                    if name == "bc":
                        self.assertEqual(solves, 0)  # fermion part of the work identically zero
                        pars = type("P", (), dict(D=G2.D, N=G2.N, device=torch.device("cpu")))
                        S = lambda c: Wilson_action(pars, BETA, c)(U, defect_mask())  # noqa: E731
                        self.assertLess((dW - (S(p_new) - S(p_old))).abs().max().item(), 1e-10 * scale)
                    elif name == "beta":
                        self.assertEqual(solves, 0)
                    else:
                        self.assertEqual(solves, 1)  # S at the old kappa is known from the block

    def test_low_acceptance(self):
        """kappa protocol with one Omelyan step per trajectory (acceptance well below 1): after every
        block the end-of-block S equals that of an independent action object fed the recorded
        (U, phi), so a stale S after a rejected trajectory (old U, new phi) would be caught; each
        jump is checked as in test_accounting."""
        Ensembles.build()
        U0 = Ensembles.starts[0][:32]
        protocol = [0.19, 0.2, 0.205, 0.21]
        act = make_action("kappa")
        block = RecordingBlock(act, tau=1.0, n_steps=1, n_traj=1, integrator="omelyan")
        block.records = []
        ne = RecordingNEMCMC(act, protocol, block=block)
        state, W, Q, obs = ne.forward(FieldState(U0), torch.Generator().manual_seed(78))
        acc = [a for *_, info in block.records for a in info["acceptance"]]
        print(f"\nlow-acceptance kappa protocol: acceptance per block {[round(a, 3) for a in acc]}")
        self.assertTrue(0.1 < sum(acc) / len(acc) < 0.7 and min(acc) < 0.9, msg=str(acc))
        self.assertEqual(len(block.records), len(protocol))
        for i, (U, phi, p, S_rec, _) in enumerate(block.records):
            ref_act = make_action("kappa")
            ref_act.phi = phi
            ref = ref_act.action(U, p)
            self.assertLess((S_rec - ref).abs().max().item(), 1e-9 * ref.abs().max().item())
            self.assertTrue(torch.equal(obs[:, i, 3], S_rec))
        self.assertEqual(len(ne.jumps), len(protocol))
        for U, phi, p_old, p_new, dW, _ in ne.jumps:
            ref_act = make_action("kappa")
            ref_act.phi = phi
            ref = ref_act.action(U, p_new) - ref_act.action(U, p_old)
            self.assertLess((dW - ref).abs().max().item(), 1e-10 * ref.abs().max().item())
        self.assertLess((W + Q + ne.last_run["S0"] - block.records[-1][3]).abs().max().item(),
                        1e-10 * block.records[-1][3].abs().max().item())


class EquilibriumLimitTest(unittest.TestCase):
    """6. A constant protocol gives W = 0 exactly, and its HMC blocks are the plain fermionic HMC:
    with reuse_guess=False (the force is a function of (U, phi) only) the same seed gives the same
    chains, and the plaquette agrees with the reference ensemble."""

    def test_constant_protocol(self):
        Ensembles.build()
        n = 6
        U0 = Ensembles.starts[-1]
        act = make_action("kappa", reuse_guess=False)
        ne = FermionNEMCMC(act, [KAPPA] * n, **HMC_KW)
        state, W, Q, obs = ne.forward(FieldState(U0, phi=torch.zeros(G2.wilson_shape(B), dtype=torch.cdouble)),
                                      torch.Generator().manual_seed(55))
        self.assertTrue(torch.equal(W, torch.zeros_like(W)))
        # plain HMC with the same random stream (phi given: the driver draws nothing before the blocks)
        gen = torch.Generator().manual_seed(55)
        ham = TwoFlavourHamiltonian(BETA, KAPPA, G2, n_gauge_substeps=N_SUB, reuse_guess=False)
        hmc = H.HMC(G2, ham, tau=HMC_KW["tau"], n_steps=HMC_KW["n_steps"], integrator=HMC_KW["integrator"])
        U = U0
        plaq = []
        for _ in range(n):
            U, _, _, _ = hmc.trajectory(U, gen)
            plaq.append(mean_plaquette(plaquette_fields(U, G2.D), G2.N))
        plaq = torch.stack(plaq, dim=1)
        self.assertLess((obs[:, :, 5] - plaq).abs().max().item(), 1e-12)
        self.assertLess((state.U - U).abs().max().item(), 1e-12)
        p_ne, e_ne = chain_mean_error(obs[:, :, 5].T)
        p_ref, e_ref = chain_mean_error(Ensembles.ref_plaquette)
        print(f"\nconstant protocol kappa = {KAPPA}: <P> NE-MCMC blocks {p_ne:.4f}({e_ne:.4f}), "
              f"reference HMC {p_ref:.4f}({e_ref:.4f})")
        self.assertLess(abs(p_ne - p_ref), 4 * math.hypot(e_ne, e_ref))


if __name__ == "__main__":
    unittest.main()
