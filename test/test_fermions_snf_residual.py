"""Residual normalizing-flow layers in the fermionic SNF (neoqcd.fermions.snf_layers):
`make_gauge_layers(..., kind="residual"|"hyper-residual", residual={...})`, the conditioning of
`FermionSNF` (`layer_conditioning`), and the driver options of main/main_snf_fermions.py.

CPU, complex128, fixed seeds. 2D 4 x 4 SU(3), beta = 5, kappa = 0.18 (the setting and the
equilibrium ensembles of test_fermions_nemcmc / test_fermions_snf). In 2D every variant is valid
(4 is a multiple of both mask moduli); n_pt = 1 with mod 2 is a valid bijection there but adds no
effective dependence (README), so it is also exercised in 4D on a 2^4 lattice. Variants:
static / hyper coefficients x (n_pt, mask_mod) in (0, 2), (1, 2), (2, 2), (0, 4), (2, 4), all with
imaginary and quadratic features (the defaults); one variant without them.
"""

import math
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import torch

from neoqcd.fermions._gauge import FlowPars, create_mask
from neoqcd.fermions.algebra import algebra_basis, algebra_components, expi, project_hermitian_traceless, random_algebra
from neoqcd.fermions.pseudofermion import TwoFlavourWilson
from neoqcd.fermions.snf import FermionNEMCMC, FieldState, effective_sample_size, jarzynski_free_energy
from neoqcd.fermions.snf_layers import (FermionSNF, layer_conditioning, make_gauge_layers, pseudofermion_action,
                                        residual_options)
from test.test_fermions_nemcmc import BETA, G2, HMC_KW, KAPPA, KAPPA2, Ensembles, chain_mean_error, linear_protocol
from test.test_fermions_snf import dense_pf_action, fd4, make_action

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = [(par, n_pt, mod) for par in ("static", "hyper") for n_pt, mod in ((0, 2), (1, 2), (2, 2), (0, 4), (2, 4))]
KIND = {"static": "residual", "hyper": "hyper-residual"}
DELTA = 0.25


def flow_pars(protocol, batch_size, D=2, T=4, L=4):
    return FlowPars(D=D, T=T, L=L, N=3, mask=create_mask(D, T, L), protocol=list(protocol), batch_size=batch_size,
                    device="cpu")


def residual_layer(par, n_pt, mod, fp, seed=0, scale=0.1, **options):
    """One residual layer with all parameters perturbed (moderate strength at DELTA)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # n_pt = 1, mod 2 in 2D
        layer = make_gauge_layers(fp, KIND[par], residual=dict(n_pt=n_pt, mask_mod=mod, **options))[0]
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in layer.parameters():
            p.add_(scale * torch.randn(p.shape, generator=gen, dtype=p.dtype))
    return layer


def random_links(shape, seed):
    return expi(random_algebra(shape + (3, 3), torch.Generator().manual_seed(seed)))


def dense_haar_logdet(layer, fp, U0, beta, delta):
    """log |det| of the layer map in orthonormal left-invariant tangent coordinates (HaarJacobianTest)."""
    T = algebra_basis(3)
    with torch.no_grad():
        U0p, _, _ = layer(U0, fp, None, beta=beta, delta_beta=delta)
    n = U0[..., 0, 0].numel() * T.shape[0]

    def f(x):
        X = torch.einsum("...a,aij->...ij", x.view(U0.shape[:-2] + (T.shape[0],)).to(U0.dtype), T)
        Up, _, _ = layer(torch.matmul(expi(X), U0), fp, None, beta=beta, delta_beta=delta)
        Y = project_hermitian_traceless(-1j * torch.matmul(Up, U0p.conj().transpose(-1, -2)))
        return algebra_components(Y).reshape(-1)

    J = torch.autograd.functional.jacobian(f, torch.zeros(n, dtype=torch.float64), vectorize=True)
    return torch.linalg.slogdet(J)


class OptionsAndConditioningTest(unittest.TestCase):
    def test_options(self):
        opt = residual_options("hyper-residual", lattice=(4, 4), n_pt=2, mask_mod=4)
        self.assertEqual((opt["parameterization"], opt["n_pt"], opt["mask_mod"], opt["coeff_init"]), ("hyper", 2, 4, 1e-3))
        self.assertEqual(residual_options("residual")["parameterization"], "static")
        for bad in (dict(n_pt=-1), dict(mask_mod=3), dict(n_pt=1.5), dict(coeff_max=-1.0), dict(hyper_depth=0),
                    dict(hyper_time_embedding_dim=3), dict(hyper_activation="elu"), dict(foo=1)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                residual_options("residual", **bad)
        with self.assertRaises(ValueError):  # extents not multiples of 4
            residual_options("residual", lattice=(4, 6), mask_mod=4)
        with self.assertRaises(ValueError):  # n_pt > 0 needs even extents
            residual_options("residual", lattice=(4, 3), n_pt=1)
        with self.assertWarns(RuntimeWarning):
            residual_options("residual", lattice=(4, 4), n_pt=1)
        with self.assertRaises(ValueError):
            make_gauge_layers(flow_pars([0.2], 1), "smearing", residual=dict(n_pt=1))
        fp = flow_pars([0.2], 1)
        layer = make_gauge_layers(fp, "residual", residual=dict(n_pt=2, mask_mod=4, coeff_init=0.0))[0]
        self.assertEqual((layer.residual_flow.n_pt, layer.residual_flow.mask_mod), (2, 4))
        self.assertEqual(layer.residual_flow.parameterization, "static")
        self.assertEqual(fp.residual_n_pt, 0)  # the caller's FlowPars is not modified
        # sharing: hyper-residual shares one layer across the protocol by default (as main_nemcmc_snf)
        fp = flow_pars([0.19, 0.2, 0.21], 1)
        shared = make_gauge_layers(fp, "hyper-residual")
        self.assertTrue(all(layer is shared[0] for layer in shared))
        per_step = make_gauge_layers(fp, "residual")
        self.assertEqual(len({id(layer) for layer in per_step}), 3)
        self.assertEqual(len({id(layer) for layer in make_gauge_layers(fp, "residual", sharing="shared")}), 1)
        with self.assertRaises(ValueError):
            make_gauge_layers(fp, "smearing", sharing="shared")
        snf = FermionSNF(make_action("kappa"), [0.19, 0.2, 0.21], shared, fp, **HMC_KW)
        self.assertEqual(sum(p.numel() for p in snf.parameters()), sum(p.numel() for p in shared[0].parameters()))

    def test_conditioning(self):
        prot = linear_protocol(0.18, 0.21, 4)
        c = layer_conditioning(0.18, prot, "time", "fraction")
        for i, (ci, di) in enumerate(c):
            self.assertAlmostEqual(ci, (i + 1) / 4, places=14)
            self.assertAlmostEqual(di, 0.25, places=14)
        # pure-gauge convention: lambda_i and forward differences, 0 for one step
        c = layer_conditioning(0.18, prot, "parameter", "step")
        self.assertEqual([x[0] for x in c], prot)
        self.assertAlmostEqual(c[0][1], prot[1] - prot[0], places=15)
        self.assertEqual(layer_conditioning(0.18, [0.2], "parameter", "step"), [(0.2, 0.0)])
        # a zero step gives an exactly zero delta; bc 1 -> 0 is decreasing: negative fractions
        c = layer_conditioning(1.0, [0.5, 0.5, 0.0], "time", "fraction")
        self.assertEqual(c, [(0.5, -0.5), (0.5, 0.0), (1.0, -0.5)])
        self.assertEqual(layer_conditioning(1.0, [1.0, 1.0], "time", "fraction"), [(0.5, 0.0), (1.0, 0.0)])


class LayerTest(unittest.TestCase):
    """(a) SU(3), (b) dlogJ = dense Haar log-Jacobian to 1e-9, (c) exact identity at zero step."""

    def check(self, par, n_pt, mod, D, T, L, seed, **options):
        fp = flow_pars([0.2], 1, D, T, L)
        layer = residual_layer(par, n_pt, mod, fp, seed=seed, **options)
        U0 = random_links((1, D, T) + (L,) * (D - 1), seed + 100)
        beta = 0.5
        with torch.no_grad():
            Up, dQ, dlogJ = layer(U0, fp, None, beta=beta, delta_beta=DELTA)
            eye = torch.eye(3, dtype=Up.dtype)
            unit = (torch.matmul(Up, Up.conj().transpose(-1, -2)) - eye).abs().max().item()
            det = (torch.linalg.det(Up) - 1).abs().max().item()
            U1, _, lj0 = layer(U0, fp, None, beta=beta, delta_beta=0.0)
        self.assertEqual(dQ, 0)
        self.assertLess(unit, 1e-13)
        self.assertLess(det, 1e-13)
        self.assertGreater((Up - U0).abs().max().item(), 2e-3)
        self.assertTrue(torch.equal(U1, U0))
        self.assertEqual(float(lj0.abs().max()), 0.0)
        sign, logdet = dense_haar_logdet(layer, fp, U0, beta, DELTA)
        diff = abs(logdet.item() - dlogJ.item())
        print(f"\n{par} n_pt={n_pt} mod={mod} D={D} {options or ''}: dlogJ {dlogJ.item():+.12f}, "
              f"dense {logdet.item():+.12f}, diff {diff:.1e}, |U'-U| {(Up - U0).abs().max().item():.2f}")
        self.assertEqual(sign.item(), 1.0)
        self.assertGreater(abs(dlogJ.item()), 3e-3)
        self.assertLess(diff, 1e-9)

    def test_variants_2d(self):
        for k, (par, n_pt, mod) in enumerate(VARIANTS):
            with self.subTest(par=par, n_pt=n_pt, mod=mod):
                self.check(par, n_pt, mod, 2, 4, 4, seed=k)

    def test_linear_real_features_2d(self):
        for par in ("static", "hyper"):
            with self.subTest(par=par):
                self.check(par, 2, 2, 2, 4, 4, seed=40, include_imag=False, quadratic=False)

    def test_4d_one_convolution(self):
        """n_pt = 1, mod 2 is effective only for D > 2: 2^4, hyper coefficients."""
        self.check("hyper", 1, 2, 4, 2, 2, seed=50)


class GradientTest(unittest.TestCase):
    """(d) d/dtheta S_pf(U'(theta), phi) through PseudofermionAction: against 4th-order finite
    differences along random parameter directions (1e-6) and against autograd through a dense
    solve (1e-9), static and hyper, two convolutions."""

    @classmethod
    def setUpClass(cls):
        Ensembles.build()
        cls.U = Ensembles.starts[1][:2].clone()
        cls.fermion = TwoFlavourWilson(KAPPA, G2, solver_tol=1e-13, action_tol=1e-14, reuse_guess=False)
        cls.fermion.refresh(cls.U, torch.Generator().manual_seed(5))

    def test_gradient(self):
        U, fermion = self.U, self.fermion
        for k, par in enumerate(("static", "hyper")):
            with self.subTest(par=par):
                fp = flow_pars([KAPPA], U.shape[0])
                layer = residual_layer(par, 2, 2, fp, seed=60 + k, scale=0.05)
                params = [p for p in layer.parameters()]

                def L(dense=False):
                    Up, _, _ = layer(U, fp, None, beta=0.5, delta_beta=DELTA)
                    S = dense_pf_action(Up, fermion.phi, KAPPA) if dense else pseudofermion_action(Up, fermion)
                    return S.sum()

                g = torch.cat([x.reshape(-1) for x in torch.autograd.grad(L(), params)])
                g_dense = torch.cat([x.reshape(-1) for x in torch.autograd.grad(L(True), params)])
                rel_dense = ((g - g_dense).norm() / g_dense.norm()).item()
                theta0 = [p.detach().clone() for p in params]
                gen = torch.Generator().manual_seed(70 + k)
                worst = 0.0
                for _ in range(3):
                    v = [torch.randn(p.shape, generator=gen, dtype=p.dtype) for p in params]
                    vflat = torch.cat([x.reshape(-1) for x in v])

                    def f(h):
                        with torch.no_grad():
                            for p, t, d in zip(params, theta0, v):
                                p.copy_(t + h * d)
                            return L().item()
                    fd = fd4(f, 1e-4)
                    f(0.0)
                    worst = max(worst, abs((g @ vflat).item() - fd) / abs(fd))
                print(f"\n{par}: |grad| {g.norm().item():.3e}, rel diff vs dense autograd {rel_dense:.1e}, "
                      f"worst directional rel diff vs FD {worst:.1e}")
                self.assertGreater(g.norm().item(), 1e-3)
                self.assertLess(rel_dense, 1e-9)
                self.assertLess(worst, 1e-6)

    def test_gradient_at_identity(self):
        """No dead point: coeff_init = 0 (the identity) has a nonzero gradient for every variant's
        linear coefficients (static_linear, or the hyper-network's output layer)."""
        U, fermion = self.U, self.fermion
        for par in ("static", "hyper"):
            with self.subTest(par=par):
                fp = flow_pars([KAPPA], U.shape[0])
                layer = make_gauge_layers(fp, KIND[par], residual=dict(n_pt=2, coeff_init=0.0))[0]
                Up, _, lj = layer(U, fp, None, beta=0.5, delta_beta=DELTA)
                self.assertTrue(torch.equal(Up.detach(), U))
                loss = (pseudofermion_action(Up, fermion) - lj).sum()
                named = dict(layer.named_parameters())
                grads = dict(zip(named, torch.autograd.grad(loss, list(named.values()))))
                key = "residual_flow.static_linear" if par == "static" else "residual_flow.mlp.4.bias"
                print(f"\n{par} at the identity: |grad| " + ", ".join(f"{n.split('.', 1)[1]} {g.norm().item():.2e}"
                                                                     for n, g in grads.items()))
                self.assertGreater(grads[key].norm().item(), 1e-3)
                self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads.values()))


class SNFTest(unittest.TestCase):
    """(e) bookkeeping for the three protocols; identity layers reproduce FermionNEMCMC;
    (f) Jarzynski in kappa with untrained residual layers against exact reweighting."""

    PROTOCOLS = {"kappa": (KAPPA, [0.195, 0.21]), "beta": (BETA, [5.15, 5.3]), "bc": (1.0, [0.5, 0.0])}

    @classmethod
    def setUpClass(cls):
        Ensembles.build()

    def make(self, name, start, protocol, B, kind="hyper-residual", perturb=0.0, **options):
        """FermionSNF with residual layers; perturb > 0 adds perturb * N(0, 1) to every parameter
        (fixed seed), so that hyper-network coefficients depend on the conditioning value."""
        act = make_action(name, reuse_guess=False)
        act.set_parameter(start)
        fp = flow_pars(protocol, B)
        layers = make_gauge_layers(fp, kind, residual=options)
        snf = FermionSNF(act, protocol, layers, fp, **HMC_KW)
        if perturb:
            gen = torch.Generator().manual_seed(77)
            with torch.no_grad():
                for p in snf.parameters():
                    p.add_(perturb * torch.randn(p.shape, generator=gen, dtype=p.dtype))
        return snf, act

    def test_accounting(self):
        U0 = Ensembles.starts[0][:8]
        for name, (start, protocol) in self.PROTOCOLS.items():
            for kind in ("residual", "hyper-residual"):
                with self.subTest(parameter=name, kind=kind):
                    snf, _ = self.make(name, start, protocol, U0.shape[0], kind, perturb=0.01 if kind == "hyper-residual"
                                       else 0.0, n_pt=2, coeff_init=0.02)
                    with torch.no_grad():
                        state, W, Q, logJ, obs = snf(FieldState(U0), torch.Generator().manual_seed(11))
                    check = make_action(name)
                    check.set_parameter(protocol[-1])
                    check.phi = state.phi
                    S_final = check.action(state.U)
                    scale = S_final.abs().max().item()
                    resid = (W + Q - (S_final - snf.last_run["S0"] - logJ)).abs().max().item()
                    print(f"\n{name} {kind}: max |W + Q - (S_final - S_0 - logJ)| = {resid:.1e}, "
                          f"<logJ> = {logJ.mean().item():+.4f}, <W> = {W.mean().item():+.4f}")
                    self.assertLess(resid, 1e-10 * scale)
                    self.assertGreater(logJ.abs().min().item(), 1e-3)

    def test_identity_is_nemcmc(self):
        U = Ensembles.starts[3][:16]
        for name, (start, protocol) in self.PROTOCOLS.items():
            for kind in ("residual", "hyper-residual"):
                with self.subTest(parameter=name, kind=kind):
                    snf, _ = self.make(name, start, protocol, U.shape[0], kind, n_pt=2, coeff_init=0.0)
                    with torch.no_grad():
                        st1, W1, Q1, logJ1, _ = snf(FieldState(U), torch.Generator().manual_seed(21))
                    act = make_action(name, reuse_guess=False)
                    act.set_parameter(start)
                    st2, W2, Q2, _ = FermionNEMCMC(act, protocol, **HMC_KW).forward(
                        FieldState(U), torch.Generator().manual_seed(21))
                    self.assertTrue(torch.equal(st1.U, st2.U))
                    self.assertLess((W1 - W2).abs().max().item(), 1e-11)
                    self.assertLess((Q1 - Q2).abs().max().item(), 1e-10)
                    self.assertLess(logJ1.abs().max().item(), 1e-11)

    def test_jarzynski(self):
        r, e_r = chain_mean_error(Ensembles.ref_det_ratio)
        dF_ref, e_ref = -math.log(r), e_r / r
        protocol = linear_protocol(KAPPA, KAPPA2, 4)
        for kind, options in (("residual", dict(coeff_init=-0.02)),
                              ("hyper-residual", dict(n_pt=2, coeff_init=0.02, perturb=0.01))):
            with self.subTest(kind=kind):
                gen = torch.Generator().manual_seed(4004)
                Ws, logJs = [], []
                for U in Ensembles.starts:
                    snf, _ = self.make("kappa", KAPPA, protocol, U.shape[0], kind, **options)
                    with torch.no_grad():
                        _, W, _, logJ, _ = snf(FieldState(U), gen)
                    Ws.append(W)
                    logJs.append(logJ)
                W, logJ = torch.cat(Ws), torch.cat(logJs)
                dF, e = jarzynski_free_energy(W)
                z = (dF - dF_ref) / math.hypot(e, e_ref)
                msg = (f"{kind} {options}: dF = {dF:.4f}({e:.4f}), exact {dF_ref:.4f}({e_ref:.4f}), {z:+.2f} sigma, "
                       f"<W> = {W.mean().item():.4f}, ESS = {effective_sample_size(W):.3f}, <logJ> = {logJ.mean().item():+.4f}")
                print("\n" + msg)
                self.assertLess(abs(z), 3.0, msg=msg)
                self.assertGreater(logJ.abs().mean().item(), 0.1, msg=msg)


def _load_main():
    import importlib.util
    for p in (ROOT / "main",):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location("main_snf_fermions", ROOT / "main" / "main_snf_fermions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LATTICE = ["--D", "2", "--T", "4", "--L", "4", "--N", "3", "--beta", "5.0", "--kappa", "0.18",
           "--tau", "0.5", "--n-steps", "2", "--n-gauge-substeps", "2", "--device", "cpu", "--quiet"]


class DriverTest(unittest.TestCase):
    """(g) main_snf_fermions.py: static and hyper residual layers train two steps and evaluate,
    a finished run resumes to a no-op, the architecture is recorded, --load-flow refuses another
    architecture and accepts the same one; option validation."""

    @classmethod
    def setUpClass(cls):
        cls.SNF = _load_main()

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_train_eval_resume_load(self):
        import json
        SNF = self.SNF
        for layers, extra in (("residual", ["--residual-n-pt", "2", "--residual-coeff-init", "0"]),
                              ("hyper-residual", ["--residual-mask-mod", "4", "--hyper-hidden-dim", "8"])):
            with self.subTest(layers=layers):
                out = self.tmp / layers
                core = LATTICE + ["--protocol", "kappa", "--final", "0.19", "--n-step", "2", "--mode", "snf",
                                  "--layers", layers, "--batch", "4", "--therm", "1", "--seed", "7",
                                  "--eval-samples", "4"]
                base = core + ["--lr", "1e-3", "--train-steps", "2", "--checkpoint-every", "1"] + extra
                SNF.main(base + ["--out", str(out)])
                rows = SNF.read_observables(out / "training.csv")
                self.assertEqual(rows["step"].size, 2)
                self.assertTrue((rows["grad_norm"] > 0).all())
                side = json.loads((out / "flow.json").read_text())
                meta = json.loads((out / "run_meta.json").read_text())
                self.assertEqual(side["architecture"]["layers"], layers)
                self.assertEqual(side["architecture"]["condition"], "time")
                self.assertEqual(side["architecture"]["sharing"], "per_step" if layers == "residual" else "shared")
                self.assertEqual(meta["run"]["flow_architecture"], side["architecture"])
                self.assertEqual([c[1] for c in side["conditioning"]], [0.5, 0.5])
                self.assertTrue(meta["finished"] and math.isfinite(meta["summary"]["dF"]))
                SNF.main(base + ["--out", str(out), "--resume"])
                self.assertEqual(SNF.read_observables(out / "samples.csv")["work"].size, 4)
                # the same architecture loads; another one is refused
                ok = self.tmp / (layers + "_load")
                load = core + ["--load-flow", str(out / "flow.pt")]
                SNF.main(load + extra + ["--out", str(ok)])
                self.assertEqual(SNF.read_observables(ok / "samples.csv")["work"].size, 4)
                other = ["--residual-n-pt", "1"] if layers == "residual" else ["--residual-mask-mod", "2"]
                for k, bad in enumerate((other, extra + ["--residual-delta", "step"])):
                    with self.assertRaises(SystemExit):
                        SNF.main(load + bad + ["--out", str(self.tmp / f"bad{k}")])

    def test_validation(self):
        SNF = self.SNF
        base = LATTICE + ["--protocol", "kappa", "--final", "0.19", "--n-step", "2", "--mode", "snf"]
        for bad in (["--layers", "smearing", "--residual-n-pt", "1"],
                    ["--layers", "smearing", "--residual-condition", "parameter"],
                    ["--layers", "residual", "--residual-n-pt", "-1"],
                    ["--layers", "residual", "--residual-mask-mod", "4", "--L", "6"],
                    ["--layers", "hyper-residual", "--hyper-time-embedding-dim", "3"]):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                SNF.parse_args(base + bad)
        args = SNF.parse_args(base + ["--layers", "hyper-residual", "--residual-n-pt", "2"])
        self.assertEqual(args.residual["n_pt"], 2)
        self.assertEqual(SNF.physics_key(args)["residual"]["parameterization"], "hyper")


if __name__ == "__main__":
    unittest.main()
