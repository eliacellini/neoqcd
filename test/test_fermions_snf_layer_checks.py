"""Checks of the SNF layers the review of 20 November did by hand (scripts jac.py, pf.py, blk.py,
jarz.py), small and fast:

- the exact Haar log-Jacobian of a 4D signed stout layer (2^3 x 4, SU(3)) for a batch of 2 and
  both signs of rho, against the dense 1024 x 1024 tangent-space Jacobian of each sample (by
  autograd), with the other sample left untouched;
- the gradient of sum_b w_b S_pf,b through a layer with non-unit weights w_b and a (B,) kappa,
  reference and fast terms, against dense autograd of the even-odd determinant solve;
- gauge equivariance f(U^g) = f(U)^g and invariance of log J for every layer kind (stout, signed
  stout, static and hyper residual with 0-2 frozen-link convolutions, mask mod 2 and 4), in 2D and
  4D;
- Jarzynski with a NON-identity signed layer (rho < 0, which the |rho| layer cannot express)
  against the exact determinant ratio (kappa 0.18 -> 0.21 on 2D 4 x 4, the ensembles of
  test_fermions_nemcmc).
"""

import math
import unittest

import torch

from neoqcd.fermions._gauge import FlowPars, create_mask
from neoqcd.fermions.algebra import (algebra_basis, algebra_components, expi, project_hermitian_traceless,
                                     random_algebra)
from neoqcd.fermions.exact import assemble_even_odd
from neoqcd.fermions.fast import FastTwoFlavourWilson
from neoqcd.fermions.lattice import Geometry, gauge_transform_links, parity
from neoqcd.fermions.pseudofermion import TwoFlavourWilson
from neoqcd.fermions.snf import FieldState, effective_sample_size, jarzynski_free_energy
from neoqcd.fermions.snf_layers import FermionSNF, make_gauge_layers, pseudofermion_action
from test.test_fermions_nemcmc import (G2, HMC_KW, KAPPA, KAPPA2, Ensembles, chain_mean_error, linear_protocol,
                                       make_action)


def flow_pars(g, protocol, batch_size=1):
    return FlowPars(D=g.D, T=g.T, L=g.L, N=g.N, mask=create_mask(g.D, g.T, g.L), protocol=list(protocol),
                    batch_size=batch_size, device="cpu")


def random_links(g, batch, seed, scale=1.0):
    return expi(scale * random_algebra(g.link_shape(batch), torch.Generator().manual_seed(seed)))


def perturb(layer, scale, seed):
    """Every parameter of the layer moved by scale * N(0, 1) (a generic, non-identity layer)."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in layer.parameters():
            p.add_(scale * torch.randn(p.shape, generator=gen, dtype=p.dtype))
    return layer


class Haar4DTest(unittest.TestCase):
    """dlogJ of a signed stout layer in 4D = log |det| of the dense Jacobian in left-invariant
    orthonormal tangent coordinates (input e^{iX} U, output -i U'(X) U'(0)^dagger on su(3)), per
    sample of a batch of 2, rho of either sign (one value per mask slot)."""

    G = Geometry(4, 4, 2, 3)

    def test_logdet_batch_and_sign(self):
        g = self.G
        fp = flow_pars(g, [0.0], 2)
        U0 = random_links(g, 2, 41, scale=0.6)
        T = algebra_basis(g.N)
        n = U0[0, ..., 0, 0].numel() * T.shape[0]
        self.assertEqual(n, 1024)
        base = torch.tensor([0.05, 0.03, 0.06, 0.02, 0.04, 0.07, 0.01, 0.035], dtype=torch.float64)
        for sign in (1.0, -1.0):
            layer = make_gauge_layers(fp, "smearing_signed", rho=0.0)[0]
            with torch.no_grad():
                layer.rho_layer.copy_((sign * base).view(fp.rho_shape))
                U0p, _, dlogJ = layer(U0, fp, None)
            dlogJ = torch.as_tensor(dlogJ).reshape(-1)
            for b in range(2):
                other = 1 - b

                def f(x, b=b):
                    X = torch.einsum("...a,aij->...ij", x.view(U0.shape[1:-2] + (T.shape[0],)).to(U0.dtype), T)
                    U = U0.clone()
                    U[b] = torch.matmul(expi(X), U0[b])
                    Up, _, _ = layer(U, fp, None)
                    Y = project_hermitian_traceless(-1j * torch.matmul(Up[b], U0p[b].conj().transpose(-1, -2)))
                    return algebra_components(Y).reshape(-1)

                J = torch.autograd.functional.jacobian(f, torch.zeros(n, dtype=torch.float64), vectorize=True)
                s, logdet = torch.linalg.slogdet(J)
                with self.subTest(sign=sign, sample=b):
                    print(f"\n4D signed stout, rho sign {sign:+.0f}, sample {b}: dense {logdet.item():.12f}, "
                          f"layer {dlogJ[b].item():.12f}")
                    self.assertEqual(s.item(), 1.0)
                    self.assertGreater(abs(dlogJ[b].item()), 1e-2)
                    self.assertLess(abs(logdet.item() - dlogJ[b].item()), 1e-10)
                # the other sample of the batch is not affected by sample b's input
                U = U0.clone()
                U[b] = torch.matmul(expi(0.1 * random_algebra(U0.shape[1:], torch.Generator().manual_seed(7))), U0[b])
                with torch.no_grad():
                    Up, _, lj = layer(U, fp, None)
                self.assertTrue(torch.equal(Up[other], U0p[other]))
                self.assertEqual(float(torch.as_tensor(lj).reshape(-1)[other]), float(dlogJ[other]))


def dense_pf_action(U, phi, kappa):
    """S_pf = |D_hat^-dagger phi_e|^2 from the dense even-odd matrix ((B,) kappa), autograd in U."""
    B = U.shape[0]
    even = (parity(G2).reshape(-1) == 0).nonzero().squeeze(-1)
    p = phi.reshape(B, G2.volume, -1)[:, even].reshape(B, -1, 1)
    Dh = assemble_even_odd(U, kappa, G2)
    y = torch.linalg.solve(Dh.conj().transpose(-1, -2), p)
    return (y.conj() * y).real.reshape(B, -1).sum(1)


class WeightedGradientTest(unittest.TestCase):
    """d/drho sum_b w_b S_pf,b(f(U)) with w = (0.7, -1.3, 2.0) and kappa = (0.16, 0.18, 0.2): the
    custom Function (reference and fast term) against dense autograd, and against the weighted sum
    of single-sample gradients."""

    def test_weights_and_batched_kappa(self):
        U = random_links(G2, 3, 51, scale=0.5)
        kappa = torch.tensor([0.16, 0.18, 0.2], dtype=torch.float64)
        w = torch.tensor([0.7, -1.3, 2.0], dtype=torch.float64)
        fp = flow_pars(G2, [0.0], 3)
        layer = make_gauge_layers(fp, "smearing_signed", rho=0.0)[0]
        with torch.no_grad():
            layer.rho_layer.copy_(torch.tensor([0.05, -0.08, 0.03, -0.06], dtype=torch.float64).view(fp.rho_shape))
        ref = TwoFlavourWilson(kappa.clone(), G2, solver_tol=1e-13, action_tol=1e-14, reuse_guess=False)
        ref.refresh(U, torch.Generator().manual_seed(52))
        fast = FastTwoFlavourWilson(kappa.clone(), G2, solver_tol=1e-13, action_tol=1e-14, reuse_guess=False)
        fast.phi = ref.phi.clone()

        def grad(action):
            U1, _, _ = layer(U, fp, None)
            return torch.autograd.grad((w * action(U1)).sum(), layer.rho_layer)[0].reshape(-1)

        g_dense = grad(lambda V: dense_pf_action(V, ref.phi, kappa))
        self.assertGreater(g_dense.abs().max().item(), 1e-2)
        for name, term in (("reference", ref), ("fast", fast)):
            g = grad(lambda V, term=term: pseudofermion_action(V, term))
            rel = ((g - g_dense).abs().max() / g_dense.abs().max()).item()
            with self.subTest(term=name):
                self.assertLess(rel, 1e-9, f"{name}: {rel:.1e}")
        # the weights act per sample: the same as the weighted single-sample gradients
        single = torch.zeros_like(g_dense)
        for b in range(3):
            t = TwoFlavourWilson(float(kappa[b]), G2, solver_tol=1e-13, action_tol=1e-14, reuse_guess=False)
            t.phi = ref.phi[b:b + 1].clone()
            fp1 = flow_pars(G2, [0.0], 1)
            U1, _, _ = layer(U[b:b + 1], fp1, None)
            single += w[b] * torch.autograd.grad(pseudofermion_action(U1, t).sum(), layer.rho_layer)[0].reshape(-1)
        self.assertLess(((single - g_dense).abs().max() / g_dense.abs().max()).item(), 1e-9)


class GaugeEquivarianceTest(unittest.TestCase):
    """f(U^g) = f(U)^g and log J(U^g) = log J(U) for every layer kind (generic parameters), with
    U^g_mu(x) = g(x) U_mu(x) g(x + mu)^dagger (`lattice.gauge_transform_links`)."""

    CASES = [("smearing", None), ("smearing_signed", None),
             ("residual", dict(n_pt=0)), ("residual", dict(n_pt=1)), ("residual", dict(n_pt=2)),
             ("residual", dict(n_pt=1, mask_mod=4)),
             ("hyper-residual", dict(n_pt=0)), ("hyper-residual", dict(n_pt=2))]

    def test_equivariance(self):
        for g in (G2, Geometry(4, 4, 2, 3)):
            U = random_links(g, 2, 61)
            omega = expi(random_algebra((2,) + g.lattice + (g.N, g.N), torch.Generator().manual_seed(62)))
            Ug = gauge_transform_links(U, omega)
            for kind, options in self.CASES:
                if options is not None and options.get("mask_mod") == 4 and g.L % 4:
                    continue
                with self.subTest(D=g.D, kind=kind, options=options):
                    fp = flow_pars(g, [0.0], 2)
                    rho = -0.04 if kind == "smearing_signed" else 0.04
                    layer = make_gauge_layers(fp, kind, rho=rho, residual=options)[0]
                    perturb(layer, 0.02 if options is None else 0.1, 63)
                    with torch.no_grad():
                        Up, _, lj = layer(U, fp, None, beta=0.7, delta_beta=0.5)
                        Upg, _, ljg = layer(Ug, fp, None, beta=0.7, delta_beta=0.5)
                    self.assertGreater((Up - U).abs().max().item(), 1e-3)  # not the identity
                    self.assertLess((Upg - gauge_transform_links(Up, omega)).abs().max().item(), 1e-12)
                    lj, ljg = torch.as_tensor(lj).reshape(-1), torch.as_tensor(ljg).reshape(-1)
                    self.assertGreater(lj.abs().max().item(), 1e-2)
                    self.assertLess((ljg - lj).abs().max().item(), 1e-10 * max(1.0, lj.abs().max().item()))


class SignedJarzynskiTest(unittest.TestCase):
    """kappa 0.18 -> 0.21 in 4 steps with signed stout layers at rho = -0.008 (a map the |rho|
    layer cannot express): <exp(-W)> = Z(0.21)/Z(0.18) from the exact determinant ratio, within
    3 sigma; log J is clearly nonzero (the layers are not the identity)."""

    def test_jarzynski(self):
        Ensembles.build()
        r, e_r = chain_mean_error(Ensembles.ref_det_ratio)
        dF_ref, e_ref = -math.log(r), e_r / r
        protocol = linear_protocol(KAPPA, KAPPA2, 4)
        gen = torch.Generator().manual_seed(4004)
        Ws, logJs = [], []
        for U in Ensembles.starts:
            act = make_action("kappa")
            act.set_parameter(KAPPA)
            fp = flow_pars(G2, protocol, U.shape[0])
            snf = FermionSNF(act, protocol, make_gauge_layers(fp, "smearing_signed", rho=-0.008), fp, **HMC_KW)
            with torch.no_grad():
                _, W, _, logJ, _ = snf(FieldState(U), gen)
            Ws.append(W)
            logJs.append(logJ)
        W, logJ = torch.cat(Ws), torch.cat(logJs)
        dF, e = jarzynski_free_energy(W)
        msg = (f"signed stout rho = -0.008: dF = {dF:.4f}({e:.4f}), exact {dF_ref:.4f}({e_ref:.4f}), pull "
               f"{(dF - dF_ref) / math.hypot(e, e_ref):+.2f}, ESS {effective_sample_size(W):.3f}, <logJ> "
               f"{logJ.mean().item():+.3f}")
        print("\n" + msg)
        self.assertLess(abs(dF - dF_ref), 3 * math.hypot(e, e_ref), msg=msg)
        self.assertGreater(logJ.mean().item(), 1.0, msg=msg)  # rho < 0: the layers expand (log J > 0)


if __name__ == "__main__":
    unittest.main()
