import unittest

import torch

import neoqcd.sun_utils as sun
from neoqcd.flow import FlowPars
from neoqcd.smearing import ResidualNormalizingFlows, generate_coefficients, generate_expQ, otimes
from neoqcd.utils import create_mask


def random_su3(shape, dtype=torch.cdouble, device=None):
    z = torch.randn(shape + (3, 3), dtype=dtype, device=device)
    q, r = torch.linalg.qr(z)
    diag = torch.diagonal(r, dim1=-2, dim2=-1)
    phase = diag / torch.clamp(diag.abs(), min=1e-14)
    q = q * phase.conj().unsqueeze(-2)
    det = torch.linalg.det(q)
    q = q / det.pow(1.0 / 3.0).unsqueeze(-1).unsqueeze(-1)
    return q


def exact_su3_exp_iQ(Q):
    eye = sun.SUN_identity(Q.shape[:-1], dtype=Q.dtype, device=Q.device)
    Q2 = sun.SUN_mul(Q, Q)
    oidid = otimes(eye, eye)
    oidQ = otimes(eye, Q) + otimes(Q, eye)
    f0, f1, f2, _, _, _ = generate_coefficients(
        Q,
        Q2,
        eye,
        oidid,
        oidQ,
        device=Q.device,
        backward=False,
    )
    return generate_expQ(Q, Q2, eye, f0, f1, f2)


def max_su3_error(U):
    eye = torch.eye(3, dtype=U.dtype, device=U.device)
    unit = sun.SUN_mul(sun.SUN_dagger(U), U)
    unit_err = (unit - eye).abs().max()
    det_err = (torch.linalg.det(U) - 1.0).abs().max()
    return max(unit_err.item(), det_err.item())


def build_layer(batch_size=1, coeff_init=1e-3, quadratic=True, include_imag=True):
    torch.manual_seed(1234)
    D, T, L = 4, 4, 4
    mask = create_mask(D, T, L)
    flow_pars = FlowPars(
        D=D,
        T=T,
        L=L,
        N=3,
        mask=mask,
        protocol=[6.0],
        batch_size=batch_size,
        device=torch.device("cpu"),
        smearing_steps_per_layer=1,
        hyper_time_embedding_dim=8,
        hyper_hidden_dim=12,
        hyper_depth=1,
        residual_include_imag=include_imag,
        residual_quadratic=quadratic,
        residual_coeff_init=coeff_init,
    )
    return ResidualNormalizingFlows(flow_pars), flow_pars


class ResidualNormalizingFlowsTest(unittest.TestCase):
    def test_residual_has_no_complex_buffers_for_nccl_ddp(self):
        layer, _ = build_layer(batch_size=1)
        complex_buffers = [name for name, buf in layer.named_buffers() if torch.is_complex(buf)]

        self.assertEqual(complex_buffers, [])

    def test_closed_form_exponential_lands_in_su3(self):
        layer, _ = build_layer(batch_size=1)
        gens = layer.generators
        coeffs = torch.tensor([0.011, -0.007, 0.005, 0.013, -0.003, 0.009, -0.004, 0.006])
        Q = torch.einsum("a,aij->ij", coeffs.to(dtype=gens.dtype), gens)

        expQ = exact_su3_exp_iQ(Q)

        self.assertLess(max_su3_error(expQ), 1e-12)

    def test_residual_forward_preserves_su3_links(self):
        layer, flow_pars = build_layer(batch_size=1, coeff_init=2e-3)
        x = random_su3(flow_pars.init_shape[:-1])

        y, _, logdet = layer(
            x,
            flow_pars.mask,
            beta=torch.tensor([6.0]),
            delta_beta=torch.tensor([0.02]),
        )

        self.assertTrue(torch.isfinite(logdet).all().item())
        self.assertLess(max_su3_error(y), 5e-12)

    def test_delta_zero_is_identity_with_zero_logdet(self):
        layer, flow_pars = build_layer(batch_size=2)
        x = random_su3(flow_pars.init_shape[:-1])

        y, _, logdet = layer(
            x,
            flow_pars.mask,
            beta=torch.tensor([5.9, 6.1]),
            delta_beta=torch.tensor([0.0, 0.0]),
        )

        self.assertLess((y - x).abs().max().item(), 1e-12)
        self.assertLess(logdet.abs().max().item(), 1e-10)

    def test_delta_beta_sign_and_magnitude_are_sensible(self):
        layer, flow_pars = build_layer(batch_size=1, coeff_init=5e-3)
        x = random_su3(flow_pars.init_shape[:-1])

        y_small, _, _ = layer(x, flow_pars.mask, beta=torch.tensor([6.0]), delta_beta=torch.tensor([0.01]))
        y_large, _, _ = layer(x, flow_pars.mask, beta=torch.tensor([6.0]), delta_beta=torch.tensor([0.02]))
        y_neg, _, _ = layer(x, flow_pars.mask, beta=torch.tensor([6.0]), delta_beta=torch.tensor([-0.01]))

        d_small = torch.linalg.vector_norm(y_small - x)
        d_large = torch.linalg.vector_norm(y_large - x)
        sign_dot = torch.sum((y_small - x).conj() * (y_neg - x)).real

        self.assertGreater(d_large.item(), d_small.item())
        self.assertLess(sign_dot.item(), 0.0)

    def test_beta_conditioning_can_change_the_map(self):
        layer, flow_pars = build_layer(batch_size=1, coeff_init=1e-3)
        with torch.no_grad():
            layer.mlp[-1].weight[:, 0].fill_(1e-4)

        x = random_su3(flow_pars.init_shape[:-1])
        y_lo, _, _ = layer(x, flow_pars.mask, beta=torch.tensor([5.7]), delta_beta=torch.tensor([0.02]))
        y_hi, _, _ = layer(x, flow_pars.mask, beta=torch.tensor([6.3]), delta_beta=torch.tensor([0.02]))

        self.assertGreater((y_lo - y_hi).abs().max().item(), 1e-8)

    def test_delta_beta_magnitude_does_not_condition_hypernetwork(self):
        layer, _ = build_layer(batch_size=1, coeff_init=1e-3)
        lin_small, quad_small, delta_small = layer._coefficients(
            beta=torch.tensor([6.0]),
            delta_beta=torch.tensor([0.01]),
            batch_size=1,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        lin_large, quad_large, delta_large = layer._coefficients(
            beta=torch.tensor([6.0]),
            delta_beta=torch.tensor([0.02]),
            batch_size=1,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )

        self.assertTrue(torch.allclose(lin_small, lin_large))
        self.assertTrue(torch.allclose(quad_small, quad_large))
        self.assertTrue(torch.allclose(delta_small, torch.tensor([0.01], dtype=torch.float64)))
        self.assertTrue(torch.allclose(delta_large, torch.tensor([0.02], dtype=torch.float64)))

    def test_global_gauge_equivariance_and_logdet_invariance(self):
        layer, flow_pars = build_layer(batch_size=1, coeff_init=1e-3)
        x = random_su3(flow_pars.init_shape[:-1])
        g = random_su3(())
        gdag = sun.SUN_dagger(g)
        x_g = sun.SUN_mul(g, sun.SUN_mul(x, gdag))

        y, _, logdet = layer(x, flow_pars.mask, beta=torch.tensor([6.0]), delta_beta=torch.tensor([0.02]))
        y_g, _, logdet_g = layer(x_g, flow_pars.mask, beta=torch.tensor([6.0]), delta_beta=torch.tensor([0.02]))
        y_expected = sun.SUN_mul(g, sun.SUN_mul(y, gdag))

        self.assertLess((y_g - y_expected).abs().max().item(), 1e-10)
        self.assertLess((logdet_g - logdet).abs().max().item(), 1e-10)

    def test_local_analytic_jacobian_matches_finite_difference(self):
        layer, flow_pars = build_layer(batch_size=1, coeff_init=2e-3)
        x = random_su3(flow_pars.init_shape[:-1])
        mu = 0
        site = (0, 0, 0, 0)
        eps = 1e-5

        y, _, jac = layer.residual_update_mu(
            x,
            mu,
            beta=torch.tensor([6.0]),
            delta_beta=torch.tensor([0.02]),
        )
        analytic = jac[(0,) + site]
        gens = layer.generators.to(dtype=x.dtype)
        finite_cols = []

        for b in range(8):
            eplus = exact_su3_exp_iQ(eps * gens[b])
            eminus = exact_su3_exp_iQ(-eps * gens[b])
            x_plus = x.clone()
            x_minus = x.clone()
            x_plus[(0, mu) + site] = sun.SUN_mul(eplus, x_plus[(0, mu) + site])
            x_minus[(0, mu) + site] = sun.SUN_mul(eminus, x_minus[(0, mu) + site])

            y_plus, _, _ = layer.residual_update_mu(
                x_plus,
                mu,
                beta=torch.tensor([6.0]),
                delta_beta=torch.tensor([0.02]),
            )
            y_minus, _, _ = layer.residual_update_mu(
                x_minus,
                mu,
                beta=torch.tensor([6.0]),
                delta_beta=torch.tensor([0.02]),
            )
            delta_y = (y_plus[(0,) + site] - y_minus[(0,) + site]) / (2.0 * eps)
            finite_cols.append(layer._tangent_coordinates(delta_y, y[(0,) + site]))

        finite = torch.stack(finite_cols, dim=-1)
        self.assertLess((analytic - finite).abs().max().item(), 5e-5)


if __name__ == "__main__":
    unittest.main()
