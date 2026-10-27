"""The term-by-term path-gradient loss equals the forward-derivative one it replaces."""

import unittest

import torch

import neoqcd.cnf as cnf
import neoqcd.sun_utils as sun
from neoqcd.cnf import CloverChairLinkDrift, path_gradient_loss
from neoqcd.cnf_io import build_cnf
from test.test_cnf_adapters import _defect_config, _flow_pars
from neoqcd.cnf_adapters import make_existing_drift


def _reference_loss(model, action_fn, U, t):
    """The previous implementation: forward derivative of xi along F_theta."""
    V = cnf._volume(U, None)
    t = cnf._as_time(t, U)
    velocity = cnf.lie_tangent(U, model.drift(U, t))
    frozen = cnf._frozen_model(model)
    residual = lambda value: cnf.path_gradient_residual(frozen, action_fn, value, t, V)[0]
    return torch.func.jvp(residual, (U,), (velocity,))


def _action(beta_scale, D):
    def action(U, t):
        return -beta_scale * t * sum(p.flatten(1).sum(1).real for p in sun.plaquette_fields(U, D))
    return action


def _perturb(module, scale=0.05):
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(scale * torch.randn_like(parameter))


def _fields():
    torch.manual_seed(21)
    clover = CloverChairLinkDrift(2, hidden_dim=4, depth=1, loop_max=1).to(dtype=torch.double)
    yield "clover-chair", clover, sun.SUN_haar((2, 2, 4, 4, 3), dtype=torch.cdouble), 2
    smearing = make_existing_drift("hyper-smearing", _flow_pars("hyper-smearing"))
    yield "hyper-smearing", smearing, sun.SUN_haar((2, 2, 4, 4, 3), dtype=torch.cdouble), 2
    pars = _flow_pars("hyper-residual", n_pt=1)
    pars.residual_mask_mod = 4
    yield "hyper-residual", make_existing_drift("hyper-residual", pars), sun.SUN_haar((2, 2, 4, 4, 3), dtype=torch.cdouble), 2
    for architecture in ("defect-hyper-smearing", "defect-hyper-residual"):
        field, _, _ = build_cnf(_defect_config(architecture=architecture, batch_size=2))
        yield architecture, field, sun.SUN_haar((2, 4, 4, 6, 6, 6, 3), dtype=torch.cdouble), 4
    field, _, _ = build_cnf(_defect_config(architecture="defect-hyper-residual", residual_mode="class", batch_size=2))
    yield "defect-hyper-residual class", field, sun.SUN_haar((2, 4, 4, 6, 6, 6, 3), dtype=torch.cdouble), 4


class TermByTermPathGradientTests(unittest.TestCase):
    def test_matches_the_forward_derivative_loss(self):
        t = torch.tensor([0.3, 0.7], dtype=torch.double)
        for name, field, U, D in _fields():
            with self.subTest(field=name):
                _perturb(field)
                action = _action(0.8, D)

                xi_ref, loss_ref = _reference_loss(field, action, U, t)
                field.zero_grad()
                loss_ref.sum().backward()
                grads_ref = [p.grad.clone() for p in field.parameters()]

                for chunk_size, directions in ((None, 1), (1, 1), (None, 8), (1, 2)):
                    field.zero_grad()
                    result = path_gradient_loss(field, action, U, t, chunk_size=chunk_size,
                                                directions_per_term=directions)
                    result["loss"].sum().backward()
                    torch.testing.assert_close(result["xi"], xi_ref.detach(), rtol=1e-10, atol=1e-12)
                    torch.testing.assert_close(result["loss"].detach(), loss_ref.detach(), rtol=1e-9, atol=1e-12)
                    for grad, ref in zip((p.grad for p in field.parameters()), grads_ref):
                        torch.testing.assert_close(grad, ref, rtol=1e-8, atol=1e-12)
                self.assertTrue(any(g.abs().max() > 0 for g in grads_ref))

    def test_terms_sum_to_the_divergence(self):
        t = torch.tensor([0.3, 0.7], dtype=torch.double)
        for name, field, U, _ in _fields():
            with self.subTest(field=name), torch.no_grad():
                _perturb(field)
                _, divergence = field(U, t)
                for directions in (1, 2, 4, 8):
                    terms = field.divergence_terms(t, directions)
                    total = sum(term(U) for term in terms)
                    torch.testing.assert_close(total, divergence, rtol=1e-12, atol=1e-12)
                self.assertEqual(len(field.divergence_terms(t, 8)) * 8, len(field.divergence_terms(t, 1)))


if __name__ == "__main__":
    unittest.main()
