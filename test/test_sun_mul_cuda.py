"""On GPU, the broadcast link product gives the PG loss and gradient of the addcmul one it replaced."""

import functools
import unittest
from unittest import mock

import torch

import neoqcd.sun_utils as sun
from neoqcd.cnf import path_gradient_loss
from neoqcd.cnf_io import build_cnf
from neoqcd.cnf_training import ladder_action


def _addcmul_mul(U, V):
    W = U[..., :, 0, None] * V[..., None, 0, :]
    for j in range(1, U.shape[-1]):
        W = torch.addcmul(W, U[..., :, j, None], V[..., None, j, :])
    return W


CONFIGS = {
    "hyper-residual": dict(domain="beta", residual_n_pt=1, residual_mask_mod=4),
    "defect-hyper-residual": dict(domain="defect", residual_n_pt=1, residual_mask_mod=4, defect_size=2,
                                  time_slice=4, space_slice=3),
    "defect-clover-chair": dict(domain="defect", residual_n_pt=0, residual_mask_mod=2, defect_size=2,
                                time_slice=4, space_slice=3),
}


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class BroadcastProductTests(unittest.TestCase):
    def test_pg_loss_and_gradient_match_the_addcmul_product(self):
        for architecture, extra in CONFIGS.items():
            with self.subTest(architecture=architecture):
                torch.manual_seed(5)
                config = dict(architecture=architecture, D=4, T=8, L=8, beta=6.0, parameter_min=0.0,
                              parameter_max=1.0, batch_size=2, hidden_dim=8, depth=1, rho_init=1e-2,
                              orsteps=1, **extra)
                field, pars, defect_mask = build_cnf(config, device="cuda")
                field = field.cuda()
                with torch.no_grad():
                    for parameter in field.parameters():
                        parameter.add_(0.05 * torch.randn_like(parameter))
                U = sun.SUN_haar((2, 4, 8, 8, 8, 8, 3), dtype=torch.cdouble).cuda()
                t = torch.tensor([0.3, 0.8], dtype=torch.double, device="cuda")
                action = functools.partial(ladder_action, pars, extra["domain"], 6.0, defect_mask, 8 ** 4)

                def run():
                    field.zero_grad(set_to_none=True)
                    result = path_gradient_loss(field, action, U, t, chunk_size=1, directions_per_term=8)
                    result["loss"].mean().backward()
                    return result["loss"].detach(), [p.grad.clone() for p in field.parameters()]

                loss, grads = run()
                with mock.patch.object(sun, "SUN_mul", _addcmul_mul):
                    loss_ref, grads_ref = run()
                torch.testing.assert_close(loss, loss_ref, rtol=1e-10, atol=1e-14)
                for grad, ref in zip(grads, grads_ref):
                    torch.testing.assert_close(grad, ref, rtol=1e-9, atol=1e-14)
                self.assertTrue(any(float(g.abs().max()) > 0 for g in grads_ref))


if __name__ == "__main__":
    unittest.main()
