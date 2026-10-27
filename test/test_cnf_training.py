import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from neoqcd import sun_utils as sun
from neoqcd.cnf import CloverChairLinkDrift, path_gradient_residual
from neoqcd.cnf_training import CNFREXTrainer, rex_swap


def checkerboard_mask(D=2, T=2, L=2):
    mask = torch.zeros((2 * D, 1, D, T, *([L] * (D - 1)), 1, 1), dtype=torch.int64)
    for mu in range(D):
        for t in range(T):
            for x in range(L):
                mask[(2 * mu + (t + x) % 2, 0, mu, t, x, 0, 0)] = 1
    return mask


def flow_pars():
    return SimpleNamespace(D=2, T=2, L=2, N=3, device=torch.device("cpu"),
                           mask=checkerboard_mask(), orsteps=0)


def defect_mask():
    defect = torch.zeros((2, 2, 2), dtype=torch.float64)
    defect[1, 0, 0] = 1.0
    return torch.stack((1.0 - defect, defect))


class CNFREXTrainerTest(unittest.TestCase):
    def test_swap_accepts_lower_cross_action_with_correct_sign(self):
        configs = torch.tensor([[[0.0]], [[1.0]]])
        parameters = torch.tensor([0.0, 2.0])
        action = lambda x, p: p * x[:, 0]
        swapped, accepted, proposed = rex_swap(configs, parameters, action, parity=0)
        self.assertEqual(accepted.tolist(), [1.0])
        self.assertEqual(proposed.tolist(), [1.0])
        self.assertEqual(swapped[:, 0, 0].tolist(), [1.0, 0.0])

    def test_action_parameter_derivative_is_preserved_at_defect_endpoint(self):
        pars = flow_pars()
        field = CloverChairLinkDrift(2, hidden_dim=4, depth=1).to(dtype=torch.float64)
        trainer = CNFREXTrainer(
            field, pars, [0.2, 1.0], torch.optim.Adam(field.parameters()),
            domain="defect", beta=2.0, defect_mask=defect_mask(), batch_size=2,
        )
        # The action implementation uses the defect polynomial at BC=1, where its
        # derivative must remain available to the continuity residual.
        U = trainer.x[0]
        p = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        derivative = torch.autograd.grad(trainer.action(U, p).sum(), p)[0]
        self.assertTrue(torch.isfinite(derivative))
        self.assertNotEqual(derivative.item(), 0.0)
        _, jvp_derivative = torch.func.jvp(
            lambda value: trainer.action(U, value).sum(),
            (p.detach(),),
            (torch.ones_like(p),),
        )
        torch.testing.assert_close(jvp_derivative, derivative.detach())
        xi, dSdt = path_gradient_residual(field, trainer.action, U, p, volume=4)
        self.assertTrue(torch.isfinite(xi).all())
        self.assertTrue(torch.isfinite(dSdt).all())

    def test_defect_action_broadcasts_a_parameter_per_chain(self):
        D, T, L, B = 4, 4, 4, 2
        defect = torch.zeros((D, T, L, L, L), dtype=torch.float64)
        defect[0, 1, 1, 1, 1] = 1.0
        mask = torch.stack((1.0 - defect, defect))
        pars = SimpleNamespace(
            D=D, T=T, L=L, N=3, device=torch.device("cpu"),
            mask=torch.zeros((2 * D, 1, D, T, L, L, L, 1, 1), dtype=torch.int64),
            orsteps=0,
        )
        field = CloverChairLinkDrift(D, hidden_dim=4, depth=1).to(dtype=torch.float64)
        trainer = CNFREXTrainer(
            field, pars, [0.2, 1.0], torch.optim.SGD(field.parameters(), lr=0.1),
            domain="defect", beta=2.0, defect_mask=mask, batch_size=B,
        )
        parameter = torch.tensor([0.35, 0.9], dtype=torch.float64)
        batched = trainer.action(trainer.x[0], parameter)
        separate = torch.stack([
            trainer.action(trainer.x[0, i:i + 1], parameter[i])[0] for i in range(B)
        ])
        torch.testing.assert_close(batched, separate)

    def test_training_step_uses_path_gradient_and_records_within_rung_pinn(self):
        pars = flow_pars()
        field = CloverChairLinkDrift(2, hidden_dim=4, depth=1).to(dtype=torch.float64)
        optimizer = torch.optim.SGD(field.parameters(), lr=0.1)
        trainer = CNFREXTrainer(field, pars, [0.2, 0.8], optimizer, batch_size=2)
        zeros = torch.zeros(1, dtype=torch.float64)

        def loss(model, action_fn, configs, parameter, volume=None, chunk_size=None, directions_per_term=1):
            coeff = model.output_layer.weight[0, 0]
            return {
                "loss": coeff.expand(configs.shape[0]),
                "xi": torch.arange(configs.shape[0], dtype=torch.float64),
                "pinn": torch.tensor(0.0),
                "F_norm": torch.ones(configs.shape[0]),
            }

        with mock.patch.object(trainer, "_sweep", return_value=(zeros, zeros)), \
                mock.patch("neoqcd.cnf_training.path_gradient_loss", side_effect=loss) as pg:
            result = trainer.step()
        self.assertEqual(pg.call_count, 2)
        self.assertEqual(len(result["pinn_by_rung"]), 2)
        self.assertEqual(result["pinn_by_rung"], [1.0, 1.0])
        self.assertTrue(result["optimizer_updated"])


if __name__ == "__main__":
    unittest.main()
