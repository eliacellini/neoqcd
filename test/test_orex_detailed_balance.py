"""Exact detailed balance of the O-REX and NEO-REX swap kernels on a three-state toy.

Every stochastic update is enumerated over its outcomes, so the swap kernel is
built exactly from the production protocol loops and checked against the joint
equilibrium distribution of the two replicas.
"""

import itertools
import math
import unittest

import torch
from torch import nn

from neoqcd.distributedPT import (
    NEOREX,
    NEOREXProtocolWrapper,
    OREXPTConfig,
    OutOfEquilibriumParallelTempering,
)


ENERGY = torch.tensor([0.0, 0.7, 1.9], dtype=torch.float64)
STATES = range(len(ENERGY))
LAMBDA_A, LAMBDA_B = 0.4, 1.7


def _action(x, parameter):
    return parameter * ENERGY[x.long()]


def _boltzmann(parameter):
    weights = torch.exp(-parameter * ENERGY)
    return weights / weights.sum()


class _ScriptedMetropolis:
    """Random-walk Metropolis on Z_3, whose random choices are read from a script.

    Each update consumes one (direction, accepted) outcome and multiplies `weight`
    by its probability, so enumerating scripts enumerates the kernel exactly.
    """

    def __init__(self, script=()):
        self.script = list(script)
        self.weight = 1.0
        self.calls = 0
        self.parameter = None

    def set_parameter(self, parameter, parameter_name=None):
        self.parameter = parameter

    def action(self, x, parameter):
        return _action(x, parameter)

    def __call__(self, x, nsteps=1, **_):
        for _ in range(nsteps):
            self.calls += 1
            direction, accepted = self.script.pop(0) if self.script else (1, False)
            proposal = (x + direction) % len(ENERGY)
            delta = _action(proposal, self.parameter) - _action(x, self.parameter)
            p_accept = float(torch.clamp(torch.exp(-delta), max=1.0))
            self.weight *= 0.5 * (p_accept if accepted else 1.0 - p_accept)
            if accepted:
                x = proposal
        return x


class _Shift(nn.Module):
    """Volume-preserving bijection whose shift depends on the parameter, and its inverse.

    Called with -delta it is not its own inverse, like a real flow layer: the swap is balanced
    only if chains going down apply `inverse`.
    """

    @staticmethod
    def _shift(bc):
        return torch.where(bc < 0.5 * (LAMBDA_A + LAMBDA_B), 1.0, 2.0)

    def forward(self, x, bc, delta_bc):
        return (x + self._shift(bc)) % len(ENERGY), torch.zeros_like(x)

    def inverse(self, x, bc, delta_bc):
        return (x - self._shift(bc)) % len(ENERGY), torch.zeros_like(x)


class _Local(nn.Module):
    """Stands in for DDP on a single process."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _outcomes(run, x0, start, target):
    """All (final state, work, probability) of one protocol run from x0."""
    probe = _ScriptedMetropolis()
    run(probe, x0, start, target)
    results = []
    for script in itertools.product(itertools.product((1, -1), (True, False)), repeat=probe.calls):
        mcmc = _ScriptedMetropolis(script)
        x, work = run(mcmc, x0, start, target)
        if mcmc.weight > 0:
            results.append((int(x.item()), float(work.item()), mcmc.weight))
    return results


def _swap_kernel(run):
    """Transition matrix on (state at LAMBDA_A, state at LAMBDA_B)."""
    pairs = list(itertools.product(STATES, STATES))
    index = {pair: i for i, pair in enumerate(pairs)}
    kernel = torch.zeros(len(pairs), len(pairs), dtype=torch.float64)
    for x, y in pairs:
        forward = _outcomes(run, x, LAMBDA_A, LAMBDA_B)
        reverse = _outcomes(run, y, LAMBDA_B, LAMBDA_A)
        for (x_new, work_x, p_x), (y_new, work_y, p_y) in itertools.product(forward, reverse):
            accept = min(1.0, math.exp(-(work_x + work_y)))
            kernel[index[(x, y)], index[(y_new, x_new)]] += p_x * p_y * accept
            kernel[index[(x, y)], index[(x, y)]] += p_x * p_y * (1.0 - accept)
    joint = torch.outer(_boltzmann(LAMBDA_A), _boltzmann(LAMBDA_B)).flatten()
    return kernel, joint


def _tensor(value):
    return torch.tensor([float(value)], dtype=torch.float64)


def _wrapper(mcmc, steps):
    return NEOREXProtocolWrapper(
        _Shift(), mcmc, steps, stochastic_update=mcmc, stochastic_steps_per_step=1, parameter_name="bc"
    )


def _neorex(mcmc, steps):
    pt = object.__new__(NEOREX)
    pt.ddp_flow = _Local(_wrapper(mcmc, steps))
    pt.B, pt.rank, pt.world_size = 1, 0, 1
    pt.device, pt.param_dtype = torch.device("cpu"), torch.float64
    return pt


class SwapDetailedBalanceTests(unittest.TestCase):
    def assertDetailedBalance(self, run):
        kernel, joint = _swap_kernel(run)
        torch.testing.assert_close(kernel.sum(dim=1), torch.ones_like(joint))
        flux = joint[:, None] * kernel
        torch.testing.assert_close(flux, flux.T, atol=1e-12, rtol=0)

    def test_orex_protocol(self):
        for steps in (1, 3):
            with self.subTest(steps=steps):
                pt = object.__new__(OutOfEquilibriumParallelTempering)
                pt.config = OREXPTConfig(
                    batch_size=1, parameter_min=LAMBDA_A, parameter_max=LAMBDA_B,
                    parameter_name="bc", orex_protocol_steps=steps,
                )
                pt.B, pt.device, pt.param_dtype, pt.parameter_name = 1, torch.device("cpu"), torch.float64, "bc"

                def run(mcmc, x0, start, target):
                    return pt._run_orex_protocol(
                        _tensor(x0), mcmc, _tensor(start), _tensor(target), torch.tensor([True])
                    )

                self.assertDetailedBalance(run)

    def test_neorex_wrapper(self):
        for steps in (1, 2):
            with self.subTest(steps=steps):
                def run(mcmc, x0, start, target):
                    return _wrapper(mcmc, steps)(_tensor(x0), _tensor(start), _tensor(target))

                self.assertDetailedBalance(run)

    def test_neorex_distributed_protocol_loops(self):
        for name in ("_eval_flow_local_protocol", "_train_flow_local_protocol"):
            with self.subTest(loop=name):
                def run(mcmc, x0, start, target):
                    out = getattr(_neorex(mcmc, 2), name)(
                        _tensor(x0), _tensor(start), _tensor(target), torch.tensor([True])
                    )
                    return out[0], out[1]

                self.assertDetailedBalance(run)


if __name__ == "__main__":
    unittest.main()
