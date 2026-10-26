"""Replica exchange training for continuous normalizing flows, without map evaluation."""

import math

import torch

import neoqcd.sun_utils as sun
from neoqcd.mcmc import HBOR
from .cnf import path_gradient_loss


@torch.no_grad()
def rex_swap(configs, parameters, action_fn, *, parity):
    """Attempt adjacent replica swaps for configs shaped (rungs, chains, ...)."""
    n_rungs = configs.shape[0]
    accepted = torch.zeros(n_rungs - 1, device=configs.device, dtype=torch.float64)
    proposed = torch.zeros_like(accepted)
    for i in range(int(parity), n_rungs - 1, 2):
        j = i + 1
        pi, pj = parameters[i], parameters[j]
        ei = action_fn(configs[i], pi)
        ej = action_fn(configs[j], pj)
        eij = action_fn(configs[j], pi)
        eji = action_fn(configs[i], pj)
        log_accept = ei + ej - eij - eji
        take = torch.log(torch.rand_like(log_accept)) < log_accept
        old_i = configs[i].clone()
        configs[i] = torch.where(take.view(-1, *([1] * (configs.dim() - 2))), configs[j], configs[i])
        configs[j] = torch.where(take.view(-1, *([1] * (configs.dim() - 2))), old_i, configs[j])
        accepted[i] = take.sum()
        proposed[i] = take.numel()
    return configs, accepted, proposed


def _batched_plaquette_weight(kappa, mu, nu, D):
    return (
        kappa[:, mu]
        * torch.roll(kappa[:, nu], 1, dims=-D + mu)
        * torch.roll(kappa[:, mu], 1, dims=-D + nu)
        * kappa[:, nu]
    )


class CNFREXTrainer:
    """Train a parameter-conditioned drift on persistent replica-exchange samples."""

    def __init__(
        self,
        field,
        flow_pars,
        parameters,
        optimizer,
        *,
        domain="beta",
        beta=6.0,
        defect_mask=None,
        batch_size=2,
        updates_per_replica=1,
        grad_clip_norm=0.0,
        initial_cfgs=None,
    ):
        self.field = field
        self.flow_pars = flow_pars
        self.optimizer = optimizer
        self.domain = str(domain).lower()
        self.beta = float(beta)
        self.defect_mask = defect_mask
        self.batch_size = int(batch_size)
        self.updates_per_replica = int(updates_per_replica)
        self.grad_clip_norm = float(grad_clip_norm)
        device = torch.device(flow_pars.device)
        self.parameters = torch.as_tensor(parameters, dtype=torch.float64, device=device).reshape(-1)
        if self.domain not in {"beta", "defect"}:
            raise ValueError("domain must be 'beta' or 'defect'")
        if int(flow_pars.N) != 3:
            raise ValueError("CNFREXTrainer supports SU(3) configurations")
        if self.parameters.numel() < 2:
            raise ValueError("replica exchange requires at least two parameter values")
        if not bool(torch.isfinite(self.parameters).all()):
            raise ValueError("replica parameters must be finite")
        differences = torch.diff(self.parameters)
        if not (bool((differences > 0).all()) or bool((differences < 0).all())):
            raise ValueError("replica parameters must be strictly monotonic")
        if self.batch_size < 2:
            raise ValueError("batch_size must be >= 2 to measure PINN within each replica")
        if self.updates_per_replica < 1:
            raise ValueError("updates_per_replica must be positive")
        if self.domain == "defect" and defect_mask is None:
            raise ValueError("defect_mask is required for domain='defect'")
        if self.domain == "beta" and bool((self.parameters < 0).any()):
            raise ValueError("beta replica parameters must be non-negative")
        if self.domain == "defect" and bool(((self.parameters < 0) | (self.parameters > 1)).any()):
            raise ValueError("defect replica parameters must lie in [0, 1]")
        if not math.isfinite(self.beta) or self.beta < 0:
            raise ValueError("beta must be finite and non-negative")

        self.volume = int(flow_pars.T) * int(flow_pars.L) ** (int(flow_pars.D) - 1)
        self.parity = 0
        self.n_updates = 0
        shape = (
            self.parameters.numel(),
            self.batch_size,
            int(flow_pars.D),
            int(flow_pars.T),
            *([int(flow_pars.L)] * (int(flow_pars.D) - 1)),
            int(flow_pars.N),
            int(flow_pars.N),
        )
        if initial_cfgs is None:
            self.x = sun.SUN_haar(shape[:-1], device=device, dtype=torch.cdouble)
        else:
            if tuple(initial_cfgs.shape) != shape:
                raise ValueError(f"initial_cfgs must have shape {shape}, got {tuple(initial_cfgs.shape)}")
            self.x = initial_cfgs.detach().clone().to(device=device, dtype=torch.cdouble)

        self.updates = []
        for p in self.parameters:
            if self.domain == "beta":
                update = HBOR(flow_pars, beta=float(p.item()), defect_par=1.0)
            else:
                update = HBOR(flow_pars, beta=self.beta, defect_par=float(p.item()))
            self.updates.append(update)

    def action(self, configs, parameter):
        """Wilson action along beta or defect parameter, differentiable in parameter."""
        p = torch.as_tensor(parameter, dtype=configs.real.dtype, device=configs.device).reshape(-1)
        if p.numel() == 1:
            p = p.expand(configs.shape[0])
        elif p.numel() != configs.shape[0]:
            raise ValueError(f"parameter must be scalar or have shape ({configs.shape[0]},)")
        plaquettes = sun.plaquette_fields(configs, int(self.flow_pars.D))
        D, N = int(self.flow_pars.D), int(self.flow_pars.N)
        result = configs.real.new_zeros(configs.shape[0])
        if self.domain == "beta":
            retr = sum(plaq.real for plaq in plaquettes)
            constant = D * (D - 1) // 2
            action0 = constant * self.volume - retr.sum(dim=tuple(range(1, D + 1))) / N
            result = p * action0
        else:
            defect = self.defect_mask.to(device=configs.device, dtype=configs.real.dtype)
            p_view = p.view((configs.shape[0],) + (1,) * (D + 1))
            kappa = defect[0].unsqueeze(0) + defect[1].unsqueeze(0) * p_view
            weighted = []
            for plaq, (mu, nu) in zip(
                plaquettes,
                ((mu, nu) for nu in range(1, D) for mu in range(nu)),
            ):
                weighted.append(_batched_plaquette_weight(kappa, mu, nu, D) * plaq.real)
            retr = torch.stack(weighted).sum(dim=0)
            constant = D * (D - 1) // 2
            result = self.beta * (
                constant * self.volume - retr.sum(dim=tuple(range(1, D + 1))) / N
            )
        return result

    def _sweep(self):
        with torch.no_grad():
            # At zero gauge coupling the first rung is exactly Haar and can be refreshed.
            zero_base = (
                self.domain == "beta" and float(self.parameters[0]) == 0.0
            ) or (self.domain == "defect" and self.beta == 0.0)
            if zero_base:
                self.x[0] = sun.SUN_haar(
                    self.x[0].shape[:-1], device=self.x.device, dtype=self.x.dtype
                )
            mask = self.flow_pars.mask.to(device=self.x.device)
            defect = None if self.defect_mask is None else self.defect_mask.to(device=self.x.device)
            for i, update in enumerate(self.updates):
                for _ in range(self.updates_per_replica):
                    self.x[i] = update(self.x[i], mask, defect)
            self.x, accepted, proposed = rex_swap(
                self.x, self.parameters, self.action, parity=self.parity
            )
            self.parity = 1 - self.parity
        return accepted, proposed

    def warmup(self, sweeps=1):
        sweeps = int(sweeps)
        if sweeps < 0:
            raise ValueError("warmup sweeps must be non-negative")
        for _ in range(sweeps):
            self._sweep()

    def step(self):
        accepted, proposed = self._sweep()

        self.optimizer.zero_grad(set_to_none=True)
        losses, pinns = [], []
        residuals = []
        finite_loss = True
        for i, parameter in enumerate(self.parameters):
            result = path_gradient_loss(
                self.field,
                self.action,
                self.x[i],
                parameter,
                volume=self.volume,
            )
            loss = result["loss"]
            xi = result["xi"]
            finite_loss = finite_loss and bool(torch.isfinite(loss).all()) and bool(torch.isfinite(xi).all())
            (loss.mean() / self.parameters.numel()).backward()
            losses.append(loss.detach().mean())
            residuals.append(xi.detach())
            pinn = self.volume * xi.var(unbiased=False)
            pinns.append(pinn.detach())

        parameters = [p for p in self.field.parameters() if p.requires_grad]
        grads = [p.grad for p in parameters if p.grad is not None]
        finite = finite_loss and bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads)
        grad_norm = float("nan")
        updated = False
        if finite:
            if self.grad_clip_norm > 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, self.grad_clip_norm).item())
            else:
                grad_norm = float(torch.linalg.vector_norm(torch.stack([g.detach().norm() for g in grads])).item())
            finite = math.isfinite(grad_norm) and all(bool(torch.isfinite(g).all()) for g in grads)
        if finite:
            self.optimizer.step()
            updated = True

        self.n_updates += 1
        swap_rates = accepted / proposed.clamp_min(1.0)
        return {
            "losses_by_rung": [float(v.cpu()) for v in losses],
            "pinn_by_rung": [float(v.cpu()) for v in pinns],
            "residuals_by_rung": [float(v.mean().cpu()) for v in residuals],
            "swap_acceptance_by_edge": [float(v) for v in swap_rates],
            "swap_acceptance": float(accepted.sum() / proposed.sum().clamp_min(1.0)),
            "grad_norm": grad_norm,
            "optimizer_updated": updated,
            "updates": self.n_updates,
        }

    def state_dict(self):
        state = {
            "parameters": self.parameters.detach().cpu(),
            "configs": self.x.detach().cpu(),
            "parity": self.parity,
            "n_updates": self.n_updates,
            "optimizer": self.optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        return state

    def load_state_dict(self, state):
        if not torch.equal(state["parameters"].cpu(), self.parameters.detach().cpu()):
            raise ValueError("checkpoint replica ladder does not match")
        if tuple(state["configs"].shape) != tuple(self.x.shape):
            raise ValueError("checkpoint replica-chain shape does not match")
        self.x.copy_(state["configs"].to(device=self.x.device, dtype=self.x.dtype))
        self.parity = int(state["parity"])
        if self.parity not in {0, 1}:
            raise ValueError("checkpoint has invalid replica-swap parity")
        self.n_updates = int(state["n_updates"])
        self.optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if torch.cuda.is_available() and "cuda_rng_state_all" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
