"""Replica exchange training for continuous normalizing flows, without map evaluation."""

import math
import time

import torch
import torch.distributed as dist

import neoqcd.sun_utils as sun
from neoqcd.distributedPT import ParallelTempering, PTConfig
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


def ladder_action(flow_pars, domain, beta, defect_mask, volume, configs, parameter):
    """Wilson action of configs at a beta (domain 'beta') or boundary coupling ('defect')."""
    p = torch.as_tensor(parameter, dtype=configs.real.dtype, device=configs.device).reshape(-1)
    if p.numel() == 1:
        p = p.expand(configs.shape[0])
    elif p.numel() != configs.shape[0]:
        raise ValueError(f"parameter must be scalar or have shape ({configs.shape[0]},)")
    plaquettes = sun.plaquette_fields(configs, int(flow_pars.D))
    D, N = int(flow_pars.D), int(flow_pars.N)
    result = configs.real.new_zeros(configs.shape[0])
    if domain == "beta":
        retr = sum(plaq.real for plaq in plaquettes)
        constant = D * (D - 1) // 2
        action0 = constant * volume - retr.sum(dim=tuple(range(1, D + 1))) / N
        result = p * action0
    else:
        defect = defect_mask.to(device=configs.device, dtype=configs.real.dtype)
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
        result = beta * (
            constant * volume - retr.sum(dim=tuple(range(1, D + 1))) / N
        )
    return result


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
        chunk_size=None,
        directions_per_term=1,
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
        # Chains per path-gradient chunk (None: all chains of a rung at once); memory only.
        self.chunk_size = None if chunk_size is None else int(chunk_size)
        self.directions_per_term = int(directions_per_term)
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
        return ladder_action(self.flow_pars, self.domain, self.beta, self.defect_mask, self.volume, configs, parameter)

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
                chunk_size=self.chunk_size,
                directions_per_term=self.directions_per_term,
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


class DistributedCNFREXTrainer:
    """Replica exchange Path Gradient training with one replica per rank.

    As the distributed PT of distributedPT: every rank keeps batch_size chains whose ladder
    labels move between ranks through ParallelTempering (exact full-action swaps decided on
    rank 0). A rank sweeps each chain with the HBOR of its label and evaluates the PG loss at
    those labels. Parameter gradients are summed over ranks, so every rank applies the same
    update to an identical field; the loss is the mean over all chains, as in CNFREXTrainer.
    """

    def __init__(
        self,
        field,
        flow_pars,
        parameters,
        optimizer,
        *,
        rank,
        world_size,
        domain="beta",
        beta=6.0,
        defect_mask=None,
        batch_size=2,
        updates_per_replica=1,
        grad_clip_norm=0.0,
        chunk_size=None,
        directions_per_term=1,
        swap_parity="alternating",
    ):
        self.field = field
        self.flow_pars = flow_pars
        self.optimizer = optimizer
        self.rank, self.world_size = int(rank), int(world_size)
        self.domain = str(domain).lower()
        self.beta = float(beta)
        self.defect_mask = defect_mask
        self.batch_size = int(batch_size)
        self.updates_per_replica = int(updates_per_replica)
        self.grad_clip_norm = float(grad_clip_norm)
        self.chunk_size = None if chunk_size is None else int(chunk_size)
        self.directions_per_term = int(directions_per_term)
        self.device = torch.device(flow_pars.device)
        self.parameters = torch.as_tensor(parameters, dtype=torch.float64, device=self.device).reshape(-1)
        if self.domain not in {"beta", "defect"}:
            raise ValueError("domain must be 'beta' or 'defect'")
        if self.parameters.numel() != self.world_size:
            raise ValueError(f"one replica per rank: {self.parameters.numel()} replicas on {self.world_size} ranks")
        ladder = torch.linspace(float(self.parameters[0]), float(self.parameters[-1]), self.world_size,
                                dtype=torch.float64, device=self.device)
        if self.world_size < 2 or not torch.allclose(self.parameters, ladder, rtol=0, atol=1e-12) \
                or not bool((torch.diff(self.parameters) > 0).all()):
            raise ValueError("the replica ladder must be increasing and evenly spaced, as in the distributed PT")
        if self.batch_size < 2:
            raise ValueError("batch_size must be >= 2")
        if self.updates_per_replica < 1:
            raise ValueError("updates_per_replica must be positive")
        if self.domain == "defect" and defect_mask is None:
            raise ValueError("defect_mask is required for domain='defect'")

        # Same field on every rank, whatever each rank's random initialization.
        with torch.no_grad():
            for tensor in list(field.parameters()) + list(field.buffers()):
                dist.broadcast(tensor.data, src=0)
        self.pt = ParallelTempering(
            PTConfig(
                batch_size=self.batch_size, parameter_min=float(self.parameters[0]),
                parameter_max=float(self.parameters[-1]), swap_parity=swap_parity,
                parameter_name="beta" if self.domain == "beta" else "bc", acceptance_rule="exact",
            ),
            self.rank, self.world_size, self.device,
        )
        self.volume = int(flow_pars.T) * int(flow_pars.L) ** (int(flow_pars.D) - 1)
        self.n_updates = 0
        shape = (self.batch_size, int(flow_pars.D), int(flow_pars.T),
                 *([int(flow_pars.L)] * (int(flow_pars.D) - 1)), int(flow_pars.N))
        self.x = sun.SUN_haar(shape, device=self.device, dtype=torch.cdouble)
        self.updates = [
            HBOR(flow_pars, beta=float(p), defect_par=1.0) if self.domain == "beta"
            else HBOR(flow_pars, beta=self.beta, defect_par=float(p))
            for p in self.parameters
        ]

    def action(self, configs, parameter):
        """Wilson action along beta or defect parameter, differentiable in parameter."""
        return ladder_action(self.flow_pars, self.domain, self.beta, self.defect_mask, self.volume, configs, parameter)

    def _labels(self):
        """Ladder index of each local chain."""
        return torch.argmin((self.pt.local_parameters[:, None] - self.parameters[None, :]).abs(), dim=1)

    def _sweep(self):
        with torch.no_grad():
            labels = self._labels()
            # At zero gauge coupling the first rung is exactly Haar and can be refreshed.
            if (self.domain == "beta" and float(self.parameters[0]) == 0.0) or (
                self.domain == "defect" and self.beta == 0.0
            ):
                base = torch.nonzero(labels == 0).squeeze(-1)
                self.x[base] = sun.SUN_haar(self.x[base].shape[:-1], device=self.device, dtype=self.x.dtype)
            mask = self.flow_pars.mask.to(device=self.device)
            defect = None if self.defect_mask is None else self.defect_mask.to(device=self.device)
            for k in torch.unique(labels).tolist():
                chains = torch.nonzero(labels == k).squeeze(-1)
                configs = self.x[chains]
                for _ in range(self.updates_per_replica):
                    configs = self.updates[k](configs, mask, defect)
                self.x[chains] = configs
            table = torch.stack([self.action(self.x, p) for p in self.parameters], dim=1)
            return self.pt.step(local_actions=table)

    def warmup(self, sweeps=1):
        sweeps = int(sweeps)
        if sweeps < 0:
            raise ValueError("warmup sweeps must be non-negative")
        for _ in range(sweeps):
            self._sweep()

    def _backward(self):
        """PG loss of the local chains at their labels, gradients summed over ranks."""
        t = self.pt.local_parameters.to(dtype=self.x.real.dtype)
        result = path_gradient_loss(self.field, self.action, self.x, t, volume=self.volume, chunk_size=self.chunk_size,
                                    directions_per_term=self.directions_per_term)
        (result["loss"].mean() / self.world_size).backward()
        for parameter in self.field.parameters():
            if parameter.requires_grad:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                dist.all_reduce(parameter.grad)
        return result

    def _gather(self, tensor):
        parts = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(parts, tensor.contiguous())
        return torch.cat(parts)

    def _clock(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def step(self):
        start = self._clock()
        stats = self._sweep()
        swept = self._clock()
        self.optimizer.zero_grad(set_to_none=True)
        result = self._backward()
        loss, xi = result["loss"].detach(), result["xi"].detach()
        backward = self._clock()

        parameters = [p for p in self.field.parameters() if p.requires_grad]
        grads = [p.grad for p in parameters]
        # Gradients are identical on all ranks after the sum; the losses are not.
        finite = torch.tensor(float(bool(torch.isfinite(loss).all()) and bool(torch.isfinite(xi).all())
                                    and all(bool(torch.isfinite(g).all()) for g in grads)), device=self.device)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        grad_norm = float("nan")
        updated = False
        if bool(finite.item()):
            if self.grad_clip_norm > 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, self.grad_clip_norm).item())
            else:
                grad_norm = float(torch.linalg.vector_norm(torch.stack([g.detach().norm() for g in grads])).item())
            if math.isfinite(grad_norm):
                self.optimizer.step()
                updated = True
        self.n_updates += 1

        labels, losses, residuals = self._gather(self._labels()), self._gather(loss), self._gather(xi)
        by_rung = [labels == k for k in range(self.world_size)]
        n_links = self.world_size - 1
        links = torch.tensor(stats["link_accepted"] + stats["link_proposed"], dtype=torch.float64, device=self.device) \
            if self.rank == 0 else torch.zeros(2 * n_links, dtype=torch.float64, device=self.device)
        dist.broadcast(links, src=0)
        accepted, proposed = links[:n_links], links[n_links:]
        # Slowest rank and largest peak memory over ranks.
        costs = torch.tensor([swept - start, backward - swept,
                              torch.cuda.max_memory_allocated(self.device) / 2**30 if self.device.type == "cuda" else 0.0],
                             dtype=torch.float64, device=self.device)
        dist.all_reduce(costs, op=dist.ReduceOp.MAX)
        return {
            "losses_by_rung": [float(losses[r].mean()) for r in by_rung],
            "pinn_by_rung": [float(self.volume * residuals[r].var(unbiased=False)) for r in by_rung],
            "residuals_by_rung": [float(residuals[r].mean()) for r in by_rung],
            "swap_acceptance_by_edge": [float(v) for v in accepted / proposed.clamp_min(1.0)],
            "swap_acceptance": float(accepted.sum() / proposed.sum().clamp_min(1.0)),
            "grad_norm": grad_norm,
            "optimizer_updated": updated,
            "updates": self.n_updates,
            "sweep_s": round(float(costs[0]), 3),
            "loss_s": round(float(costs[1]), 3),
            "peak_memory_gb": round(float(costs[2]), 2),
        }

    def state_dict(self):
        """State shared by all ranks, saved by rank 0 with the field."""
        return {
            "distributed": True,
            "world_size": self.world_size,
            "parameters": self.parameters.detach().cpu(),
            "parameter_matrix": self.pt.parameter_matrix.detach().cpu() if self.rank == 0 else None,
            "pt_step": self.pt._pt_step,
            "n_updates": self.n_updates,
            "optimizer": self.optimizer.state_dict(),
        }

    def shard_state_dict(self):
        """State of this rank: its chains, their labels and its random streams."""
        shard = {
            "rank": self.rank,
            "configs": self.x.detach().cpu(),
            "local_parameters": self.pt.local_parameters.detach().cpu(),
            "torch_rng_state": torch.get_rng_state(),
        }
        if self.device.type == "cuda":
            shard["cuda_rng_state"] = torch.cuda.get_rng_state(self.device)
        return shard

    def load_state_dict(self, state, shard):
        if not state.get("distributed") or int(state["world_size"]) != self.world_size:
            raise ValueError("checkpoint was not written by a distributed run on this many ranks")
        if not torch.equal(state["parameters"].cpu(), self.parameters.detach().cpu()):
            raise ValueError("checkpoint replica ladder does not match")
        if int(shard["rank"]) != self.rank or tuple(shard["configs"].shape) != tuple(self.x.shape):
            raise ValueError("checkpoint shard does not belong to this rank")
        if self.rank == 0:
            self.pt.parameter_matrix.copy_(state["parameter_matrix"].to(self.pt.parameter_matrix))
        self.pt._scatter_parameters()
        if not torch.equal(self.pt.local_parameters.cpu(), shard["local_parameters"].to(self.pt.local_parameters.dtype)):
            raise ValueError("checkpoint chain labels disagree with the replica ladder state")
        self.pt._pt_step = int(state["pt_step"])
        self.x.copy_(shard["configs"].to(device=self.device, dtype=self.x.dtype))
        self.n_updates = int(state["n_updates"])
        self.optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(shard["torch_rng_state"].cpu())
        if self.device.type == "cuda" and "cuda_rng_state" in shard:
            torch.cuda.set_rng_state(shard["cuda_rng_state"], self.device)
