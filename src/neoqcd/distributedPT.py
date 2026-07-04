import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from dataclasses import dataclass
from typing import Optional, Sequence, Any
from contextlib import nullcontext
 
 ####################################################################
############# Vanilla Parallel Tempering (PT) ######################
 ####################################################################


def _call_action_fn(fn, x: torch.Tensor, parameter: torch.Tensor, parameter_name: str):
    try:
        return fn(x, parameter, parameter_name=parameter_name)
    except TypeError:
        try:
            return fn(x, parameter)
        except TypeError:
            return fn(x)


def _set_parameter_on_target(target: Any, parameter: torch.Tensor, parameter_name: str):
    if target is None:
        raise TypeError("Cannot set parameter on None target")

    if hasattr(target, "set_parameter"):
        setter = target.set_parameter
        try:
            setter(parameter, parameter_name=parameter_name)
            return
        except TypeError:
            setter(parameter)
            return

    if hasattr(target, parameter_name):
        setattr(target, parameter_name, parameter)
        return

    raise AttributeError(
        f"Target of type {type(target).__name__!r} does not expose "
        f"'set_parameter(...)' or attribute '{parameter_name}'."
    )


def _action_on_target(target: Any, x: torch.Tensor, parameter: torch.Tensor, parameter_name: str):
    if target is None:
        raise TypeError("Cannot evaluate action on None target")

    if hasattr(target, "action"):
        return _call_action_fn(target.action, x, parameter, parameter_name)
    if hasattr(target, "theory"):
        return _call_action_fn(target.theory, x, parameter, parameter_name)

    raise AttributeError(
        f"Target of type {type(target).__name__!r} does not expose "
        "'action(...)' or 'theory(...)'."
    )
 
@dataclass
class PTConfig:
    batch_size: int          # replicas per node
    parameter_min: float
    parameter_max: float
    swap_parity: str = "alternating"   # "even" | "odd" | "alternating"
    parameter_dtype: torch.dtype = torch.float64
    # General tempering parameter support:
    # - parameter_name: e.g. "beta" or "bc"
    # - acceptance_rule: "exact" | "linear" | "auto"
    #   exact:   uses full action differences via local_actions (general, BC-safe)
    #   linear:  uses ΔS=(λ_j-λ_i)(O_j-O_i) via local_energies/local_observable
    #   auto:    picks exact when local_actions is passed, else linear
    parameter_name: str = "beta"
    acceptance_rule: str = "exact"
 
 
class ParallelTempering:
    """
    Manages tempering-parameter labels and swaps between replicas.

    Usage:
        pt = ParallelTempering(config, rank, world_size, device)

        # in your loop:
        actions = azioni_per_ogni_parametro(pt.local_parameters)  # (B, world_size)
        pt.step(local_actions=actions)
    """

    def __init__(self, config: PTConfig, rank: int, world_size: int, device: torch.device):
        self.config     = config
        self.rank       = rank
        self.world_size = world_size
        self.device     = device
        self.parameter_name = str(config.parameter_name)
        self.acceptance_rule = str(config.acceptance_rule).lower()
        self.param_dtype = config.parameter_dtype
        self.param_min = config.parameter_min
        self.param_max = config.parameter_max
        self.B          = config.batch_size
        self._pt_step   = 0
        # For current linear protocols this is a single constant scalar.
        # It is computed on rank 0 and broadcast so every rank has the same value.
        self.delta_parameter = torch.zeros((), device=device, dtype=self.param_dtype)

        if config.swap_parity not in {"even", "odd", "alternating"}:
            raise ValueError(f"swap_parity must be 'even', 'odd', or 'alternating', got: {config.swap_parity!r}")
        if self.acceptance_rule not in {"exact", "linear", "auto"}:
            raise ValueError(
                f"acceptance_rule must be 'exact', 'linear', or 'auto', got: {config.acceptance_rule!r}"
            )

        # rank 0 holds the global parameter matrix
        if rank == 0:
            params = torch.linspace(
                self.param_min,
                self.param_max,
                world_size,
                dtype=self.param_dtype,
            )
            params[0] = self.param_min
            params[-1] = self.param_max
            if world_size > 1:
                self.delta_parameter = (params[1] - params[0]).to(device=device, dtype=self.param_dtype)
            # (B, n_nodes): each column is constant at its rank parameter value
            self.parameter_matrix = params.view(1, world_size).repeat(self.B, 1).contiguous().to(device)
        else:
            self.parameter_matrix = None
        dist.broadcast(self.delta_parameter, src=0)

        # each node receives its local parameter column
        self.local_parameters = torch.zeros(self.B, device=device, dtype=self.param_dtype)
        self._scatter_parameters()

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def step(
        self,
        local_energies: Optional[torch.Tensor] = None,
        local_actions: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        One PT step: gather local statistics, swap on rank 0, scatter new labels.

        Args:
            local_energies:
                (batch_size,) observable used in the linear rule
                ΔS = (λ_j - λ_i) * (O_j - O_i).
                This is exact for beta-PT with O=energy.
            local_actions:
                (batch_size, world_size) exact-action mode.
                local_actions[b, k] must be S(x_b_local; λ_k), i.e. the local
                configuration evaluated at every ladder parameter.
                Useful for PT on boundary-condition couplings (bc PT).

        Returns:
            dict with n_accepted, n_proposed (only rank 0 has values > 0)
        """
        parity = self._current_parity()
        stats  = self._attempt_swaps(local_energies, local_actions, parity)
        self._scatter_parameters()
        self._pt_step += 1
        return stats

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
 
    def _current_parity(self) -> str:
        if self.config.swap_parity == "alternating":
            return "even" if self._pt_step % 2 == 0 else "odd"
        return self.config.swap_parity

    def _attempt_swaps(
        self,
        local_energies: Optional[torch.Tensor],
        local_actions: Optional[torch.Tensor],
        parity: str,
    ) -> dict:
        mode = self.acceptance_rule
        if mode == "auto":
            if local_actions is not None:
                mode = "exact"
            elif local_energies is not None:
                mode = "linear"
            else:
                raise ValueError("acceptance_rule='auto': pass local_actions or local_energies.")

        if mode == "exact":
            if local_actions is None:
                raise ValueError(
                    "acceptance_rule='exact' requires local_actions with shape (batch_size, world_size)."
                )
            return self._attempt_swaps_exact_actions(local_actions, parity)

        if mode == "linear":
            if local_energies is None:
                raise ValueError(
                    "acceptance_rule='linear' requires local_energies/local_observable with shape (batch_size,)."
                )
            return self._attempt_swaps_linear_observable(local_energies, parity)

        raise RuntimeError(f"Unsupported acceptance mode: {mode!r}")

    def _attempt_swaps_linear_observable(self, local_observable: torch.Tensor, parity: str) -> dict:
        local_observable = local_observable.to(dtype=self.param_dtype)
        # --- gather energies ---
        if self.rank == 0:
            gather_list = [
                torch.zeros(self.B, device=self.device, dtype=local_observable.dtype)
                for _ in range(self.world_size)
            ]
        else:
            gather_list = None
        dist.gather(local_observable, gather_list=gather_list, dst=0)

        n_accepted, n_proposed = 0, 0
        n_links = max(0, self.world_size - 1)
        link_accepted = [0] * n_links
        link_proposed = [0] * n_links

        if self.rank == 0:
            # observable_matrix: (B, n_nodes)
            observable_matrix = torch.stack(gather_list, dim=1)

            # parameter order for each batch element
            param_order = torch.argsort(self.parameter_matrix, dim=1)

            start = 0 if parity == "even" else 1
            b_idx = torch.arange(self.B, device=self.device)

            for k in range(start, self.world_size - 1, 2):
                node_i = param_order[:, k]
                node_j = param_order[:, k + 1]

                lam_i = self.parameter_matrix[b_idx, node_i]
                lam_j = self.parameter_matrix[b_idx, node_j]
                obs_i = observable_matrix[b_idx, node_i]
                obs_j = observable_matrix[b_idx, node_j]

                # Metropolis linear rule: ΔS = (λ_j - λ_i)(O_j - O_i)
                delta_s = (lam_j - lam_i) * (obs_j - obs_i)
                accept = delta_s >= torch.rand(self.B, device=self.device).log()

                self.parameter_matrix[b_idx, node_i] = torch.where(accept, lam_j, lam_i)
                self.parameter_matrix[b_idx, node_j] = torch.where(accept, lam_i, lam_j)

                n_accepted += accept.sum().item()
                n_proposed += self.B
                link_accepted[int(k)] += int(accept.sum().item())
                link_proposed[int(k)] += self.B

        return {
            "n_accepted": int(n_accepted),
            "n_proposed": int(n_proposed),
            "link_accepted": link_accepted,
            "link_proposed": link_proposed,
        }

    def _attempt_swaps_exact_actions(self, local_actions: torch.Tensor, parity: str) -> dict:
        if local_actions.ndim != 2:
            raise ValueError(
                f"local_actions must have shape (batch_size, world_size), got {tuple(local_actions.shape)}"
            )
        if int(local_actions.shape[0]) != self.B or int(local_actions.shape[1]) != self.world_size:
            raise ValueError(
                "local_actions must have shape "
                f"({self.B}, {self.world_size}), got {tuple(local_actions.shape)}"
            )

        local_actions = local_actions.to(dtype=self.param_dtype)
        if self.rank == 0:
            gather_list = [
                torch.zeros((self.B, self.world_size), device=self.device, dtype=local_actions.dtype)
                for _ in range(self.world_size)
            ]
        else:
            gather_list = None
        dist.gather(local_actions, gather_list=gather_list, dst=0)

        n_accepted, n_proposed = 0, 0
        n_links = max(0, self.world_size - 1)
        link_accepted = [0] * n_links
        link_proposed = [0] * n_links

        if self.rank == 0:
            # action_cube[b, node, k] = S(x_b_from_node ; lambda_k)
            action_cube = torch.stack(gather_list, dim=1)
            param_order = torch.argsort(self.parameter_matrix, dim=1)

            start = 0 if parity == "even" else 1
            b_idx = torch.arange(self.B, device=self.device)

            for k in range(start, self.world_size - 1, 2):
                node_i = param_order[:, k]
                node_j = param_order[:, k + 1]

                lam_i = self.parameter_matrix[b_idx, node_i]
                lam_j = self.parameter_matrix[b_idx, node_j]

                s_ii = action_cube[b_idx, node_i, node_i]
                s_ij = action_cube[b_idx, node_i, node_j]
                s_ji = action_cube[b_idx, node_j, node_i]
                s_jj = action_cube[b_idx, node_j, node_j]

                # Exact Metropolis swap test:
                # ΔS = S(x_i, λ_j) + S(x_j, λ_i) - S(x_i, λ_i) - S(x_j, λ_j)
                delta_s = (s_ij + s_ji) - (s_ii + s_jj)
                accept = delta_s >= torch.rand(self.B, device=self.device).log()

                self.parameter_matrix[b_idx, node_i] = torch.where(accept, lam_j, lam_i)
                self.parameter_matrix[b_idx, node_j] = torch.where(accept, lam_i, lam_j)

                n_accepted += int(accept.sum().item())
                n_proposed += self.B
                link_accepted[int(k)] += int(accept.sum().item())
                link_proposed[int(k)] += self.B

        return {
            "n_accepted": int(n_accepted),
            "n_proposed": int(n_proposed),
            "link_accepted": link_accepted,
            "link_proposed": link_proposed,
        }

    def _scatter_parameters(self):
        """rank 0 sends each node its current parameter column."""
        if self.rank == 0:
            scatter_list = list(self.parameter_matrix.T.contiguous())
        else:
            scatter_list = None
        recv = torch.zeros(self.B, device=self.device, dtype=self.param_dtype)
        dist.scatter(recv, scatter_list=scatter_list, src=0)
        self.local_parameters = recv
 
 
@dataclass
class OREXPTConfig(PTConfig):
    orex_protocol_steps: int = 8
    orex_mcmc_steps_per_step: int = 1
    orex_schedule: str = "even_odd"  # "even_odd" | "alternating" | "even" | "odd"
    save_work_values: bool = False


class OutOfEquilibriumParallelTempering(ParallelTempering):
    """
    Distributed O-REX parallel tempering.

    The tempering-parameter labels are exchanged as in ParallelTempering, but each proposed
    exchange first drives both candidate configurations along opposite protocol
    protocols and accepts/rejects with the accumulated protocol work.
    """

    def __init__(self, config: OREXPTConfig, rank: int, world_size: int, device: torch.device):
        if int(config.orex_protocol_steps) < 0:
            raise ValueError(
                f"orex_protocol_steps must be >= 0, got {config.orex_protocol_steps}"
            )
        if int(config.orex_mcmc_steps_per_step) < 0:
            raise ValueError(
                "orex_mcmc_steps_per_step must be >= 0, "
                f"got {config.orex_mcmc_steps_per_step}"
            )
        if config.orex_schedule not in {"even_odd", "alternating", "even", "odd"}:
            raise ValueError(
                "orex_schedule must be 'even_odd', 'alternating', 'even', or 'odd', "
                f"got {config.orex_schedule!r}"
            )
        super().__init__(config, rank, world_size, device)
        self.config: OREXPTConfig = config
        self._pending_pairs = []

    def set_parameter(self, mcmc, parameter: torch.Tensor):
        """
        Set the runtime tempering parameter on the MCMC object.
        This is the single entry point used by O-REX and can be reused by NEO-REX.
        """
        targets = [mcmc]
        if hasattr(mcmc, "sampler"):
            targets.append(mcmc.sampler)
        last_error = None
        for target in targets:
            try:
                _set_parameter_on_target(target, parameter, self.parameter_name)
                return
            except Exception as err:
                last_error = err
        raise TypeError(
            "Unable to set protocol parameter on mcmc object. "
            f"Tried targets {[type(t).__name__ for t in targets]} with parameter_name={self.parameter_name!r}."
        ) from last_error

    def action(self, mcmc, x: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the action S(x; parameter) for the current protocol parameter.
        This is the single entry point used by O-REX and can be reused by NEO-REX.
        """
        targets = [mcmc]
        if hasattr(mcmc, "sampler"):
            targets.append(mcmc.sampler)
        last_error = None
        for target in targets:
            try:
                return _action_on_target(target, x, parameter, self.parameter_name)
            except Exception as err:
                last_error = err
        raise TypeError(
            "Unable to evaluate protocol action from mcmc object. "
            f"Tried targets {[type(t).__name__ for t in targets]} with parameter_name={self.parameter_name!r}."
        ) from last_error

    def action_difference(
        self,
        mcmc,
        x: torch.Tensor,
        parameter_old: torch.Tensor,
        parameter_new: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate S(x; parameter_new) - S(x; parameter_old).
        Specialized targets can implement this without two full action calls.
        """
        targets = [mcmc]
        if hasattr(mcmc, "sampler"):
            targets.append(mcmc.sampler)
        last_error = None
        for target in targets:
            if not hasattr(target, "action_difference"):
                continue
            try:
                return target.action_difference(
                    x,
                    parameter_old,
                    parameter_new,
                    parameter_name=self.parameter_name,
                )
            except TypeError:
                try:
                    return target.action_difference(x, parameter_old, parameter_new)
                except Exception as err:
                    last_error = err
            except Exception as err:
                last_error = err
        if last_error is not None:
            raise TypeError(
                "Unable to evaluate specialized protocol action difference from mcmc object. "
                f"Tried targets {[type(t).__name__ for t in targets]} with parameter_name={self.parameter_name!r}."
            ) from last_error
        return self.action(mcmc, x, parameter_new) - self.action(mcmc, x, parameter_old)

    @torch.no_grad()
    def step(self, x: torch.Tensor, mcmc) -> tuple[torch.Tensor, dict]:
        """
        One O-REX PT step.

        Args:
            x: local configurations, shape (batch_size, ...)
            mcmc: MCMC wrapper exposing either:
                - set_parameter(...) and action(...), or
                - sampler with compatible methods/attributes.

        Returns:
            (new_x, stats). Accepted candidates keep the O-REX transformed
            configuration; rejected candidates are restored to their pre-protocol
            configuration.
        """
        if int(x.shape[0]) != self.B:
            raise ValueError(f"x batch dimension must be {self.B}, got {x.shape[0]}")

        schedule = self.config.orex_schedule
        if schedule == "even_odd":
            total_stats = self._empty_stats()
            for parity in ("even", "odd"):
                x, stats = self._step_parity(x, mcmc, parity)
                total_stats = self._merge_stats(total_stats, stats)
            self._pt_step += 1
            return x, self._finalize_stats(total_stats)
        if schedule == "alternating":
            parity = "even" if self._pt_step % 2 == 0 else "odd"
        else:
            parity = schedule

        x_out, stats = self._step_parity(x, mcmc, parity)
        self._pt_step += 1
        return x_out, self._finalize_stats(stats)

    def _step_parity(self, x: torch.Tensor, mcmc, parity: str) -> tuple[torch.Tensor, dict]:
        x_old = x.clone()
        start_parameters = self.local_parameters.clone()
        target_parameters, active = self._prepare_orex_targets(parity)

        x_proposed, local_work = self._run_orex_protocol(
            x=x,
            mcmc=mcmc,
            start_parameters=start_parameters,
            target_parameters=target_parameters,
            active=active,
        )
        accept, stats = self._accept_orex_work(local_work)

        accept_view = self._sample_mask_view(accept, x)
        x_out = torch.where(accept_view, x_proposed, x_old)

        self._scatter_parameters()
        self.set_parameter(mcmc, self.local_parameters)
        return x_out, stats

    def _empty_stats(self) -> dict:
        return {
            "n_accepted": 0,
            "n_proposed": 0,
            "_work_sum": 0.0,
            "_work_sum_sq": 0.0,
            "_work_count": 0,
            "_work_values": [],
            "_link_accepted": [0] * max(0, self.world_size - 1),
            "_link_proposed": [0] * max(0, self.world_size - 1),
            "_link_work_sum": [0.0] * max(0, self.world_size - 1),
            "_link_work_sum_sq": [0.0] * max(0, self.world_size - 1),
            "_link_work_count": [0] * max(0, self.world_size - 1),
            "_link_work_values": [[] for _ in range(max(0, self.world_size - 1))],
        }

    def _merge_stats(self, left: dict, right: dict) -> dict:
        n_links = max(0, self.world_size - 1)
        left_acc = left.get("_link_accepted", [0] * n_links)
        right_acc = right.get("_link_accepted", [0] * n_links)
        left_prop = left.get("_link_proposed", [0] * n_links)
        right_prop = right.get("_link_proposed", [0] * n_links)
        left_wsum = left.get("_link_work_sum", [0.0] * n_links)
        right_wsum = right.get("_link_work_sum", [0.0] * n_links)
        left_wsum_sq = left.get("_link_work_sum_sq", [0.0] * n_links)
        right_wsum_sq = right.get("_link_work_sum_sq", [0.0] * n_links)
        left_wcount = left.get("_link_work_count", [0] * n_links)
        right_wcount = right.get("_link_work_count", [0] * n_links)
        left_wvalues = left.get("_link_work_values", [[] for _ in range(n_links)])
        right_wvalues = right.get("_link_work_values", [[] for _ in range(n_links)])
        return {
            "n_accepted": int(left.get("n_accepted", 0)) + int(right.get("n_accepted", 0)),
            "n_proposed": int(left.get("n_proposed", 0)) + int(right.get("n_proposed", 0)),
            "_work_sum": float(left.get("_work_sum", 0.0)) + float(right.get("_work_sum", 0.0)),
            "_work_sum_sq": float(left.get("_work_sum_sq", 0.0)) + float(right.get("_work_sum_sq", 0.0)),
            "_work_count": int(left.get("_work_count", 0)) + int(right.get("_work_count", 0)),
            "_work_values": list(left.get("_work_values", [])) + list(right.get("_work_values", [])),
            "_link_accepted": [
                int(left_acc[k]) + int(right_acc[k])
                for k in range(n_links)
            ],
            "_link_proposed": [
                int(left_prop[k]) + int(right_prop[k])
                for k in range(n_links)
            ],
            "_link_work_sum": [
                float(left_wsum[k]) + float(right_wsum[k])
                for k in range(n_links)
            ],
            "_link_work_sum_sq": [
                float(left_wsum_sq[k]) + float(right_wsum_sq[k])
                for k in range(n_links)
            ],
            "_link_work_count": [
                int(left_wcount[k]) + int(right_wcount[k])
                for k in range(n_links)
            ],
            "_link_work_values": [
                list(left_wvalues[k]) + list(right_wvalues[k])
                for k in range(n_links)
            ],
        }

    def _finalize_stats(self, stats: dict) -> dict:
        work_count = int(stats.get("_work_count", 0))
        n_links = max(0, self.world_size - 1)
        link_acc = [int(v) for v in stats.get("_link_accepted", [0] * n_links)]
        link_prop = [int(v) for v in stats.get("_link_proposed", [0] * n_links)]
        link_wsum = [float(v) for v in stats.get("_link_work_sum", [0.0] * n_links)]
        link_wsum_sq = [float(v) for v in stats.get("_link_work_sum_sq", [0.0] * n_links)]
        link_wcount = [int(v) for v in stats.get("_link_work_count", [0] * n_links)]
        link_wmean = [
            (link_wsum[k] / link_wcount[k]) if link_wcount[k] > 0 else float("nan")
            for k in range(n_links)
        ]
        link_wsem = []
        for k in range(n_links):
            cnt = int(link_wcount[k])
            if cnt <= 1:
                link_wsem.append(float("nan"))
                continue
            var_unbiased = (link_wsum_sq[k] - (link_wsum[k] * link_wsum[k] / float(cnt))) / float(cnt - 1)
            var_unbiased = max(var_unbiased, 0.0)
            link_wsem.append(float(np.sqrt(var_unbiased / float(cnt))))
        return {
            "n_accepted": int(stats.get("n_accepted", 0)),
            "n_proposed": int(stats.get("n_proposed", 0)),
            "work_mean": (float(stats.get("_work_sum", 0.0)) / work_count)
            if work_count
            else float("nan"),
            "work_sum": float(stats.get("_work_sum", 0.0)),
            "work_sum_sq": float(stats.get("_work_sum_sq", 0.0)),
            "work_count": work_count,
            "work_values": self._finalize_work_values(stats.get("_work_values", [])),
            "link_accepted": link_acc,
            "link_proposed": link_prop,
            "link_work_mean": link_wmean,
            "link_work_sem": link_wsem,
            "link_work_sum": link_wsum,
            "link_work_sum_sq": link_wsum_sq,
            "link_work_count": link_wcount,
            "link_work_values": self._finalize_link_work_values(
                stats.get("_link_work_values", [[] for _ in range(n_links)])
            ),
        }

    def _finalize_work_values(self, work_values) -> np.ndarray:
        if not getattr(self.config, "save_work_values", False):
            return np.empty(0, dtype=np.float64)
        arrays = [
            np.asarray(values, dtype=np.float64).reshape(-1)
            for values in work_values
            if np.asarray(values).size
        ]
        if not arrays:
            return np.empty(0, dtype=np.float64)
        return np.concatenate(arrays).astype(np.float64, copy=False)

    def _finalize_link_work_values(self, link_work_values) -> list[np.ndarray]:
        n_links = max(0, self.world_size - 1)
        if not getattr(self.config, "save_work_values", False):
            return [np.empty((0, self.B), dtype=np.float64) for _ in range(n_links)]
        out = []
        for values in link_work_values:
            arrays = []
            for value in values:
                arr = np.asarray(value, dtype=np.float64)
                if not arr.size:
                    continue
                arrays.append(arr.reshape(1, -1))
            if arrays:
                out.append(np.concatenate(arrays, axis=0).astype(np.float64, copy=False))
            else:
                out.append(np.empty((0, self.B), dtype=np.float64))
        while len(out) < n_links:
            out.append(np.empty((0, self.B), dtype=np.float64))
        return out[:n_links]

    def _sample_mask_view(self, mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return mask.to(device=x.device, dtype=torch.bool).view(
            (self.B,) + (1,) * (x.dim() - 1)
        )

    def _prepare_orex_targets(self, parity: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rank == 0:
            parameter_order = torch.argsort(self.parameter_matrix, dim=1)
            target_matrix = self.parameter_matrix.clone()
            active_matrix = torch.zeros(
                (self.B, self.world_size),
                device=self.device,
                dtype=torch.uint8,
            )
            self._pending_pairs = []
            start = 0 if parity == "even" else 1
            b_idx = torch.arange(self.B, device=self.device)

            for k in range(start, self.world_size - 1, 2):
                node_i = parameter_order[:, k].clone()
                node_j = parameter_order[:, k + 1].clone()

                parameter_i = self.parameter_matrix[b_idx, node_i]
                parameter_j = self.parameter_matrix[b_idx, node_j]

                target_matrix[b_idx, node_i] = parameter_j
                target_matrix[b_idx, node_j] = parameter_i
                active_matrix[b_idx, node_i] = 1
                active_matrix[b_idx, node_j] = 1
                self._pending_pairs.append((k, node_i, node_j))

            target_scatter = list(target_matrix.T.contiguous())
            active_scatter = list(active_matrix.T.contiguous())
        else:
            target_scatter = None
            active_scatter = None

        target = torch.zeros(self.B, device=self.device, dtype=self.param_dtype)
        active_uint8 = torch.zeros(self.B, device=self.device, dtype=torch.uint8)
        dist.scatter(target, scatter_list=target_scatter, src=0)
        dist.scatter(active_uint8, scatter_list=active_scatter, src=0)
        return target, active_uint8.to(dtype=torch.bool)

    @torch.no_grad()
    def _run_orex_protocol(
        self,
        x: torch.Tensor,
        mcmc,
        start_parameters: torch.Tensor,
        target_parameters: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not bool(active.any().item()):
            return x, torch.zeros(self.B, device=self.device, dtype=self.param_dtype)

        n_protocol = int(self.config.orex_protocol_steps)
        n_mcmc = int(self.config.orex_mcmc_steps_per_step)
        work = torch.zeros(self.B, device=self.device, dtype=self.param_dtype)
        x_curr = x
        parameter_delta = target_parameters - start_parameters
        active_view = self._sample_mask_view(active, x)

        if n_protocol == 0:
            dwork = self.action_difference(mcmc, x_curr, start_parameters, target_parameters)
            work = torch.where(active, dwork.to(dtype=self.param_dtype), 0.0)
            return x_curr, work

        for n in range(n_protocol):
            lambda_0 = n / n_protocol
            lambda_1 = (n + 1) / n_protocol
            parameter_0 = start_parameters + lambda_0 * parameter_delta
            parameter_1 = start_parameters + lambda_1 * parameter_delta

            dwork = self.action_difference(mcmc, x_curr, parameter_0, parameter_1)
            work = work + torch.where(active, dwork.to(dtype=self.param_dtype), 0.0)

            if n_mcmc > 0:
                self.set_parameter(mcmc, parameter_1)
                x_updated = mcmc(
                    x_curr,
                    nsteps=n_mcmc,
                    ensembles=False,
                    use_tqdm=False,
                )
                x_curr = torch.where(active_view, x_updated, x_curr)

        return x_curr, work

    def _accept_orex_work(self, local_work: torch.Tensor) -> tuple[torch.Tensor, dict]:
        local_work = local_work.to(dtype=self.param_dtype)
        if self.rank == 0:
            gather_list = [
                torch.zeros(self.B, device=self.device, dtype=self.param_dtype)
                for _ in range(self.world_size)
            ]
        else:
            gather_list = None
        dist.gather(local_work, gather_list=gather_list, dst=0)

        n_accepted, n_proposed = 0, 0
        work_sum_total = 0.0
        work_sum_sq_total = 0.0
        work_count = 0
        work_values = []
        link_accepted = [0] * max(0, self.world_size - 1)
        link_proposed = [0] * max(0, self.world_size - 1)
        link_work_sum = [0.0] * max(0, self.world_size - 1)
        link_work_sum_sq = [0.0] * max(0, self.world_size - 1)
        link_work_count = [0] * max(0, self.world_size - 1)
        link_work_values = [[] for _ in range(max(0, self.world_size - 1))]

        if self.rank == 0:
            work_matrix = torch.stack(gather_list, dim=1)
            accept_matrix = torch.zeros(
                (self.B, self.world_size),
                device=self.device,
                dtype=torch.uint8,
            )
            b_idx = torch.arange(self.B, device=self.device)

            for link_idx, node_i, node_j in self._pending_pairs:
                parameter_i = self.parameter_matrix[b_idx, node_i]
                parameter_j = self.parameter_matrix[b_idx, node_j]
                pair_work = work_matrix[b_idx, node_i] + work_matrix[b_idx, node_j]
                log_u = torch.rand(self.B, device=self.device, dtype=self.param_dtype).log()
                accept = torch.isfinite(pair_work) & ((-pair_work) >= log_u)

                self.parameter_matrix[b_idx, node_i] = torch.where(accept, parameter_j, parameter_i)
                self.parameter_matrix[b_idx, node_j] = torch.where(accept, parameter_i, parameter_j)
                accept_matrix[b_idx, node_i] = accept.to(dtype=torch.uint8)
                accept_matrix[b_idx, node_j] = accept.to(dtype=torch.uint8)

                n_accepted += int(accept.sum().item())
                n_proposed += self.B
                link_accepted[int(link_idx)] += int(accept.sum().item())
                link_proposed[int(link_idx)] += self.B
                finite_work = pair_work[torch.isfinite(pair_work)]
                if finite_work.numel() > 0:
                    work_sum_total += float(finite_work.sum().item())
                    work_sum_sq_total += float((finite_work * finite_work).sum().item())
                    work_count += int(finite_work.numel())
                    link_work_sum[int(link_idx)] += float(finite_work.sum().item())
                    link_work_sum_sq[int(link_idx)] += float((finite_work * finite_work).sum().item())
                    link_work_count[int(link_idx)] += int(finite_work.numel())
                    if getattr(self.config, "save_work_values", False):
                        work_values.append(finite_work.detach().cpu().numpy().copy())
                if getattr(self.config, "save_work_values", False):
                    link_work_values[int(link_idx)].append(pair_work.detach().cpu().numpy().copy())

            accept_scatter = list(accept_matrix.T.contiguous())
        else:
            accept_scatter = None

        accept_uint8 = torch.zeros(self.B, device=self.device, dtype=torch.uint8)
        dist.scatter(accept_uint8, scatter_list=accept_scatter, src=0)
        stats = {
            "n_accepted": n_accepted,
            "n_proposed": n_proposed,
            "_work_sum": work_sum_total,
            "_work_sum_sq": work_sum_sq_total,
            "_work_count": work_count,
            "_work_values": work_values,
            "_link_accepted": link_accepted,
            "_link_proposed": link_proposed,
            "_link_work_sum": link_work_sum,
            "_link_work_sum_sq": link_work_sum_sq,
            "_link_work_count": link_work_count,
            "_link_work_values": link_work_values,
        }
        return accept_uint8.to(dtype=torch.bool), stats


@dataclass
class NEOREXConfig(OREXPTConfig):
    flow_lr: float = 1e-3
    flow_train: bool = True
    grad_clip_norm: float = 0.0
    max_consecutive_nan_grad_updates: int = 5


class NEOREXProtocolWrapper(nn.Module):
    """
    Shared stochastic normalizing-flow protocol wrapper for NEO-REX.

    The wrapped deterministic NF layer is reused at every internal protocol
    step. Each deterministic block can be followed by stochastic blocks at the
    new tempering parameter; those stochastic blocks are applied under no_grad
    and contribute zero net nonequilibrium work at fixed parameter.
    """

    def __init__(
        self,
        nf_layer: nn.Module,
        theory,
        n_protocol_steps: int,
        work_mode: str = "logdet",
        stochastic_update: Optional[nn.Module] = None,
        stochastic_steps_per_step: int = 1,
        nf_first: bool = True,
        parameter_name: str = "beta",
    ):
        super().__init__()
        n_protocol_steps = int(n_protocol_steps)
        if n_protocol_steps < 1:
            raise ValueError(f"n_protocol_steps must be >= 1, got {n_protocol_steps}")
        stochastic_steps_per_step = int(stochastic_steps_per_step)
        if stochastic_steps_per_step < 0:
            raise ValueError(
                "stochastic_steps_per_step must be >= 0, "
                f"got {stochastic_steps_per_step}"
            )
        work_mode = str(work_mode).lower()
        if work_mode not in {"logdet", "work"}:
            raise ValueError("work_mode must be 'logdet' or 'work'")
        self.nf_layer = nf_layer
        self.theory = theory
        self.n_protocol_steps = n_protocol_steps
        self.work_mode = work_mode
        self.stochastic_update = stochastic_update
        self.stochastic_steps_per_step = stochastic_steps_per_step
        self.nf_first = bool(nf_first)
        self.parameter_name = str(parameter_name)

    def set_parameter(self, target: Any, parameter: torch.Tensor):
        _set_parameter_on_target(target, parameter, self.parameter_name)

    def action(self, x: torch.Tensor, parameter: torch.Tensor):
        try:
            return _action_on_target(self.theory, x, parameter, self.parameter_name)
        except AttributeError:
            return _call_action_fn(self.theory, x, parameter, self.parameter_name)

    def action_change(
        self,
        x_old: torch.Tensor,
        x_new: torch.Tensor,
        parameter_old: torch.Tensor,
        parameter_new: torch.Tensor,
    ):
        if hasattr(self.theory, "local_nf_action_change"):
            try:
                return self.theory.local_nf_action_change(
                    x_old,
                    x_new,
                    parameter_old,
                    parameter_new,
                    parameter_name=self.parameter_name,
                )
            except TypeError:
                return self.theory.local_nf_action_change(
                    x_old,
                    x_new,
                    parameter_old,
                    parameter_new,
                )
        return self.action(x_new, parameter_new) - self.action(x_old, parameter_old)

    def _call_nf(self, x: torch.Tensor, parameter: torch.Tensor, delta_parameter: torch.Tensor):
        try:
            out = self.nf_layer(
                x,
                **{
                    self.parameter_name: parameter,
                    f"delta_{self.parameter_name}": delta_parameter,
                },
            )
        except TypeError:
            out = self.nf_layer(x, parameter, delta_parameter)
        if not isinstance(out, tuple) or len(out) < 2:
            raise TypeError(
                "NEO-REX NF layer must return at least (x_new, work_or_logdet)."
            )
        return out[0], out[1]

    def _run_stochastic_blocks(self, x: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
        if self.stochastic_update is None or self.stochastic_steps_per_step <= 0:
            return x

        x_curr = x.detach()
        for _ in range(self.stochastic_steps_per_step):
            self.set_parameter(self.stochastic_update, parameter)
            with torch.no_grad():
                update_out = self.stochastic_update(x_curr)
            if isinstance(update_out, tuple):
                if len(update_out) == 0:
                    raise TypeError(
                        "stochastic_update returned an empty tuple; expected x or (x, ...)."
                    )
                x_curr = update_out[0]
            else:
                x_curr = update_out
            x_curr = x_curr.detach()
        return x_curr

    def _run_nf_block(
        self,
        x: torch.Tensor,
        parameter_0: torch.Tensor,
        parameter_1: torch.Tensor,
        delta_parameter: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        work_dtype = x.real.dtype if torch.is_complex(x) else x.dtype
        x_next, aux = self._call_nf(x, parameter_0, delta_parameter)
        aux = aux.to(device=x.device, dtype=work_dtype)
        if self.work_mode == "work":
            dwork = aux
        else:
            dwork = self.action_change(x, x_next, parameter_0, parameter_1) - aux
        return x_next, dwork

    def forward(
        self,
        x: torch.Tensor,
        beta_start: torch.Tensor,
        beta_target: torch.Tensor,
        nf_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        work_dtype = x.real.dtype if torch.is_complex(x) else x.dtype
        if nf_only:
            delta_beta = beta_target - beta_start
            return self._run_nf_block(x, beta_start, beta_target, delta_beta)

        x_curr = x
        parameter_delta = beta_target - beta_start
        work = torch.zeros(x.shape[0], device=x.device, dtype=work_dtype)

        for n in range(self.n_protocol_steps):
            lambda_0 = n / self.n_protocol_steps
            lambda_1 = (n + 1) / self.n_protocol_steps
            parameter_0 = beta_start + lambda_0 * parameter_delta
            parameter_1 = beta_start + lambda_1 * parameter_delta
            delta_parameter = parameter_1 - parameter_0

            if not self.nf_first:
                x_curr = self._run_stochastic_blocks(x_curr, parameter_0)

            x_next, dwork = self._run_nf_block(
                x_curr,
                parameter_0,
                parameter_1,
                delta_parameter,
            )
            work = work + dwork
            x_curr = x_next

            if self.nf_first:
                x_curr = self._run_stochastic_blocks(x_curr, parameter_1)

        return x_curr, work


class NEOREX(OutOfEquilibriumParallelTempering):
    """
    Neural nonequilibrium replica exchange (NEO-REX).

    A shared DDP-wrapped SNF protocol proposes the out-of-equilibrium
    transformation. During adaptation, the same computed work is used for the
    global work loss and then detached for the Metropolis accept/reject step.
    In production, call with train_flow=False.
    """

    def __init__(
        self,
        flow: nn.Module,
        config: NEOREXConfig,
        rank: int,
        world_size: int,
        device: torch.device,
        optimizer_cls=torch.optim.Adam,
        optimizer_param_groups: Optional[Sequence[dict]] = None,
    ):
        super().__init__(config, rank, world_size, device)
        self.config: NEOREXConfig = config
        self.flow = flow.to(device)
        if device.type == "cuda":
            self.ddp_flow = DDP(self.flow, device_ids=[device.index])
        else:
            self.ddp_flow = DDP(self.flow)

        if optimizer_param_groups is None:
            self.optimizer = optimizer_cls(self.ddp_flow.parameters(), lr=config.flow_lr)
        else:
            self.optimizer = optimizer_cls(optimizer_param_groups)
        self.consecutive_nan_grad_updates = 0

    @torch.enable_grad()
    def neorex_step(self, x: torch.Tensor, train_flow: bool = True) -> tuple[torch.Tensor, dict]:
        if int(x.shape[0]) != self.B:
            raise ValueError(f"x batch dimension must be {self.B}, got {x.shape[0]}")

        schedule = self.config.orex_schedule
        if schedule == "even_odd":
            total_stats = self._empty_neorex_stats()
            for parity in ("even", "odd"):
                x, stats = self._neorex_step_parity(x, parity, train_flow=train_flow)
                total_stats = self._merge_neorex_stats(total_stats, stats)
            self._pt_step += 1
            return x, self._finalize_neorex_stats(total_stats)
        if schedule == "alternating":
            parity = "even" if self._pt_step % 2 == 0 else "odd"
        else:
            parity = schedule

        x_out, stats = self._neorex_step_parity(x, parity, train_flow=train_flow)
        self._pt_step += 1
        return x_out, self._finalize_neorex_stats(stats)

    def _empty_neorex_stats(self) -> dict:
        n_links = max(0, self.world_size - 1)
        stats = self._empty_stats()
        stats.update(
            {
                "_flow_loss_sum": 0.0,
                "_flow_loss_count": 0,
                "_flow_loss_link_sum": [0.0] * n_links,
                "_flow_loss_link_count": [0] * n_links,
                "applied_updates": 0,
                "skipped_nan_grad_updates": 0,
            }
        )
        return stats

    def _merge_neorex_stats(self, left: dict, right: dict) -> dict:
        n_links = max(0, self.world_size - 1)
        left_link_sum = left.get("_flow_loss_link_sum", [0.0] * n_links)
        right_link_sum = right.get("_flow_loss_link_sum", [0.0] * n_links)
        left_link_count = left.get("_flow_loss_link_count", [0] * n_links)
        right_link_count = right.get("_flow_loss_link_count", [0] * n_links)
        merged = self._merge_stats(left, right)
        merged.update(
            {
                "_flow_loss_sum": float(left.get("_flow_loss_sum", 0.0))
                + float(right.get("_flow_loss_sum", 0.0)),
                "_flow_loss_count": int(left.get("_flow_loss_count", 0))
                + int(right.get("_flow_loss_count", 0)),
                "_flow_loss_link_sum": [
                    float(left_link_sum[k]) + float(right_link_sum[k])
                    for k in range(n_links)
                ],
                "_flow_loss_link_count": [
                    int(left_link_count[k]) + int(right_link_count[k])
                    for k in range(n_links)
                ],
                "applied_updates": int(left.get("applied_updates", 0))
                + int(right.get("applied_updates", 0)),
                "skipped_nan_grad_updates": int(left.get("skipped_nan_grad_updates", 0))
                + int(right.get("skipped_nan_grad_updates", 0)),
            }
        )
        return merged

    def _finalize_neorex_stats(self, stats: dict) -> dict:
        finalized = self._finalize_stats(stats)
        loss_count = int(stats.get("_flow_loss_count", 0))
        n_links = max(0, self.world_size - 1)
        link_sum = [float(v) for v in stats.get("_flow_loss_link_sum", [0.0] * n_links)]
        link_count = [int(v) for v in stats.get("_flow_loss_link_count", [0] * n_links)]
        finalized.update(
            {
                "flow_loss": (float(stats.get("_flow_loss_sum", 0.0)) / loss_count)
                if loss_count
                else float("nan"),
                "flow_loss_count": loss_count,
                "flow_loss_link_mean": [
                    (link_sum[k] / link_count[k]) if link_count[k] > 0 else float("nan")
                    for k in range(n_links)
                ],
                "flow_loss_link_count": link_count,
                "applied_updates": int(stats.get("applied_updates", 0)),
                "skipped_nan_grad_updates": int(stats.get("skipped_nan_grad_updates", 0)),
                "consecutive_nan_grad_updates": int(self.consecutive_nan_grad_updates),
            }
        )
        return finalized

    def _neorex_step_parity(
        self,
        x: torch.Tensor,
        parity: str,
        train_flow: bool,
    ) -> tuple[torch.Tensor, dict]:
        x_old = x.detach().clone()
        start_betas = self.local_parameters.clone()
        target_betas, active = self._prepare_orex_targets(parity)

        if bool(train_flow):
            x_proposed, local_work, train_stats = self._train_flow_local_protocol(
                x,
                start_betas,
                target_betas,
                active,
            )
        else:
            n_links = max(0, self.world_size - 1)
            with torch.no_grad():
                x_proposed, local_work = self._eval_flow_local_protocol(
                    x,
                    start_betas,
                    target_betas,
                    active,
                )
            train_stats = {
                "_flow_loss_sum": 0.0,
                "_flow_loss_count": 0,
                "_flow_loss_link_sum": [0.0] * n_links,
                "_flow_loss_link_count": [0] * n_links,
                "applied_updates": 0,
                "skipped_nan_grad_updates": 0,
            }

        active_view = self._sample_mask_view(active, x)
        x_proposed = torch.where(active_view, x_proposed.detach(), x_old)
        local_work = torch.where(
            active,
            local_work.detach().to(dtype=self.param_dtype),
            torch.zeros_like(local_work.detach(), dtype=self.param_dtype),
        )
        accept, stats = self._accept_orex_work(local_work)
        stats.update(train_stats)

        accept_view = self._sample_mask_view(accept, x)
        x_out = torch.where(accept_view, x_proposed, x_old)

        self._scatter_parameters()
        return x_out, stats

    def _run_stochastic_blocks_active(
        self,
        module: nn.Module,
        x: torch.Tensor,
        parameter: torch.Tensor,
        active_idx: torch.Tensor,
    ) -> torch.Tensor:
        if active_idx.numel() == 0:
            return x.detach()
        x_active = x.index_select(0, active_idx)
        parameter_active = parameter.index_select(0, active_idx)
        x_updated = module._run_stochastic_blocks(x_active, parameter_active)
        out = x.detach().clone()
        out.index_copy_(0, active_idx, x_updated.detach())
        return out

    def _run_nf_block_active(
        self,
        x: torch.Tensor,
        parameter_0: torch.Tensor,
        parameter_1: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        active_idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
        x_active = x.index_select(0, active_idx)
        parameter_0_active = parameter_0.index_select(0, active_idx)
        parameter_1_active = parameter_1.index_select(0, active_idx)

        x_next_active, dwork_active = self.ddp_flow(
            x_active,
            parameter_0_active,
            parameter_1_active,
            nf_only=True,
        )
        zero_anchor = dwork_active.sum() * 0.0
        dwork = torch.zeros(x.shape[0], device=x.device, dtype=self.param_dtype) + zero_anchor
        if active_idx.numel() == 0:
            return x, dwork

        dwork = dwork.index_copy(0, active_idx, dwork_active.to(dtype=self.param_dtype))
        x_next = x.clone()
        x_next.index_copy_(0, active_idx, x_next_active)
        return x_next, dwork

    def _eval_flow_local_protocol(
        self,
        x: torch.Tensor,
        start_betas: torch.Tensor,
        target_betas: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        module = self.ddp_flow.module
        x_curr = x
        beta_delta = target_betas - start_betas
        total_work = torch.zeros(x.shape[0], device=x.device, dtype=self.param_dtype)
        active_idx = torch.nonzero(active, as_tuple=False).squeeze(-1)

        for n in range(int(module.n_protocol_steps)):
            lambda_0 = n / int(module.n_protocol_steps)
            lambda_1 = (n + 1) / int(module.n_protocol_steps)
            beta_0 = start_betas + lambda_0 * beta_delta
            beta_1 = start_betas + lambda_1 * beta_delta

            if not bool(module.nf_first):
                x_curr = self._run_stochastic_blocks_active(module, x_curr, beta_0, active_idx)

            x_next, dwork = self._run_nf_block_active(x_curr, beta_0, beta_1, active)
            total_work = total_work + dwork.detach().to(dtype=self.param_dtype)
            x_curr = x_next.detach()

            if bool(module.nf_first):
                x_curr = self._run_stochastic_blocks_active(module, x_curr, beta_1, active_idx)

        return x_curr, total_work

    def _train_flow_local_protocol(
        self,
        x: torch.Tensor,
        start_betas: torch.Tensor,
        target_betas: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        module = self.ddp_flow.module
        x_curr = x
        beta_delta = target_betas - start_betas
        total_work = torch.zeros(x.shape[0], device=x.device, dtype=self.param_dtype)
        active_idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
        total_stats = {
            "_flow_loss_sum": 0.0,
            "_flow_loss_count": 0,
            "_flow_loss_link_sum": [0.0] * max(0, self.world_size - 1),
            "_flow_loss_link_count": [0] * max(0, self.world_size - 1),
            "applied_updates": 0,
            "skipped_nan_grad_updates": 0,
        }

        for n in range(int(module.n_protocol_steps)):
            lambda_0 = n / int(module.n_protocol_steps)
            lambda_1 = (n + 1) / int(module.n_protocol_steps)
            beta_0 = start_betas + lambda_0 * beta_delta
            beta_1 = start_betas + lambda_1 * beta_delta

            if not bool(module.nf_first):
                x_curr = self._run_stochastic_blocks_active(module, x_curr, beta_0, active_idx)

            x_in = x_curr.detach()
            x_next, dwork = self._run_nf_block_active(x_in, beta_0, beta_1, active)
            step_stats = self._train_flow_from_work(dwork, active)
            total_stats["applied_updates"] += int(step_stats.get("applied_updates", 0))
            total_stats["skipped_nan_grad_updates"] += int(
                step_stats.get("skipped_nan_grad_updates", 0)
            )

            total_work = total_work + dwork.detach().to(dtype=self.param_dtype)
            x_curr = x_next.detach()

            if bool(module.nf_first):
                x_curr = self._run_stochastic_blocks_active(module, x_curr, beta_1, active_idx)

        link_loss_sum, link_loss_count = self._flow_loss_link_moments(total_work.detach(), active)
        total_stats["_flow_loss_link_sum"] = link_loss_sum
        total_stats["_flow_loss_link_count"] = link_loss_count
        total_stats["_flow_loss_sum"] = float(sum(link_loss_sum))
        total_stats["_flow_loss_count"] = int(sum(link_loss_count))
        return x_curr, total_work, total_stats

    def _train_flow_from_work(self, local_work: torch.Tensor, active: torch.Tensor) -> dict:
        n_links = max(0, self.world_size - 1)
        if not any(p.requires_grad for p in self.ddp_flow.parameters()):
            return {
                "_flow_loss_sum": 0.0,
                "_flow_loss_count": 0,
                "_flow_loss_link_sum": [0.0] * n_links,
                "_flow_loss_link_count": [0] * n_links,
                "applied_updates": 0,
                "skipped_nan_grad_updates": 0,
            }

        self.optimizer.zero_grad(set_to_none=True)
        finite_active = active & torch.isfinite(local_work.detach())
        local_count = finite_active.to(dtype=torch.long).sum()
        global_count = local_count.clone()
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
        global_count_int = int(global_count.item())

        if global_count_int <= 0:
            return {
                "_flow_loss_sum": 0.0,
                "_flow_loss_count": 0,
                "_flow_loss_link_sum": [0.0] * n_links,
                "_flow_loss_link_count": [0] * n_links,
                "applied_updates": 0,
                "skipped_nan_grad_updates": 0,
            }

        local_sum = torch.where(
            finite_active,
            local_work,
            torch.zeros_like(local_work),
        ).sum()
        loss = local_sum * (self.world_size / float(global_count_int))

        loss.backward()
        has_nonfinite_grads, grad_error_message = self._check_nonfinite_flow_gradients()
        if has_nonfinite_grads:
            self.optimizer.zero_grad(set_to_none=True)
            self.consecutive_nan_grad_updates += 1
            if self.rank == 0:
                print(
                    "Warning: skipped NEO-REX flow update due to non-finite gradients "
                    f"(consecutive_nan_grad_updates={self.consecutive_nan_grad_updates}/"
                    f"{int(self.config.max_consecutive_nan_grad_updates)}). "
                    f"{grad_error_message}",
                    flush=True,
                )
            if self.consecutive_nan_grad_updates >= int(self.config.max_consecutive_nan_grad_updates):
                raise FloatingPointError(
                    "Stopping NEO-REX after too many consecutive non-finite-gradient updates: "
                    f"{self.consecutive_nan_grad_updates} >= "
                    f"{int(self.config.max_consecutive_nan_grad_updates)}. "
                    f"Last error: {grad_error_message}"
                )
            return {
                "_flow_loss_sum": 0.0,
                "_flow_loss_count": 0,
                "_flow_loss_link_sum": [0.0] * n_links,
                "_flow_loss_link_count": [0] * n_links,
                "applied_updates": 0,
                "skipped_nan_grad_updates": 1,
            }

        self.consecutive_nan_grad_updates = 0
        if float(self.config.grad_clip_norm) > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.ddp_flow.parameters(),
                max_norm=float(self.config.grad_clip_norm),
            )
        self.optimizer.step()

        return {
            "_flow_loss_sum": 0.0,
            "_flow_loss_count": 0,
            "_flow_loss_link_sum": [0.0] * n_links,
            "_flow_loss_link_count": [0] * n_links,
            "applied_updates": 1,
            "skipped_nan_grad_updates": 0,
        }

    @torch.no_grad()
    def _flow_loss_link_moments(self, local_work: torch.Tensor, active: torch.Tensor):
        n_links = max(0, self.world_size - 1)
        if n_links == 0:
            return [], []

        local_work = local_work.detach().to(dtype=self.param_dtype)
        active_u8 = active.to(dtype=torch.uint8)

        if self.rank == 0:
            work_gather = [
                torch.zeros(self.B, device=self.device, dtype=self.param_dtype)
                for _ in range(self.world_size)
            ]
            active_gather = [
                torch.zeros(self.B, device=self.device, dtype=torch.uint8)
                for _ in range(self.world_size)
            ]
        else:
            work_gather = None
            active_gather = None

        dist.gather(local_work, gather_list=work_gather, dst=0)
        dist.gather(active_u8, gather_list=active_gather, dst=0)

        link_sum_t = torch.zeros(n_links, device=self.device, dtype=self.param_dtype)
        link_count_t = torch.zeros(n_links, device=self.device, dtype=torch.long)

        if self.rank == 0:
            work_matrix = torch.stack(work_gather, dim=1)
            active_matrix = torch.stack(active_gather, dim=1).to(dtype=torch.bool)
            b_idx = torch.arange(self.B, device=self.device)

            for link_idx, node_i, node_j in self._pending_pairs:
                wi = work_matrix[b_idx, node_i]
                wj = work_matrix[b_idx, node_j]
                ai = active_matrix[b_idx, node_i]
                aj = active_matrix[b_idx, node_j]
                fi = ai & torch.isfinite(wi)
                fj = aj & torch.isfinite(wj)

                if fi.any():
                    link_sum_t[int(link_idx)] += wi[fi].sum()
                    link_count_t[int(link_idx)] += fi.to(dtype=torch.long).sum()
                if fj.any():
                    link_sum_t[int(link_idx)] += wj[fj].sum()
                    link_count_t[int(link_idx)] += fj.to(dtype=torch.long).sum()

        dist.broadcast(link_sum_t, src=0)
        dist.broadcast(link_count_t, src=0)

        return (
            [float(v.item()) for v in link_sum_t],
            [int(v.item()) for v in link_count_t],
        )

    @staticmethod
    def _finite_summary(tensor: torch.Tensor) -> dict:
        with torch.no_grad():
            finite = torch.isfinite(tensor)
            n_bad = int((~finite).sum().item())
            total = int(tensor.numel())
            finite_vals = tensor[finite]
            if finite_vals.numel() == 0:
                return {
                    "bad": n_bad,
                    "total": total,
                    "min": float("nan"),
                    "max": float("nan"),
                    "absmax": float("nan"),
                    "l2": float("nan"),
                }
            return {
                "bad": n_bad,
                "total": total,
                "min": float(finite_vals.min().item()),
                "max": float(finite_vals.max().item()),
                "absmax": float(finite_vals.abs().max().item()),
                "l2": float(torch.linalg.vector_norm(finite_vals).item()),
            }

    def _check_nonfinite_flow_gradients(self):
        local_bad = torch.zeros((), device=self.device, dtype=torch.int32)
        bad_name = None
        bad_grad = None
        bad_param = None

        for name, param in self.ddp_flow.named_parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all().item():
                local_bad.fill_(1)
                bad_name = name
                bad_grad = param.grad.detach()
                bad_param = param.detach()
                break

        global_bad = local_bad.clone()
        dist.all_reduce(global_bad, op=dist.ReduceOp.SUM)
        if int(global_bad.item()) == 0:
            return False, None

        if bad_name is None:
            return True, (
                f"rank {self.rank}: non-finite gradients detected on another rank; "
                f"bad ranks={int(global_bad.item())}/{self.world_size}"
            )

        grad_stats = self._finite_summary(bad_grad)
        param_stats = self._finite_summary(bad_param)
        return True, (
            f"rank {self.rank}: non-finite gradient in `{bad_name}`; "
            f"grad bad={grad_stats['bad']}/{grad_stats['total']}, "
            f"grad finite range=[{grad_stats['min']:.6g}, {grad_stats['max']:.6g}], "
            f"grad absmax={grad_stats['absmax']:.6g}, grad l2={grad_stats['l2']:.6g}; "
            f"param bad={param_stats['bad']}/{param_stats['total']}, "
            f"param finite range=[{param_stats['min']:.6g}, {param_stats['max']:.6g}], "
            f"param absmax={param_stats['absmax']:.6g}; "
            f"bad ranks={int(global_bad.item())}/{self.world_size}"
        )

    def save_flow(self, path: str):
        if self.rank == 0:
            torch.save(self.ddp_flow.module.state_dict(), path)

    def load_flow(self, path: str):
        if self.rank == 0:
            map_loc = {"cpu": "cpu"} if self.device.type == "cpu" else None
            state = torch.load(path, map_location=map_loc)
            self.ddp_flow.module.load_state_dict(state)
        for p in self.ddp_flow.parameters():
            dist.broadcast(p.data, src=0)


 ####################################################################
 ########### Neural Training Parallel Tempering (PT) ################
 ####################################################################

@dataclass
class NeuralPTConfig(PTConfig):
    # Keep neural PT backward-compatible with existing energy-based workflows.
    acceptance_rule: str = "linear"
    lr: float = 1e-3
    # number of network updates per PT step
    net_updates_per_step: int = 1
    # local chunk size for gradient accumulation (<=0 disables)
    chunk_size: int = 0
    # global gradient-norm clipping (<=0 disables)
    grad_clip_norm: float = 0.0
    # stop training after this many consecutive updates with non-finite gradients
    max_consecutive_nan_grad_updates: int = 5

class NeuralParallelTempering(ParallelTempering):
    """
    Parallel Tempering with a shared neural network across all ranks.

    The network receives (x, parameter) as input and is updated at every step
    via a user-defined loss function.

    Args:
        net:            nn.Module, not yet wrapped with DDP
        loss_fn:        Callable(net_output, x, local_parameters) -> scalar tensor
        config:         NeuralPTConfig
        rank / world_size / device: standard distributed arguments
        optimizer_cls:  optimizer class (default: Adam)
    """

    def __init__(
        self,
        net: nn.Module,
        config: NeuralPTConfig,
        rank: int,
        world_size: int,
        device: torch.device,
        optimizer_cls=torch.optim.Adam,
        optimizer_param_groups: Optional[Sequence[dict]] = None,
    ):
        super().__init__(config, rank, world_size, device)

        self.config: NeuralPTConfig = config

        # move model to the correct device before wrapping with DDP
        self.net = net.to(device)

        # pass device_ids only for NCCL (GPU); gloo (CPU) does not need it
        if device.type == "cuda":
            self.ddp_net = DDP(self.net, device_ids=[device.index])
        else:
            self.ddp_net = DDP(self.net)

        if optimizer_param_groups is None:
            self.optimizer = optimizer_cls(self.ddp_net.parameters(), lr=config.lr)
        else:
            self.optimizer = optimizer_cls(optimizer_param_groups)
        self.consecutive_nan_grad_updates = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def neural_step(
        self,
        x: torch.Tensor,                # (B, ...) local configurations
        local_energies: torch.Tensor,   # (B,)     for the PT swap criterion
        do_pt_swap: bool = True,
    ) -> dict:
        """
        Combined step: network update + optional PT replica swap.

        1. Updates the network on (x, local_parameters)  [net_updates_per_step times]
        2. Optionally performs a PT swap and scatters the new labels

        Returns:
            dict with keys: loss, loss_unavg, n_accepted, n_proposed
        """
        net_loss, loss_unavg, applied_updates, skipped_nan_updates = self._update_net(x)

        # Safety: all ranks must agree, otherwise collectives in self.step() deadlock.
        self._assert_global_flag_consensus(do_pt_swap, name="do_pt_swap")
        pt_stats = {"n_accepted": 0, "n_proposed": 0}
        if do_pt_swap:
            pt_stats = self.step(local_energies)   # inherited from ParallelTempering

        return {
            "loss": net_loss,
            "loss_unavg": loss_unavg,
            "applied_updates": int(applied_updates),
            "skipped_nan_grad_updates": int(skipped_nan_updates),
            "consecutive_nan_grad_updates": int(self.consecutive_nan_grad_updates),
            **pt_stats,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _finite_summary(tensor: torch.Tensor) -> dict:
        with torch.no_grad():
            finite = torch.isfinite(tensor)
            n_bad = int((~finite).sum().item())
            total = int(tensor.numel())
            finite_vals = tensor[finite]
            if finite_vals.numel() == 0:
                return {
                    "bad": n_bad,
                    "total": total,
                    "min": float("nan"),
                    "max": float("nan"),
                    "absmax": float("nan"),
                    "l2": float("nan"),
                }
            return {
                "bad": n_bad,
                "total": total,
                "min": float(finite_vals.min().item()),
                "max": float(finite_vals.max().item()),
                "absmax": float(finite_vals.abs().max().item()),
                "l2": float(torch.linalg.vector_norm(finite_vals).item()),
            }

    def _check_nonfinite_gradients(self):
        local_bad = torch.zeros((), device=self.device, dtype=torch.int32)
        bad_name = None
        bad_grad = None
        bad_param = None

        for name, param in self.ddp_net.named_parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all().item():
                local_bad.fill_(1)
                bad_name = name
                bad_grad = param.grad.detach()
                bad_param = param.detach()
                break

        global_bad = local_bad.clone()
        dist.all_reduce(global_bad, op=dist.ReduceOp.SUM)
        if int(global_bad.item()) == 0:
            return False, None

        if bad_name is None:
            return True, (
                f"rank {self.rank}: non-finite gradients detected on another rank; "
                f"bad ranks={int(global_bad.item())}/{self.world_size}"
            )

        grad_stats = self._finite_summary(bad_grad)
        param_stats = self._finite_summary(bad_param)
        return True, (
            f"rank {self.rank}: non-finite gradient in `{bad_name}`; "
            f"grad bad={grad_stats['bad']}/{grad_stats['total']}, "
            f"grad finite range=[{grad_stats['min']:.6g}, {grad_stats['max']:.6g}], "
            f"grad absmax={grad_stats['absmax']:.6g}, grad l2={grad_stats['l2']:.6g}; "
            f"param bad={param_stats['bad']}/{param_stats['total']}, "
            f"param finite range=[{param_stats['min']:.6g}, {param_stats['max']:.6g}], "
            f"param absmax={param_stats['absmax']:.6g}; "
            f"loss parameter range=[{self.local_parameters.min().item():.6g}, {self.local_parameters.max().item():.6g}], "
            f"delta_parameter={self.delta_parameter.item():.6g}, "
            f"bad ranks={int(global_bad.item())}/{self.world_size}"
        )

    def _update_net(self, x: torch.Tensor):
        """
        Forward + backward + optimizer step, repeated net_updates_per_step times.
        Returns:
            (global_mean_loss: float, local_loss_unavg: torch.Tensor)
        """
        total_loss = 0.0
        loss_unavg_accum = None
        applied_updates = 0
        skipped_nan_updates = 0
        batch_size = int(x.shape[0])
        use_chunking = bool(self.config.chunk_size and int(self.config.chunk_size) > 0)
        chunk_size = int(self.config.chunk_size) if use_chunking else batch_size
        chunk_size = max(1, min(batch_size, chunk_size))
        last_loss_unavg = None

        for _ in range(self.config.net_updates_per_step):
            self.optimizer.zero_grad()

            if chunk_size >= batch_size:
                # local_parameters shape: (B,) — exact broadcasting depends on your architecture
                loss_unavg = self.ddp_net(x, self.local_parameters, self.delta_parameter)
                loss = loss_unavg.mean()  # DDP requires a scalar loss; it will average across ranks
                # DDP automatically averages gradients across all ranks
                loss.backward()
            else:
                perm = torch.randperm(batch_size, device=x.device)
                chunks = perm.split(chunk_size)
                n_chunks = len(chunks)
                loss_unavg = None
                for i, idx in enumerate(chunks):
                    sync_ctx = self.ddp_net.no_sync() if i < (n_chunks - 1) else nullcontext()
                    with sync_ctx:
                        loss_unavg_chunk = self.ddp_net(
                            x[idx], self.local_parameters[idx], self.delta_parameter
                        )
                        if loss_unavg is None:
                            loss_unavg = torch.empty(
                                (batch_size,),
                                device=loss_unavg_chunk.device,
                                dtype=loss_unavg_chunk.dtype,
                            )
                        loss_unavg[idx] = loss_unavg_chunk.detach()
                        (loss_unavg_chunk.mean() / n_chunks).backward()
            last_loss_unavg = loss_unavg.detach().clone()

            has_nonfinite_grads, grad_error_message = self._check_nonfinite_gradients()
            if has_nonfinite_grads:
                self.optimizer.zero_grad(set_to_none=True)
                self.consecutive_nan_grad_updates += 1
                skipped_nan_updates += 1
                if self.rank == 0:
                    print(
                        "Warning: skipped optimizer update due to non-finite gradients "
                        f"(consecutive_nan_grad_updates={self.consecutive_nan_grad_updates}/"
                        f"{int(self.config.max_consecutive_nan_grad_updates)}). "
                        f"{grad_error_message}",
                        flush=True,
                    )
                if self.consecutive_nan_grad_updates >= int(self.config.max_consecutive_nan_grad_updates):
                    raise FloatingPointError(
                        "Stopping run after too many consecutive non-finite-gradient updates: "
                        f"{self.consecutive_nan_grad_updates} >= "
                        f"{int(self.config.max_consecutive_nan_grad_updates)}. "
                        f"Last error: {grad_error_message}"
                    )
                # Skip the remaining net updates for this PT step and continue with next MCMC step.
                break

            self.consecutive_nan_grad_updates = 0
            if float(self.config.grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.ddp_net.parameters(),
                    max_norm=float(self.config.grad_clip_norm),
                )
            self.optimizer.step()
            applied_updates += 1

            # Log a globally consistent scalar loss (mean over ranks).
            loss_detached = loss.detach() if chunk_size >= batch_size else loss_unavg.mean().detach()
            dist.all_reduce(loss_detached, op=dist.ReduceOp.SUM)
            loss_detached = loss_detached / self.world_size
            total_loss += loss_detached.item()
            if loss_unavg_accum is None:
                loss_unavg_accum = loss_unavg.detach().clone()
            else:
                loss_unavg_accum += loss_unavg.detach()
        if applied_updates > 0:
            loss_unavg_accum = loss_unavg_accum / applied_updates
            mean_loss = total_loss / applied_updates
        else:
            mean_loss = float("nan")
            if loss_unavg_accum is None:
                if last_loss_unavg is not None:
                    loss_unavg_accum = last_loss_unavg
                else:
                    loss_unavg_accum = torch.full((batch_size,), float("nan"), device=x.device, dtype=x.dtype)
        return mean_loss, loss_unavg_accum, applied_updates, skipped_nan_updates

    def _assert_global_flag_consensus(self, flag: bool, name: str = "flag"):
        """
        Ensure a boolean runtime flag is identical across ranks.
        Needed when downstream code performs collectives conditionally.
        """
        v = torch.tensor(1 if flag else 0, device=self.device, dtype=torch.int32)
        v_min = v.clone()
        v_max = v.clone()
        dist.all_reduce(v_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(v_max, op=dist.ReduceOp.MAX)
        if v_min.item() != v_max.item():
            raise RuntimeError(
                f"Inconsistent `{name}` across ranks: collectives would deadlock."
            )

    # ------------------------------------------------------------------
    # Checkpoint helpers (rank 0 only)
    # ------------------------------------------------------------------

    def save_net(self, path: str):
        """Save the underlying module state dict from rank 0."""
        if self.rank == 0:
            torch.save(self.ddp_net.module.state_dict(), path)

    def load_net(self, path: str):
        """Load checkpoint on rank 0, then broadcast parameters to all ranks."""
        if self.rank == 0:
            map_loc = {"cpu": "cpu"} if self.device.type == "cpu" else None
            state = torch.load(path, map_location=map_loc)
            self.ddp_net.module.load_state_dict(state)
        # explicit broadcast from rank 0 to guarantee all ranks use the same weights
        for p in self.ddp_net.parameters():
            dist.broadcast(p.data, src=0)
