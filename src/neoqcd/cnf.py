"""Clover/chair gauge-equivariant continuous flows on SU(3) lattice links.

The drift is represented in the orthonormal Hermitian basis T_a=lambda_a/2,
with left velocity dU/dt = i F_a T_a U.  Its Haar divergence is the trace of
the eight left-invariant directional derivatives of F, with the loop paths
held fixed because paths that wrap back onto their own link are excluded.
"""

import itertools
import math
from functools import lru_cache

import torch
from torch import nn
from torch.func import functional_call

from . import sun_utils as sun
from .smearing import _su3_generators


def _pairs(n, strict=False):
    return [(i, j) for i in range(n) for j in range(i + int(strict), n)]


def _shift(field, direction, amount):
    # Keep the repository's convention: a positive shift is torch.roll(+1).
    return torch.roll(field, shifts=amount, dims=1 + direction)


def _link(U, direction, sign):
    link = U[:, direction]
    return link if sign > 0 else sun.SUN_dagger(_shift(link, direction, -1))


def _path_product(U, steps):
    offset = [0] * U.shape[1]
    product = None
    for direction, sign in steps:
        link = _link(U, direction, sign)
        for axis, amount in enumerate(offset):
            if amount:
                link = _shift(link, axis, amount)
        product = link if product is None else sun.SUN_mul(product, link)
        offset[direction] += sign
    return product


def _uses_periodic_link(steps, direction, start, extents):
    """Whether a path on this torus traverses the positive link at the origin."""
    position = list(start)
    for axis, sign in steps:
        base = position.copy()
        if sign < 0:
            base[axis] += 1
        if axis == direction and all(base[i] % extents[i] == 0 for i in range(len(extents))):
            return True
        position[axis] -= sign
    return False


@lru_cache(maxsize=None)
def _clover_paths(dimension):
    paths = []
    for mu in range(dimension):
        for nu in range(mu + 1, dimension):
            a, b = (mu, 1), (nu, 1)
            am, bm = (mu, -1), (nu, -1)
            paths.extend(([a, b, am, bm], [b, am, bm, a], [am, bm, a, b], [bm, a, b, am]))
    return tuple(tuple(path) for path in paths)


@lru_cache(maxsize=None)
def _rectangle_paths(dimension, mu, loop_max):
    paths = []
    for nu in range(dimension):
        if nu == mu:
            continue
        for sign in (1, -1):
            for a in range(1, loop_max + 1):
                for b in range(1, loop_max + 1):
                    if a == b == 1:
                        continue
                    for j in range(a):
                        paths.append(
                            [(mu, 1)] * (a - 1 - j)
                            + [(nu, sign)] * b
                            + [(mu, -1)] * a
                            + [(nu, -sign)] * b
                            + [(mu, 1)] * j
                        )
    return tuple(tuple(path) for path in paths)


@lru_cache(maxsize=None)
def _chair_paths(dimension, mu):
    directions = [(axis, sign) for axis in range(dimension) for sign in (1, -1)]
    paths = []
    for steps in itertools.product(directions, repeat=5):
        position = [0] * dimension
        position[mu] = 1
        previous = (mu, 1)
        used = {((0,) * dimension, mu)}
        valid = True
        for axis, sign in steps:
            base = position.copy()
            if sign < 0:
                base[axis] -= 1
            if (axis, sign) == (previous[0], -previous[1]) or (tuple(base), axis) in used:
                valid = False
                break
            used.add((tuple(base), axis))
            position[axis] += sign
            previous = (axis, sign)
        if valid and not any(position) and len({axis for axis, _ in steps} | {mu}) == 3:
            paths.append(list(steps))
    return tuple(tuple(path) for path in paths)


class _LayerNorm(nn.Module):
    """Elementary LayerNorm supports the higher derivatives in the PG loss."""

    def __init__(self, size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        variance = (x - mean).square().mean(-1, keepdim=True)
        return (x - mean) * torch.rsqrt(variance + self.eps) * self.weight + self.bias


class CloverChairLinkDrift(nn.Module):
    """SU(3) version of CloverChairLinkDrift, without coarse or conditional inputs."""

    def __init__(self, D, hidden_dim=16, depth=2, loop_max=2, zero_init=True, time_range=None):
        super().__init__()
        self.D = int(D)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.loop_max = int(loop_max)
        if self.D < 2:
            raise ValueError("CloverChairLinkDrift requires D >= 2")
        if self.hidden_dim < 1 or self.depth < 0 or self.loop_max < 1:
            raise ValueError("hidden_dim and loop_max must be positive; depth must be non-negative")
        # The time encoder sees (t - t_min) / (t_max - t_min); None keeps the raw parameter.
        if time_range is not None:
            time_range = tuple(float(v) for v in time_range)
            if len(time_range) != 2 or not all(math.isfinite(v) for v in time_range):
                raise ValueError("time_range must be two finite values")
            if not time_range[1] > time_range[0]:
                raise ValueError("time_range must satisfy t_min < t_max")
        self.time_range = time_range

        # Keep complex generators as ordinary attributes: Module.to(dtype=...) must not cast them.
        self._generators = _su3_generators(dtype=torch.cdouble, device="cpu")
        commutator = torch.einsum("aij,bjk->abik", self._generators, self._generators)
        commutator = commutator - commutator.transpose(0, 1)
        trace = torch.einsum("abij,cji->abc", commutator, self._generators)
        self._structure_constants = (-2j * trace).real

        self.k0 = 2 * (self.D - 1)
        self.n_rectangles = 2 * (self.D - 1) * (sum(range(1, self.loop_max + 1)) * self.loop_max - 1)
        self.n_chairs = len(_chair_paths(self.D, 0))
        self.K = self.k0 + self.n_rectangles + self.n_chairs
        self.n_leaves = 2 * (self.D - 1) ** 2
        self.n_vec = self.K + self.n_leaves
        self._rect_paths = tuple(_rectangle_paths(self.D, mu, self.loop_max) for mu in range(self.D))
        self._chair_path_data = tuple(_chair_paths(self.D, mu) for mu in range(self.D))
        clover_paths = _clover_paths(self.D)
        self.free_leaf_indices = tuple(
            tuple(i for i, path in enumerate(clover_paths)
                  if not _uses_periodic_link(path, mu, (0,) * self.D, (10**9,) * self.D))
            for mu in range(self.D)
        )

        small_vec = self.k0 + self.n_leaves
        large = range(small_vec, self.n_vec)
        sym = _pairs(small_vec) + [(i, j) for i in range(self.k0) for j in large]
        anti = _pairs(small_vec, strict=True) + [(i, j) for i in range(self.k0) for j in large]
        gram = _pairs(self.k0) + [(i, j) for i in range(self.k0) for j in range(self.k0, self.K)]
        for name, pairs in (("sym", sym), ("anti", anti), ("gram", gram)):
            self.register_buffer(f"_{name}_i", torch.tensor([i for i, _ in pairs]), persistent=False)
            self.register_buffer(f"_{name}_j", torch.tensor([j for _, j in pairs]), persistent=False)
        # Free-leaf lists have the same size by lattice symmetry in the unwrapped lattice.
        self.n_free_leaves = len(self.free_leaf_indices[0])
        if self.n_free_leaves != self.n_leaves:
            raise RuntimeError("unexpected clover-leaf count")

        n_in = len(gram) + self.D + self.n_vec + len(sym)
        n_out = self.n_vec + len(anti)
        self.input_norm = _LayerNorm(n_in)
        self.input_layer = nn.Linear(n_in, self.hidden_dim)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                _LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, 2 * self.hidden_dim), nn.SiLU(),
                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            )
            for _ in range(self.depth)
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_out)
        if zero_init:
            nn.init.zeros_(self.output_layer.weight)
            nn.init.zeros_(self.output_layer.bias)
        self.time_encoder = nn.Sequential(
            nn.Linear(1, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), _LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
        )

    def _validate_U(self, U):
        if not torch.is_tensor(U) or not torch.is_complex(U):
            raise ValueError("U must be a complex SU(3) lattice tensor")
        if U.ndim != self.D + 4 or int(U.shape[1]) != self.D or tuple(U.shape[-2:]) != (3, 3):
            raise ValueError(f"U must have shape (B, {self.D}, L..., 3, 3)")
        if any(int(n) < 1 for n in U.shape[2:-2]):
            raise ValueError("lattice extents must be positive")

    def _time(self, t, U):
        t = torch.as_tensor(t, dtype=U.real.dtype, device=U.device).reshape(-1)
        if t.numel() == 1:
            t = t.expand(U.shape[0]).clone()
        elif t.numel() != U.shape[0]:
            raise ValueError(f"t must be scalar or have one value per chain ({U.shape[0]})")
        return t

    def _staples(self, U):
        extents = tuple(int(n) for n in U.shape[2:-2])
        per_direction = []
        for mu in range(self.D):
            channels = []
            for nu in range(self.D):
                if nu == mu:
                    continue
                for sign in (1, -1):
                    paths = [(nu, sign), (mu, -1), (nu, -sign)]
                    # The path is based at n+mu; the initial link is U_mu(n).
                    start = tuple(-int(i == mu) for i in range(self.D))
                    safe = not _uses_periodic_link(paths, mu, start, extents)
                    channels.append(
                        _shift(_path_product(U, paths), mu, 1) if safe else torch.zeros_like(U[:, mu])
                    )
            for paths in self._rect_paths[mu]:
                safe = not _uses_periodic_link(
                    paths, mu, tuple(-int(i == mu) for i in range(self.D)), extents
                )
                channels.append(
                    _shift(_path_product(U, paths), mu, 1) if safe else torch.zeros_like(U[:, mu])
                )
            for paths in self._chair_path_data[mu]:
                safe = not _uses_periodic_link(
                    paths, mu, tuple(-int(i == mu) for i in range(self.D)), extents
                )
                channels.append(
                    _shift(_path_product(U, paths), mu, 1) if safe else torch.zeros_like(U[:, mu])
                )
            per_direction.append(torch.stack(channels, dim=1))
        return torch.stack(per_direction, dim=1)

    def _leaves(self, U):
        paths = _clover_paths(self.D)
        extents = tuple(int(n) for n in U.shape[2:-2])
        per_direction = []
        for mu in range(self.D):
            channels = []
            indices = self.free_leaf_indices[mu]
            for index in indices:
                path = paths[index]
                safe = not _uses_periodic_link(path, mu, (0,) * self.D, extents)
                channels.append(_path_product(U, path) if safe else torch.zeros_like(U[:, mu]))
            per_direction.append(torch.stack(channels, dim=1))
        return torch.stack(per_direction, dim=1)

    def _basis(self, U):
        generators = self._generators.to(device=U.device, dtype=U.dtype)
        structure = self._structure_constants.to(device=U.device, dtype=U.real.dtype)
        return generators, structure

    @staticmethod
    def _coordinates(W, generators):
        antihermitian = W - sun.SUN_dagger(W)
        hermitian = (-0.5j) * antihermitian
        trace = sun.SUN_trace(hermitian).real / 3.0
        eye = torch.eye(3, dtype=W.dtype, device=W.device)
        hermitian = hermitian - trace[..., None, None] * eye
        channels = W.shape[2]
        flat = hermitian.movedim(2, -3).reshape(-1, channels, 3, 3)
        coeff = 2.0 * torch.einsum("aij,ncji->nca", generators, flat).real
        coeff = coeff.reshape(*W.shape[:2], *W.shape[3:-2], channels, 8)
        scalar = (sun.SUN_trace(W).real / 3.0).movedim(2, -1)
        return scalar, coeff

    def _context_and_vectors(self, U, S, C, generators, structure):
        W = sun.SUN_mul(U.unsqueeze(2), S)
        W = torch.cat((W[:, :, :self.k0], C, W[:, :, self.k0:]), dim=2)
        scalar, vector = self._coordinates(W, generators)
        # Invariant Gram features use the open staples themselves, independent of U_l.
        si = S.index_select(2, self._gram_i)
        sj = S.index_select(2, self._gram_j)
        gram = (sun.SUN_trace(sun.SUN_mul(si, sun.SUN_dagger(sj))).real / 3.0).movedim(2, -1)
        direction = torch.eye(self.D, dtype=U.real.dtype, device=U.device)
        view = (1, self.D) + (1,) * (U.ndim - 4) + (self.D,)
        direction = direction.view(view).expand(U.shape[0], self.D, *U.shape[2:-2], self.D)
        context = torch.cat((gram, direction), dim=-1)
        dot = (vector.index_select(-2, self._sym_i) * vector.index_select(-2, self._sym_j)).sum(-1)
        inputs = torch.cat((context, scalar, dot), dim=-1)
        return inputs, vector, structure

    def _drift_given(self, U, S, C, t, generators, structure):
        inputs, vector, structure = self._context_and_vectors(U, S, C, generators, structure)
        if self.time_range is not None:
            t = (t - self.time_range[0]) / (self.time_range[1] - self.time_range[0])
        t_emb = self.time_encoder(t[:, None]).view(
            U.shape[0], *([1] * (self.D + 1)), self.hidden_dim
        )
        h = torch.nn.functional.silu(self.input_layer(self.input_norm(inputs))) + t_emb
        for block in self.blocks:
            h = h + block(h)
        out = self.output_layer(h)
        a, b = out[..., :self.n_vec], out[..., self.n_vec:]
        anti_i = vector.index_select(-2, self._anti_i)
        anti_j = vector.index_select(-2, self._anti_j)
        cross = torch.einsum("...pa,...pb,abc->...pc", anti_i, anti_j, structure)
        return (a.unsqueeze(-1) * vector).sum(-2) + (b.unsqueeze(-1) * cross).sum(-2)

    def drift(self, U, t):
        self._validate_U(U)
        t = self._time(t, U)
        generators, structure = self._basis(U)
        S, C = self._staples(U), self._leaves(U)
        return self._drift_given(U, S, C, t, generators, structure)

    def forward(self, U, t):
        self._validate_U(U)
        t = self._time(t, U)
        generators, structure = self._basis(U)
        S, C = self._staples(U), self._leaves(U)
        f = lambda value: self._drift_given(value, S, C, t, generators, structure)
        tangent = (1j * generators).view(8, *([1] * (U.ndim - 2)), 3, 3) @ U.unsqueeze(0)
        F, dF = torch.func.vmap(lambda direction: torch.func.jvp(f, (U,), (direction,)))(tangent)
        divergence = sum(dF[a, ..., a] for a in range(8)).flatten(1).sum(1)
        return F[0], divergence

    @property
    def checkpoint_semantics(self):
        return {
            "field": type(self).__name__,
            "D": self.D,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "loop_max": self.loop_max,
            "time_range": self.time_range,
        }


def coeff_to_matrix(coeff):
    """Map real coefficients (..., 8) to the Hermitian Lie-algebra matrix F_a T_a."""
    if coeff.shape[-1] != 8 or coeff.is_complex():
        raise ValueError("coeff must be real with final dimension 8")
    generators = _su3_generators(dtype=torch.cdouble, device=coeff.device).to(
        dtype=torch.complex128 if coeff.dtype == torch.float64 else torch.complex64
    )
    return torch.einsum("...a,aij->...ij", coeff.to(generators.dtype), generators)


def _batch_step(h, F):
    h = torch.as_tensor(h, dtype=F.dtype, device=F.device).reshape(-1)
    if h.numel() == 1:
        h = h.expand(F.shape[0])
    elif h.numel() != F.shape[0]:
        raise ValueError(f"step size must be scalar or have one value per chain ({F.shape[0]})")
    return h.view(F.shape[0], *([1] * (F.ndim - 1)))


def apply_velocity(U, coeff, h=1.0):
    """Apply the left SU(3) exponential exp(i h F_a T_a) to every link."""
    if coeff.shape != U.shape[:-2] + (8,):
        raise ValueError("coeff must have shape U.shape[:-2] + (8,)")
    H = coeff_to_matrix(coeff * _batch_step(h, coeff))
    exponent = 1j * H
    return torch.matmul(torch.matrix_exp(exponent), U)


def lie_tangent(U, coeff):
    """Tangent i F_a T_a U at U (not the finite exponential update)."""
    return torch.matmul(1j * coeff_to_matrix(coeff), U)


def _as_time(t, U):
    t = torch.as_tensor(t, dtype=U.real.dtype, device=U.device).reshape(-1)
    if t.numel() == 1:
        return t.expand(U.shape[0]).clone()
    if t.numel() != U.shape[0]:
        raise ValueError(f"time must be scalar or have one value per chain ({U.shape[0]})")
    return t


def _volume(U, volume):
    volume = math.prod(U.shape[2:-2]) if volume is None else int(volume)
    if volume <= 0:
        raise ValueError("volume must be positive")
    return volume


def path_gradient_residual(model, action_fn, U, t, volume=None):
    """Return (xi, dSdt/V), where xi=(dS/dt-div F+F.grad S)/V.

    ``action_fn(U, t)`` must return one action per chain and be samplewise in its batch.
    """
    V = _volume(U, volume)
    t = _as_time(t, U)
    coeff, divergence = model(U, t)
    velocity = lie_tangent(U, coeff)
    _, dSdt = torch.func.jvp(
        lambda time: action_fn(U, time), (t,), (torch.ones_like(t),)
    )
    if dSdt.shape != (U.shape[0],):
        raise ValueError(f"action_fn must return one value per chain ({U.shape[0]},)")
    _, FgradS = torch.func.jvp(
        lambda value: action_fn(value, t), (U,), (velocity,)
    )
    return (dSdt - divergence + FgradS) / V, dSdt / V


def _frozen_model(model):
    state = {name: value.detach() for name, value in model.named_parameters()}
    state.update(dict(model.named_buffers()))
    return lambda U, t: functional_call(model, state, (U, t))


def _pinn(xi, t, volume):
    # Keep distinct physical parameter values (REX rungs) out of one another's variance.
    groups = torch.unique(t.detach())
    total = xi.new_zeros(())
    count = 0
    for value in groups:
        sample = xi[t.detach() == value]
        if sample.numel() > 1:
            total = total + sample.var(unbiased=False) * sample.numel()
            count += sample.numel()
    return volume * total / max(count, 1)


def path_gradient_loss(model, action_fn, U, t, volume=None):
    """Path-gradient training loss and the same-rung PINN residual variance."""
    V = _volume(U, volume)
    t = _as_time(t, U)
    coeff = model.drift(U, t)
    velocity = lie_tangent(U, coeff)
    frozen = _frozen_model(model)
    residual = lambda value: path_gradient_residual(frozen, action_fn, value, t, V)[0]
    xi, loss = torch.func.jvp(residual, (U,), (velocity,))
    return {
        "loss": loss,
        "xi": xi.detach(),
        "pinn": _pinn(xi.detach(), t, V),
        "F_norm": (coeff.detach().square().flatten(1).sum(1) / V).sqrt(),
    }


def integrate_flow(U, t0, t1, model, n_steps=32):
    """Integrate a time-dependent left SU(3) flow with CF4 and its Haar log-Jacobian.

    The log-Jacobian is the CF4 quadrature of the exact field divergence and converges as
    n_steps grows; the finite-step map does not have an exact discrete log-Jacobian.
    """
    if int(n_steps) != n_steps or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    t0, t1 = _as_time(t0, U), _as_time(t1, U)
    h = (t1 - t0) / int(n_steps)
    logdet = torch.zeros(U.shape[0], dtype=U.real.dtype, device=U.device)

    def step(value, coeff, dt):
        return apply_velocity(value, coeff, dt)

    for i in range(int(n_steps)):
        time = t0 + i * h
        F1, d1 = model(U, time)
        Y2 = step(U, F1, h / 2)
        F2, d2 = model(Y2, time + h / 2)
        F3, d3 = model(step(U, F2, h / 2), time + h / 2)
        F4, d4 = model(step(Y2, F3 - F1 / 2, h), time + h)
        U = step(U, (3 * F1 + 2 * F2 + 2 * F3 - F4) / 12, h)
        U = step(U, (-F1 + 2 * F2 + 2 * F3 + 3 * F4) / 12, h)
        logdet = logdet + h * (d1 + 2 * d2 + 2 * d3 + d4) / 6
    return U, logdet


class CNFProtocolLayer(nn.Module):
    """Adapter matching the NEO-REX/SNF protocol-layer call convention."""

    def __init__(self, field, n_steps=32):
        super().__init__()
        if int(n_steps) != n_steps or n_steps < 1:
            raise ValueError("n_steps must be a positive integer")
        self.field = field
        self.n_steps = int(n_steps)
        self.D = field.D
        self.full_lattice = True
        self.smearing_jacobian_version = None

    @property
    def checkpoint_semantics(self):
        field_semantics = getattr(self.field, "checkpoint_semantics", None)
        if callable(field_semantics):
            field_semantics = field_semantics()
        if field_semantics is None:
            field_semantics = {
                "field": type(self.field).__name__,
                "D": int(self.field.D),
                "parameters": {name: list(value.shape) for name, value in self.field.named_parameters()},
            }
        return {
            "layer": "cnf",
            "training_config": getattr(self.field, "cnf_config", None),
            "field_config": field_semantics,
            "generator_basis": "Gell-Mann/2; Tr(T_a T_b)=1/2",
            "integrator": "CF4",
            "n_steps": self.n_steps,
        }

    def forward(self, cfgs, parameter, delta_parameter, rho_layer=None):
        del rho_layer
        return integrate_flow(
            cfgs, parameter, torch.as_tensor(parameter, dtype=cfgs.real.dtype, device=cfgs.device)
            + torch.as_tensor(delta_parameter, dtype=cfgs.real.dtype, device=cfgs.device),
            self.field, self.n_steps,
        )


__all__ = [
    "CloverChairLinkDrift", "CNFProtocolLayer", "apply_velocity", "coeff_to_matrix",
    "integrate_flow", "lie_tangent", "path_gradient_loss", "path_gradient_residual",
]
