"""Topological charge of SU(3) configurations in D = 4: Wilson cooling and the clover charge.

Q is measured as in the SNF paper (arXiv:2610.20747): `cooling_steps` sweeps of Wilson cooling (60
there), then the clover charge

    Q = sum_x q(x),   q(x) = [Tr F01 F23 - Tr F02 F13 + Tr F03 F12] / (4 pi^2),
    F_{mu nu}(x) = [(C - C^dagger) - Tr(C - C^dagger) / 3] / (8 i),

with C_{mu nu}(x) the sum of the four plaquettes around x in the (mu, nu) plane (the clover), all with
the same orientation.

Cooling sweep: every link, in the checkerboard blocks m = eo + 2 mu of create_mask in order, and in every
link the SU(2) subgroups 0, 1, 2 in turn (as the heat bath of mcmc.HBOR); each subgroup update replaces
the link U by V U, with V the SU(2) element that maximizes Re Tr(V R), R the subgroup projection of U S
(S the sum of the staples of the standard Wilson action): V = W^dagger, W = R / sqrt(det R). No
re-unitarization: the SU(2) factors are unitary to rounding, and after 60 sweeps the links are in SU(3)
to ~1e-13 (a whole-lattice re-unitarization would hold ~5 copies of the links: 37 GB at 30^4, batch 16).
The fixed SU(2) subgroups make cooling covariant only under diagonal gauge transformations; the cooled
charge of a generic gauge transform differs slightly before Q settles on an integer.

Neighbours follow sun.pstaple: "x + mu" is torch.roll(field, 1, mu)[x]. That reverses the four
directions with respect to the usual convention, a rotation of the lattice: Q keeps its sign.

The torch functions (cool_torch, clover_charge_torch) are the references; cool and clover_charge run the
kernels of neoqcd.fused (snf_cool, snf_topological_charge) where they apply (CUDA, complex128, T x L^3; on
CPU tensors only with emulate=True, for the tests) and the torch functions otherwise.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

import neoqcd.fused as fused
import neoqcd.sun_utils as sun
from neoqcd.mcmc import HBOR, SU2_left_multiply_SU3, SU3_compute_RK
from neoqcd.utils import create_mask

D = 4
_PLANE_PAIRS = (((0, 1), (2, 3), 1.0), ((0, 2), (1, 3), -1.0), ((0, 3), (1, 2), 1.0))


def _check(cfgs):
    if cfgs.dim() != 8 or int(cfgs.shape[1]) != D or tuple(cfgs.shape[-2:]) != (3, 3):
        raise ValueError(f"expected SU(3) configurations (B, 4, T, L, L, L, 3, 3), got {tuple(cfgs.shape)}")


def _cooling_hbor(cfgs):
    pars = SimpleNamespace(N=3, D=D, orsteps=0, device=cfgs.device)
    return HBOR(pars, beta=1.0)


def _cool_block(hbor, flat, links, m, cfgs):
    # One block of links: each subgroup of each link moved to the maximum of Re Tr(V R)
    staple = hbor._gathered_staples(cfgs, flat, links, m, None)
    new = flat.index_select(1, links.links[m])
    colors = torch.arange(3, device=cfgs.device)
    for phb in (0, 1, 2):
        Rk = SU3_compute_RK(new, staple, colors, phb)
        k = torch.sqrt(sun.det_small(Rk).abs())
        good = k > 0
        safe = torch.where(good, k, torch.ones_like(k))
        V = sun.SUN_dagger(Rk / safe.unsqueeze(-1).unsqueeze(-1))
        proposal = SU2_left_multiply_SU3(new, V, phb)
        new = torch.where(good.unsqueeze(-1).unsqueeze(-1), proposal, new)
    return new


@torch.no_grad()
def cool_torch(cfgs, steps):
    """`steps` sweeps of Wilson cooling of every link, on a copy (the reference of fused.cooling)."""
    _check(cfgs)
    T, L = int(cfgs.shape[2]), int(cfgs.shape[3])
    if tuple(cfgs.shape[3:6]) != (L, L, L):
        raise ValueError("cooling uses the checkerboard masks of create_mask: a T x L^3 lattice")
    hbor = _cooling_hbor(cfgs)
    links = hbor._links_for_mask(create_mask(D, T, L).to(cfgs.device))
    flat = sun.flat_links(cfgs, D).clone()
    for _ in range(int(steps)):
        for m in range(2 * D):
            flat.index_copy_(1, links.links[m], _cool_block(hbor, flat, links, m, cfgs))
    return flat.reshape(cfgs.shape)


def _roll(field, shifts):
    # field (B, T, L, L, L, 3, 3); shifts {direction: +-1}: the field at x + sum s_d d ("x + d" = roll +1)
    for d, s in shifts.items():
        field = torch.roll(field, s, dims=1 + d)
    return field


def _clover_a(cfgs, mu, nu):
    # A_{mu nu} = (C - C^dagger) - Tr(C - C^dagger) / 3 of the clover C_{mu nu}
    U_mu, U_nu = cfgs[:, mu], cfgs[:, nu]
    mul, dag = sun.SUN_mul, sun.SUN_dagger
    leaf1 = mul(mul(mul(U_mu, _roll(U_nu, {mu: 1})), dag(_roll(U_mu, {nu: 1}))), dag(U_nu))
    leaf2 = mul(mul(mul(U_nu, dag(_roll(U_mu, {nu: 1, mu: -1}))), dag(_roll(U_nu, {mu: -1}))), _roll(U_mu, {mu: -1}))
    leaf3 = mul(mul(mul(dag(_roll(U_mu, {mu: -1})), dag(_roll(U_nu, {mu: -1, nu: -1}))),
                    _roll(U_mu, {mu: -1, nu: -1})), _roll(U_nu, {nu: -1}))
    leaf4 = mul(mul(mul(dag(_roll(U_nu, {nu: -1})), _roll(U_mu, {nu: -1})), _roll(U_nu, {nu: -1, mu: 1})), dag(U_mu))
    C = leaf1 + leaf2 + leaf3 + leaf4
    A = C - dag(C)
    eye = torch.eye(3, dtype=A.dtype, device=A.device)
    return A - (sun.SUN_trace(A) / 3.0)[..., None, None] * eye


@torch.no_grad()
def clover_charge_density_torch(cfgs):
    """q(x) of each configuration, (B, T, L, L, L) float64 (the reference of the kernel)."""
    _check(cfgs)
    total = torch.zeros(cfgs[:, 0].shape[:-2], dtype=cfgs.real.dtype, device=cfgs.device)
    for (a, b), (c, d), sign in _PLANE_PAIRS:
        retr = sun.SUN_trace(sun.SUN_mul(_clover_a(cfgs, a, b), _clover_a(cfgs, c, d))).real
        total = total + sign * retr
    return -total / (256.0 * math.pi ** 2)


def clover_charge_torch(cfgs):
    """The clover charge Q of each configuration, (B,) float64, without cooling."""
    return clover_charge_density_torch(cfgs).sum(dim=tuple(range(1, D + 1)))


def cool(cfgs, steps, use_kernels=True, emulate=False):
    """Wilson cooling (fused.cooling where it applies, cool_torch otherwise), on a copy."""
    if use_kernels:
        out = fused.cooling(cfgs, steps, emulate=emulate)
        if out is not None:
            return out
    return cool_torch(cfgs, steps)


def clover_charge(cfgs, use_kernels=True, emulate=False):
    """The clover charge (fused.clover_charge where it applies, clover_charge_torch otherwise)."""
    if use_kernels:
        out = fused.clover_charge(cfgs, emulate=emulate)
        if out is not None:
            return out
    return clover_charge_torch(cfgs)


@torch.no_grad()
def topological_charge(cfgs, cooling_steps=60, use_kernels=True, emulate=False):
    """Q of each configuration after `cooling_steps` sweeps of Wilson cooling (SNF paper: 60), (B,) float64.
    The configurations are not changed."""
    if int(cfgs.shape[0]) == 0:
        return torch.zeros(0, dtype=torch.float64, device=cfgs.device)
    cooled = cool(cfgs, cooling_steps, use_kernels=use_kernels, emulate=emulate) if cooling_steps > 0 else cfgs
    return clover_charge(cooled, use_kernels=use_kernels, emulate=emulate)
