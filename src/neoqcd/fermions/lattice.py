"""Lattice geometry shared by the fermion modules.

Conventions, taken from the pure-gauge code (neoqcd.sun_utils):
- gauge links `U` have shape (B, D, T, L, ..., L, N, N); direction 0 is time;
- `U[:, mu]` is the link from x to x + mu, where "x + mu" is `torch.roll(., +1, axis)` along the
  lattice axis of mu. In array indices this means that the field "at x + mu" seen from site n is
  the entry at n - 1 (`torch.roll(f, 1)[n] = f[n - 1]`): the forward direction of the code runs
  towards decreasing array index. The plaquette of sun_utils is invariant under
  U_mu(x) -> Omega(x) U_mu(x) Omega(x + mu)^dagger with this reading, and not with the opposite
  one (test_fermions_lattice);
- a Wilson fermion field has shape (B, T, L, ..., L, S, N) with S = 2^(D // 2) spin components;
  a staggered field has shape (B, T, L, ..., L, N).
Lattice axes are 1 .. D for both link components `U[:, mu]` and fermion fields.
"""

from dataclasses import dataclass

import torch

from neoqcd.fermions._gauge import SUN_dagger, SUN_mul


@dataclass(frozen=True)
class Geometry:
    """Lattice extent and gauge group: D dimensions, time extent T, spatial extent L, SU(N)."""

    D: int
    T: int
    L: int
    N: int

    @classmethod
    def from_pars(cls, pars):
        """Build from any object with attributes D, T, L, N (FlowPars or a SimpleNamespace)."""
        return cls(int(pars.D), int(pars.T), int(pars.L), int(pars.N))

    @property
    def lattice(self):
        return (self.T,) + (self.L,) * (self.D - 1)

    @property
    def volume(self):
        v = 1
        for n in self.lattice:
            v *= n
        return v

    @property
    def spin(self):
        return 2 ** (self.D // 2)

    @property
    def has_even_extents(self):
        """True when every extent is even, so that the checkerboard is consistent across the wrap."""
        return all(n % 2 == 0 for n in self.lattice)

    def link_shape(self, batch_size):
        return (batch_size, self.D) + self.lattice + (self.N, self.N)

    def wilson_shape(self, batch_size):
        return (batch_size,) + self.lattice + (self.spin, self.N)

    def staggered_shape(self, batch_size):
        return (batch_size,) + self.lattice + (self.N,)


def axis(mu):
    """Tensor axis of lattice direction mu for `U[:, mu]` and for fermion fields."""
    return 1 + int(mu)


def shift(field, mu, steps):
    """field(x + steps * mu): `torch.roll` by +steps along the axis of mu (code convention).

    In array indices, `shift(f, mu, 1)[n] = f[n - 1]` along the axis of mu.
    """
    return torch.roll(field, int(steps), dims=axis(mu))


def parity(geometry, device=None):
    """(t + x + y + z) mod 2 on the lattice, shape `geometry.lattice`."""
    grids = torch.meshgrid(*[torch.arange(n, device=device) for n in geometry.lattice], indexing="ij")
    return sum(grids) % 2


def even_odd_masks(geometry, device=None, dtype=torch.bool):
    """Even and odd site masks of shape (1,) + lattice + (1, 1), broadcastable to Wilson fields.

    Nearest-neighbour hops change parity only if every extent is even; this is checked here.
    """
    if not geometry.has_even_extents:
        raise ValueError(f"even-odd decomposition needs even extents, got lattice {geometry.lattice}")
    p = parity(geometry, device=device)
    shape = (1,) + geometry.lattice + (1, 1)
    even = (p == 0).reshape(shape).to(dtype)
    odd = (p == 1).reshape(shape).to(dtype)
    return even, odd


def fermion_boundary_links(U, time_sign=-1.0):
    """Links with the fermion boundary phase absorbed into the time direction.

    With the code's shift convention the time links at array index t = 0 are the ones that cross
    the boundary in both hops (x -> x + 0 at t = 0 reads psi at t = T - 1, and x -> x - 0 at
    t = T - 1 reads psi at t = 0 through U_0 at t = 0): multiplying them by `time_sign` imposes
    antiperiodic (-1) or periodic (+1) fermion boundary conditions in time. A new tensor is
    returned; the gauge field `U` itself is untouched. Spatial directions stay periodic.
    """
    sign = float(time_sign)
    if sign == 1.0:
        return U
    scale = torch.ones(U.shape[2], dtype=U.real.dtype, device=U.device)
    scale[0] = sign
    view = (1, 1, U.shape[2]) + (1,) * (U.dim() - 3)
    U0 = U[:, 0:1] * scale.view(view)
    return torch.cat((U0, U[:, 1:]), dim=1)


def gauge_transform_links(U, omega):
    """U_mu(x) -> Omega(x) U_mu(x) Omega(x + mu)^dagger for a site field `omega` (B, lattice, N, N)."""
    out = []
    for mu in range(U.shape[1]):
        out.append(SUN_mul(SUN_mul(omega, U[:, mu]), SUN_dagger(shift(omega, mu, 1))))
    return torch.stack(out, dim=1)


def gauge_transform_field(psi, omega):
    """psi(x) -> Omega(x) psi(x) for a Wilson field (B, lattice, S, N) or staggered (B, lattice, N)."""
    if psi.dim() == omega.dim() - 1:  # staggered: colour index last, no spin index
        return torch.einsum("...ab,...b->...a", omega, psi)
    return torch.einsum("...ab,...sb->...sa", omega, psi)
