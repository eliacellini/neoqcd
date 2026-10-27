import torch

### SU(N) utils

def SUN_identity(init_shape, dtype=torch.cdouble, device=None):
    id = torch.ones(init_shape, dtype=dtype, device=device)
    return torch.diag_embed(id)

def hot_SU2_start_nonuniform(init_shape_SU2, dtype=torch.cdouble, device=None):
    x0 = torch.rand(init_shape_SU2, dtype=dtype, device=device)
    nrm = torch.norm(x0, dim = -1)
    x0 = x0 / nrm.unsqueeze(-1)

    sign = torch.tensor([1, -1], device=x0.device, dtype=x0.real.dtype).view([1] * (len(init_shape_SU2) - 1) + [2])
    x1 = torch.conj((x0 * sign).roll(1,-1))

    return torch.cat((x0.unsqueeze(-2),x1.unsqueeze(-2)),-2)

def SUN_determinant(cfgs):
    return torch.linalg.det(cfgs)


def SU2_haar(shape, device=None, dtype=torch.float64, generator=None):
    """Draw exact Haar SU(2) first-row pairs with shape `shape`."""
    g = torch.randn(tuple(shape) + (4,), dtype=dtype, device=device, generator=generator)
    g = g / torch.linalg.vector_norm(g, dim=-1, keepdim=True)
    return torch.complex(g[..., 0], g[..., 1]), torch.complex(g[..., 2], g[..., 3])


def SU3_from_two_rows(u, v):
    """Complete two complex rows to an SU(3) matrix by Gram-Schmidt."""
    u = u / torch.linalg.vector_norm(u, dim=-1, keepdim=True)
    v = v - (u.conj() * v).sum(-1, keepdim=True) * u
    v = v / torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    w = torch.stack((u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1],
                     u[..., 2] * v[..., 0] - u[..., 0] * v[..., 2],
                     u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]), dim=-1)
    return torch.stack((u, v, w.conj()), dim=-2)


def SU3_unitarize(U):
    """Project links back to SU(3) from their first two rows."""
    return SU3_from_two_rows(U[..., 0, :], U[..., 1, :])


def SUN_haar(init_shape, device=None, dtype=torch.cdouble, generator=None):
    """Draw exact Haar SU(2) or SU(3) matrices for the given `(..., N)` shape."""
    init_shape = tuple(init_shape)
    N = init_shape[-1]
    shape = init_shape[:-1]
    real_dtype = torch.float64 if dtype == torch.cdouble else torch.float32
    if N == 2:
        a, b = SU2_haar(shape, device=device, dtype=real_dtype, generator=generator)
        first = torch.stack((a, b), dim=-1)
        second = torch.stack((-b.conj(), a.conj()), dim=-1)
        return torch.stack((first, second), dim=-2).to(dtype=dtype)
    if N == 3:
        z = torch.randn(tuple(shape) + (2, 3), dtype=dtype, device=device, generator=generator)
        return SU3_from_two_rows(z[..., 0, :], z[..., 1, :])
    raise ValueError(f"Haar sampling supports SU(2) and SU(3), got SU({N})")

def SUN_trace(cfgs):
    return torch.einsum('...ii', cfgs)

def check_determinant_SUN(cfgs, tol):
    dtm = SUN_determinant(cfgs)
    return (dtm - 1.0).abs() > tol

def SUN_dagger(cfgs):
    return torch.conj(torch.transpose(cfgs,-2,-1))

def SUN_mul(U,V):
    if U.is_cuda:
        # One broadcast product and one sum: on an A100 forward + backward is 2.6x faster than
        # an addcmul per column, whose indexed slices each zero-fill a tensor in the backward
        # (main/bench_sun_mul.py), and 5x faster than matmul, for 18% more peak memory.
        return (U.unsqueeze(-1) * V.unsqueeze(-3)).sum(-2)
    return torch.einsum('...ij,...jk->...ik',U,V)

def plaquette_SUN(cfgs, D, N):
    retr = torch.zeros(cfgs[:,0].shape[:-2], dtype=cfgs.real.dtype, device=cfgs.device)
    dims = range(1,D+1)
    for nu in range(1,D):
        for mu in range(0,nu):
            plaq = SUN_mul(SUN_dagger(cfgs[:,nu]),cfgs[:,mu])
            plaq = SUN_mul(plaq,torch.roll(cfgs,1,dims=(-D + mu - 2))[:,nu])
            plaq = SUN_mul(plaq,SUN_dagger(torch.roll(cfgs,1,dims=(-D + nu - 2)))[:,mu])
            retr += SUN_trace(plaq).real
    return torch.mean(retr, tuple(dims))/(D*(D-1)/2)/N

def pstaple(cfgs, mu, nu, D):
    # Slice out the link direction before rolling. Rolling the full D-direction
    # tensor needlessly moves every matrix field for each individual staple.
    U_nu, U_mu = cfgs[:, nu], cfgs[:, mu]
    U_nu_xpmu = torch.roll(U_nu, 1, dims=(-D + mu - 2))
    U_mu_xpnu = torch.roll(U_mu, 1, dims=(-D + nu - 2))
    staple = SUN_mul(U_nu_xpmu, SUN_dagger(U_mu_xpnu))
    return SUN_mul(staple, SUN_dagger(U_nu))

def nstaple(cfgs, mu, nu, D):
    U_nu, U_mu = cfgs[:, nu], cfgs[:, mu]
    U_nu_xpmu_xmnu = torch.roll(
        U_nu, (1, -1), dims=(-D + mu - 2, -D + nu - 2)
    )
    U_mu_xmnu = torch.roll(U_mu, -1, dims=(-D + nu - 2))
    U_nu_xmnu = torch.roll(U_nu, -1, dims=(-D + nu - 2))
    staple = SUN_mul(SUN_dagger(U_nu_xpmu_xmnu), SUN_dagger(U_mu_xmnu))
    return SUN_mul(staple, U_nu_xmnu)
        
# Plaquette fields are independent of the action coupling and can be reused while the configuration
# remains unchanged.
def plaquette_retr(cfgs, mu, nu, D):
    U_mu, U_nu = cfgs[:, mu], cfgs[:, nu]
    U_nu_xpmu = torch.roll(U_nu, 1, dims=-D + mu - 2)
    U_mu_xpnu = torch.roll(U_mu, 1, dims=-D + nu - 2)
    A = SUN_mul(U_mu, U_nu_xpmu)
    B = SUN_mul(U_nu, U_mu_xpnu)
    return (A * B.conj()).real.sum((-2, -1))


def plaquette_fields(cfgs, D):
    return [plaquette_retr(cfgs, mu, nu, D) for nu in range(1, D) for mu in range(nu)]


def mean_plaquette(plaquettes, N):
    retr = sum(plaquettes, plaquettes[0].new_zeros(plaquettes[0].shape))
    return torch.mean(retr / N, tuple(range(1, retr.dim()))) / len(plaquettes)


# Flat gather support for checkerboard link updates.  A link (mu, x) is stored at mu * V + x.
def flat_links(cfgs, D):
    return cfgs.flatten(1, D + 1)


class MaskedLinks:
    """Precomputed flat indices for selected links and the six links in each staple pair."""

    def __init__(self, mask, D):
        self.D = int(D)
        lattice = tuple(mask.shape[3:3 + self.D])
        self.V = int(mask[0, 0, 0].numel())
        ids = torch.arange(self.V, device=mask.device).reshape(lattice)
        shift = lambda amount, axis: torch.roll(ids, amount, dims=axis).flatten()
        self.sites, self.links, self.staple_links = [], [], []
        for m in range(2 * self.D):
            mu = m // 2
            if mask[m, 0, :mu].any() or mask[m, 0, mu + 1:].any():
                raise ValueError(f"mask {m} selects links in a direction other than {mu}")
            sites = mask[m, 0, mu].to(dtype=torch.bool).flatten().nonzero().squeeze(-1)
            per_nu = []
            for nu in range(self.D):
                if nu == mu:
                    continue
                xpmu, xpnu, xmnu = shift(1, mu), shift(1, nu), shift(-1, nu)
                y = xmnu[sites]
                per_nu.append(torch.stack((nu * self.V + xpmu[sites],
                                           mu * self.V + xpnu[sites],
                                           nu * self.V + sites,
                                           nu * self.V + xpmu[y],
                                           mu * self.V + y,
                                           nu * self.V + y)))
            self.sites.append(sites)
            self.links.append(mu * self.V + sites)
            self.staple_links.append(per_nu)


class MaskedLinksCache:
    """Small, owner-scoped cache for persistent FlowPars masks."""

    def __init__(self, max_entries=8):
        self.max_entries = int(max_entries)
        self._entries = {}

    def get(self, mask, D, dmask=None):
        key = (
            id(mask),
            id(dmask) if dmask is not None else None,
            int(D),
            str(mask.device),
            None if dmask is None else str(dmask.device),
        )
        entry = self._entries.get(key)
        if (entry is not None and entry[0] is mask and entry[1] is dmask
                and entry[2] == mask._version
                and entry[3] == (None if dmask is None else dmask._version)):
            return entry[4]

        effective_mask = mask if dmask is None else mask * dmask.unsqueeze(0)
        links = MaskedLinks(effective_mask, D)
        if entry is None and len(self._entries) >= self.max_entries:
            self._entries.pop(next(iter(self._entries)))
        self._entries[key] = (
            mask,
            dmask,
            mask._version,
            None if dmask is None else dmask._version,
            links,
        )
        return links


def masked_links_cache_for(owner):
    """Get the bounded index cache shared by the layers of one FlowPars owner."""
    cache = getattr(owner, "_masked_links_cache_owner", None)
    if cache is None:
        cache = MaskedLinksCache()
        try:
            setattr(owner, "_masked_links_cache_owner", cache)
        except (AttributeError, TypeError):
            pass
    return cache


def staples_at(flat, links):
    """Return upper/lower plaquette staples for flat indices with shape (6, n)."""
    U = flat.index_select(1, links.flatten()).unflatten(1, links.shape)
    U_nu_xpmu, U_mu_xpnu, U_nu, U_nu_ypmu, U_mu_y, U_nu_y = U.unbind(1)
    upper = SUN_mul(SUN_mul(U_nu_xpmu, SUN_dagger(U_mu_xpnu)), SUN_dagger(U_nu))
    lower = SUN_mul(SUN_mul(SUN_dagger(U_nu_ypmu), SUN_dagger(U_mu_y)), U_nu_y)
    return upper, lower


def plaq_index(D, mu, nu):
    if int(D) < 2 or mu == nu or not (0 <= mu < D and 0 <= nu < D):
        raise ValueError(f"invalid plaquette directions D={D}, mu={mu}, nu={nu}")
    # Orient the D-1 plaquettes touching mu into positive/negative slots. This
    # preserves the historical D=4 ordering while also covering SU(2)/D=2.
    return (nu - mu - 1) % D


def invert_left_update(left_factor, Up, tol=1e-12, max_iter=100):
    """Solve X(U) U = Up for a left update by X(U) in SU(N), given left_factor(U) = X(U).

    Fixed-point iteration U <- X(U)^dagger Up, started at U = Up: it converges when X varies
    slowly with U, as for the small steps of a flow. X is recomputed from U at every iteration, so
    rounding does not accumulate. Links with X = 1 stay at Up. Returns U, the number of iterations
    and the largest residual |X(U) U - Up|; the caller decides what to do if it is above tol.
    """
    U = Up
    for iterations in range(max_iter + 1):
        X = left_factor(U)
        residual = (SUN_mul(X, U) - Up).abs().amax()
        if residual <= tol or iterations == max_iter:
            return U, iterations, residual
        U = SUN_mul(SUN_dagger(X), Up)
