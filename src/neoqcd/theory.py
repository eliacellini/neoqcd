import torch
import neoqcd.sun_utils as sun


def _as_parameter_vector(parameter, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    p = torch.as_tensor(parameter, device=device, dtype=dtype)
    if p.dim() == 0:
        p = p.expand(batch_size)
    p = p.reshape(-1)
    if int(p.numel()) != int(batch_size):
        raise ValueError(f"parameter must have shape ({batch_size},), got ({p.numel()},)")
    return p


def build_nf_changed_link_mask(flow_pars, device=None) -> torch.Tensor:
    """
    Link support modified by DefectSNFProtocolLayer._embed_small_cfgs.
    Shape: (D, T, L, ..., L).
    """
    defect = flow_pars.defect
    if defect is None:
        raise ValueError("NF halo action requires flow_pars.defect")
    D = int(flow_pars.D)
    T = int(flow_pars.T)
    L = int(flow_pars.L)
    t = int(defect.time_slice)
    s = int(defect.space_slice)
    dsize = int(defect.dsize)
    if device is None:
        device = flow_pars.device

    mask = torch.zeros((D, T) + (L,) * (D - 1), dtype=torch.bool, device=device)
    temporal_slice = (slice(s - 1, dsize + s + 1),) * (D - 1)
    spatial_slice = (slice(s, dsize + s + 1),) * (D - 1)

    mask[(0, t) + temporal_slice] = True
    for mu in range(1, D):
        mask[(mu, t) + spatial_slice] = True
        mask[(mu, t - 1) + spatial_slice] = True
    return mask


def _plaquette_support_from_links(link_mask: torch.Tensor, mu: int, nu: int) -> torch.Tensor:
    return (
        link_mask[mu]
        | torch.roll(link_mask[mu], 1, dims=nu)
        | link_mask[nu]
        | torch.roll(link_mask[nu], 1, dims=mu)
    )


def build_nf_halo_plaquette_masks(flow_pars, defect_mask: torch.Tensor, device=None) -> dict[tuple[int, int], torch.Tensor]:
    """
    Plaquette anchors whose Wilson term can change during a local NF block.

    Includes plaquettes touched by links embedded by the NF and plaquettes whose
    coupling depends on the defect parameter.
    """
    if device is None:
        device = flow_pars.device
    D = int(flow_pars.D)
    changed_links = build_nf_changed_link_mask(flow_pars, device=device)
    defect_links = defect_mask.to(device=device, dtype=torch.bool)[1]
    masks = {}
    for nu in range(1, D):
        for mu in range(0, nu):
            masks[(mu, nu)] = (
                _plaquette_support_from_links(changed_links, mu, nu)
                | _plaquette_support_from_links(defect_links, mu, nu)
            )
    return masks


def _bbox_slices(mask: torch.Tensor):
    idx = torch.nonzero(mask, as_tuple=False)
    if idx.numel() == 0:
        return None
    return tuple(
        slice(int(idx[:, dim].min().item()), int(idx[:, dim].max().item()) + 1)
        for dim in range(mask.dim())
    )


def _shift_slices(slices, dim: int, shift: int):
    shifted = list(slices)
    sl = shifted[dim]
    shifted[dim] = slice(sl.start + shift, sl.stop + shift)
    return tuple(shifted)


def _slices_inside(slices, shape) -> bool:
    return all(sl.start >= 0 and sl.stop <= int(size) for sl, size in zip(slices, shape))


def _kappa_for_parameter(defect: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
    view_shape = (int(parameter.numel()),) + (1,) * (defect.dim() - 1)
    p = parameter.view(view_shape)
    return defect[0].unsqueeze(0) + defect[1].unsqueeze(0) * p


def S_halo(
    cfgs: torch.Tensor,
    defect_mask: torch.Tensor,
    beta: float,
    defect_par,
    plaquette_masks: dict[tuple[int, int], torch.Tensor],
    D: int,
    N: int,
) -> torch.Tensor:
    """
    Wilson action restricted to a fixed plaquette halo.

    The halo masks select plaquette anchors. This is intended for differences
    where all terms outside the halo cancel exactly.
    """
    work_dtype = cfgs.real.dtype if torch.is_complex(cfgs) else cfgs.dtype
    batch_size = int(cfgs.shape[0])
    parameter = _as_parameter_vector(defect_par, batch_size, cfgs.device, work_dtype)
    defect = defect_mask.to(device=cfgs.device, dtype=work_dtype)
    kappa = _kappa_for_parameter(defect, parameter)
    out = torch.zeros(batch_size, dtype=work_dtype, device=cfgs.device)
    lattice_shape = tuple(int(v) for v in cfgs.shape[2:-2])

    for nu in range(1, int(D)):
        for mu in range(0, nu):
            mask = plaquette_masks[(mu, nu)].to(device=cfgs.device, dtype=torch.bool)
            bbox = _bbox_slices(mask)
            if bbox is None:
                continue

            mu_shift = _shift_slices(bbox, mu, -1)
            nu_shift = _shift_slices(bbox, nu, -1)
            if not (_slices_inside(mu_shift, lattice_shape) and _slices_inside(nu_shift, lattice_shape)):
                plaq = sun.SUN_mul(sun.pstaple(cfgs, mu, nu, int(D)), cfgs[:, mu])
                retr = sun.SUN_trace(plaq).real
                weight = (
                    kappa[:, mu]
                    * torch.roll(kappa, 1, dims=(-int(D) + mu))[:, nu]
                    * torch.roll(kappa, 1, dims=(-int(D) + nu))[:, mu]
                    * kappa[:, nu]
                )
                term = (6.0 - weight * retr / float(N)) * mask.to(dtype=work_dtype)
                out = out + float(beta) * term.sum(dim=tuple(range(1, int(D) + 1)))
                continue

            U_mu = cfgs[(slice(None), mu) + bbox + (slice(None), slice(None))]
            U_nu = cfgs[(slice(None), nu) + bbox + (slice(None), slice(None))]
            U_nu_rm = cfgs[(slice(None), nu) + mu_shift + (slice(None), slice(None))]
            U_mu_rn = cfgs[(slice(None), mu) + nu_shift + (slice(None), slice(None))]
            plaq = sun.SUN_mul(
                sun.SUN_mul(
                    sun.SUN_mul(U_nu_rm, sun.SUN_dagger(U_mu_rn)),
                    sun.SUN_dagger(U_nu),
                ),
                U_mu,
            )
            retr = sun.SUN_trace(plaq).real
            weight = (
                kappa[(slice(None), mu) + bbox]
                * kappa[(slice(None), nu) + mu_shift]
                * kappa[(slice(None), mu) + nu_shift]
                * kappa[(slice(None), nu) + bbox]
            )
            local_mask = mask[bbox].to(dtype=work_dtype)
            term = (6.0 - weight * retr / float(N)) * local_mask
            out = out + float(beta) * term.sum(dim=tuple(range(1, int(D) + 1)))
    return out


def local_nf_action_change(
    x_old: torch.Tensor,
    x_new: torch.Tensor,
    defect_mask: torch.Tensor,
    beta: float,
    parameter_old,
    parameter_new,
    plaquette_masks: dict[tuple[int, int], torch.Tensor],
    D: int,
    N: int,
) -> torch.Tensor:
    return S_halo(
        x_new,
        defect_mask,
        beta,
        parameter_new,
        plaquette_masks,
        D,
        N,
    ) - S_halo(
        x_old,
        defect_mask,
        beta,
        parameter_old,
        plaquette_masks,
        D,
        N,
    )

class PriorSUN:
    def __init__(self, flow_pars, beta, therm_steps, mcmc_steps, defect_mask=None, defect_par=1.0):
        # Import locally to avoid a module-level circular dependency:
        # mcmc -> theory.Wilson_action and PriorSUN -> mcmc.HBOR.
        from neoqcd.mcmc import HBOR
        self.init_shape = flow_pars.init_shape
        self.action = Wilson_action(flow_pars, beta, defect_par)
        self.update = HBOR(flow_pars, beta, defect_par)
        self.beta = beta
        self.therm_steps = therm_steps
        self.mcmc_steps = mcmc_steps
        self.mask = flow_pars.mask
        self.device = flow_pars.device
        self.defect_mask = defect_mask

    def __call__(self, cfgs=None):
        if cfgs != None:
            for i in range(self.mcmc_steps):
                cfgs = self.update(cfgs, self.mask, self.defect_mask)
        else:
            cfgs = self.therm()
        return cfgs, self.action(cfgs, self.defect_mask)
    
    @torch.no_grad()
    def therm(self):
        cfgs = sun.SUN_identity(self.init_shape, device=self.device)
        for i in range(self.therm_steps):
            cfgs = self.update(cfgs, self.mask, self.defect_mask)
        return cfgs
    
    def save_thermalized(self,x,path):
        torch.save(x,path)

    def load_thermalized(self,path):
        x = torch.load(path)
        return x.to(self.device)
    

class Wilson_action:
    def __init__(self, flow_pars, beta, defect_par=1.0):
        self.D = flow_pars.D
        self.N = flow_pars.N
        self.device = flow_pars.device

        self.beta = beta
        self.defect_par = defect_par

    def __call__(self, cfgs, defect):
        retr = torch.zeros(cfgs[:,0].shape[:-2], dtype=cfgs.real.dtype, device=cfgs.device)

        if (self.defect_par < 1.0):
            kappa = defect[0,:] + defect[1,:] * self.defect_par

        for nu in range(1, self.D):
            for mu in range(0, nu):
                plaq = sun.SUN_mul(sun.pstaple(cfgs, mu, nu, self.D), cfgs[:,mu])
                
                if (self.defect_par < 1.0):
                    kappamunu = kappa[mu] * torch.roll(kappa, 1, dims=(-self.D + mu))[nu] * torch.roll(kappa, 1, dims=(-self.D + nu))[mu] * kappa[nu]
                    retr += kappamunu * sun.SUN_trace(plaq).real
                else:
                    retr += sun.SUN_trace(plaq).real

        dims = range(1, self.D + 1)
        return self.beta*torch.sum(6.0 - retr/self.N, tuple(dims))
