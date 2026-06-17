import torch
import torch.utils.checkpoint as checkpoint
import neoqcd.sun_utils as sun
from neoqcd.theory import Wilson_action
from typing import Optional


def _su2_sign(view_ndim, device, dtype):
    return torch.tensor([1, -1], device=device, dtype=dtype).view([1] * view_ndim + [2])


class NEMCMC_update(torch.nn.Module):
    def __init__(self, flow_pars, beta, defect_mask=None, defect_par=1.0):
        super().__init__()
        self.action = Wilson_action(flow_pars, beta, defect_par)
        self.update = HBOR(flow_pars, beta, defect_par)
        self.beta = beta
        self.D = flow_pars.D
        self.device = flow_pars.device
        self.nupdates = flow_pars.updates_per_layer
        self.hierarchical_levels = flow_pars.hierarchical_levels

        self.defect_mask = defect_mask
        if self.hierarchical_levels > 0:
            self.hierarchical_rectangles = flow_pars.hierarchical_rectangles
            self.hierarchical_nupdates = flow_pars.hierarchical_updates

    def set_parameter(self, parameter: torch.Tensor, parameter_name: str = "bc"):
        if str(parameter_name) != "bc":
            raise ValueError(
                f"NEMCMC_update supports only parameter_name='bc', got {parameter_name!r}"
            )
        p = torch.as_tensor(parameter, dtype=torch.float64, device=self.device).reshape(-1)
        if p.numel() == 0:
            raise ValueError("parameter cannot be empty")
        p0 = float(p[0].item())
        if not bool(torch.allclose(p, torch.full_like(p, p0))):
            raise ValueError(
                "NEMCMC_update expects a scalar/shared parameter across the local batch. "
                "Use DefectMCMCAdapter for per-sample parameters."
            )
        self.action.defect_par = p0
        self.update.set_defect_parameter(p0)

    def forward(self, cfgs, flow_pars, rho_root=None):
        s_old = self.action(cfgs, self.defect_mask)

        for u in range(self.nupdates):
            cfgs = self.update(cfgs, flow_pars.mask, self.defect_mask)
            if self.hierarchical_levels > 0:
                cfgs = self.hierarchical(cfgs, 0, flow_pars)
        
        s_new = self.action(cfgs, self.defect_mask)
        return cfgs, s_new - s_old, 0

    def hierarchical(self, cfgs, level, flow_pars):
        for hu in range(self.hierarchical_nupdates[level]):
            small_cfgs = flow_pars.defect.cut_defect(cfgs, buffer=self.hierarchical_rectangles[level])
            small_cfgs = self.update(small_cfgs, flow_pars.smeared_defect_mask[level]*flow_pars.small_mask[level], flow_pars.small_defect_mask[level])
            cfgs = flow_pars.defect.embedding(small_cfgs, cfgs, buffer=self.hierarchical_rectangles[level])
        if level < self.hierarchical_levels - 1:
            cfgs = self.hierarchical(cfgs, level + 1, flow_pars)
        else:
            return cfgs

        return cfgs


class HBOR():
    def __init__(self, flow_pars, beta, defect_par=1.0, hierarchical=False):
        self.N = flow_pars.N
        self.D = flow_pars.D
        self.orsteps = flow_pars.orsteps
        self.device = flow_pars.device

        self.beta = beta
        self.defect_par = defect_par

        if self.N==2:
            self.hb_update=self.SU2_link_update_hb
            self.haar_hb_update=self.SU2_link_haar_hb
            self.over_update=self.SU2_link_update_over
        elif self.N==3:
            self.hb_update=self.SU3_link_update_hb
            self.haar_hb_update=self.SU3_link_haar_hb
            self.over_update=self.SU3_link_update_over

        if self.defect_par < 1.0:
            self.sum_of_staples = self.sum_of_staples_defect
            self.heatbath = self.heatbath_defect
        else:
            self.sum_of_staples = self.sum_of_staples_standard
            self.heatbath = self.heatbath_standard

    def set_defect_parameter(self, defect_par: float):
        self.defect_par = float(defect_par)
        if self.defect_par < 1.0:
            self.sum_of_staples = self.sum_of_staples_defect
            self.heatbath = self.heatbath_defect
        else:
            self.sum_of_staples = self.sum_of_staples_standard
            self.heatbath = self.heatbath_standard


    def __call__(self, cfgs, mask, defect):

        rn_shape = list(cfgs.shape[:-2])
        if self.N == 3:
            rn_shape.insert(1, 18)
        elif self.N == 2:
            rn_shape.insert(1, 6)
        rn_shape = tuple(rn_shape)
        #extracting random numbers
        rand = torch.rand(rn_shape, device=cfgs.device, dtype=cfgs.real.dtype)

        #heatbath
        for mu in range(self.D):
            for eo in range(2):
                current_mask = mask[eo+mu*2,:]
                cfgs_new = self.hb_update(cfgs, mu, rand, defect)
                cfgs = current_mask * cfgs_new + (1-current_mask) * cfgs

        #overrelaxation
        for o in range(self.orsteps):
            for mu in range(self.D):
                for eo in range(2):
                    current_mask = mask[eo+mu*2,:]
                    cfgs_new = self.over_update(cfgs, mu, defect)
                    cfgs = current_mask * cfgs_new + (1-current_mask) * cfgs
        return cfgs

    def haar(self, cfgs):
        rn_shape = list(cfgs.shape[:-2])
        rn_shape.insert(1, 18)
        rn_shape = tuple(rn_shape)
        #extracting random numbers
        rand = torch.rand(rn_shape, device=cfgs.device, dtype=cfgs.real.dtype)

        for rr in range(rn_shape[1]//6):
            rand[:,rr*6] = 1.0 - 2.0 * rand[:,rr*6]
            rand[:,rr*6+4] = 1.0 - 2.0 * rand[:,rr*6+4]
            rand[:,rr*6+5] = 2.0 * torch.pi * rand[:,rr*6+5]

        return self.haar_hb_update(cfgs, rand)
    
    def sum_of_staples_defect(self, cfgs, mu, defect):
        staple = torch.zeros(cfgs[:,0].shape, dtype=cfgs.dtype, device=cfgs.device)
        kappa = defect[0,:] + defect[1,:] * self.defect_par

        for nu in range(self.D):
            if nu != mu:
                kappamunu = kappa[mu] * torch.roll(kappa, 1, dims=(-self.D + mu))[nu] * torch.roll(kappa, 1, dims=(-self.D + nu))[mu] * kappa[nu]
                staple += kappamunu.unsqueeze(-1).unsqueeze(-1) * sun.pstaple(cfgs, mu, nu, self.D)
                
                kappamunu = kappa[mu] * torch.roll(kappa, (1,-1), dims=(-self.D + mu,-self.D + nu))[nu] * torch.roll(kappa, -1, dims=(-self.D + nu))[mu] * torch.roll(kappa, -1, dims=(-self.D + nu))[nu]
                staple += kappamunu.unsqueeze(-1).unsqueeze(-1) * sun.nstaple(cfgs, mu, nu, self.D)

        return staple

    def sum_of_staples_standard(self, cfgs, mu, defect=None):
        staple = torch.zeros(cfgs[:,0].shape, dtype=cfgs.dtype, device=cfgs.device)

        for nu in range(self.D):
            if nu != mu:
                staple += sun.pstaple(cfgs, mu, nu, self.D)
                staple += sun.nstaple(cfgs, mu, nu, self.D)

        return staple

    def SU2toSU3(self, Ak, phb):
        idx0 = torch.arange(phb, device=Ak.device)
        idx1 = torch.arange(phb, 2, device=Ak.device)
        A = torch.zeros(Ak.shape[:-2] + (2, 1), dtype=Ak.dtype, device=Ak.device)
        B = torch.cat((torch.index_select(Ak, -1, idx0), A, torch.index_select(Ak, -1, idx1)), -1)
        A = torch.zeros(Ak.shape[:-2] + (1, 2), dtype=Ak.dtype, device=Ak.device)
        C = torch.ones(Ak.shape[:-2] + (1, 1), dtype=Ak.dtype, device=Ak.device)
        AA = torch.cat((torch.index_select(A, -1, idx0), C, torch.index_select(A, -1, idx1)), -1)
        return torch.cat((torch.index_select(B, -2, idx0), AA, torch.index_select(B, -2, idx1)), -2)

    def heatbath_standard(self, staples, rand, factor):
        k = torch.sqrt(sun.SUN_determinant(staples).abs())
        xi = factor / (self.beta * k)

        #WARNING: if rand[:,0] and one of the two other rand are simultaneously 0 a NaN occurs in the gradient of mod. Probably very unlikely
        lam2 = (- 0.5 * xi * (torch.log(1.0 - rand[:,0,:]) + torch.cos(2.0 * torch.pi *(1.0 - rand[:,1,:]))**2 * torch.log(1.0 - rand[:,2,:])))
        acc = (1.0 - lam2 >= rand[:,3,:]**2)
        x0 = (1.0 - 2.0 * lam2) * acc

        #x0=x0.detach() # This detach prevents (unlikely) NaN in the gradient of the square root mod
        stheta = torch.sqrt(1.0 -  (1.0 - 2.0 * rand[:,4,:])**2)
        mod = torch.sqrt(1.0 - x0**2)

        A = torch.complex(x0, mod * (1.0 - 2.0 * rand[:,4,:])).unsqueeze(-1)
        B = (stheta*mod*torch.complex(torch.cos(2.0 * torch.pi * rand[:,5,:]), torch.sin(2.0 * torch.pi * rand[:,5,:]))).unsqueeze(-1)
        Xa = torch.ones(k.shape + (1,), dtype=A.dtype, device=A.device) * A
        Xb = torch.ones(k.shape + (1,), dtype=B.dtype, device=B.device) * B
        X = torch.cat((Xa,Xb),-1)

        v1 = torch.conj((X * _su2_sign(len(X.shape) - 1, X.device, X.real.dtype)).roll(1,-1))
        Xmat = torch.cat((X.unsqueeze(-2),v1.unsqueeze(-2)),-2)
        acc = acc.unsqueeze(-1).unsqueeze(-1)

        return sun.SUN_mul(Xmat, sun.SUN_dagger(staples / k.unsqueeze(-1).unsqueeze(-1))), acc

    def heatbath_defect(self, staples, rand, factor):
        k = torch.sqrt(sun.SUN_determinant(staples).abs())
        #this guarantees that links whose sum of staples has zero determinant do not generate nans. To do the heatbath a random SU(2) matrix is generated instead.
        k0 = k > 0.0
        k = k * k0 + torch.ones_like(k) * ~k0
        xi = factor / (self.beta * k)

        #WARNING: if rand[:,0] and one of the two other rand are simultaneously 0 a NaN occurs in the gradient of mod. Probably very unlikely
        #the second addend generates a random SU(2) matrix for staples that have zero determinant
        lam2 = (- 0.5 * xi * (torch.log(1.0 - rand[:,0,:]) + torch.cos(2.0 * torch.pi *(1.0 - rand[:,1,:]))**2 * torch.log(1.0 - rand[:,2,:]))) #* k0 #+ rand[:,0,:]**2 * ~k0
        acc = (1.0 - lam2 >= rand[:,3,:]**2)
        x0 = ((1.0 - 2.0 * lam2) * k0) * acc + (1.0 - 2.0 * rand[:,0,:]) * ~k0

        #x0=x0.detach() # This detach prevents (unlikely) NaN in the gradient of the square root mod
        stheta = torch.sqrt(1.0 - (1.0 - 2.0 * rand[:,4,:])**2)
        mod = torch.sqrt(1.0 - x0**2)

        A = torch.complex(x0, mod * (1.0 - 2.0 * rand[:,4,:])).unsqueeze(-1)
        B = (stheta*mod*torch.complex(torch.cos(2.0 * torch.pi * rand[:,5,:]), torch.sin(2.0 * torch.pi * rand[:,5,:]))).unsqueeze(-1)
        Xa = torch.ones(k.shape + (1,), dtype=A.dtype, device=A.device) * A
        Xb = torch.ones(k.shape + (1,), dtype=B.dtype, device=B.device) * B
        X = torch.cat((Xa,Xb),-1)

        v1 = torch.conj((X * _su2_sign(len(X.shape) - 1, X.device, X.real.dtype)).roll(1,-1))
        Xmat = torch.cat((X.unsqueeze(-2),v1.unsqueeze(-2)),-2)
        acc = acc.unsqueeze(-1).unsqueeze(-1)
        k0 = k0.unsqueeze(-1).unsqueeze(-1)

        return sun.SUN_mul(Xmat, sun.SUN_dagger(staples / k.unsqueeze(-1).unsqueeze(-1))) * k0 + Xmat * ~k0, acc

    def SU2_link_update_hb(self, cfgs, mu, rand, defect):
        staples = self.sum_of_staples(cfgs, mu, defect)
        
        Xmat, acc = self.heatbath(staples, rand[:,0:6,mu,:], 2.0)

        new_cfgs = Xmat * acc + cfgs * ~acc
        return new_cfgs.unsqueeze(1)

    def SU2_link_update_over(self, cfgs, mu, defect):
        staples = self.sum_of_staples(cfgs, mu, defect)
        k = torch.sqrt(sun.SUN_determinant(staples).abs())

        #this guarantees that links whose sum of staples has zero determinant do not generate nans
        k0 = k > 0
        k = k * k0 + torch.ones_like(k) * ~k0
        k0 = k0.unsqueeze(-1).unsqueeze(-1)

        staples = sun.SUN_dagger(staples / k.unsqueeze(-1).unsqueeze(-1))
        new_cfgs = sun.SUN_mul(sun.SUN_mul(staples, sun.SUN_dagger(cfgs)), staples) * k0 + cfgs * ~k0
        return new_cfgs.unsqueeze(1)

    def SU3_link_update_hb(self, cfgs, mu, rand, defect):
        staples = checkpoint.checkpoint(self.sum_of_staples, cfgs, mu, defect, use_reentrant=False)
        phb_steps = 3
        new_cfgs = cfgs[:,mu]
        sel = torch.tensor([0, 1, 2], device=new_cfgs.device)
        for phb in range(phb_steps):
            Rk = SU3_compute_RK(new_cfgs, staples, sel, phb)
            Ak, acc = self.heatbath(Rk, rand[:,6*phb:6*(phb+1),mu,:], 3.0 / 2.0)
            Ak = self.SU2toSU3(Ak, phb)
            new_cfgs = sun.SUN_mul(Ak, new_cfgs) * acc + new_cfgs * ~acc
        return new_cfgs.unsqueeze(1)

    def SU3_link_update_over(self, cfgs, mu, defect):
        staples = checkpoint.checkpoint(self.sum_of_staples, cfgs, mu, defect, use_reentrant=False)
        phb_steps = 3
        new_cfgs = cfgs[:,mu]
        sel = torch.tensor([0, 1, 2], device=new_cfgs.device)
        for phb in range(phb_steps):
            Rk = SU3_compute_RK(new_cfgs, staples, sel, phb)
            k = torch.sqrt(sun.SUN_determinant(Rk).abs())

            #this guarantees that links whose sum of staples has zero determinant do not generate nans
            k0 = k > 0
            k = k * k0 + torch.ones_like(k) * ~k0

            Rk = Rk / k.unsqueeze(-1).unsqueeze(-1)
            i0 = torch.tensor([0], device=Rk.device)
            i1 = torch.tensor([1], device=Rk.device)
            a = (SU2_element(Rk,i1,i1) * SU2_element(Rk,i1,i1) + SU2_element(Rk,i0,i1) * SU2_element(Rk,i1,i0)).unsqueeze(-1)
            b = (-SU2_element(Rk,i1,i1) * SU2_element(Rk,i0,i1) - SU2_element(Rk,i0,i1) * SU2_element(Rk,i0,i0)).unsqueeze(-1)
            v0 = torch.cat((a,b),-1)
            v1 = torch.conj((v0 * _su2_sign(len(v0.shape) - 1, v0.device, v0.real.dtype)).roll(1,-1))
            Ak = torch.cat((v0.unsqueeze(-2),v1.unsqueeze(-2)),-2)
            #bring Ak back to SU(3)
            Ak = self.SU2toSU3(Ak, phb)

            #new link is Ak*old_link
            #k=0 cases are left untouched
            k0 = k0.unsqueeze(-1).unsqueeze(-1)
            new_cfgs = sun.SUN_mul(Ak, new_cfgs) * k0 + new_cfgs * ~k0
        return new_cfgs.unsqueeze(1)

    #haar measure sampling (UNTESTED)
    def haar_heatbath(self, rand, cfgs_shape):
        x0 = rand[:,0,:]
        acc = (1.0 - x0**2 >= rand[:,3,:]**2)

        x0 = x0 * acc
        stheta = torch.sqrt(1.0 - rand[:,4,:]**2)
        mod = torch.sqrt(1.0 - x0**2)

        A = torch.complex(x0, mod*rand[:,4,:]).unsqueeze(-1)
        B = (stheta*mod*torch.complex(torch.cos(rand[:,5,:]), torch.sin(rand[:,5,:]))).unsqueeze(-1)
        Xa = torch.ones(cfgs_shape[:-2] + (1,), dtype=A.dtype, device=A.device) * A
        Xb = torch.ones(cfgs_shape[:-2] + (1,), dtype=B.dtype, device=B.device) * B
        v0 = torch.cat((Xa,Xb),-1)

        v1 = torch.conj((v0 * _su2_sign(len(v0.shape) - 1, v0.device, v0.real.dtype)).roll(1,-1))
        Xmat = torch.cat((v0.unsqueeze(-2),v1.unsqueeze(-2)),-2)
        acc = acc.unsqueeze(-1).unsqueeze(-1)

        return Xmat, acc

    def SU2_link_haar_hb(self, cfgs, rand):
        Xmat, acc = self.haar_heatbath(rand[:,0:6], cfgs.shape)
        new_cfgs = Xmat * acc + cfgs * ~acc
        return new_cfgs

    def SU3_link_haar_hb(self, cfgs, rand):
        phb_steps = 3
        new_cfgs = cfgs
        for phb in range(phb_steps):
            Xmat, acc = self.haar_heatbath(rand[:,6*phb:6*(phb+1),:], cfgs.shape)
            Ak = sun.SU2toSU3(Xmat, phb)
            new_cfgs = sun.SUN_mul(Ak, new_cfgs) * acc + new_cfgs * ~acc

        return new_cfgs

def SU2_element(mat, i, j):
    return torch.index_select(torch.index_select(mat,-2,i),-1,j).squeeze(-1).squeeze(-1)

def SU3_compute_RK(new_cfgs, staples, sel, phb):
    sel2 = sel[sel != phb]
    Rk = sun.SUN_mul(new_cfgs, staples)# SUN_dagger(staples))

    #generate SU(2) subgroup matrix from Rk
    a = 0.5*(SU2_element(Rk,sel2[0],sel2[0]) + torch.conj(SU2_element(Rk,sel2[1],sel2[1]))).unsqueeze(-1)
    b = 0.5*(SU2_element(Rk,sel2[0],sel2[1]) - torch.conj(SU2_element(Rk,sel2[1],sel2[0]))).unsqueeze(-1)
    v0 = torch.cat((a,b),-1)
    v1 = torch.conj((v0 * _su2_sign(len(v0.shape) - 1, v0.device, v0.real.dtype)).roll(1,-1))
    Rk = torch.cat((v0.unsqueeze(-2),v1.unsqueeze(-2)),-2)
    return Rk


class DefectMCMCAdapter:
    """
    Runtime-parameterized defect MCMC/action adapter.

    This exposes the same interface used by O-REX and NEO-REX:
    - set_parameter(parameter, parameter_name=...)
    - action(x, parameter, parameter_name=...)
    - __call__(x, nsteps=..., ...)
    """

    def __init__(
        self,
        flow_pars,
        beta: float,
        defect_mask: torch.Tensor,
        batch_size: int,
        parameter_name: str = "bc",
        default_parameter: float = 1.0,
    ):
        self.flow_pars = flow_pars
        self.beta = float(beta)
        self.defect_mask = defect_mask
        self.batch_size = int(batch_size)
        self.parameter_name = str(parameter_name)
        self._current_parameter: Optional[torch.Tensor] = torch.full(
            (self.batch_size,),
            float(default_parameter),
            dtype=torch.float64,
        )
        self._hbor_cache: dict[float, HBOR] = {}
        self._action_cache: dict[float, Wilson_action] = {}

    def _normalize_parameter(
        self,
        parameter: torch.Tensor,
        *,
        device: Optional[torch.device] = None,
        expected_batch_size: Optional[int] = None,
        allow_any_batch: bool = False,
    ) -> torch.Tensor:
        p = torch.as_tensor(parameter, dtype=torch.float64)
        expected = self.batch_size if expected_batch_size is None else int(expected_batch_size)
        if p.dim() == 0:
            p = p.expand(expected)
        if p.dim() != 1:
            raise ValueError(
                f"parameter must be a scalar or 1D tensor, got {tuple(p.shape)}"
            )
        if not allow_any_batch and int(p.shape[0]) != expected:
            raise ValueError(
                f"parameter must have shape ({expected},), got {tuple(p.shape)}"
            )
        if device is not None:
            p = p.to(device=device)
        return p.detach()

    def _get_hbor(self, defect_par: float) -> HBOR:
        if defect_par not in self._hbor_cache:
            self._hbor_cache[defect_par] = HBOR(
                self.flow_pars,
                beta=self.beta,
                defect_par=defect_par,
            )
        return self._hbor_cache[defect_par]

    def _get_action(self, defect_par: float) -> Wilson_action:
        if defect_par not in self._action_cache:
            self._action_cache[defect_par] = Wilson_action(
                self.flow_pars,
                beta=self.beta,
                defect_par=defect_par,
            )
        return self._action_cache[defect_par]

    def set_parameter(self, parameter: torch.Tensor, parameter_name: str = "bc"):
        if str(parameter_name) != self.parameter_name:
            raise ValueError(
                f"Unsupported parameter_name={parameter_name!r}, expected {self.parameter_name!r}"
            )
        self._current_parameter = self._normalize_parameter(parameter, allow_any_batch=True)

    def action(
        self,
        x: torch.Tensor,
        parameter: torch.Tensor,
        parameter_name: str = "bc",
    ) -> torch.Tensor:
        if str(parameter_name) != self.parameter_name:
            raise ValueError(
                f"Unsupported parameter_name={parameter_name!r}, expected {self.parameter_name!r}"
            )
        p = self._normalize_parameter(
            parameter,
            device=x.device,
            expected_batch_size=int(x.shape[0]),
        )
        out = torch.empty(x.shape[0], device=x.device, dtype=torch.float64)
        defect_mask = self.defect_mask.to(device=x.device)
        for p_val in torch.unique(p):
            sel = torch.nonzero(p == p_val, as_tuple=False).squeeze(-1)
            if sel.numel() == 0:
                continue
            act = self._get_action(float(p_val.item()))
            vals = act(x.index_select(0, sel), defect_mask).to(dtype=out.dtype)
            out.index_copy_(0, sel, vals)
        return out

    @torch.no_grad()
    def __call__(
        self,
        x: torch.Tensor,
        nsteps: int = 1,
        ensembles: bool = False,
        use_tqdm: bool = False,
    ) -> torch.Tensor:
        del ensembles, use_tqdm
        if self._current_parameter is None:
            raise RuntimeError("Call set_parameter(...) before using DefectMCMCAdapter.")

        p = self._current_parameter.to(device=x.device)
        out = x
        mask = self.flow_pars.mask.to(device=x.device)
        defect_mask = self.defect_mask.to(device=x.device)
        for _ in range(int(nsteps)):
            out_next = out.clone()
            for p_val in torch.unique(p):
                sel = torch.nonzero(p == p_val, as_tuple=False).squeeze(-1)
                if sel.numel() == 0:
                    continue
                updater = self._get_hbor(float(p_val.item()))
                updated = updater(
                    out.index_select(0, sel),
                    mask,
                    defect_mask,
                )
                out_next.index_copy_(0, sel, updated)
            out = out_next
        return out
