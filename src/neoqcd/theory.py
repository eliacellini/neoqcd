import torch
import neoqcd.sun_utils as sun

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
