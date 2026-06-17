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

def SUN_trace(cfgs):
    return torch.einsum('...ii', cfgs)

def check_determinant_SUN(cfgs, tol):
    dtm = SUN_determinant(cfgs)
    return (dtm - 1.0).abs() > tol

def SUN_dagger(cfgs):
    return torch.conj(torch.transpose(cfgs,-2,-1))

def SUN_mul(U,V):
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
    pstaple = SUN_mul(torch.roll(cfgs,1,dims=(-D + mu - 2))[:,nu], SUN_dagger(torch.roll(cfgs,1,dims=(-D + nu - 2)))[:,mu])
    return SUN_mul(pstaple, SUN_dagger(cfgs[:,nu]))

def nstaple(cfgs, mu, nu, D):
    nstaple = SUN_mul(SUN_dagger(torch.roll(cfgs,(1,-1),dims=(-D + mu - 2,-D + nu - 2))[:,nu]), SUN_dagger(torch.roll(cfgs,-1,dims=(-D + nu - 2)))[:,mu])
    return SUN_mul(nstaple, torch.roll(cfgs,-1,dims=(-D + nu - 2))[:,nu])
        
#CHANGE THIS TO MAKE IT SYMMETRIC IN MU AND NU?
def plaq_index(D, mu, nu): 
    if D==4:
        return (nu + 3 - mu) % 4
        #if mu == 0:
        #    return (nu + 3) % 4
            #if nu == 1:
            #    return 0
            #elif nu == 2:
            #    return 1
            #elif nu == 3:
            #    return 2
        #elif mu == 1:
        #    return (nu + 2) % 4
            #if nu == 2:
            #    return 0
            #elif nu == 3:
            #    return 1
            #elif nu == 0:
            #    return 2
        #elif mu == 2:
        #    return (nu + 1) % 4
            #if nu == 3:
            #    return 0
            #elif nu == 0:
            #    return 1
            #elif nu == 1:
            #    return 2
        #elif mu == 3:
        #    return nu
            #if nu == 0:
            #    return 0
            #elif nu == 1:
            #    return 1
            #elif nu == 2:
            #    return 2
