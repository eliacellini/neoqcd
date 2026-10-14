import torch
import numpy as np
import time
import os
import tempfile
import warnings
from neoqcd.obs import Obs
import neoqcd.sun_utils as sun
from neoqcd.smearing_jacobian import LOGDET_VERSION
#from scipy.special import logsumexp
from scipy.optimize import fsolve

### flow utils

def get_lr(optimizer):
    for p in optimizer.param_groups:
        return p["lr"]

def grab(var):
    if torch.is_tensor(var):
        return var.detach().cpu().numpy()
    else:
        return var

def _uses_exact_classic_smearing(model):
    return any(
        type(module).__name__ == "Smearing" and getattr(module, "N", None) == 3
        for module in model.modules()
    )


def _validate_smearing_checkpoint(model, checkpoint, allow_legacy_smearing=False):
    if not _uses_exact_classic_smearing(model):
        return
    version = checkpoint.get("smearing_jacobian_version")
    if version == LOGDET_VERSION:
        return
    if allow_legacy_smearing:
        warnings.warn(
            "Loading classic SU(3) smearing weights without the current full Haar-logdet version; "
            "the stored optimizer state may have been trained with a different determinant loss.",
            RuntimeWarning,
        )
        return
    raise ValueError(
        "This model uses classic SU(3) smearing with the full Haar logdet, but the checkpoint "
        f"has smearing_jacobian_version={version!r}. Pass allow_legacy_smearing=True only after "
        "intentionally accepting the old determinant convention."
    )


def save(model, optimizers, path, schedulers=None):
    checkpoint = {
        "checkpoint_format_version": 2,
        "model_state_dict": model.state_dict(),
        "optimizers_state_dict": [opt.state_dict() for opt in optimizers],
        "schedulers_state_dict": (
            None if schedulers is None else [scheduler.state_dict() for scheduler in schedulers]
        ),
    }
    if _uses_exact_classic_smearing(model):
        checkpoint["smearing_jacobian_version"] = LOGDET_VERSION

    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    fd, temporary_path = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def load_weights(model, path, device=None, allow_legacy_smearing=False):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    checkpoint = torch.load(path, map_location=device)
    _validate_smearing_checkpoint(model, checkpoint, allow_legacy_smearing)
    model.load_state_dict(checkpoint['model_state_dict'])


def load_optimizer(model, optimizers, path, device=None, schedulers=None,
                   allow_legacy_smearing=False):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    checkpoint = torch.load(path, map_location=device)
    _validate_smearing_checkpoint(model, checkpoint, allow_legacy_smearing)
    opts_sd = checkpoint['optimizers_state_dict']
    if len(opts_sd) != len(optimizers):
        raise ValueError(f"checkpoint has {len(opts_sd)} optimizers, got {len(optimizers)}")
    for opt, state in zip(optimizers, opts_sd):
        opt.load_state_dict(state)
    if schedulers is not None:
        schedulers_sd = checkpoint.get("schedulers_state_dict")
        if schedulers_sd is None or len(schedulers_sd) != len(schedulers):
            raise ValueError("checkpoint scheduler state does not match the supplied schedulers")
        for scheduler, state in zip(schedulers, schedulers_sd):
            scheduler.load_state_dict(state)

def write(history,root):
    ess_file = root + '_ESS.dat'
    loss_var_file = root + '_lossvar.dat'
    loss_file = root + '_loss.dat'

    with open(ess_file, 'a') as f:
        for item in history['ESS']:
            f.write("%.17g\n" % float(item))

    with open(loss_var_file, 'a') as f:
        for item in history['var_loss']:
            f.write("%.17g\n" % float(item))

    with open(loss_file, 'a') as f:
        for item in history['loss']:
            f.write("%.17g\n" % float(item))

def print_metrics(history_file,history, era, epoch, avg_last_N_epochs, t0):
    with open(history_file, 'a') as f:
        f.write(f'\n == Era {era} | Epoch {epoch} metrics ==\n')
        for key, val in history.items():
            avgd = np.mean(val[-avg_last_N_epochs:])
            f.write(f'\t{key} {avgd:g}\n')
        f.write(str(time.time()-t0))

### Work/data analysis

def analysis_and_output(work, work2, expw, expw2, heat, action, actionrw, action0, runs, details, log_file, dat_file):

    #gamma_method analysis + print .log file

    sval=2.0
    #Loss
    work_obs = Obs(work, runs)
    gamma_method(work_obs, sval, log_file, 'work')
    
    #variance
    varw_obs = Obs(work2, runs) - work_obs**2
    gamma_method(varw_obs, sval, log_file, 'var')
    
    #heat_s
    heat_obs = Obs(heat, runs)
    gamma_method(heat_obs, sval, log_file, 'heat')
    
    #deltaF
    expw_obs = Obs(expw, runs)
    expw_obs.gamma_method(S=sval)
    deltaF = -np.log(expw_obs) + work[0][0]
    gamma_method(deltaF, sval, log_file, 'deltaF')
    
    actrw_obs = Obs(actionrw, runs) / expw_obs
    gamma_method(actrw_obs, sval, log_file, 'action_rw')
    
    act_obs = Obs(action, runs)
    gamma_method(act_obs, sval, log_file, 'action')
    
    if action0 is not None:
        act0_obs = Obs(action0, runs)
        gamma_method(act0_obs, sval, log_file, 'action_0')
    
    #heat_tot
    heat_tot_obs = heat_obs - act_obs + actrw_obs
    gamma_method(heat_tot_obs, sval, log_file, 'heat_tot')
    
    #ESS
    expw2_obs = Obs(expw2, runs)
    ESS_obs = expw_obs**2 / expw2_obs
    gamma_method(ESS_obs, sval, log_file, 'ESS')
    
    #print .dat file
    #work \DeltaF ESS var(work) action action_rw heat heat_tot

    with open(dat_file, 'a') as fff:
        fff.write(details + ' ' + str(work_obs.value) + ' ' + str(work_obs.dvalue) + ' ' 
                  + str(deltaF.value) + ' ' + str(deltaF.dvalue) + ' ' 
                  + str(ESS_obs.value) + ' ' + str(ESS_obs.dvalue) + ' ' 
                  + str(varw_obs.value) + ' ' + str(varw_obs.dvalue) + ' ' 
                  + str(act_obs.value) + ' ' + str(act_obs.dvalue) + ' ' 
                  + str(actrw_obs.value) + ' ' + str(actrw_obs.dvalue) + ' ' 
                  + str(heat_obs.value) + ' ' + str(heat_obs.dvalue) + ' ' 
                  + str(heat_tot_obs.value) + ' ' + str(heat_tot_obs.dvalue) + '\n')

def gamma_method(obs, sval, log_file, obsname):
    #format
    #Obs, dObs, ddObs, tau_int(Obs), dtau_int(Obs)
    obs.gamma_method(S=sval)
    with open(log_file, 'a') as ff:
        ff.write(obsname + '    ' + str(obs.value)+' '+str(obs.dvalue)+' '+str(obs.ddvalue)+' '+str(np.mean(list(obs.e_tauint.values())))+' '+str(np.mean(list(obs.e_dtauint.values())))+'\n')
    

### Scale setting Necco-Sommer 5.7<beta<6.92

def ar0(beta):
    return np.exp( - 1.6804 - 1.7331 * (beta - 6.0) + 0.7849 * (beta - 6.0)**2 - 0.4428 * (beta - 6.0)**3)

def betafromar0(ar0_input):
    return fsolve(lambda beta : ar0_input - ar0(beta) , 6.0*np.ones(len(ar0_input)))

### mask utils

def create_mask(D, T, L):
    mask_shape = (2*D, D, T)
    for i in range(1,D):
        mask_shape += (L,)

    mask = torch.zeros(mask_shape, dtype = int)
    for mu in range(D):
        for t in range(T):
            for x in range(L):
                if(D>2):
                    for y in range(L):
                        if(D>3):
                            for z in range(L):
                                parity = t + x + y + z
                                mask[parity%2+mu*2,mu,t,x,y,z] = 1
                        else:
                            parity = t + x + y
                            mask[parity%2+mu*2,mu,t,x,y] = 1
                else:
                    parity = t + x
                    mask[parity%2+mu*2,mu,t,x] = 1

    return mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

class Defect():
    def __init__(self, D, T, L, dsize, time_slice=4, space_slice=4):
        self.D = D
        self.T = T
        self.L = L
        self.dsize = dsize
        self.time_slice = time_slice
        self.space_slice = space_slice

    def create_defect_mask(self):
    
        kappa_shape = (self.D, self.T)
        for _ in range(1, self.D):
            kappa_shape += (self.L,)
        drange = range(self.space_slice, self.dsize+self.space_slice)

        kappa_mask_d = torch.zeros(kappa_shape, dtype = int)

        for x in drange:
            if (self.D>2):
                for y in drange:
                    if (self.D>3):
                        for z in drange:
                            kappa_mask_d[0, self.time_slice, x, y, z] = 1
                    else:
                        kappa_mask_d[0, self.time_slice, x, y] = 1
            else:
                kappa_mask_d[0, self.time_slice, x] = 1

        kappa_mask_nod = 1 - kappa_mask_d
        kappa_mask = torch.cat((kappa_mask_nod.unsqueeze(0), kappa_mask_d.unsqueeze(0)), 0)

        return kappa_mask

    def cut_defect(self, phi, field='cfgs', buffer=2):
        t = self.time_slice
        s = self.space_slice
        spatial = tuple(
            slice(s - buffer, s + self.dsize + buffer)
            for _ in range(self.D - 1)
        )
        if field == 'cfgs':
            dphi = phi[(slice(None), slice(None), slice(t - buffer, t + buffer)) + spatial + (slice(None), slice(None))]
        elif field == 'mask':
            dphi = phi[(slice(None), slice(None), slice(None), slice(t - buffer, t + buffer)) + spatial]
        else:
            raise ValueError("field must be 'cfgs' or 'mask'")
        return dphi

    def site_indices(self, buffer, device=None):
        """Map sites of a cut box back to flattened sites of the full lattice."""
        volume = self.T * self.L ** (self.D - 1)
        ids = torch.arange(volume, device=device).reshape(
            (1, 1, self.T) + (self.L,) * (self.D - 1)
        )
        t, s = self.time_slice, self.space_slice
        slices = (slice(None), slice(None), slice(t - buffer, t + buffer)) + tuple(
            slice(s - buffer, s + self.dsize + buffer) for _ in range(self.D - 1)
        )
        return ids[slices][0, 0]

    def box_interior_mask(self, buffer, device=None):
        """Select links whose complete staple remains inside this nonperiodic box."""
        volume = self.T * self.L ** (self.D - 1)
        lattice = torch.arange(volume, device=device).reshape(
            (self.T,) + (self.L,) * (self.D - 1)
        )
        box = self.site_indices(buffer, device=device)

        def exact_shift(amount, direction):
            return torch.roll(box, amount, dims=direction) == torch.roll(
                lattice, amount, dims=direction
            ).flatten()[box]

        directions = []
        for mu in range(self.D):
            inside = exact_shift(1, mu)
            for nu in range(self.D):
                if nu != mu:
                    inside = inside & exact_shift(1, nu) & exact_shift(-1, nu)
            directions.append(inside)
        return torch.stack(directions)[None, ..., None, None]

    def embedding(self, small_cfgs, cfgs, buffer=2):
        t = self.time_slice
        s = self.space_slice
        out = cfgs.clone()
        center_spatial = tuple(slice(s - 1, s + self.dsize + 1) for _ in range(self.D - 1))
        local_center = tuple(slice(buffer - 1, buffer + self.dsize + 1) for _ in range(self.D - 1))
        out[(slice(None), 0, t) + center_spatial] = small_cfgs[(slice(None), 0, buffer) + local_center]
        target = tuple(slice(s, s + self.dsize + 1) for _ in range(self.D - 1))
        local = tuple(slice(buffer, buffer + self.dsize + 1) for _ in range(self.D - 1))
        for d in range(1, self.D):
            out[(slice(None), d, t) + target] = small_cfgs[(slice(None), d, buffer) + local]
            out[(slice(None), d, t - 1) + target] = small_cfgs[(slice(None), d, buffer - 1) + local]
        return out


class DefectBox:
    """A nested MCMC box; only links with every staple wholly inside are updated."""

    def __init__(self, defect, mask, defect_mask, buffer, outer=None):
        self.defect = defect
        self.buffer = int(buffer)
        self.D = defect.D
        self.outer = outer
        interior = defect.box_interior_mask(buffer, device=mask.device)
        self.mask = defect.cut_defect(mask, field='mask', buffer=buffer) & interior
        self.links = sun.MaskedLinks(self.mask, self.D)
        self.defect_mask = (
            None if defect_mask is None
            else defect.cut_defect(defect_mask, buffer=buffer)
        )

        sites = defect.site_indices(buffer, device=mask.device).flatten()
        volume = defect.T * defect.L ** (self.D - 1)
        box_volume = int(sites.numel())
        box_links = torch.cat(self.links.links)
        lattice_links = (box_links // box_volume) * volume + sites[box_links % box_volume]
        if outer is None:
            self.box_links = box_links
            self.lattice_links = lattice_links
            self.target_links = lattice_links
            return

        if outer.buffer <= self.buffer:
            raise ValueError(
                f"the box with buffer {self.buffer} is not inside buffer {outer.buffer}"
            )
        outer_sites = defect.site_indices(outer.buffer, device=mask.device).flatten()
        if not bool(torch.isin(lattice_links, outer.lattice_links).all()):
            raise ValueError(
                f"the box with buffer {self.buffer} is not inside buffer {outer.buffer}"
            )
        outer_positions = torch.full((volume,), -1, dtype=torch.long, device=mask.device)
        outer_positions[outer_sites] = torch.arange(outer_sites.numel(), device=mask.device)
        positions = outer_positions[sites[box_links % box_volume]]
        if bool((positions < 0).any()):
            raise ValueError(
                f"the box with buffer {self.buffer} is not inside buffer {outer.buffer}"
            )
        self.box_links = box_links
        self.lattice_links = lattice_links
        self.target_links = (box_links // box_volume) * outer_sites.numel() + positions

    def cut(self, cfgs):
        if self.outer is None:
            return self.defect.cut_defect(cfgs, buffer=self.buffer)
        shift = self.outer.buffer - self.buffer
        return cfgs[(slice(None), slice(None)) + (slice(shift, -shift),) * self.D +
                    (slice(None), slice(None))]

    def embed(self, small_cfgs, cfgs):
        updated = sun.flat_links(small_cfgs, self.D).index_select(1, self.box_links)
        return sun.flat_links(cfgs, self.D).index_copy(
            1, self.target_links, updated
        ).reshape(cfgs.shape)

def create_around_defect_mask(D, dsize, buffer=2):

    time_slice = buffer
    kappa_shape = (D, 2*buffer)
    for _ in range(1, D):
        kappa_shape += (dsize+2*buffer,)
    drange = range(buffer-1, dsize+buffer+1)

    kappa_mask_d = torch.zeros(kappa_shape, dtype = int)

    for x in drange:
        if (D>2):
            for y in drange:
                if (D>3):
                    for z in drange:
                        kappa_mask_d[0, time_slice, x, y, z] = 1
                else:
                    kappa_mask_d[0, time_slice, x, y] = 1
        else:
            kappa_mask_d[0, time_slice, x] = 1

    drange = range(buffer, dsize+buffer+1)
    for d in range(1, D):
        for x in drange:
            if (D>2):
                for y in drange:
                    if (D>3):
                        for z in drange:
                            kappa_mask_d[d, time_slice, x, y, z] = 1
                            kappa_mask_d[d, time_slice-1, x, y, z] = 1
                    else:
                        kappa_mask_d[d, time_slice, x, y] = 1
                        kappa_mask_d[d, time_slice-1, x, y] = 1
            else:
                kappa_mask_d[d, time_slice, x] = 1
                kappa_mask_d[d, time_slice-1, x] = 1

    return kappa_mask_d.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

def create_defect_plaquette_mask(D, dsize, defect_mask, buffer=2):

    defect = defect_mask[1]
    defect_plaq_shape = (2*D, D*(D-1)//2, 2*buffer)
    for _ in range(1, D):
        defect_plaq_shape += (dsize+2*buffer,)

    defect_plaq_mask = torch.zeros(defect_plaq_shape, dtype = int)

    for mu in range(D):
        for nu in range(D):
            if nu != mu:
                defect_plaq_mask[2*mu, 2 * sun.plaq_index(D, mu, nu), :] = torch.maximum( torch.maximum(defect[mu], torch.roll(defect, 1, dims=(-D + mu))[nu]), torch.maximum(torch.roll(defect, 1, dims=(-D + nu))[mu], defect[nu]) ).unsqueeze(0).unsqueeze(0)
                defect_plaq_mask[2*mu+1, 2 * sun.plaq_index(D, mu, nu), :] = torch.maximum( torch.maximum(defect[mu], torch.roll(defect, 1, dims=(-D + mu))[nu]), torch.maximum(torch.roll(defect, 1, dims=(-D + nu))[mu], defect[nu]) ).unsqueeze(0).unsqueeze(0)
                defect_plaq_mask[2*mu, 2 * sun.plaq_index(D, mu, nu) + 1, :] = torch.maximum( torch.maximum(defect[mu], torch.roll(defect, (1,-1), dims=(-D + mu,-D + nu))[nu]), torch.maximum(torch.roll(defect, -1, dims=(-D + nu))[mu], torch.roll(defect, -1, dims=(-D + nu))[nu])).unsqueeze(0).unsqueeze(0)
                defect_plaq_mask[2*mu+1, 2 * sun.plaq_index(D, mu, nu) + 1, :] = torch.maximum( torch.maximum(defect[mu], torch.roll(defect, (1,-1), dims=(-D + mu,-D + nu))[nu]), torch.maximum(torch.roll(defect, -1, dims=(-D + nu))[mu], torch.roll(defect, -1, dims=(-D + nu))[nu])).unsqueeze(0).unsqueeze(0)
    
    return defect_plaq_mask

def create_defect_plaquette_mask_full(D, dsize, smeared_defect_mask, buffer=2):

    defect_plaq_shape = (2*D, D*(D-1)//2, 2*buffer)
    for _ in range(1, D):
        defect_plaq_shape += (dsize+2*buffer,)

    defect_plaq_mask_full = torch.zeros(defect_plaq_shape, dtype = int)

    for mu in range(D):
        for nu in range(D):
            if nu != mu:
                defect_plaq_mask_full[2*mu, 2 * sun.plaq_index(D, mu, nu), :] = smeared_defect_mask[mu,:]
                defect_plaq_mask_full[2*mu+1, 2 * sun.plaq_index(D, mu, nu), :] = smeared_defect_mask[mu,:]
                defect_plaq_mask_full[2*mu, 2 * sun.plaq_index(D, mu, nu) + 1, :] = smeared_defect_mask[mu,:]
                defect_plaq_mask_full[2*mu+1, 2 * sun.plaq_index(D, mu, nu) + 1, :] = smeared_defect_mask[mu,:]

    return defect_plaq_mask_full

def set_defect_rho_from_fits(xval, nstep, rhofitpars, rho_shape, defect_mask, D):
    rho = 0.0001 * torch.ones(rho_shape)
    t1b_spl = rhofitpars[0]
    t1e_spl = rhofitpars[1]
    t1c_spl = rhofitpars[2]
    t2_spl = rhofitpars[3]
    t2b_spl = rhofitpars[4]
    t2e_spl = rhofitpars[5]
    t2c_spl = rhofitpars[6]
    sp_spl = rhofitpars[7]
    ex_spl = rhofitpars[8]

    T = rho_shape[3]
    L = rho_shape[4]

    for sm in range(rho_shape[0]):
        for m in range(rho_shape[1]):
            for pl in range(rho_shape[2]):
                for t in range(rho_shape[3]):
                    for x in range(rho_shape[4]):
                        for y in range(rho_shape[5]):
                            for z in range(rho_shape[6]):
                                parity = t + x + y + z
                                mu = m//2
                                if m%2 == parity%2:
                                    if mu == 0: #link to be smeared is temporal
                                        if defect_mask[1, 0, t, x, y, z] == 1:#link to be smeared IS on defect
                                            nnx, nny, nnz = nn_plaq_i(pl, x, y, z, L)
                                            
                                            nu  = dir_plaq_i(pl)
                                            if (defect_mask[1, 0, t, nnx, nny, nnz] == 0):#temporal link of nearest neighbour in nu direction IS NOT on defect -- we are on the boundary of the defect in the nu direction
                                                nodef_ctr = 0
                                                for mu2 in range(1,4): #loop on nearest neighbours on spatial directions, count how many of them have the temporal link NOT on defect
                                                    nnx2, nny2, nnz2 = nn_dir_i(mu2, x, y, z, L)
                                                    if defect_mask[1, 0, t, nnx2, nny2, nnz2] == 0: 
                                                        nodef_ctr += 1
                                                    nnx3, nny3, nnz3 = nn_dir_i(-mu2, x, y, z, L)
                                                    if defect_mask[1, 0, t, nnx3, nny3, nnz3] == 0: 
                                                        nodef_ctr += 1 

                                                if nodef_ctr == 1: #on the face of the defect
                                                    rhoval = torch.tensor(t1b_spl(xval) / nstep) 
                                                elif nodef_ctr == 2: #on the edge of the defect
                                                    rhoval = torch.tensor(t1e_spl(xval) / nstep)
                                                elif nodef_ctr == 3: #on the corner of the defect
                                                    rhoval = torch.tensor(t1c_spl(xval) / nstep)
                                                else:
                                                    rhoval = 1e-8

                                                if rhoval > 1e-8:
                                                    rho[sm,m,pl,t,x,y,z] = rhoval

                                            else:#nearest neighbour temporal link IS on defect
                                                nodef_ctr = 0
                                                for mu2 in range(1,4): #loop on nearest neighbours on spatial directions, count how many of them have the temporal link NOT on defect
                                                    nnx2, nny2, nnz2 = nn_dir_i(mu2, x, y, z, L)
                                                    if defect_mask[1, 0, t, nnx2, nny2, nnz2] == 0: 
                                                        nodef_ctr += 1
                                                    nnx3, nny3, nnz3 = nn_dir_i(-mu2, x, y, z, L)
                                                    if defect_mask[1, 0, t, nnx3, nny3, nnz3] == 0: 
                                                        nodef_ctr += 1 

                                                if nodef_ctr == 0: #inside the defect
                                                    rhoval = torch.tensor(t2_spl(xval) / nstep) 
                                                elif nodef_ctr == 1: #on the face of the defect
                                                    rhoval = torch.tensor(t2b_spl(xval) / nstep)
                                                elif nodef_ctr == 2: #on the edge of the defect
                                                    rhoval = torch.tensor(t2e_spl(xval) / nstep)
                                                elif nodef_ctr == 3: #on the corner of the defect
                                                    rhoval = torch.tensor(t2c_spl(xval) / nstep)
                                                else:
                                                    rhoval = 1e-8

                                                if rhoval > 1e-8:
                                                    rho[sm,m,pl,t,x,y,z] = rhoval


                                                # nnx2, nny2, nnz2 = nn_dir_i(-nu, x, y, z, L)
                                                # if (defect_mask[1, 0, t, nnx2, nny2, nnz2] == 0): #if the other nearest neighbour (opposite direction) is not on defect
                                                #     rhoval = torch.tensor(t2b_spl(xval) / nstep) #boundary
                                                # else:
                                                #     rhoval = torch.tensor(t2_spl(xval) / nstep) #bulk
                                                # if rhoval > 1e-8:
                                                #     rho_rhoval[sm,m,pl,t,x,y,z] = torch.sqrt(rhoval)
                                        
                                        else: #link to be smeared IS NOT on defect
                                            nnx, nny, nnz = nn_plaq_i(pl, x, y, z, L)
                                            if (defect_mask[1, mu, t, nnx, nny, nnz] == 1): #if temporal link of nearest neighbour in nu direction is on defect
                                                rhoval = torch.tensor(ex_spl(xval) / nstep)
                                                if rhoval > 1e-8:
                                                    rho[sm,m,pl,t,x,y,z] = rhoval
                                    
                                    else: #link to be smeared is spatial
                                        nnx, nny, nnz = nn_dir_i(mu, x, y, z, L)
                                        def1 = defect_mask[1, 0, t, x, y, z]
                                        def2 = defect_mask[1, 0, t, nnx, nny, nnz]

                                        def3 = defect_mask[1, 0, (t+1)%T, x, y, z]
                                        def4 = defect_mask[1, 0, (t+1)%T, nnx, nny, nnz]

                                        #link is part of a plaquette with two temporal defect links
                                        if (def1 == 1 and def2 == 1 and pl == 2 * sun.plaq_index(D, mu, 0)) or (def3 == 1 and def4 == 1 and pl == (2 * sun.plaq_index(D, mu, 0) + 1)): 
                                            rhoval = torch.tensor(sp_spl(xval) / nstep)
                                            if rhoval > 1e-8:
                                                 rho[sm,m,pl,t,x,y,z] = rhoval
                                        #or link is part of a plaquette with only one temporal defect link
                                        elif (((def1 == 0 and def2 == 1) or (def1 == 1 and def2 == 0)) and pl == 2 * sun.plaq_index(D, mu, 0)) or (((def3 == 0 and def4 == 1) or (def3 == 1 and def4 == 0)) and pl == (2 * sun.plaq_index(D, mu, 0) + 1)):
                                            rhoval = torch.tensor(ex_spl(xval) / nstep)
                                            if rhoval > 1e-8:
                                                 rho[sm,m,pl,t,x,y,z] = rhoval
    return rho

def nn_plaq_i(pl, x, y, z, L):
    if pl == 0:
        return (L+x-1)%L, y, z
    elif pl == 1:
        return (x+1)%L, y, z
    elif pl == 2:
        return x, (L+y-1)%L, z
    elif pl == 3:
        return x, (y+1)%L, z
    elif pl == 4:
        return x, y, (L+z-1)%L
    elif pl == 5:
        return x, y, (z+1)%L
    else:
        return None

def dir_plaq_i(pl):
    if pl == 0:
        return 1
    elif pl == 1:
        return -1
    elif pl == 2:
        return 2
    elif pl == 3:
        return -2
    elif pl == 4:
        return 3
    elif pl == 5:
        return -3
    else:
        return None

def nn_dir_i(mu, x, y, z, L):
    if mu == 1:
        return (L+x-1)%L, y, z
    elif mu == 2:
        return x, (L+y-1)%L, z
    elif mu == 3:
        return x, y, (L+z-1)%L
    elif mu == -1:
        return (x+1)%L, y, z
    elif mu == -2:
        return x, (y+1)%L, z
    elif mu == -3:
        return x, y, (z+1)%L
    else:
        return None
