import torch
import numpy as np
import pickle
from neoqcd.mcmc import NEMCMC_update
from neoqcd.smearing import CouplingLayer, DefectCouplingLayer, ResidualCouplingLayer
from neoqcd.utils import grab, set_defect_rho_from_fits
from neoqcd.nn import KISS_HTConv1D, HyperTimeConv1D, time_embedding,HTCNN

class Flow(torch.nn.Module):
    def __init__(self, flow_pars, flow_type='nemcmc', protocol_type='beta', beta=6.0, defect_mask=None, rhofit_file=None, scale_fac=0.0001):
        super().__init__()
        self.defect_mask = defect_mask

        if flow_pars.set_rho == 1:
            rhofitpars = self.init_rhofitpars(flow_pars.rho_shape_type, beta, flow_pars.defect.dsize, rhofit_file)
            if flow_pars.fit_train == 1 and flow_pars.rho_shape_type != 3:
                self.rhofitpars = torch.nn.Parameter(rhofitpars).to(flow_pars.device)
            else:
                self.rhofitpars = rhofitpars
        else:
            self.rhofitpars = None

        layers=[]
        step = 0
        snf_layer_type = str(getattr(flow_pars, "nf_layer_type", "smearing")).lower()
        for protocol_par in flow_pars.protocol:
            #add smearing layer
            if (flow_type == 'snf'):
                rho_step = self.initialize_rho(step, flow_pars).to(flow_pars.device)

                if snf_layer_type == "residual":
                    if flow_pars.rho_shape_type == 3:
                        raise ValueError("nf_layer_type='residual' is a standard SNF layer; do not use defect coupling")
                    smr_step = ResidualCouplingLayer(flow_pars, rho_step, (step+1)/flow_pars.nstep)
                elif flow_pars.rho_shape_type == 3:
                    smr_step = DefectCouplingLayer(flow_pars, rho_step, (step+1)/flow_pars.nstep)
                else:
                    smr_step = CouplingLayer(flow_pars, rho_step, (step+1)/flow_pars.nstep)
                layers.append(smr_step)

            #add nemcmc layer
            if (protocol_type == 'beta'):
                beta = protocol_par
                stb_step = NEMCMC_update(flow_pars, beta)
            elif (protocol_type == 'bc'):
                bc_coupling = protocol_par
                stb_step = NEMCMC_update(flow_pars, beta, defect_mask=defect_mask, defect_par=bc_coupling)
            layers.append(stb_step)

            step += 1
        
        self.layers = torch.nn.ModuleList(layers).to(flow_pars.device)

    def __call__(self, x, s0, flow_pars, int_meas=1):
        x, Q, logJ, int_obs = self.forward(x, s0, flow_pars, int_meas=int_meas)
        st = self.layers[-1].action(x, self.defect_mask)
        w = st - s0 - Q - 2.0*logJ
        return x, w, st, Q, 2.0*logJ, int_obs

    def _layer_rho(self, flow_pars, step):
        if flow_pars.fit_train == 1:
            return self.set_rho_fits(flow_pars, step)
        return None

    def _forward_layer(self, layer, x, flow_pars, step):
        rho_layer = self._layer_rho(flow_pars, step)
        if isinstance(layer, NEMCMC_update):
            return layer.forward(x, flow_pars, rho_layer)

        nstep = int(flow_pars.nstep)
        if nstep <= 0:
            beta_step = 0.0
            delta_step = 0.0
        else:
            idx = int(max(0, min(step, nstep - 1)))
            beta_step = float(flow_pars.protocol[idx])
            if nstep == 1:
                delta_step = 0.0
            elif idx < nstep - 1:
                delta_step = float(flow_pars.protocol[idx + 1] - flow_pars.protocol[idx])
            else:
                delta_step = float(flow_pars.protocol[idx] - flow_pars.protocol[idx - 1])

        return layer.forward(
            x,
            flow_pars,
            rho_layer,
            beta=beta_step,
            delta_beta=delta_step,
        )

    def forward(self, x, s0, flow_pars, int_meas=1):
        batch_size = x.shape[0]
        device = x.device
        dtype = s0.dtype if torch.is_tensor(s0) else (x.real.dtype if torch.is_complex(x) else x.dtype)

        Q = torch.zeros(batch_size, device=device, dtype=dtype)
        logJ = torch.zeros(batch_size, device=device, dtype=dtype)
        int_obs = torch.zeros((batch_size, int_meas, 5), device=device, dtype=dtype)

        stochastic_step = 0
        m = 0
        nstep_per_meas = max(1, flow_pars.nstep // int_meas)

        for layer in self.layers:
            x, dQ, dlogJ = self._forward_layer(layer, x, flow_pars, stochastic_step)
            Q += dQ
            logJ += dlogJ
            # index stochastic_step runs on stochastic updates
            if isinstance(layer, NEMCMC_update):
                if (stochastic_step + 1) % nstep_per_meas == 0 and m < int_meas:
                    st = layer.action(x, self.defect_mask)
                    int_obs[:, m, 0] = st - s0 - Q - 2.0 * logJ #work
                    int_obs[:, m, 1] = Q #heat
                    int_obs[:, m, 2] = 2.0 * logJ #jacobian
                    int_obs[:, m, 3] = st #action
                    int_obs[:, m, 4] = float(flow_pars.protocol[stochastic_step]) #coupling
                    m += 1
                stochastic_step += 1

        return x, Q, logJ, int_obs

    # regular call up to 2*tb-layer with nograd
    def up_to_block_nograd(self, x, s0, flow_pars, tb):
        if 2 * tb >= len(self.layers):
            raise IndexError(f"Target block {tb} is out of range for {len(self.layers)} layers")
        with torch.no_grad():
            x, Q, logJ = self.forward_up_to_layer(x, flow_pars, 2 * tb)
        x, dQ, dlogJ = self._forward_layer(self.layers[2 * tb], x, flow_pars, tb) #forward only on the smearing layer of the target block!
        Q += dQ  #unused for now
        logJ += dlogJ

        action_layer = None
        for layer in self.layers[2 * tb:]:
            if isinstance(layer, NEMCMC_update):
                action_layer = layer
                break
        if action_layer is None:
            raise RuntimeError("No stochastic update layer found to evaluate the action")

        st = action_layer.action(x, self.defect_mask)
        w = st - s0 - Q - 2.0*logJ
        return x, w, st, Q, 2.0*logJ

    #regular forward up to tl-th layer
    def forward_up_to_layer(self, x, flow_pars, tl):
        dtype = x.real.dtype if torch.is_complex(x) else x.dtype
        Q = torch.zeros(x.shape[0], device=x.device, dtype=dtype)
        logJ = torch.zeros(x.shape[0], device=x.device, dtype=dtype)
        stochastic_step = 0
        for layer in self.layers[0:tl]:
            x, dQ, dlogJ = self._forward_layer(layer, x, flow_pars, stochastic_step)
            Q += dQ
            logJ += dlogJ
            if isinstance(layer, NEMCMC_update):
                stochastic_step += 1
        return x, Q, logJ

    def compute_metrics(self, w):
        wd = w.detach()
        wdm = wd.mean()
        expwm = torch.mean(torch.exp(-(wd-wdm)))
        DF = - torch.log(expwm) + wdm
        ess = (expwm**2)/torch.mean(torch.exp(-2.0*(wd-wdm)))
        return DF, ess

    def sample_(self, x, s0):
        with torch.no_grad():
            return self(x, s0)

    #functions for setting rho from fits for standard coupling layers
    def rho_fit(self, x, a, b, d):
        #return a*np.tanh(b*x) + c - d*x
        return a*torch.tanh(b*x) - d*x

    def set_rho_fits(self, flow_pars, step):
        nstep = flow_pars.nstep
        x = (step+1)/nstep
        if flow_pars.rho_shape_type != 3:
            return self.rho_fit(x, self.rhofitpars[0], self.rhofitpars[1]*nstep, self.rhofitpars[2])/nstep * torch.ones(flow_pars.rho_shape)
            #return torch.sqrt(torch.tensor(rho_fit(x, a, b*nstep, c0 + c1*nstep, d)/nstep) * torch.ones(self.flow_pars.rho_shape))
        else:
            return set_defect_rho_from_fits(x, nstep, self.rhofitpars, flow_pars.rho_shape, flow_pars.small_defect_mask[-1], flow_pars.D)

    def initialize_rho(self, step, flow_pars):
        #setting smearing parameter from file
        if flow_pars.set_rho == 0:
            return torch.zeros(flow_pars.rho_shape)
        #initializing smearing parameter from fits
        elif flow_pars.set_rho == 1:
            return self.set_rho_fits(flow_pars, step)
        #or randomly
        elif flow_pars.set_rho == 2:
            return flow_pars.scale_fac * torch.rand(flow_pars.rho_shape)

    def init_rhofitpars(self, rho_shape_type, beta, dsize, rhofit_file):
        if rho_shape_type != 3:
            if rhofit_file is not None:
                return torch.tensor(np.loadtxt(rhofit_file))
            else:
                return torch.tensor([0.0001, 0.1, 0.0])
        else:
            if rhofit_file is not None:
                with open(rhofit_file + 'b' + str(beta) + 'd' + str(dsize) + 't1c_spl.pkl', 'rb') as f:
                    t1c_spl = pickle.load(f)
                with open(rhofit_file + 'b' + str(beta) + 'd' + str(dsize) + 't2c_spl.pkl', 'rb') as f:
                    t2c_spl = pickle.load(f)
                with open(rhofit_file + 'b' + str(beta) + 'd' + str(dsize) + 'sp_spl.pkl', 'rb') as f:
                    sp_spl = pickle.load(f)
                with open(rhofit_file + 'b' + str(beta) + 'd' + str(dsize) + 'ex_spl.pkl', 'rb') as f:
                    ex_spl = pickle.load(f)

                if dsize > 2:
                    with open(rhofit_file + 'b' + str(beta) + 'd' + str(dsize) + 't1b_spl.pkl', 'rb') as f:
                        t1b_spl = pickle.load(f)
                    with open(rhofit_file + 'b' + str(beta) + 'd' + str(dsize) + 't1e_spl.pkl', 'rb') as f:
                        t1e_spl = pickle.load(f)
                    with open(rhofit_file + 'b' + str(beta) + 'd' + str(dsize) + 't2_spl.pkl', 'rb') as f:
                        t2_spl = pickle.load(f)
                    with open(rhofit_file + 'b' + str(beta) + 'd' + str(dsize) + 't2b_spl.pkl', 'rb') as f:
                        t2b_spl = pickle.load(f)
                    with open(rhofit_file + 'b' + str(beta) + 'd' + str(dsize) + 't2e_spl.pkl', 'rb') as f:
                        t2e_spl = pickle.load(f)
                else:
                    t1b_spl = None
                    t1e_spl = None
                    t2_spl = None
                    t2b_spl = None
                    t2e_spl = None

                return [t1b_spl, t1e_spl, t1c_spl, t2_spl, t2b_spl, t2e_spl, t2c_spl, sp_spl, ex_spl]
            else:
                return None

    def print_rhofitpars(self, flow_pars, rhofit_file):
        if flow_pars.fit_train == 1:
            with open(rhofit_file, 'a') as ff:
                ff.write(str(self.rhofitpars[0]) + ' ' + str(self.rhofitpars[1]) + ' ' + str(self.rhofitpars[2]))

    def print_parameters(self, file, file_full, flow_pars, defect_plaq_mask, defect_plaq_mask_full):
        b = 0
        for layer in self.layers:
            if (b%2 == 0):#selecting only smearing
                if hasattr(layer, "smearing") and getattr(layer.smearing, "use_hyper_smearing", False):
                    rr_layer = layer.smearing.hyper_smearing().detach()
                else:
                    rr_layer = layer.rho_layer
                with open(file, 'a') as ff:
                    if (flow_pars.rho_shape_type != 3):
                        rr = torch.mean(rr_layer, dim=1) #average over masks
                    else:
                        rr = rr_layer*defect_plaq_mask #remove unused parameters in defect coupling layers
                    for sm in range(flow_pars.smearing_steps_per_layer):
                        par = grab(rr[sm,:])**2
                        if (flow_pars.rho_shape_type == 0 or flow_pars.rho_shape_type == 1):
                            for p in par:
                                ff.write(str(p) + ' ')
                        elif (flow_pars.rho_shape_type == 2):
                            par = np.mean(par, axis=(1,2,3,4)) #average over sites
                            for p in par:
                                ff.write(str(p) + ' ')
                        elif (flow_pars.rho_shape_type == 3):
                            for m in range(par.shape[0]):
                                for pl in range(par.shape[1]):
                                    for t in range(par.shape[2]):
                                        for x in range(par.shape[3]):
                                            for y in range(par.shape[4]):
                                                for z in range(par.shape[5]):
                                                    rhoval = par[m,pl,t,x,y,z]
                                                    parity = t + x + y + z
                                                    if rhoval > 0.0 and m%2 == parity%2:
                                                        ff.write(str(b//2) + ' ' + str(m) + ' ' + str(pl) + ' ' + str(t) + ' ' + str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(rhoval) + '\n')

                    #ff.write('\n')
                with open(file_full, 'a') as ff:
                    if (flow_pars.rho_shape_type == 3):
                        rr = rr_layer*defect_plaq_mask_full
                        for sm in range(flow_pars.smearing_steps_per_layer):
                            par = grab(rr[sm,:])**2
                            for m in range(par.shape[0]):
                                for pl in range(par.shape[1]):
                                    for t in range(par.shape[2]):
                                        for x in range(par.shape[3]):
                                            for y in range(par.shape[4]):
                                                for z in range(par.shape[5]):
                                                    rhoval = par[m,pl,t,x,y,z]
                                                    parity = t + x + y + z
                                                    if rhoval > 0.0 and m%2 == parity%2:
                                                        ff.write(str(b//2) + ' ' + str(m) + ' ' + str(pl) + ' ' + str(t) + ' ' + str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(rhoval) + '\n')
            b += 1

class FlowPars():
    def __init__(self, D, T, L, N, mask, protocol, batch_size, device=None, 
                 rho_shape_type=0, set_rho=0, fit_train=0, 
                 orsteps=0, updates_per_layer=1, 
                 hierarchical_levels=0, hierarchical_updates=None, hierarchical_rectangles=None,
                 smearing_steps_per_layer=1, rho_scale_fac=0.0158113883, use_nn=False, use_kiss = True,
		         F=9, rho_init_nn=1e-4, hidden_mlp=16, in_channels=60, out_channels=2, kernel_size=3, hidden_sizes=[8], 
                 residual = True, hyper=True,
                 use_hyper_smearing=False, hyper_smearing_mode="shared",
                 hyper_time_embedding_dim=8, hyper_hidden_dim=16, hyper_rho_init=1e-3,
                 hyper_depth=2, hyper_activation="silu",
                 hyper_normalize_by_nstep=True, hyper_rho_eps=0.0,
                 hyper_scale_by_delta=False,
                 hyper_rho_max=0.0,
                 nf_layer_type="smearing",
                 residual_include_imag=True, residual_quadratic=True,
                 residual_coeff_init=1e-3, residual_coeff_max=0.0,
                 defect=None, smeared_defect_mask=None, small_mask=None, small_defect_mask=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        self.N = N
        self.D = D
        self.T = T
        self.L = L
        self.mask = mask
        self.defect = defect
        self.smeared_defect_mask = smeared_defect_mask
        self.small_mask = small_mask
        self.small_defect_mask = small_defect_mask
        self.rho_shape_type = rho_shape_type

        self.protocol = protocol
        self.nstep = len(protocol)

        self.set_rho = set_rho
        self.fit_train = fit_train
        self.scale_fac = rho_scale_fac

        self.updates_per_layer = updates_per_layer
        self.hierarchical_levels = hierarchical_levels
        self.hierarchical_updates = [] if hierarchical_updates is None else hierarchical_updates
        self.hierarchical_rectangles = [] if hierarchical_rectangles is None else hierarchical_rectangles
        self.smearing_steps_per_layer = smearing_steps_per_layer
        self.orsteps = orsteps

        self.use_nn = use_nn
        self.residual = residual
        self.use_hyper_smearing = bool(use_hyper_smearing)
        self.hyper_smearing_mode = str(hyper_smearing_mode)
        self.hyper_time_embedding_dim = int(hyper_time_embedding_dim)
        self.hyper_hidden_dim = int(hyper_hidden_dim)
        self.hyper_depth = int(hyper_depth)
        self.hyper_activation = str(hyper_activation)
        self.hyper_rho_init = float(hyper_rho_init)
        self.hyper_normalize_by_nstep = bool(hyper_normalize_by_nstep)
        self.hyper_rho_eps = float(hyper_rho_eps)
        self.hyper_scale_by_delta = bool(hyper_scale_by_delta)
        self.hyper_rho_max = float(hyper_rho_max)
        self.nf_layer_type = str(nf_layer_type)
        self.residual_include_imag = bool(residual_include_imag)
        self.residual_quadratic = bool(residual_quadratic)
        self.residual_coeff_init = float(residual_coeff_init)
        self.residual_coeff_max = float(residual_coeff_max)
        if self.use_nn and not self.use_hyper_smearing:
            embedding = time_embedding(F, 1.0, device=device)
            layer =  KISS_HTConv1D if use_kiss else HyperTimeConv1D
            self.nn_pars = {'embedding': embedding, 'rho_init':rho_init_nn,'layer': layer,
                            'in_channels':in_channels,'out_channels':out_channels,'hidden_mlp':hidden_mlp,'device':device, 
                            'hidden_sizes': hidden_sizes, 'kernel_size':kernel_size,'residual':self.residual, 'hyper':hyper }
            self.net = self._init_net(**self.nn_pars)

        #shape of smearing parameter
        #isotropic
        if rho_shape_type == 0:
            self.rho_shape = (smearing_steps_per_layer, 2*D, 1, 1, 1, 1, 1)
        #anisotropic
        elif rho_shape_type == 1:
            self.rho_shape = (smearing_steps_per_layer, 2*D, D*(D-1)//2, 1, 1, 1, 1)
        #general
        elif rho_shape_type == 2:
            self.rho_shape = (smearing_steps_per_layer, 2*D, D*(D-1)//2, T, L, L, L)
        #defect
        elif rho_shape_type == 3:
            self.rho_shape = (smearing_steps_per_layer, 2*D, D*(D-1)//2, 4, defect.dsize+4, defect.dsize+4, defect.dsize+4)

        #configuration and Jacobian shapes
        if rho_shape_type == 3:
            effT = 4
            effL = defect.dsize + 4
        else:
            effT = T
            effL = L
        
        init_shape = [batch_size, D, T]
        jac_shape = [batch_size, effT]

        for i in range(D-1):
            init_shape += [L]
            jac_shape += [effL]
        init_shape += [N]
        jac_shape += [N*N, N*N]
        self.init_shape=tuple(init_shape)
        self.jac_shape=tuple(jac_shape)

        self.device = device

    def _init_net(self,**kwargs):
        return HTCNN(**kwargs)
    


    
