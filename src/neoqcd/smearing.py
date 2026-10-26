import torch
import torch.utils.checkpoint as checkpoint
import numpy as np
import neoqcd.sun_utils as sun
from neoqcd.smearing_jacobian import SmearingStep as ExactSmearingStep
from neoqcd.smearing_jacobian import StepQuantities, exp_coefficients, inverse_smearing_step, smearing_step
from neoqcd.nn import HTCNN
from neoqcd.utils import set_defect_rho_from_fits


#functions for smearing
def xi0(w, w2):
    return torch.where(torch.abs(w) > 0.005, torch.sin(w)/w, 1. - 1./6.*w2*(1. - 1./20.*w2*(1. - 1./42.*w2)))

def xi1(w, w2):
    return torch.where(torch.abs(w) > 0.005, torch.cos(w)/w2 - torch.sin(w)/w**3,
                       -1./3. + w2*(1./30. + w2*(-1./840 + 1./45360.*w2)))

def xi2(w, w2, xizero, xione):
    return torch.where(torch.abs(w) > 0.005, 1./w2*(xizero + 3. * xione), -1./15. + w2*(1./210. - w2/7560.))

#stuff for Jacobian
def otimes(A,B):
    return torch.einsum('...ij,...kl->...ijkl',A,B)

def oplus(A,B):
    return torch.einsum('...kj,...il->...ijkl',A,B)

def starprod(A,B):
    return torch.einsum('...inml,...njkm->...ijkl',A,B)

def starprodmat(A,B):
    return torch.einsum('...ijkn,...nl->...ijkl',A,B)

def matstarprod(A,B):
    return torch.einsum('...in,...njkl->...ijkl',A,B)

def gradjacprod(A,B):
    return torch.einsum('...mn,...mijn->...ij',A,B) ###VERIFICARE INDICI!

def graddjacprod(A,B):
    return torch.einsum('...mnkl,...mijnkl->...ij',A,B) ###VERIFICARE INDICI!

def compute_d2expQdQ2(id, B1, B2, Q, Q2, dB1dQ, dB2dQ, df1dQ, df2dQ, f2):

    f2 = f2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    M = torch.einsum('...li,...jk,...ab->...aijklb', id, id, B1)
    M += torch.einsum('...lk,...aijb->...aijklb', Q, dB1dQ)
    M += torch.einsum('...li,...jk,...ab->...aijklb', id, Q, B2)
    M += torch.einsum('...li,...jk,...ab->...aijklb', Q, id, B2)
    M += torch.einsum('...lk,...aijb->...aijklb', Q2, dB2dQ)
    M += torch.einsum('...ji,...ak,...lb->...aijklb', df1dQ, id, id)
    M += torch.einsum('...ji,...ak,...lb->...aijklb', df2dQ, id, Q)
    M += f2 * torch.einsum('...ak,...li,...jb->...aijklb', id, id, id)
    M += torch.einsum('...ji,...lb,...ak->...aijklb', df2dQ, id, Q)
    M += f2 * torch.einsum('...ai,...jk,...lb->...aijklb', id, id, id)

    return M

def generate_coefficients(Q, Q2, id, oidid, oidQ, device, backward = False):
    if not backward:
        c0 = (Q * Q2.mT).sum((-2, -1)).real / 3.0
        c1 = 0.5 * sun.SUN_trace(Q2).real
        f, b1, b2 = exp_coefficients(c0, c1)
        f0, f1, f2 = (f[..., i, None, None] for i in range(3))
        B1 = sum(b1[..., i, None, None] * (id if i == 0 else Q if i == 1 else Q2) for i in range(3))
        B2 = sum(b2[..., i, None, None] * (id if i == 0 else Q if i == 1 else Q2) for i in range(3))
        d2expQdQ2 = torch.zeros_like(Q).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return f0, f1, f2, B1, B2, d2expQdQ2

    c0 = sun.SUN_determinant(Q)
    c1 = .5 * sun.SUN_trace(Q2)
    c0max = 2.0*(c1/3.0)**1.5

    sgnc0 = torch.real(torch.sgn(c0))
    c0 = torch.abs(c0)

    theta = torch.arccos(c0/c0max)
    u = torch.sqrt(c1/3.0) * torch.cos(theta/3.0)
    w = torch.sqrt(c1) * torch.sin(theta/3.0)
    u2 = u**2
    w2 = w**2
    cw = torch.cos(w)
    eu = torch.cos(u) + 1.j*torch.sin(u)
    eu2 = eu**2
    eum = eu**(-1)

    xizero = xi0(w, w2)
    xione = xi1(w, w2)

    h0 = (u2 - w2)*eu2 + eum*(8.*u2*cw + 2.j*u*(3.*u2 + w2)*xizero)
    h1 = 2.*u*eu2 - eum*(2.*u*cw - 1.j*(3.*u2 - w2)*xizero)
    h2 = eu2 - eum*(cw + 3.j*u*xizero)

    r10 = 2.*(u + 1.j*(u2 - w2))*eu2 + 2.*eum*(4.*u*(2.-1.j*u)*cw + 1.j*(9.*u2 + w2 - 1.j*u*(3.*u2 + w2)) * xizero)
    r11 = 2.*(1. + 2.j*u)*eu2 + eum*(-2.*(1.-1.j*u)*cw + 1.j*(6.*u + 1.j*(w2 - 3.*u2))*xizero)
    r12 = 2.j*eu2 + 1.j*eum*(cw - 3.*(1.-1.j*u)*xizero)
    r20 = -2.*eu2 + 2.j*u*eum*(cw + (1.+4.j*u)*xizero + 3.*u2*xione)
    r21 = -1.j*eum*(cw + (1.+2.j*u)*xizero - 3.*u2*xione)
    r22 = eum * (xizero - 3.j*u*xione)

    den = 9.*u2 - w2
    den2 = 2. * den**2
    v3u2mw2 = 3.*u2 - w2
    v15u2pw2 = 15.*u2 + w2

    f0 = h0 / den
    f1 = h1 / den
    f2 = h2 / den

    b10 = (2.*u*r10 + v3u2mw2 * r20 - 2.*v15u2pw2*f0) / den2
    b11 = (2.*u*r11 + v3u2mw2 * r21 - 2.*v15u2pw2*f1) / den2
    b12 = (2.*u*r12 + v3u2mw2 * r22 - 2.*v15u2pw2*f2) / den2
    b20 = (r10 - 3.*u*r20 - 24.*u*f0) / den2
    b21 = (r11 - 3.*u*r21 - 24.*u*f1) / den2
    b22 = (r12 - 3.*u*r22 - 24.*u*f2) / den2

    f0 = torch.where(sgnc0 > 0, f0, torch.conj(f0))
    f1 = torch.where(sgnc0 > 0, f1, -torch.conj(f1))
    f2 = torch.where(sgnc0 > 0, f2, torch.conj(f2))

    b10 = torch.where(sgnc0 > 0, b10, torch.conj(b10))
    b11 = torch.where(sgnc0 > 0, b11, -torch.conj(b11))
    b12 = torch.where(sgnc0 > 0, b12, torch.conj(b12))
    b20 = torch.where(sgnc0 > 0, b20, -torch.conj(b20))
    b21 = torch.where(sgnc0 > 0, b21, torch.conj(b21))
    b22 = torch.where(sgnc0 > 0, b22, -torch.conj(b22))

    B1 = b10.unsqueeze(-1).unsqueeze(-1) * id + b11.unsqueeze(-1).unsqueeze(-1) * Q +\
         + b12.unsqueeze(-1).unsqueeze(-1) * Q2
    B2 = b20.unsqueeze(-1).unsqueeze(-1) * id + b21.unsqueeze(-1).unsqueeze(-1) * Q +\
         + b22.unsqueeze(-1).unsqueeze(-1) * Q2

    d2expQdQ2 = torch.zeros(
        Q.shape,
        dtype=Q.dtype,
        device=Q.device,
    ).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    if backward:
        xitwo = xi2(w, w2, xizero, xione)

        dr10du = 2.*eu2*(1. + 4.j*u - 2.*u2 + 2.*w2) - 2.*eum*(4.*(-2. + 4.j*u + u2)*cw +\
                 1.j*(18.j*u2 + 3.*u2*u + 2.j*w2 + u*(-18. + w2))*xizero)
        dr10dw_w = -4.j*eu2 + 2.*eum*((1.j + u)*cw + (1.j - 7.*u + 4.j*u2)*xizero +\
                                      u2*(9.j + 3.*u)*xione)
        dr11du = 8.*eu2*(1.j - u) + eum*(2.*(2.j + u)*cw + (12.*u - 3.j*u2 + 1.j*(6. + w2))*xizero)
        dr11dw_w = eum*(-cw + (1. - 2.j*u)*xizero + 3.*u*(2.j + u)*xione)
        dr12du = -4.*eu2 + eum*(cw + 3.j*(2.j + u)*xizero)
        dr12dw_w = -eum*(1.j*xizero + 3.*(1.j + u)*xione)
        dr20du = -4.j*eu2 + 2.*eum*((1.j + u)*cw + (1.j - 7.*u + 4.j*u2)*xizero + 3.*u2*(3.j + u)*xione)
        dr20dw_w = 2.j*eum*u*((1 + 4.j*u)*xione - xizero - 3.*u2*xitwo)
        dr21du = eum*(-cw + (1. - 2.j*u)*xizero + 3.*u*(2.j + u)*xione)
        dr21dw_w = 1.j*eum*(-(1. + 2.j*u)*xione + xizero - 3.*u2*xitwo)
        dr22du = -eum*(3.*(1.j + u)*xione + 1.j*xizero)
        dr22dw_w = eum*(xione + 3.j*u*xitwo)

        df0du = (r10 - f0 * 18.* u) / den
        df0dw_w = (r20 + 2. * f0) / den
        df1du = (r11 - f1 * 18.* u) / den
        df1dw_w = (r21 + 2. * f1) / den
        df2du = (r12 - f2 * 18.* u) / den
        df2dw_w = (r22 + 2. * f2) / den

        v4udden = 4./den
        v36udden = 9.*v4udden*u

        db10du = 1./den2 * (2.*r10 + 2.*u*dr10du + 6.*u*r20 + v3u2mw2*dr20du - 60.*u*f0 - 2.*v15u2pw2*df0du) -\
                 b10*v36udden
        db11du = 1./den2 * (2.*r11 + 2.*u*dr11du + 6.*u*r21 + v3u2mw2*dr21du - 60.*u*f1 - 2.*v15u2pw2*df1du) -\
                 b11*v36udden
        db12du = 1./den2 * (2.*r12 + 2.*u*dr12du + 6.*u*r22 + v3u2mw2*dr22du - 60.*u*f2 - 2.*v15u2pw2*df2du) -\
                 b12*v36udden

        db10dw_w = 1./den2 * (2.*u*dr10dw_w - 2.*r20 + v3u2mw2*dr20dw_w - 4.*f0 - 2.*v15u2pw2*df0dw_w) +\
                   b10*v4udden
        db11dw_w = 1./den2 * (2.*u*dr11dw_w - 2.*r21 + v3u2mw2*dr21dw_w - 4.*f1 - 2.*v15u2pw2*df1dw_w) +\
                   b11*v4udden
        db12dw_w = 1./den2 * (2.*u*dr12dw_w - 2.*r22 + v3u2mw2*dr22dw_w - 4.*f2 - 2.*v15u2pw2*df2dw_w) +\
                   b12*v4udden

        db20du = 1./den2 * (dr10du - 3.*r20 - 3.*u*dr20du - 24.*f0 - 24.*u*df0du) - b20*v36udden
        db21du = 1./den2 * (dr11du - 3.*r21 - 3.*u*dr21du - 24.*f1 - 24.*u*df1du) - b21*v36udden
        db22du = 1./den2 * (dr12du - 3.*r22 - 3.*u*dr22du - 24.*f2 - 24.*u*df2du) - b22*v36udden

        db20dw_w = 1./den2 * (dr10dw_w - 3.*u*dr20dw_w - 24.*u*df0dw_w) + b20*v4udden
        db21dw_w = 1./den2 * (dr11dw_w - 3.*u*dr21dw_w - 24.*u*df1dw_w) + b21*v4udden
        db22dw_w = 1./den2 * (dr12dw_w - 3.*u*dr22dw_w - 24.*u*df2dw_w) + b22*v4udden

        den = den.unsqueeze(-1).unsqueeze(-1)
        Qu = .5 / den * Q2 + u.unsqueeze(-1).unsqueeze(-1) / den * Q
        Qw = -3.*u.unsqueeze(-1).unsqueeze(-1)/(2.*den) * Q2 + v3u2mw2.unsqueeze(-1).unsqueeze(-1)/(2.*den) * Q

        db10dQ = db10du.unsqueeze(-1).unsqueeze(-1) * Qu + db10dw_w.unsqueeze(-1).unsqueeze(-1) * Qw
        db11dQ = db11du.unsqueeze(-1).unsqueeze(-1) * Qu + db11dw_w.unsqueeze(-1).unsqueeze(-1) * Qw
        db12dQ = db12du.unsqueeze(-1).unsqueeze(-1) * Qu + db12dw_w.unsqueeze(-1).unsqueeze(-1) * Qw
        db20dQ = db20du.unsqueeze(-1).unsqueeze(-1) * Qu + db20dw_w.unsqueeze(-1).unsqueeze(-1) * Qw
        db21dQ = db21du.unsqueeze(-1).unsqueeze(-1) * Qu + db21dw_w.unsqueeze(-1).unsqueeze(-1) * Qw
        db22dQ = db22du.unsqueeze(-1).unsqueeze(-1) * Qu + db22dw_w.unsqueeze(-1).unsqueeze(-1) * Qw

        dB1dQ = oplus(db10dQ, id) + oplus(db11dQ, Q) + b11.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * oidid +\
                oplus(db12dQ, Q2) + b12.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * oidQ
        dB2dQ = oplus(db20dQ, id) + oplus(db21dQ, Q) + b21.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * oidid +\
                oplus(db22dQ, Q2) + b22.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * oidQ

        df1dQ = b11.unsqueeze(-1).unsqueeze(-1) * Q + b21.unsqueeze(-1).unsqueeze(-1) * Q2
        df2dQ = b12.unsqueeze(-1).unsqueeze(-1) * Q + b22.unsqueeze(-1).unsqueeze(-1) * Q2

        d2expQdQ2 = compute_d2expQdQ2(id, B1, B2, Q, Q2, dB1dQ, dB2dQ, df1dQ, df2dQ, f2)

    return f0.unsqueeze(-1).unsqueeze(-1), f1.unsqueeze(-1).unsqueeze(-1), f2.unsqueeze(-1).unsqueeze(-1), B1, B2, d2expQdQ2

def generate_omega(C, U):
    return sun.SUN_mul(C, sun.SUN_dagger(U))

def generate_Q(omega, N):
    antiherm = sun.SUN_dagger(omega) - omega
    return (
        0.5 * 1.j * antiherm
        - 0.5
        * 1.j
        / N
        * sun.SUN_trace(antiherm).unsqueeze(-1).unsqueeze(-1)
        * sun.SUN_identity(antiherm.shape[:-1], dtype=antiherm.dtype, device=antiherm.device)
    )
    #return 0.5 * 1.j * (SUN_dagger(omega) - omega) - 0.5 * 1.j/N * SUN_trace(SUN_dagger(omega) - omega).unsqueeze(-1).unsqueeze(-1)

def generate_expQ(Q, Q2, id, f0, f1, f2):
    return f0 * id + f1 * Q + f2 * Q2

def generate_dQ_domega(N, oidid, id):
    A = oidid
    B = oplus(id, id)
    return -1.j*(.5 * A - .5/N * B)

def generate_dexpQ_dQ(Q, Q2, B1, B2, f1, f2, id, oidid, oidQ):
    M = oplus(Q, B1)
    M += oplus(Q2, B2)
    M += f1.unsqueeze(-1).unsqueeze(-1)*oidid
    M += f2.unsqueeze(-1).unsqueeze(-1)*oidQ
    return M

def generate_domega_dU(C, id):
    return -otimes(id, sun.SUN_dagger(C))

def generate_domega_dC(U, oidid):
    return starprodmat(oidid, sun.SUN_dagger(U))

def generate_dQ_dU(dQdomega, id, C):
    B = generate_domega_dU(C, id)
    return starprod(dQdomega, B)

def generate_dQ_dC(dQdomega, oidid, U):
    B = generate_domega_dC(U, oidid)
    return starprod(dQdomega, B)

def generate_dexpQ_dU(dexpQdQ, dQ_dU):
    return starprod(dexpQdQ, dQ_dU)

def generate_Jacobian_U(dexpQdU, U, expQ, oidid):
    return starprodmat(dexpQdU, U) + matstarprod(expQ, oidid)

def generate_Jacobian_C(dexpQdQ, dQdC, U):
    A = starprod(dexpQdQ, dQdC)
    return starprodmat(A, U)

def generate_Jacobian_gradients(id, U, dQdU, dQdC, expQ, dexpQdU, dexpQdQ, d2expQdQ2):

    dJudU = torch.einsum('...cijd,...acdmrn,...mklr->...aijkln', dQdU, d2expQdQ2, dQdU)
    dJudU = torch.einsum('...aijkln,...nb->...aijklb', dJudU, U)
    dJudU += torch.einsum('...akli,...jb->...aijklb', dexpQdU, id)
    dJudU += torch.einsum('...aijk,...lb->...aijklb', dexpQdU, id)  

    #dJcdC = torch.einsum('...cijd,...acdmrn,...mklr->...aijkln', dQdC, d2expQdQ2, dQdC)
    #dJcdC = torch.einsum('...aijkln,...nb->...aijklb', dJcdC, U)

    dJcudUC = torch.einsum('...cijd,...acdmrn,...mklr->...aijkln', dQdC, d2expQdQ2, dQdU)
    dJcudUC = torch.einsum('...aijkln,...nb->...aijklb', dJcudUC, U)
    dJcudUC += torch.einsum('...amrk,...mijr,...lb->...aijklb', dexpQdQ, dQdC, id)

    return dJudU, dJcudUC#, dJcdC

def Jacobian_reshape(jac, jac_shape):
    return torch.einsum('...ijkl->...iljk',jac).reshape(jac_shape)

def det_Jac(J):
    return torch.linalg.det(J)


class _ConstantSpline:
    def __init__(self, value: float):
        self.value = float(value)

    def __call__(self, x):
        del x
        return self.value



class HyperSmearing(torch.nn.Module):
    """
    Time-conditioned hypernetwork for smearing coefficients rho.

    Modes:
    - "shared": one rho value per smearing sub-step, broadcast to all links/plaquettes/sites.
    - "per_link": one rho per (link-parity index m, plaquette index pl) per smearing sub-step.
    - "class": one rho per defect class per smearing sub-step, expanded via fixed class masks.
      Staples outside every class get rho = 0, and Smearing leaves links with no classified
      staple untransformed.
    """

    CLASS_NAMES = ("t1b", "t1e", "t1c", "t2", "t2b", "t2e", "t2c", "sp", "ex")

    def __init__(self, flow_pars, time):
        super().__init__()
        self.mode = str(getattr(flow_pars, "hyper_smearing_mode", "shared")).lower()
        if self.mode not in {"shared", "per_link", "class"}:
            raise ValueError(
                f"hyper_smearing_mode must be 'shared', 'per_link', or 'class', got {self.mode!r}"
            )

        self.D = int(flow_pars.D)
        self.nstep = max(1, int(flow_pars.nstep))
        self.time = float(time)
        self.rho_shape = tuple(flow_pars.rho_shape)
        self.smearing_steps = int(flow_pars.smearing_steps_per_layer)
        self.n_m = int(2 * self.D)
        self.n_pl = int(max(2 * (self.D - 1), self.D * (self.D - 1) // 2))
        self.normalize_by_nstep = bool(getattr(flow_pars, "hyper_normalize_by_nstep", True))
        self.rho_eps = float(getattr(flow_pars, "hyper_rho_eps", 0.0))
        self.scale_by_delta = bool(getattr(flow_pars, "hyper_scale_by_delta", False))
        self.rho_max = float(getattr(flow_pars, "hyper_rho_max", 0.0))

        emb_dim = int(getattr(flow_pars, "hyper_time_embedding_dim", 8))
        if emb_dim <= 0 or emb_dim % 2 != 0:
            raise ValueError(f"hyper_time_embedding_dim must be a positive even integer, got {emb_dim}")
        self.scalar_emb_dim = emb_dim
        hidden_dim = int(getattr(flow_pars, "hyper_hidden_dim", 16))
        if hidden_dim <= 0:
            raise ValueError(f"hyper_hidden_dim must be >= 1, got {hidden_dim}")
        depth = int(getattr(flow_pars, "hyper_depth", 2))
        if depth <= 0:
            raise ValueError(f"hyper_depth must be >= 1, got {depth}")
        activation_name = str(getattr(flow_pars, "hyper_activation", "silu")).lower()
        activation_factories = {
            "silu": torch.nn.SiLU,
            "gelu": torch.nn.GELU,
            "tanh": torch.nn.Tanh,
            "relu": torch.nn.ReLU,
        }
        if activation_name not in activation_factories:
            raise ValueError(
                "hyper_activation must be one of "
                f"{sorted(activation_factories)}, got {activation_name!r}"
            )
        activation_factory = activation_factories[activation_name]

        n_freq = emb_dim // 2
        freqs = 2.0 ** torch.arange(n_freq, dtype=torch.float64)
        self.register_buffer("freqs", freqs)

        if self.mode == "shared":
            self.coeffs_per_step = 1
        elif self.mode == "per_link":
            self.coeffs_per_step = self.n_m * self.n_pl
        else:
            if int(flow_pars.rho_shape_type) != 3:
                raise ValueError("hyper_smearing_mode='class' requires rho_shape_type=3")
            self.coeffs_per_step = len(self.CLASS_NAMES)
            class_masks = self._build_class_masks(flow_pars)
            self.register_buffer("class_masks", class_masks.to(dtype=torch.float64))

        # embedding = fourier(beta) || [beta, time]
        in_dim = self.scalar_emb_dim + 2
        out_dim = self.smearing_steps * self.coeffs_per_step
        mlp_layers = []
        current_dim = in_dim
        for _ in range(depth):
            mlp_layers.append(torch.nn.Linear(current_dim, hidden_dim))
            mlp_layers.append(activation_factory())
            current_dim = hidden_dim
        mlp_layers.append(torch.nn.Linear(current_dim, out_dim))
        self.mlp = torch.nn.Sequential(*mlp_layers)
        self.mlp = self.mlp.to(dtype=torch.float64, device=flow_pars.device)

        rho_init = float(getattr(flow_pars, "hyper_rho_init", 1e-3))
        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.fill_(rho_init)

    def _to_scalar(self, value, default: float) -> torch.Tensor:
        if value is None:
            return torch.tensor(default, dtype=self.freqs.dtype, device=self.freqs.device)
        tensor = torch.as_tensor(value, dtype=self.freqs.dtype, device=self.freqs.device)
        if tensor.numel() == 0:
            return torch.tensor(default, dtype=self.freqs.dtype, device=self.freqs.device)
        return tensor.reshape(-1)[0]

    def _fourier_scalar(self, value: torch.Tensor) -> torch.Tensor:
        arg = self.freqs * value
        return torch.cat((torch.sin(arg), torch.cos(arg)), dim=0)

    def _to_vector(self, value, default: float) -> torch.Tensor:
        if value is None:
            return torch.tensor([default], dtype=self.freqs.dtype, device=self.freqs.device)
        tensor = torch.as_tensor(value, dtype=self.freqs.dtype, device=self.freqs.device).reshape(-1)
        if tensor.numel() == 0:
            return torch.tensor([default], dtype=self.freqs.dtype, device=self.freqs.device)
        return tensor

    def _fourier_vector(self, values: torch.Tensor) -> torch.Tensor:
        arg = values.unsqueeze(-1) * self.freqs.unsqueeze(0)
        return torch.cat((torch.sin(arg), torch.cos(arg)), dim=-1)

    def _runtime_embedding(self, beta=None) -> torch.Tensor:
        beta_v = self._to_vector(beta, self.time)
        if beta_v.numel() == 1:
            t = torch.tensor(self.time, dtype=self.freqs.dtype, device=self.freqs.device)
            beta_s = beta_v[0]
            beta_emb = self._fourier_scalar(beta_s)
            return torch.cat(
                (
                    beta_emb,
                    torch.stack((beta_s, t)),
                ),
                dim=0,
            )

        beta_emb = self._fourier_vector(beta_v)
        tcol = torch.full((beta_v.numel(), 1), self.time, dtype=self.freqs.dtype, device=self.freqs.device)
        return torch.cat((beta_emb, beta_v.unsqueeze(-1), tcol), dim=-1)

    def _build_class_masks(self, flow_pars) -> torch.Tensor:
        if flow_pars.small_defect_mask is None or len(flow_pars.small_defect_mask) == 0:
            raise ValueError(
                "hyper_smearing_mode='class' requires flow_pars.small_defect_mask[-1]"
            )

        defect_mask_cpu = flow_pars.small_defect_mask[-1].detach().cpu()
        masks = []
        # set_defect_rho_from_fits initializes rho with 1e-4 baseline;
        # here we extract only class-active entries by activating one class at a time.
        for class_idx in range(len(self.CLASS_NAMES)):
            class_spl = [_ConstantSpline(0.0) for _ in range(len(self.CLASS_NAMES))]
            class_spl[class_idx] = _ConstantSpline(float(self.nstep))
            rho_class = set_defect_rho_from_fits(
                xval=self.time,
                nstep=self.nstep,
                rhofitpars=class_spl,
                rho_shape=self.rho_shape,
                defect_mask=defect_mask_cpu,
                D=self.D,
            )
            active = (rho_class > 1e-3).to(dtype=torch.float64)
            masks.append(active[0])  # remove smearing-step dimension
        return torch.stack(masks, dim=0).to(device=flow_pars.device)

    def _expand_coeffs(self, coeffs: torch.Tensor) -> torch.Tensor:
        # coeffs shape: (smearing_steps, coeffs_per_step) or (batch, smearing_steps, coeffs_per_step)
        spatial_rank = len(self.rho_shape) - 3
        if coeffs.dim() == 2:
            if self.mode == "shared":
                view = (self.smearing_steps,) + (1,) * (len(self.rho_shape) - 1)
                return coeffs.view(view).expand(self.rho_shape)
            if self.mode == "per_link":
                view = (self.smearing_steps, self.n_m, self.n_pl) + (1,) * spatial_rank
                return coeffs.view(view).expand(self.rho_shape)
            return torch.einsum("sc,c...->s...", coeffs, self.class_masks)

        if coeffs.dim() == 3:
            bsz = int(coeffs.shape[0])
            if self.mode == "shared":
                view = (bsz, self.smearing_steps) + (1,) * (len(self.rho_shape) - 1)
                return coeffs.view(view).expand((bsz,) + self.rho_shape)
            if self.mode == "per_link":
                view = (bsz, self.smearing_steps, self.n_m, self.n_pl) + (1,) * spatial_rank
                return coeffs.view(view).expand((bsz,) + self.rho_shape)
            return torch.einsum("bsc,c...->bs...", coeffs, self.class_masks)

        raise ValueError(f"Unsupported coeffs shape {tuple(coeffs.shape)}")

    def _apply_signed_floor(self, rho: torch.Tensor, active_mask: torch.Tensor = None) -> torch.Tensor:
        if self.rho_eps <= 0.0:
            return rho
        sign = torch.where(rho >= 0.0, torch.ones_like(rho), -torch.ones_like(rho))
        rho_nz = sign * (torch.abs(rho) + self.rho_eps)
        if active_mask is None:
            return rho_nz
        return torch.where(active_mask > 0, rho_nz, torch.zeros_like(rho_nz))

    def forward(self, beta=None, delta_beta=None) -> torch.Tensor:
        emb = self._runtime_embedding(beta=beta)
        if emb.dim() == 1:
            coeffs = self.mlp(emb).view(self.smearing_steps, self.coeffs_per_step)
        else:
            coeffs = self.mlp(emb).view(emb.shape[0], self.smearing_steps, self.coeffs_per_step)
        rho = self._expand_coeffs(coeffs)
        if self.normalize_by_nstep:
            rho = rho / float(self.nstep)
        if self.mode == "class":
            active_base = (self.class_masks.sum(dim=0, keepdim=True) > 0).to(dtype=rho.dtype)
            if rho.dim() == len(self.rho_shape):
                active_mask = active_base.expand_as(rho)
            else:
                active_mask = active_base.unsqueeze(0).expand_as(rho)
        else:
            active_mask = None
        if self.rho_max > 0.0:
            rho = self.rho_max * torch.tanh(rho / self.rho_max)
        if self.scale_by_delta:
            delta_v = self._to_vector(delta_beta, 0.0).to(device=rho.device, dtype=rho.dtype)
            if rho.dim() == len(self.rho_shape):
                if delta_v.numel() == 1:
                    rho = rho * delta_v.reshape(-1)[0]
                else:
                    rho = rho.unsqueeze(0).expand(delta_v.numel(), *rho.shape)
                    rho = rho * delta_v.view((delta_v.numel(),) + (1,) * (rho.dim() - 1))
            else:
                bsz = int(rho.shape[0])
                if delta_v.numel() == 1:
                    delta_v = delta_v.expand(bsz)
                elif delta_v.numel() != bsz:
                    raise ValueError(
                        f"delta_beta has invalid size {delta_v.numel()} for rho scaling "
                        f"(expected 1 or {bsz})"
                    )
                rho = rho * delta_v.view((bsz,) + (1,) * (rho.dim() - 1))
        rho = self._apply_signed_floor(rho, active_mask=active_mask)
        return rho

    def classified_links(self) -> torch.Tensor:
        """Class mode: (D, *lattice) mask of the links with at least one classified staple."""
        union = (self.class_masks.sum(dim=0) > 0).any(dim=1)  # (m, *lattice)
        return union.view(self.D, 2, *union.shape[1:]).any(dim=1)


def _su3_generators(dtype=torch.cdouble, device=None):
    real_dtype = torch.float64 if dtype in {torch.cdouble, torch.complex128} else torch.float32
    zero = torch.tensor(0.0, dtype=real_dtype, device=device)
    one = torch.tensor(1.0, dtype=real_dtype, device=device)
    i = torch.tensor(1.0j, dtype=dtype, device=device)

    gens = []
    for _ in range(8):
        gens.append(torch.zeros((3, 3), dtype=dtype, device=device))

    gens[0][0, 1] = one
    gens[0][1, 0] = one

    gens[1][0, 1] = -i
    gens[1][1, 0] = i

    gens[2][0, 0] = one
    gens[2][1, 1] = -one

    gens[3][0, 2] = one
    gens[3][2, 0] = one

    gens[4][0, 2] = -i
    gens[4][2, 0] = i

    gens[5][1, 2] = one
    gens[5][2, 1] = one

    gens[6][1, 2] = -i
    gens[6][2, 1] = i

    inv_sqrt3 = torch.tensor(1.0 / np.sqrt(3.0), dtype=real_dtype, device=device)
    gens[7][0, 0] = inv_sqrt3
    gens[7][1, 1] = inv_sqrt3
    gens[7][2, 2] = -2.0 * inv_sqrt3

    return 0.5 * torch.stack(gens, dim=0)


def _project_hermitian_traceless(H, N):
    H = 0.5 * (H + sun.SUN_dagger(H))
    tr = sun.SUN_trace(H).unsqueeze(-1).unsqueeze(-1) / float(N)
    eye = sun.SUN_identity(H.shape[:-1], dtype=H.dtype, device=H.device)
    return H - tr * eye


def _as_batched_parameter(value, batch_size, dtype, device, default=0.0):
    if value is None:
        out = torch.full((batch_size,), float(default), dtype=dtype, device=device)
    else:
        out = torch.as_tensor(value, dtype=dtype, device=device).reshape(-1)
        if out.numel() == 0:
            out = torch.full((batch_size,), float(default), dtype=dtype, device=device)
        elif out.numel() == 1 and batch_size != 1:
            out = out.expand(batch_size)
        elif out.numel() != batch_size:
            raise ValueError(f"parameter has shape ({out.numel()},), expected 1 or {batch_size}")
    return out


class ResidualNormalizingFlows(torch.nn.Module):
    """
    Gradient-based residual SU(3) coupling flow.

    The implemented local potential is polynomial in the real/imaginary traces
    of oriented loops containing the active link:

        phi = lin_i z_i + 1/2 H_ij z_i z_j

    The forward map uses the explicit gradient of this potential and an exact
    local tangent-space Jacobian. Hypernetwork coefficients depend globally on
    beta, while delta_beta multiplies the exponential generator.
    Optional frozen-link convolutions extend the plaquette paths without
    introducing dependence on other active links.
    """

    def __init__(self, flow_pars, time=1.0):
        super().__init__()
        self.N = int(flow_pars.N)
        if self.N != 3:
            raise ValueError("ResidualNormalizingFlows currently supports SU(3) only")
        self.D = int(flow_pars.D)
        self.device = flow_pars.device
        self.time = float(time)
        self.steps = int(getattr(flow_pars, "smearing_steps_per_layer", 1))
        self.parameterization = str(getattr(flow_pars, "residual_parameterization", "hyper")).lower()
        if self.parameterization not in {"hyper", "static"}:
            raise ValueError(f"Unknown residual parameterization {self.parameterization!r}")
        self.include_imag = bool(getattr(flow_pars, "residual_include_imag", True))
        self.use_quadratic = bool(getattr(flow_pars, "residual_quadratic", True))
        self.coeff_max = float(getattr(flow_pars, "residual_coeff_max", 0.0))
        n_pt_value = getattr(flow_pars, "residual_n_pt", 0)
        mask_mod_value = getattr(flow_pars, "residual_mask_mod", 2)
        try:
            self.n_pt = int(n_pt_value)
            self.mask_mod = int(mask_mod_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("residual_n_pt and residual_mask_mod must be integers") from exc
        if self.n_pt != n_pt_value or self.mask_mod != mask_mod_value:
            raise ValueError("residual_n_pt and residual_mask_mod must be integers")
        if self.n_pt < 0:
            raise ValueError("residual_n_pt must be non-negative")
        if self.mask_mod not in (2, 4):
            raise ValueError("residual_mask_mod must be 2 or 4")

        self.n_oriented_plaquettes = 2 * (self.D - 1)
        self.n_features = self.n_oriented_plaquettes * (2 if self.include_imag else 1)
        self.coeffs_per_step = self.n_features
        if self.use_quadratic:
            self.coeffs_per_step += self.n_features * self.n_features

        # Keep this constant out of buffers: NCCL DDP cannot broadcast ComplexDouble buffers.
        self.generators = _su3_generators(dtype=torch.cdouble, device=torch.device("cpu"))
        coeff_init = float(getattr(flow_pars, "residual_coeff_init", getattr(flow_pars, "hyper_rho_init", 1e-3)))
        if self.n_pt:
            eta_init = 0.01 / (2.0 * max(1, self.D - 1))
            self.conv_eta = torch.nn.Parameter(
                torch.full(
                    (self.steps, self.n_pt, self.D, self.D - 1),
                    eta_init,
                    dtype=torch.float64,
                    device=flow_pars.device,
                )
            )
        if self.parameterization == "static":
            self.static_linear = torch.nn.Parameter(
                torch.full((self.steps, self.n_features), coeff_init, dtype=torch.float64, device=flow_pars.device)
            )
            if self.use_quadratic:
                self.static_quadratic = torch.nn.Parameter(
                    torch.zeros(
                        (self.steps, self.n_features, self.n_features),
                        dtype=torch.float64,
                        device=flow_pars.device,
                    )
                )
        else:
            emb_dim = int(getattr(flow_pars, "hyper_time_embedding_dim", 8))
            if emb_dim <= 0 or emb_dim % 2 != 0:
                raise ValueError(f"hyper_time_embedding_dim must be a positive even integer, got {emb_dim}")
            hidden_dim = int(getattr(flow_pars, "hyper_hidden_dim", 16))
            depth = int(getattr(flow_pars, "hyper_depth", 2))
            activation_name = str(getattr(flow_pars, "hyper_activation", "silu")).lower()
            activation_factories = {
                "silu": torch.nn.SiLU,
                "gelu": torch.nn.GELU,
                "tanh": torch.nn.Tanh,
                "relu": torch.nn.ReLU,
            }
            if activation_name not in activation_factories:
                raise ValueError(
                    "hyper_activation must be one of "
                    f"{sorted(activation_factories)}, got {activation_name!r}"
                )

            freqs = 2.0 ** torch.arange(emb_dim // 2, dtype=torch.float64)
            self.register_buffer("freqs", freqs)
            in_dim = emb_dim + 1  # Fourier(beta) || beta
            out_dim = self.steps * self.coeffs_per_step
            layers = []
            current_dim = in_dim
            for _ in range(depth):
                layers.append(torch.nn.Linear(current_dim, hidden_dim))
                layers.append(activation_factories[activation_name]())
                current_dim = hidden_dim
            layers.append(torch.nn.Linear(current_dim, out_dim))
            self.mlp = torch.nn.Sequential(*layers).to(dtype=torch.float64, device=flow_pars.device)
            with torch.no_grad():
                self.mlp[-1].weight.zero_()
                self.mlp[-1].bias.zero_()
                self.mlp[-1].bias.view(self.steps, self.coeffs_per_step)[:, : self.n_features].fill_(coeff_init)

    def _fourier(self, beta):
        arg = beta.unsqueeze(-1) * self.freqs.unsqueeze(0)
        return torch.cat((torch.sin(arg), torch.cos(arg)), dim=-1)

    def _coefficients(self, beta, delta_beta, batch_size, dtype, device):
        if self.parameterization == "static":
            delta_v = _as_batched_parameter(delta_beta, batch_size, torch.float64, device, 0.0)
            lin = self.static_linear.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
            if self.coeff_max > 0.0:
                lin = self.coeff_max * torch.tanh(lin / self.coeff_max)
            if self.use_quadratic:
                quad = self.static_quadratic.to(device=device, dtype=dtype)
                quad = 0.5 * (quad + quad.transpose(-1, -2))
                quad = quad.unsqueeze(0).expand(batch_size, -1, -1, -1)
            else:
                quad = torch.zeros(
                    (batch_size, self.steps, self.n_features, self.n_features),
                    dtype=dtype,
                    device=device,
                )
            return lin, quad, delta_v

        beta_v = _as_batched_parameter(beta, batch_size, torch.float64, self.freqs.device, self.time)
        delta_v = _as_batched_parameter(delta_beta, batch_size, torch.float64, self.freqs.device, 0.0)
        emb = torch.cat(
            (
                self._fourier(beta_v),
                beta_v.unsqueeze(-1),
            ),
            dim=-1,
        )
        raw = self.mlp(emb).view(batch_size, self.steps, self.coeffs_per_step)
        if self.coeff_max > 0.0:
            raw = self.coeff_max * torch.tanh(raw / self.coeff_max)

        lin = raw[..., : self.n_features].to(device=device, dtype=dtype)
        if self.use_quadratic:
            quad = raw[..., self.n_features :].view(
                batch_size, self.steps, self.n_features, self.n_features
            )
            quad = 0.5 * (quad + quad.transpose(-1, -2))
            quad = quad.to(device=device, dtype=dtype)
        else:
            quad = torch.zeros(
                (batch_size, self.steps, self.n_features, self.n_features),
                dtype=dtype,
                device=device,
            )
        delta_v = delta_v.to(device=device, dtype=dtype)
        return lin, quad, delta_v

    def _oriented_staples(self, cfgs, mu):
        staples = []
        for nu in range(self.D):
            if nu == mu:
                continue
            staples.append(sun.pstaple(cfgs, mu, nu, self.D))
            staples.append(sun.nstaple(cfgs, mu, nu, self.D))
        return torch.stack(staples, dim=1)

    def _features(self, staples, U):
        loops = sun.SUN_mul(staples, U.unsqueeze(1))
        tr = sun.SUN_trace(loops) / float(self.N)
        if self.include_imag:
            return torch.cat((tr.real, tr.imag), dim=1)
        return tr.real

    def _feature_weights(self, features, lin, quad):
        lin_view = lin.view(lin.shape + (1,) * (features.dim() - 2))
        return lin_view + torch.einsum(
            "bij,bj...->bi...",
            quad,
            features,
        )

    def _complex_coefficients_from_weights(self, weights, delta_beta):
        npl = self.n_oriented_plaquettes
        delta_view = delta_beta.view((delta_beta.shape[0],) + (1,) * (weights.dim() - 1))
        if self.include_imag:
            wr = weights[:, :npl]
            wi = weights[:, npl:]
            return delta_view * (wr - 1.j * wi) / float(self.N)
        return delta_view * weights / float(self.N)

    def _staple_sum(self, staples, weights, delta_beta):
        alpha = self._complex_coefficients_from_weights(weights, delta_beta)
        return torch.sum(alpha.unsqueeze(-1).unsqueeze(-1) * sun.SUN_dagger(staples), dim=1)

    def _delta_features(self, staples, delta_U):
        delta_loops = sun.SUN_mul(staples, delta_U.unsqueeze(1))
        delta_tr = sun.SUN_trace(delta_loops) / float(self.N)
        if self.include_imag:
            return torch.cat((delta_tr.real, delta_tr.imag), dim=1)
        return delta_tr.real

    def _project_delta_omega(self, delta_omega):
        antiherm = sun.SUN_dagger(delta_omega) - delta_omega
        eye = sun.SUN_identity(antiherm.shape[:-1], dtype=antiherm.dtype, device=antiherm.device)
        return (
            0.5j * antiherm
            - 0.5j
            / float(self.N)
            * sun.SUN_trace(antiherm).unsqueeze(-1).unsqueeze(-1)
            * eye
        )

    def _tangent_coordinates(self, delta_U, U_out):
        left = sun.SUN_mul(delta_U, sun.SUN_dagger(U_out))
        herm = _project_hermitian_traceless(-1.j * left, self.N)
        gens = self.generators.to(dtype=U_out.dtype, device=U_out.device)
        return 2.0 * torch.einsum("...ij,aji->...a", herm, gens).real

    def _local_tangent_jacobian(self, staples, U, C, dexpQdQ, expQ, U_out, quad, delta_beta):
        gens = self.generators.to(dtype=U.dtype, device=U.device)
        view = (1,) * (U.dim() - 2) + (self.N, self.N)
        cols = []
        Udag = sun.SUN_dagger(U)
        omega = sun.SUN_mul(C, Udag)
        del omega
        for b in range(gens.shape[0]):
            Tb = gens[b].view(view)
            delta_U = 1.j * sun.SUN_mul(Tb, U)
            delta_features = self._delta_features(staples, delta_U)
            delta_weights = torch.einsum("bij,bj...->bi...", quad, delta_features)
            delta_C = self._staple_sum(staples, delta_weights, delta_beta)
            delta_Udag = -1.j * sun.SUN_mul(Udag, Tb)
            delta_omega = sun.SUN_mul(delta_C, Udag) + sun.SUN_mul(C, delta_Udag)
            delta_Q = self._project_delta_omega(delta_omega)
            delta_expQ = torch.einsum("...mn,...imnj->...ij", delta_Q, dexpQdQ)
            delta_U_out = sun.SUN_mul(delta_expQ, U) + sun.SUN_mul(expQ, delta_U)
            cols.append(self._tangent_coordinates(delta_U_out, U_out))
        return torch.stack(cols, dim=-1)

    def _runtime_mask(self, mask, cfgs):
        lattice_shape = tuple(cfgs.shape[2:-2])
        if len(lattice_shape) != self.D:
            raise ValueError(
                f"configuration lattice has {len(lattice_shape)} dimensions, expected {self.D}"
            )
        if self.mask_mod == 2:
            if mask.shape[0] != 2 * self.D:
                raise ValueError(
                    f"residual mask has leading dimension {mask.shape[0]}, expected {2 * self.D}"
                )
            if self.n_pt and any(size % self.mask_mod for size in lattice_shape):
                raise ValueError(
                    "residual_n_pt > 0 with residual_mask_mod=2 requires even periodic lattice extents"
                )
            return mask

        if any(size % self.mask_mod for size in lattice_shape):
            raise ValueError(
                "residual_mask_mod=4 requires every periodic lattice extent to be divisible by 4"
            )
        coords = torch.zeros(lattice_shape, dtype=torch.long, device=cfgs.device)
        for axis, size in enumerate(lattice_shape):
            view = [1] * self.D
            view[axis] = size
            coords = coords + torch.arange(size, device=cfgs.device).view(view)
        colors = coords.remainder(self.mask_mod)
        runtime_mask = torch.zeros(
            (self.mask_mod * self.D, 1, self.D, *lattice_shape, 1, 1),
            dtype=mask.dtype,
            device=cfgs.device,
        )
        for mu in range(self.D):
            for color in range(self.mask_mod):
                runtime_mask[color + self.mask_mod * mu, 0, mu] = (colors == color).unsqueeze(-1).unsqueeze(-1)
        return runtime_mask

    def _frozen_convolution(self, values, step, point):
        # Staple helpers run opposite to a link; their daggers are link-like.
        eta = self.conv_eta[int(step), int(point)].to(dtype=values.real.dtype, device=values.device)
        values_next = values.clone()
        for mu in range(self.D):
            update = torch.zeros_like(values[:, mu])
            nu_index = 0
            for nu in range(self.D):
                if nu == mu:
                    continue
                update = update + eta[mu, nu_index] * (
                    sun.SUN_dagger(sun.pstaple(values, mu, nu, self.D))
                    + sun.SUN_dagger(sun.nstaple(values, mu, nu, self.D))
                )
                nu_index += 1
            values_next[:, mu] = values[:, mu] + update
        return values_next

    def _staples_and_generator(self, cfgs, mu, lin, quad, delta_v, step, current_mask):
        U = cfgs[:, mu]
        if self.n_pt:
            if current_mask is None:
                raise ValueError("current_mask is required for residual_n_pt > 0")
            values = (1.0 - current_mask) * cfgs
            for point in range(self.n_pt):
                values = self._frozen_convolution(values, step, point)
            staples = self._oriented_staples(values, mu)
        else:
            staples = self._oriented_staples(cfgs, mu)
        features = self._features(staples, U)
        weights = self._feature_weights(features, lin, quad)
        C = self._staple_sum(staples, weights, delta_v)

        eye = sun.SUN_identity(U.shape[:-1], dtype=U.dtype, device=U.device)
        omega = generate_omega(C, U)
        Q = generate_Q(omega, self.N)
        Q2 = sun.SUN_mul(Q, Q)
        oidid = otimes(eye, eye)
        oidQ = otimes(eye, Q) + otimes(Q, eye)
        f0, f1, f2, B1, B2, _ = generate_coefficients(Q, Q2, eye, oidid, oidQ, self.device, backward=False)
        expQ = generate_expQ(Q, Q2, eye, f0, f1, f2)
        return staples, C, expQ, (Q, Q2, B1, B2, f1, f2, eye, oidid, oidQ)

    def _step_coefficients(self, cfgs, beta, delta_beta, step):
        lin_all, quad_all, delta_v = self._coefficients(
            beta, delta_beta, int(cfgs.shape[0]), cfgs.real.dtype, cfgs.device
        )
        return lin_all[:, int(step)], quad_all[:, int(step)], delta_v

    def residual_update_mu(self, cfgs, mu, beta=None, delta_beta=None, step=0, current_mask=None):
        real_dtype = cfgs.real.dtype
        lin, quad, delta_v = self._step_coefficients(cfgs, beta, delta_beta, step)

        U = cfgs[:, mu]
        if bool(torch.all(delta_v == 0.0).item()):
            ident = torch.eye(8, dtype=real_dtype, device=cfgs.device)
            jac = ident.view((1,) * (U.dim() - 2) + (8, 8)).expand(U.shape[:-2] + (8, 8))
            detjac = torch.ones(U.shape[:-2], dtype=real_dtype, device=cfgs.device)
            return U.clone(), detjac, jac

        staples, C, expQ, exp_terms = self._staples_and_generator(cfgs, mu, lin, quad, delta_v, step, current_mask)
        dexpQdQ = generate_dexpQ_dQ(*exp_terms)
        U_out = sun.SUN_mul(expQ, U)

        jac = self._local_tangent_jacobian(staples, U, C, dexpQdQ, expQ, U_out, quad, delta_v)
        detjac = torch.linalg.det(jac).abs()
        return U_out, detjac, jac

    def forward(self, cfgs, mask, dmasking=False, dmask=None, beta=None, delta_beta=None):
        if dmasking and (self.n_pt or self.mask_mod != 2):
            raise ValueError(
                "cropped/dmasking residual layers require residual_n_pt=0 and residual_mask_mod=2"
            )
        real_dtype = cfgs.real.dtype
        mask = self._runtime_mask(mask, cfgs)
        ones = torch.ones(mask[0].shape, dtype=real_dtype, device=cfgs.device).squeeze(-1).squeeze(-1)
        dlogJ = torch.zeros(cfgs.shape[0], dtype=real_dtype, device=cfgs.device)
        if cfgs.shape[0] == 0:
            # Inactive DDP ranks must accumulate zero gradients for every parameter.
            for parameter in self.parameters():
                dlogJ = dlogJ + parameter.sum() * 0.0
            return cfgs, 0, dlogJ
        dims = tuple(np.arange(1, self.D + 2))

        for sm in range(self.steps):
            for mu in range(self.D):
                for eo in range(self.mask_mod):
                    if dmasking:
                        current_mask = mask[eo + mu * self.mask_mod, :] * dmask
                    else:
                        current_mask = mask[eo + mu * self.mask_mod, :]

                    U_new, jac, _ = self.residual_update_mu(
                        cfgs,
                        mu,
                        beta=beta,
                        delta_beta=delta_beta,
                        step=sm,
                        current_mask=current_mask,
                    )
                    cfgs_new = cfgs.clone()
                    cfgs_new[:, mu] = U_new
                    cfgs = current_mask * cfgs_new + (1 - current_mask) * cfgs

                    active = current_mask.squeeze(-1).squeeze(-1)
                    masked_jac = active * jac.unsqueeze(1) + (1 - active) * ones
                    dlogJ = dlogJ + torch.sum(
                        torch.log(torch.clamp(masked_jac, min=1e-12)),
                        dims,
                    )
        return cfgs, 0, dlogJ

    def inverse(self, cfgs, mask, dmasking=False, dmask=None, beta=None, delta_beta=None, tol=1e-12, max_iter=100):
        """
        Inverse of forward with the same arguments: the masked updates are undone in reverse
        order. The generator of a mask depends on the updated link only through the link itself
        (its loops and staples use links the mask leaves fixed), so each mask is inverted by the
        fixed-point iteration of sun.invert_left_update. Returns (cfgs, 0, logdet of the inverse
        map); iteration counts and the largest residual are left in self.last_inverse_stats.
        """
        if dmasking and (self.n_pt or self.mask_mod != 2):
            raise ValueError(
                "cropped/dmasking residual layers require residual_n_pt=0 and residual_mask_mod=2"
            )
        real_dtype = cfgs.real.dtype
        mask = self._runtime_mask(mask, cfgs)
        ones = torch.ones(mask[0].shape, dtype=real_dtype, device=cfgs.device).squeeze(-1).squeeze(-1)
        dlogJ = torch.zeros(cfgs.shape[0], dtype=real_dtype, device=cfgs.device)
        dims = tuple(np.arange(1, self.D + 2))
        stats = {"iterations": [], "residual": 0.0}

        for sm in reversed(range(self.steps)):
            lin, quad, delta_v = self._step_coefficients(cfgs, beta, delta_beta, sm)
            if bool(torch.all(delta_v == 0.0).item()):
                continue
            for mu in reversed(range(self.D)):
                for eo in reversed(range(self.mask_mod)):
                    if dmasking:
                        current_mask = mask[eo + mu * self.mask_mod, :] * dmask
                    else:
                        current_mask = mask[eo + mu * self.mask_mod, :]
                    active_mu = current_mask[:, mu]
                    frozen = cfgs

                    def left_factor(U_mu, mu=mu, current_mask=current_mask, active_mu=active_mu, frozen=frozen):
                        trial = frozen.clone()
                        trial[:, mu] = U_mu
                        _, _, expQ, _ = self._staples_and_generator(trial, mu, lin, quad, delta_v, sm, current_mask)
                        eye = torch.eye(self.N, dtype=expQ.dtype, device=expQ.device)
                        return active_mu * expQ + (1 - active_mu) * eye

                    U_mu, iterations, residual = sun.invert_left_update(
                        left_factor, cfgs[:, mu], tol=tol, max_iter=max_iter
                    )
                    if not residual <= tol:
                        raise RuntimeError(
                            f"residual-flow inverse did not converge for mask {eo + mu * self.mask_mod}: "
                            f"residual {float(residual):.3e} after {iterations} iterations"
                        )
                    cfgs = cfgs.clone()
                    cfgs[:, mu] = U_mu
                    _, jac, _ = self.residual_update_mu(
                        cfgs, mu, beta=beta, delta_beta=delta_beta, step=sm, current_mask=current_mask
                    )
                    active = current_mask.squeeze(-1).squeeze(-1)
                    masked_jac = active * jac.unsqueeze(1) + (1 - active) * ones
                    dlogJ = dlogJ - torch.sum(torch.log(torch.clamp(masked_jac, min=1e-12)), dims)
                    stats["iterations"].append(iterations)
                    stats["residual"] = max(stats["residual"], float(residual))
        self.last_inverse_stats = stats
        return cfgs, 0, dlogJ


class ResidualCouplingLayer(torch.nn.Module):
    """
    Standard full-lattice SNF coupling layer backed by ResidualNormalizingFlows.

    Both smearing implementations return the full Haar logdet.
    """

    def __init__(self, flow_pars, rho_layer, time):
        super().__init__()
        if int(getattr(flow_pars, "rho_shape_type", 0)) == 3:
            raise ValueError("ResidualCouplingLayer is standard/full-lattice; use no defect rho_shape_type")
        self.register_buffer("rho_layer", rho_layer.detach().clone())
        self.residual_flow = ResidualNormalizingFlows(flow_pars, time)

    def forward(self, cfgs, flow_pars, rho_layer, is_training=False, beta=None, delta_beta=None):
        del rho_layer, is_training
        cfgs, dQ, dlogJ = self.residual_flow(
            cfgs,
            flow_pars.mask,
            beta=beta,
            delta_beta=delta_beta,
        )
        return cfgs, dQ, dlogJ

    def inverse(self, cfgs, flow_pars, rho_layer, beta=None, delta_beta=None, **kwargs):
        """Inverse of forward with the same beta and delta_beta."""
        del rho_layer
        return self.residual_flow.inverse(cfgs, flow_pars.mask, beta=beta, delta_beta=delta_beta, **kwargs)


class CouplingLayer(torch.nn.Module):
    def __init__(self, flow_pars, rho_layer, time):
        super().__init__()
        self.fit_train = flow_pars.fit_train
        self.use_rho_functional = bool(getattr(flow_pars, "use_rho_functional", False))
        self.use_hyper_smearing = bool(getattr(flow_pars, "use_hyper_smearing", False))
        if self.fit_train == 0 and not self.use_hyper_smearing and not self.use_rho_functional:
            self.rho_layer = torch.nn.Parameter(rho_layer)
        else:
            # Keep attribute for compatibility with diagnostics code paths.
            self.register_buffer("rho_layer", rho_layer.detach().clone())
        #sistemare la rete!
        self.smearing = Smearing(flow_pars, time)
        #self.register_buffer('time',self.smearing.t)

    def _rho(self, rho_layer):
        if self.use_rho_functional:
            return rho_layer
        if self.smearing.use_hyper_smearing:
            return None
        return self.rho_layer if self.fit_train == 0 else rho_layer

    def forward(self, cfgs, flow_pars, rho_layer, is_training=False, beta=None, delta_beta=None):
        return self.smearing(
            cfgs,
            flow_pars.mask,
            self._rho(rho_layer),
            beta=beta,
            delta_beta=delta_beta,
        )

    def inverse(self, cfgs, flow_pars, rho_layer, beta=None, delta_beta=None, **kwargs):
        """Inverse of forward with the same rho_layer, beta and delta_beta."""
        return self.smearing.inverse(
            cfgs,
            flow_pars.mask,
            self._rho(rho_layer),
            beta=beta,
            delta_beta=delta_beta,
            **kwargs,
        )

class DefectCouplingLayer(CouplingLayer):
    def __init__(self, flow_pars, rho_layer, time):
        super().__init__(flow_pars, rho_layer, time)
    
    def forward(self, cfgs, flow_pars, rho_layer, is_training=False, beta=None, delta_beta=None):
        rho_r = self._rho(rho_layer)

        small_cfgs = flow_pars.defect.cut_defect(cfgs, buffer=2)
        small_cfgs, _, dlogJ = self.smearing(
            small_cfgs,
            flow_pars.small_mask[-1],
            rho_r,
            dmasking=True,
            dmask=flow_pars.smeared_defect_mask[-1],
            beta=beta,
            delta_beta=delta_beta,
        )

        if is_training:
            return small_cfgs, 0, dlogJ
        else:
            return flow_pars.defect.embedding(small_cfgs, cfgs, buffer=2), 0, dlogJ

    def inverse(self, cfgs, flow_pars, rho_layer, beta=None, delta_beta=None, **kwargs):
        """Inverse of forward with the same rho_layer, beta and delta_beta, on the full lattice."""
        small_cfgs = flow_pars.defect.cut_defect(cfgs, buffer=2)
        small_cfgs, _, dlogJ = self.smearing.inverse(
            small_cfgs,
            flow_pars.small_mask[-1],
            self._rho(rho_layer),
            dmasking=True,
            dmask=flow_pars.smeared_defect_mask[-1],
            beta=beta,
            delta_beta=delta_beta,
            **kwargs,
        )
        return flow_pars.defect.embedding(small_cfgs, cfgs, buffer=2), 0, dlogJ

class Smearing(torch.nn.Module):
    def __init__(self, flow_pars, time):
        super().__init__()
        self.N = flow_pars.N
        self.D = flow_pars.D
        self.rho_shape = tuple(flow_pars.rho_shape)
        self.jac_shape = flow_pars.jac_shape
        self.rho_shape_type = flow_pars.rho_shape_type
        self.use_nn = flow_pars.use_nn
        self.device = flow_pars.device
        self.smearing_steps_per_layer = flow_pars.smearing_steps_per_layer   
        self.fit_train = flow_pars.fit_train
        self.use_hyper_smearing = bool(getattr(flow_pars, "use_hyper_smearing", False))
        self.explicit_backward = bool(getattr(flow_pars, "explicit_backward", False))
        self._smearing_links_cache = sun.masked_links_cache_for(flow_pars)

        if self.use_hyper_smearing:
            self.hyper_smearing = HyperSmearing(flow_pars, time)

        if self.use_nn and not self.use_hyper_smearing:
            self.net = flow_pars.net
            self.compute_staples = self.stout_staples_nn
            self.residual = flow_pars.residual
        else:
            self.compute_staples = self.stout_staples
        
        self.t = time

    def _runtime_rho(self, cfgs, rho_layer, beta, delta_beta):
        real_dtype = cfgs.real.dtype
        bs = cfgs.shape[0]
        if self.use_hyper_smearing:
            rho_runtime = self.hyper_smearing(beta=beta, delta_beta=delta_beta).to(
                device=cfgs.device,
                dtype=real_dtype,
            )
        else:
            if rho_layer is None:
                raise ValueError("rho_layer is required when use_hyper_smearing=False")
            rho_runtime = rho_layer.to(device=cfgs.device, dtype=real_dtype)
        batched_rho = (
            self.use_hyper_smearing
            and rho_runtime.dim() == (len(self.rho_shape) + 1)
        )
        if batched_rho:
            n_rho = int(rho_runtime.shape[0])
            if n_rho == 1 and bs > 1:
                rho_runtime = rho_runtime.expand(bs, *rho_runtime.shape[1:])
            elif n_rho != bs:
                raise ValueError(
                    f"Batched hyper-smearing rho has leading dim {n_rho}, expected {bs}"
                )
        return rho_runtime, batched_rho

    def _mask_rho(self, rho_runtime, batched_rho, sm, m):
        if not self.use_hyper_smearing:
            return torch.abs(rho_runtime[sm, m, :])
        return rho_runtime[:, sm, m, :] if batched_rho else rho_runtime[sm, m, :]

    def forward(self, cfgs, mask, rho_layer, dmasking=False, dmask=None, beta=None, delta_beta=None):
        real_dtype = cfgs.real.dtype
        bs = cfgs.shape[0]
        dlogJ = torch.zeros(bs, dtype=real_dtype, device=cfgs.device)
        dims = tuple(np.arange(1,self.D+2))
        rho_runtime, batched_rho = self._runtime_rho(cfgs, rho_layer, beta, delta_beta)

        use_selected_links = self.N == 3 and not self.use_nn
        selected_links = (
            self._links_for_smearing_mask(mask, dmask if dmasking else None)
            if use_selected_links else None
        )
        flat = sun.flat_links(cfgs, self.D) if use_selected_links else None
        cfg_shape = tuple(cfgs.shape)

        for sm in range(self.smearing_steps_per_layer):
            for mu in range(self.D):
                for eo in range(2):
                    m = eo + mu * 2
                    rho = self._mask_rho(rho_runtime, batched_rho, sm, m)

                    if use_selected_links:
                        sites = selected_links.sites[m]
                        if sites.numel() == 0:
                            continue

                        def selected_staples(flat_cfgs, rho_value, m=m, links=selected_links, cfg_shape=cfg_shape):
                            return self._selected_stout_staples(
                                flat_cfgs, cfg_shape, links, m, rho_value
                            )

                        if torch.is_grad_enabled():
                            C = checkpoint.checkpoint(
                                selected_staples,
                                flat,
                                rho,
                                use_reentrant=False,
                                preserve_rng_state=False,
                            )
                        else:
                            C = selected_staples(flat, rho)
                        U = flat.index_select(1, selected_links.links[m])
                        step = ExactSmearingStep.apply if self.explicit_backward else smearing_step
                        U_out, logdet = step(U, C)
                        flat = flat.index_copy(1, selected_links.links[m], U_out)
                        dlogJ = dlogJ + logdet.sum(dim=1)
                        continue

                    # SU(2) keeps the legacy full-lattice implementation; upstream's selected
                    # exact Jacobian is specifically the SU(3) stout path.
                    legacy_dmask = self._class_dmask(dmask if dmasking else None)
                    current_mask = mask[m, :] if legacy_dmask is None else mask[m, :] * legacy_dmask

                    cfgs_new, jac = self.stout_smearing(cfgs, mu, rho)
                    cfgs = current_mask * cfgs_new + (1-current_mask) * cfgs
                    ones = torch.ones(mask[0].shape, dtype=real_dtype, device=cfgs.device).squeeze(-1).squeeze(-1)
                    active = current_mask.squeeze(-1).squeeze(-1)
                    if self.N == 3:
                        dlogJ += torch.sum(active * jac.squeeze(1), dims)
                    else:
                        masked_jac = active * jac + (1 - active) * ones
                        dlogJ += torch.sum(torch.log(torch.clamp(masked_jac, min=1e-12)), dims)
        if use_selected_links:
            cfgs = flat.reshape(cfg_shape)
        return cfgs, 0, dlogJ

    def inverse(self, cfgs, mask, rho_layer, dmasking=False, dmask=None, beta=None, delta_beta=None,
                tol=1e-12, max_iter=100):
        """
        Inverse of forward (SU(3) stout path) with the same rho arguments: the masked updates are
        undone in reverse order. The staples of a mask contain only links it leaves fixed, so they
        are the same before and after its update and each link is inverted on its own.
        Returns (cfgs, 0, logdet of the inverse map); iteration counts and the largest residual
        are left in self.last_inverse_stats.
        """
        if self.N != 3 or self.use_nn:
            raise NotImplementedError("the inverse is implemented for the SU(3) stout path only")
        rho_runtime, batched_rho = self._runtime_rho(cfgs, rho_layer, beta, delta_beta)
        links = self._links_for_smearing_mask(mask, dmask if dmasking else None)
        flat = sun.flat_links(cfgs, self.D)
        cfg_shape = tuple(cfgs.shape)
        dlogJ = torch.zeros(cfgs.shape[0], dtype=cfgs.real.dtype, device=cfgs.device)
        stats = {"iterations": [], "residual": 0.0}

        for sm in reversed(range(self.smearing_steps_per_layer)):
            for m in reversed(range(2 * self.D)):
                if links.sites[m].numel() == 0:
                    continue
                rho = self._mask_rho(rho_runtime, batched_rho, sm, m)
                C = self._selected_stout_staples(flat, cfg_shape, links, m, rho)
                U, logdet, iterations, residual = inverse_smearing_step(
                    flat.index_select(1, links.links[m]), C, tol=tol, max_iter=max_iter
                )
                if not residual <= tol:
                    raise RuntimeError(
                        f"stout inverse did not converge for mask {m}: residual {float(residual):.3e} "
                        f"after {iterations} iterations"
                    )
                flat = flat.index_copy(1, links.links[m], U)
                dlogJ = dlogJ + logdet.sum(dim=1)
                stats["iterations"].append(iterations)
                stats["residual"] = max(stats["residual"], float(residual))
        self.last_inverse_stats = stats
        return flat.reshape(cfg_shape), 0, dlogJ

    def _class_dmask(self, dmask):
        """
        In class mode, restrict dmask to the links with at least one classified staple: the
        others are not transformed (rather than smeared with rho = 0). The tensor is cached so
        that the link-index cache, keyed on dmask's identity, keeps working.
        """
        if not (self.use_hyper_smearing and self.hyper_smearing.mode == "class"):
            return dmask
        key = (id(dmask), None if dmask is None else dmask._version, self.hyper_smearing.class_masks.device)
        cached = getattr(self, "_class_dmask_cache", None)
        if cached is not None and cached[0] == key and cached[1] is dmask:
            return cached[2]
        classified = self.hyper_smearing.classified_links()
        if dmask is None:
            effective = classified.view(1, *classified.shape, 1, 1).to(dtype=torch.float64)
        else:
            effective = dmask * classified.view(dmask.shape).to(dtype=dmask.dtype, device=dmask.device)
        self._class_dmask_cache = (key, dmask, effective)
        return effective

    def _links_for_smearing_mask(self, mask, dmask=None):
        return self._smearing_links_cache.get(mask, self.D, self._class_dmask(dmask))

    def _rho_at_sites(self, rho, channel, sites, batch_size, lattice_shape):
        rho = rho.to(device=sites.device)
        nsites = int(sites.numel())
        if self.rho_shape_type == 0:
            values = rho.reshape(-1)
            if values.numel() == 1:
                return values.reshape(1, 1).expand(batch_size, nsites)
            if values.numel() == batch_size:
                return values.reshape(batch_size, 1).expand(batch_size, nsites)
            raise ValueError(f"Unsupported isotropic rho shape {tuple(rho.shape)} for batch={batch_size}")
        if self.rho_shape_type == 1:
            nslots = max(2 * (self.D - 1), self.D * (self.D - 1) // 2)
            if rho.numel() == nslots:
                return rho.reshape(nslots)[channel].reshape(1, 1).expand(batch_size, nsites)
            if rho.numel() == batch_size * nslots:
                return rho.reshape(batch_size, nslots)[:, channel].reshape(batch_size, 1).expand(batch_size, nsites)
            raise ValueError(f"Unsupported anisotropic rho shape {tuple(rho.shape)} for batch={batch_size}")
        if self.rho_shape_type not in (2, 3):
            raise ValueError(f"Unknown rho_shape_type {self.rho_shape_type}")

        spatial_ndim = len(lattice_shape)
        if rho.dim() == spatial_ndim + 1:
            if rho.shape[0] <= channel or tuple(rho.shape[1:]) != tuple(lattice_shape):
                raise ValueError(f"General rho shape {tuple(rho.shape)} does not match lattice {lattice_shape}")
            values = rho[channel].reshape(-1).index_select(0, sites).reshape(1, nsites)
            return values.expand(batch_size, nsites)
        if rho.dim() == spatial_ndim + 2:
            if rho.shape[1] <= channel or tuple(rho.shape[2:]) != tuple(lattice_shape):
                raise ValueError(f"Batched general rho shape {tuple(rho.shape)} does not match lattice {lattice_shape}")
            rho_batch = rho.shape[0]
            if rho_batch == 1 and batch_size > 1:
                values = rho[:, channel].expand(batch_size, *lattice_shape)
            elif rho_batch == batch_size:
                values = rho[:, channel]
            else:
                raise ValueError(f"General rho batch={rho_batch}, expected 1 or {batch_size}")
            return values.reshape(batch_size, -1).index_select(1, sites)
        raise ValueError(f"Unsupported general rho shape {tuple(rho.shape)}")

    def _selected_stout_staples(self, flat, cfg_shape, links, m, rho):
        mu = m // 2
        batch_size = int(cfg_shape[0])
        lattice_shape = tuple(cfg_shape[2:-2])
        sites = links.sites[m]
        staples = torch.zeros(
            (batch_size, int(sites.numel()), self.N, self.N), dtype=flat.dtype, device=flat.device
        )
        other_directions = [nu for nu in range(self.D) if nu != mu]
        for indices, nu in zip(links.staple_links[m], other_directions):
            upper, lower = sun.staples_at(flat, indices)
            pidx = 2 * sun.plaq_index(self.D, mu, nu)
            positive = self._rho_at_sites(rho, pidx, sites, batch_size, lattice_shape)
            negative = self._rho_at_sites(rho, pidx + 1, sites, batch_size, lattice_shape)
            staples = staples + positive[..., None, None] * sun.SUN_dagger(upper)
            staples = staples + negative[..., None, None] * sun.SUN_dagger(lower)
        return staples

    def stout_smearing(self, cfgs, mu, rho):
        #C = stout_staples(cfgs, mu, D, rho)
        if torch.is_grad_enabled():
            C = checkpoint.checkpoint(
                self.compute_staples,
                cfgs,
                mu,
                rho,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            C = self.compute_staples(cfgs, mu, rho)
    
        U = cfgs[:,mu]
        if self.N == 3:
            step = ExactSmearingStep.apply if self.explicit_backward else smearing_step
            U_out, logdet = step(U, C)
            return U_out.unsqueeze(1), logdet.unsqueeze(1)

        id = sun.SUN_identity(U.shape[:-1], dtype=U.dtype, device=U.device)
        N = U.shape[-1]
        omega = generate_omega(C, U)
        Q = generate_Q(omega, N)
        Q2 = sun.SUN_mul(Q, Q)

        oidid = otimes(id, id)
        oidQ = otimes(id, Q) + otimes(Q, id)

        f0, f1, f2, B1, B2, _ = generate_coefficients(Q, Q2, id, oidid, oidQ, self.device, backward=False)
        expQ = generate_expQ(Q, Q2, id, f0, f1, f2)
        dexpQdQ = generate_dexpQ_dQ(Q, Q2, B1, B2, f1, f2, id, oidid, oidQ)

        dQdomega = generate_dQ_domega(N, oidid, id)
        dQdU = generate_dQ_dU(dQdomega, id, C)
        dexpQdU = generate_dexpQ_dU(dexpQdQ, dQdU)
        Jacobian_U = generate_Jacobian_U(dexpQdU, U, expQ, oidid)
        detjac = det_Jac(Jacobian_reshape(Jacobian_U, self.jac_shape)).abs()
    
        return sun.SUN_mul(expQ, U).unsqueeze(1), detjac.unsqueeze(1)

    def stout_staples(self, cfgs, mu, rho):
        staple = torch.zeros(cfgs[:,0].shape, dtype=cfgs.dtype, device=cfgs.device)

        for nu in range(self.D):
            if nu != mu:
                #isotropic rho
                if self.rho_shape_type == 0:
                    if int(rho.numel()) == int(cfgs.shape[0]) and int(cfgs.shape[0]) > 1:
                        rho_batch = rho.reshape(int(cfgs.shape[0]))
                        rho_view = rho_batch.reshape((int(cfgs.shape[0]),) + (1,) * (staple.dim() - 1))
                    elif int(rho.numel()) == 1:
                        rho_view = rho.reshape((1,) * staple.dim())
                    else:
                        raise ValueError(
                            "Unsupported isotropic rho shape: expected one scalar or one value per "
                            f"sample, got shape {tuple(rho.shape)} for batch={int(cfgs.shape[0])}"
                        )
                    prho_mul = rho_view
                    nrho_mul = rho_view
                #anisotropic rho
                elif self.rho_shape_type == 1:
                    n_plaq = self.D * (self.D - 1) // 2
                    # The existing D=4 plaq_index convention addresses the
                    # six (positive/negative) link-parity channels as one
                    # flat axis, not 2*n_plaq channels.
                    n_slots = max(2 * (self.D - 1), n_plaq)
                    pidx = 2 * sun.plaq_index(self.D, mu, nu)
                    nidx = pidx + 1
                    batch_size = int(cfgs.shape[0])
                    if int(rho.numel()) == batch_size * n_slots and batch_size > 1:
                        rho_links = rho.reshape(batch_size, n_slots)
                        prho_mul = rho_links[:, pidx].reshape((batch_size,) + (1,) * (staple.dim() - 1))
                        nrho_mul = rho_links[:, nidx].reshape((batch_size,) + (1,) * (staple.dim() - 1))
                    elif int(rho.numel()) == n_slots:
                        rho_links = rho.reshape(n_slots)
                        prho_mul = rho_links[pidx].reshape((1,) * staple.dim())
                        nrho_mul = rho_links[nidx].reshape((1,) * staple.dim())
                    else:
                        raise ValueError(
                            "Unsupported anisotropic rho shape: expected "
                            f"{n_slots} static or {batch_size * n_slots} batched values, "
                            f"got shape {tuple(rho.shape)}"
                        )
                #general rho
                elif self.rho_shape_type == 2 or self.rho_shape_type == 3:
                    pidx = 2 * sun.plaq_index(self.D, mu, nu)
                    nidx = pidx + 1
                    if rho.dim() == self.D + 1:
                        prho_mul = rho[pidx].unsqueeze(-1).unsqueeze(-1)
                        nrho_mul = rho[nidx].unsqueeze(-1).unsqueeze(-1)
                    elif rho.dim() == self.D + 2:
                        prho_mul = rho[:, pidx].unsqueeze(-1).unsqueeze(-1)
                        nrho_mul = rho[:, nidx].unsqueeze(-1).unsqueeze(-1)
                    else:
                        raise ValueError(
                            f"Unsupported rho shape for general smearing: {tuple(rho.shape)}"
                        )
                staple += prho_mul * sun.SUN_dagger(sun.pstaple(cfgs, mu, nu, self.D)) + nrho_mul * sun.SUN_dagger(sun.nstaple(cfgs, mu, nu, self.D))

        return staple

    def stout_staples_nn(self, cfgs, mu, rho):
        staple = torch.zeros(cfgs[:,0].shape, dtype=cfgs.dtype, device=cfgs.device)
        staple_list = []
        trace_list = []

        for nu in range(self.D):
            if nu != mu:
                staple_list.append(sun.pstaple(cfgs, mu, nu, self.D))
                staple_list.append(sun.nstaple(cfgs, mu, nu, self.D))
        
        if self.use_nn:
            for il in range(6):
                for jl in range(il+1,6+il):
                    kl = jl % 6
                    trace_list.append(sun.SUN_trace(sun.SUN_mul(staple_list[il], sun.SUN_dagger(staple_list[kl]))))
            #Maybe:
            Nw = len(trace_list)
            bs,_,T,L,L,L,_,_ = cfgs.shape
            trace_list = torch.stack(trace_list).reshape((bs,Nw,T,L,L,L)) #reshape needed since the "appended" dimension is in 0, needed shape [Bs, 2*Channels, T,L,L,L] with 2*Channels from complex number
            #print(trace_list.shape)
            trace_list = torch.cat([trace_list.real,trace_list.imag],dim=1)

        il = 0
        for nu in range(self.D):
            if nu != mu:
                #isotropic rho
                if self.rho_shape_type == 0:
                    if self.use_nn:
                        #time = torch.tensor([1.0])
                        out = self.net(trace_list, self.t)
                        prho_mul, nrho_mul = out[:,0], out[:,1] #shape [Bs,T,L,L,L], assuming two output channels
                        prho_mul = prho_mul.unsqueeze(-1).unsqueeze(-1)    #shape [Bs,T,L,L,L,1,1]
                        nrho_mul = nrho_mul.unsqueeze(-1).unsqueeze(-1)    #shape [Bs,T,L,L,L,1,1]
                        if self.residual:
                            #print(rho,prho_mul.mean(), nrho_mul.mean())
                            prho_mul = rho + prho_mul   #shape [Bs,T,L,L,L,1,1]
                            nrho_mul = rho + nrho_mul
                    else:
                        prho_mul = rho #just a number
                        nrho_mul = rho #just a number
                #anisotropic rho
                elif self.rho_shape_type == 1:
                    prho_mul = rho[2 * sun.plaq_index(self.D, mu, nu)] #just a number
                    nrho_mul = rho[2 * sun.plaq_index(self.D, mu, nu) + 1] #just a number
                #general rho
                elif self.rho_shape_type == 2 or self.rho_shape_type == 3:
                    pidx = 2 * sun.plaq_index(self.D, mu, nu)
                    nidx = pidx + 1
                    if rho.dim() == 5:
                        prho_mul = rho[pidx].unsqueeze(-1).unsqueeze(-1)
                        nrho_mul = rho[nidx].unsqueeze(-1).unsqueeze(-1)
                    elif rho.dim() == 6:
                        prho_mul = rho[:, pidx].unsqueeze(-1).unsqueeze(-1)
                        nrho_mul = rho[:, nidx].unsqueeze(-1).unsqueeze(-1)
                    else:
                        raise ValueError(
                            f"Unsupported rho shape for general smearing: {tuple(rho.shape)}"
                        )
                staple += prho_mul * sun.SUN_dagger(staple_list[il]) + nrho_mul * sun.SUN_dagger(staple_list[il+1])
                il += 2
        return staple 

    #def stout_smearing(self, cfgs, mu, D, rho, jac_shape):
    #    #C = stout_staples(cfgs, mu, D, rho)
    #    C = checkpoint.checkpoint(stout_staples, cfgs, mu, D, rho, use_reentrant=False)

    #    new_cfgs, Jacobian_U = SmearingStep.apply(cfgs[:,mu], C)
    #    detjac = det_Jac(Jacobian_reshape(Jacobian_U, jac_shape)).abs()

    #    return new_cfgs.unsqueeze(1), detjac.unsqueeze(1)


class SmearingStep(torch.autograd.Function):
    """Legacy `(U, C) -> (U', full Jacobian)` interface retained for compatibility.

    The stable exact logdet implementation is exported as ``ExactSmearingStep`` and is used by
    ``Smearing.stout_smearing`` for SU(3).
    """

    @staticmethod
    def forward(U, C):
        id = sun.SUN_identity(U.shape[:-1], dtype=U.dtype, device=U.device)
        N = U.shape[-1]
        omega = generate_omega(C, U)
        Q = generate_Q(omega, N)
        Q2 = sun.SUN_mul(Q, Q)

        oidid = otimes(id, id)
        oidQ = otimes(id, Q) + otimes(Q, id)

        f0, f1, f2, B1, B2, _ = generate_coefficients(Q, Q2, id, oidid, oidQ, U.device, backward=False)
        expQ = generate_expQ(Q, Q2, id, f0, f1, f2)
        dexpQdQ = generate_dexpQ_dQ(Q, Q2, B1, B2, f1, f2, id, oidid, oidQ)

        dQdomega = generate_dQ_domega(N, oidid, id)
        dQdU = generate_dQ_dU(dQdomega, id, C)
        dexpQdU = generate_dexpQ_dU(dexpQdQ, dQdU)
        Jacobian_U = generate_Jacobian_U(dexpQdU, U, expQ, oidid)
    
        return sun.SUN_mul(expQ, U), Jacobian_U

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        # ctx is a context object that can be used to stash information
        # for backward computation
        U, C = inputs
        ctx.save_for_backward(U, C)

    @staticmethod
    def backward(ctx, grad_Up, grad_Ju):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        U, C = ctx.saved_tensors
        id = sun.SUN_identity(U.shape[:-1], dtype=U.dtype, device=U.device)
        N = U.shape[-1]
        omega = generate_omega(C, U)
        Q = generate_Q(omega, N)
        Q2 = sun.SUN_mul(Q, Q)

        oidid = otimes(id, id)
        oidQ = otimes(id, Q) + otimes(Q, id)

        f0, f1, f2, B1, B2, d2expQdQ2 = generate_coefficients(Q, Q2, id, oidid, oidQ, U.device, backward=True)
        expQ = generate_expQ(Q, Q2, id, f0, f1, f2)
        dexpQdQ = generate_dexpQ_dQ(Q, Q2, B1, B2, f1, f2, id, oidid, oidQ)

        dQdomega = generate_dQ_domega(N, oidid, id)
        dQdU = generate_dQ_dU(dQdomega, id, C)
        dexpQdU = generate_dexpQ_dU(dexpQdQ, dQdU)
        Jacobian_U = generate_Jacobian_U(dexpQdU, U, expQ, oidid)

        dQdC = generate_dQ_dC(dQdomega, oidid, U)
        Jacobian_C = generate_Jacobian_C(dexpQdQ, dQdC, U)

        dJudU, dJudC = generate_Jacobian_gradients(id, U, dQdU, dQdC, expQ, dexpQdU, dexpQdQ, d2expQdQ2)

        return gradjacprod(grad_Up, Jacobian_U) + graddjacprod(grad_Ju, dJudU), gradjacprod(grad_Up, Jacobian_C) + graddjacprod(grad_Ju, dJudC)
