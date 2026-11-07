"""One parametric joint action for N_f = 2 Wilson fermions, three protocol parameters.

    S(U, phi; lambda) = S_g(U; beta, bc) + S_pf(U, phi; kappa),

with lambda one of "beta", "kappa", "bc" (chosen by `parameter_name`), the other two parameters
fixed at construction (phase 3/4 of fermions_plan.md, "Tre protocolli, una interfaccia").

- S_g is `neoqcd.theory.Wilson_action` (via `hmc.WilsonGaugeAction`), with the defect weights of
  `theory.plaquette_weight`: the link coupling is c = defect_mask[0] + defect_mask[1] * bc and the
  weight of a plaquette is the product of the c of its four links. The defect is gauge-only
  (Bonanno et al., arXiv:2404.14151): the fermions never see bc.
- S_pf = phi^dagger (D_hat^dagger D_hat)^-1 phi (`pseudofermion.TwoFlavourWilson`, even-odd by
  default). The pseudofermion field phi is state: it lives in the one fermion term held by the
  action (`QCDAction.fermion`), is drawn by `refresh` and by every HMC trajectory, and is shared
  with the Hamiltonians built by `hamiltonian(parameter)`.

Cost of a jump at fixed fields, S(U, phi; p_new) - S(U, phi; p_old) (`action_difference`):
- beta: (p_new - p_old) dS_g/dbeta = (p_new - p_old) S_g(U; beta = 1, bc), no solve;
- kappa: S_pf(kappa_new) - S_pf(kappa_old) at fixed phi, two CG solves (the NE-MCMC driver
  keeps S at the old value and spends one);
- bc: gauge only, sum over plaquettes of (beta/N) (w_old - w_new) Re tr U_p, no solve.

Parameters are a float or a (B,) tensor (one value per configuration). For bc a (B,) tensor gives
per-configuration plaquette weights of shape (B,) + lattice (`bc_plaquette_weights`), which
`hmc.WilsonGaugeAction` accepts as they are.

kappa updates: `ParametricTwoFlavourWilson` is a `TwoFlavourWilson` with `set_kappa`. Nothing is
cached per kappa in TwoFlavourWilson (the Dirac operator is rebuilt from `self.kappa` at every
call), so changing kappa costs nothing; the only kappa-dependent cache is eta^dagger eta = S_pf at
the refresh links, which this subclass uses only while kappa equals the kappa of the refresh. The
last CG solution X is kept across kappa changes as the initial guess (it is a good one for nearby
kappa and only affects the iteration count).
"""

import torch

from neoqcd.fermions._gauge import plaquette_fields
from neoqcd.fermions.hmc import Hamiltonian, WilsonGaugeAction
from neoqcd.fermions.pseudofermion import TwoFlavourWilson

__all__ = ["PARAMETER_NAMES", "bc_link_couplings", "bc_plaquette_weights", "ParametricTwoFlavourWilson",
           "JointHamiltonian", "QCDAction"]

PARAMETER_NAMES = ("beta", "kappa", "bc")


def _as_parameter(value):
    """A float for scalars (python numbers, 0-dim or one-element tensors), else a float64 (B,) tensor."""
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.reshape(-1)[0].item())
        if value.dim() != 1:
            raise ValueError(f"a parameter must be a scalar or a (B,) tensor, got shape {tuple(value.shape)}")
        return value.detach().to(torch.float64)
    return float(value)


def _same_parameter(a, b):
    if a is None or b is None:
        return False
    if torch.is_tensor(a) or torch.is_tensor(b):
        a, b = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
        return bool(torch.equal(*torch.broadcast_tensors(a, b)))
    return float(a) == float(b)


def bc_link_couplings(defect_mask, bc):
    """c = defect_mask[0] + defect_mask[1] * bc on the links: (D,) + lattice for a scalar bc,
    (B, D) + lattice for a (B,) tensor."""
    bc = _as_parameter(bc)
    if torch.is_tensor(bc):
        view = (-1,) + (1,) * (defect_mask.dim() - 1)
        bc = bc.to(dtype=defect_mask.dtype, device=defect_mask.device).view(view)
        return defect_mask[0].unsqueeze(0) + defect_mask[1].unsqueeze(0) * bc
    return defect_mask[0] + defect_mask[1] * bc


def bc_plaquette_weights(defect_mask, bc, D):
    """Plaquette weights {(mu, nu): w} (mu < nu) of `theory.Wilson_action` with `defect_par = bc`.

    w_{mu nu}(x) = c_mu(x) c_nu(x + mu) c_mu(x + nu) c_nu(x) as in `theory.plaquette_weight`
    (x + mu = roll by +1 along the axis of mu). Shape lattice for a scalar bc, (B,) + lattice for a
    (B,) tensor (batched defect weights, kept local to this module).
    """
    c = bc_link_couplings(defect_mask, bc)
    link_axis = c.dim() - D - 1  # 0 (scalar bc) or 1 (batched)
    weights = {}
    for nu in range(1, D):
        for mu in range(nu):
            c_mu, c_nu = c.select(link_axis, mu), c.select(link_axis, nu)
            weights[(mu, nu)] = (c_mu * torch.roll(c_nu, 1, dims=-D + mu)
                                 * torch.roll(c_mu, 1, dims=-D + nu) * c_nu)
    return weights


class ParametricTwoFlavourWilson(TwoFlavourWilson):
    """`TwoFlavourWilson` whose kappa can change between calls at fixed phi (module doc)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._refresh_kappa = None

    def set_kappa(self, kappa):
        self.kappa = _as_parameter(kappa)

    def set_phi(self, phi):
        """Install a pseudofermion field (e.g. carried by a `FieldState`); drops the refresh cache."""
        self.phi = phi
        self.X = None
        self._refresh_links = None
        self._refresh_action = None
        self._refresh_kappa = None

    def refresh(self, U, generator=None):
        super().refresh(U, generator)
        self._refresh_kappa = self.kappa

    def action_difference(self, U_old, U_new):
        if _same_parameter(self.kappa, self._refresh_kappa):
            return super().action_difference(U_old, U_new)
        return self.action(U_new) - self.action(U_old)


class JointHamiltonian(Hamiltonian):
    """Fermion term on the outer level, gauge term on the inner one (as `TwoFlavourHamiltonian`),
    but with a given (shared) fermion term. Attributes `fermion` and `gauge`."""

    def __init__(self, fermion, gauge, n_gauge_substeps=1):
        self.fermion = fermion
        self.gauge = gauge
        super().__init__([([fermion], 1), ([gauge], n_gauge_substeps)])


class QCDAction:
    """S(U, phi; lambda) = S_g(U; beta, bc) + S_pf(U, phi; kappa), lambda = `parameter_name`.

    Args:
        geometry: `lattice.Geometry`.
        parameter_name: "beta", "kappa" or "bc", the protocol parameter.
        beta, kappa: couplings; the one named by `parameter_name` is the initial parameter value.
        bc: defect parameter (default 1, no defect); needs `defect_mask` (shape (2, D) + lattice,
            the layout of `utils.Defect`/`DefectMCMCAdapter`) unless bc stays 1 and the protocol
            is not "bc".
        n_gauge_substeps: inner gauge steps per outer fermion step of the Hamiltonians.
        Extra keyword arguments go to `TwoFlavourWilson` (time_sign, even_odd, solver_tol, ...).

    Interface (that of a target for distributedPT/Flow):
        set_parameter(value), parameter;
        action(U, parameter=None) -> (B,)                 needs phi (call `refresh` first);
        gauge_action(U, parameter=None), fermion_action(U, parameter=None);
        action_difference(U, p_old, p_new) -> (B,)        at fixed (U, phi), see module doc;
        refresh(U, generator)                             phi heat bath at the current parameter;
        hamiltonian(parameter=None) -> JointHamiltonian   for the HMC blocks;
        phi (get/set), fermion (the shared ParametricTwoFlavourWilson, holding phi and its stats).
    `hamiltonian(p)` sets the current parameter to p, because its fermion term is the shared one.
    """

    def __init__(self, geometry, parameter_name, beta, kappa, bc=1.0, defect_mask=None,
                 n_gauge_substeps=1, **fermion_kwargs):
        if parameter_name not in PARAMETER_NAMES:
            raise ValueError(f"parameter_name must be one of {PARAMETER_NAMES}, got {parameter_name!r}")
        self.geometry = geometry
        self.parameter_name = parameter_name
        self.beta = _as_parameter(beta)
        self.kappa = _as_parameter(kappa)
        self.bc = _as_parameter(bc)
        if defect_mask is not None:
            defect_mask = torch.as_tensor(defect_mask, dtype=torch.float64)
            expected = (2, geometry.D) + geometry.lattice
            if tuple(defect_mask.shape) != expected:
                raise ValueError(f"defect_mask must have shape {expected}, got {tuple(defect_mask.shape)}")
        elif parameter_name == "bc":
            raise ValueError("a bc protocol needs a defect_mask")
        self.defect_mask = defect_mask
        self.n_gauge_substeps = int(n_gauge_substeps)
        self.fermion = ParametricTwoFlavourWilson(self.kappa, geometry, **fermion_kwargs)

    # ---- parameter -------------------------------------------------------------------------
    @property
    def parameter(self):
        return getattr(self, self.parameter_name)

    def set_parameter(self, value):
        setattr(self, self.parameter_name, _as_parameter(value))
        if self.parameter_name == "kappa":
            self.fermion.set_kappa(self.kappa)

    def _couplings(self, parameter):
        """(beta, kappa, bc) with the protocol parameter replaced by `parameter` if given."""
        c = {"beta": self.beta, "kappa": self.kappa, "bc": self.bc}
        if parameter is not None:
            c[self.parameter_name] = _as_parameter(parameter)
        return c["beta"], c["kappa"], c["bc"]

    # ---- pseudofermion state ---------------------------------------------------------------
    @property
    def phi(self):
        return self.fermion.phi

    @phi.setter
    def phi(self, value):
        self.fermion.set_phi(value)

    def refresh(self, U, generator=None):
        """phi = D_hat^dagger eta at the current kappa: S_pf(U) = eta^dagger eta afterwards."""
        self.fermion.set_kappa(self.kappa)
        self.fermion.refresh(U, generator)

    # ---- gauge part -----------------------------------------------------------------------
    def plaquette_weights(self, bc):
        if self.defect_mask is None:
            if not _same_parameter(bc, 1.0):
                raise ValueError("bc != 1 needs a defect_mask")
            return None
        return bc_plaquette_weights(self.defect_mask, bc, self.geometry.D)

    def gauge_term(self, parameter=None):
        beta, _, bc = self._couplings(parameter)
        return WilsonGaugeAction(beta, self.geometry, self.plaquette_weights(bc))

    def gauge_action(self, U, parameter=None):
        return self.gauge_term(parameter).action(U)

    # ---- fermion part ---------------------------------------------------------------------
    def fermion_action(self, U, parameter=None):
        """S_pf(U, phi; kappa) at the kappa of `parameter` (one solve); the current kappa is kept."""
        _, kappa, _ = self._couplings(parameter)
        saved = self.fermion.kappa
        self.fermion.set_kappa(kappa)
        try:
            return self.fermion.action(U)
        finally:
            self.fermion.set_kappa(saved)

    # ---- joint action ---------------------------------------------------------------------
    def action(self, U, parameter=None):
        return self.gauge_action(U, parameter) + self.fermion_action(U, parameter)

    def action_difference(self, U, p_old, p_new):
        """S(U, phi; p_new) - S(U, phi; p_old) at fixed fields, shape (B,)."""
        name = self.parameter_name
        if name == "beta":
            beta_old, beta_new = _as_parameter(p_old), _as_parameter(p_new)
            dbeta = beta_new - beta_old
            if torch.is_tensor(dbeta):
                dbeta = dbeta.to(device=U.device)
            slope = WilsonGaugeAction(1.0, self.geometry, self.plaquette_weights(self.bc)).action(U)
            return dbeta * slope
        if name == "kappa":
            return self.fermion_action(U, p_new) - self.fermion_action(U, p_old)
        # bc: gauge only, (beta/N) sum_p (w_old - w_new) Re tr U_p
        D, N = self.geometry.D, self.geometry.N
        w_old = bc_plaquette_weights(self.defect_mask, p_old, D)
        w_new = bc_plaquette_weights(self.defect_mask, p_new, D)
        planes = ((mu, nu) for nu in range(1, D) for mu in range(nu))
        total = None
        for (mu, nu), retr in zip(planes, plaquette_fields(U, D)):
            dw = (w_old[(mu, nu)] - w_new[(mu, nu)]).to(device=retr.device, dtype=retr.dtype)
            term = dw * retr
            total = term if total is None else total + term
        total = total.sum(tuple(range(1, total.dim())))
        beta = self.beta
        if torch.is_tensor(beta):
            beta = beta.to(device=U.device)
        return beta * total / float(N)

    # ---- HMC -------------------------------------------------------------------------------
    def hamiltonian(self, parameter=None):
        """JointHamiltonian at `parameter` (default: the current one), which becomes current."""
        if parameter is not None:
            self.set_parameter(parameter)
        self.fermion.set_kappa(self.kappa)
        return JointHamiltonian(self.fermion, self.gauge_term(), self.n_gauge_substeps)
