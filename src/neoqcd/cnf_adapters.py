"""Continuous drift fields induced by the existing SU(3) SNF layers."""

import torch
from torch import nn

import neoqcd.sun_utils as sun
from neoqcd.smearing import (
    ResidualNormalizingFlows,
    Smearing,
    _su3_generators,
    generate_Q,
    generate_omega,
)
from neoqcd.nn import HyperTimeScalar, time_embedding


def _matrix_to_coordinates(matrix, generators):
    return 2.0 * torch.einsum("...ij,aji->...a", matrix, generators).real


class ExistingDrift(nn.Module):
    """The infinitesimal SU(3) generator of an existing smearing layer.

    The layer is interpreted per unit change of its protocol parameter. Its
    finite SNF update is recovered by integrating this field, not by applying
    the original discrete update.
    """

    def __init__(self, architecture, flow_pars, rho_init=1e-3):
        super().__init__()
        self.architecture = str(architecture).lower()
        self.D = int(flow_pars.D)
        if int(flow_pars.N) != 3:
            raise ValueError("ExistingDrift supports SU(3) only")
        if int(getattr(flow_pars, "rho_shape_type", 0)) == 3:
            raise ValueError("ExistingDrift requires full-lattice fields, not cropped defect layers")
        self.use_rho_functional = bool(getattr(flow_pars, "use_rho_functional", False))
        if bool(getattr(flow_pars, "use_nn", False)):
            raise ValueError(
                "use_nn is the legacy 4D-only smearing path; its fixed-time CNN cannot accept "
                "the per-rung protocol parameter used by the continuous field"
            )
        if self.use_rho_functional and self.architecture != "smearing":
            raise ValueError("rho_functional is available only with the smearing architecture")

        # Keep constant generators out of registered buffers for NCCL DDP.
        self.generators = _su3_generators(dtype=torch.cdouble, device=torch.device("cpu"))
        self.steps = int(getattr(flow_pars, "smearing_steps_per_layer", 1))
        self.mask_mod = 2

        if self.architecture in {"smearing", "hyper-smearing"}:
            is_hyper = self.architecture == "hyper-smearing"
            if bool(getattr(flow_pars, "use_hyper_smearing", False)) != is_hyper:
                raise ValueError(f"FlowPars use_hyper_smearing must match {self.architecture!r}")
            self.smearing = Smearing(flow_pars, time=1.0)
            if is_hyper:
                self.rho = None
            elif self.use_rho_functional:
                if int(flow_pars.nstep) < 1:
                    raise ValueError("rho_functional requires at least one protocol step")
                self.rho = None
                self.rho_functional = HyperTimeScalar(
                    time_embedding(flow_pars.F, T=2.0, device=flow_pars.device),
                    flow_pars.rho_shape,
                    flow_pars.nstep,
                    flow_pars.scale_fac,
                    hidden_dim=flow_pars.hidden_time_mlp,
                    device=flow_pars.device,
                )
                # The same Linear modules are registered both directly and inside time_mlp;
                # functional_call's frozen parameter swap mis-restores these aliases per rung.
                del self.rho_functional.fc1
                del self.rho_functional.fc2
            else:
                self.rho = nn.Parameter(
                    torch.full(flow_pars.rho_shape, float(rho_init), dtype=torch.float64,
                               device=flow_pars.device)
                )
        elif self.architecture in {"residual", "hyper-residual"}:
            parameterization = "static" if self.architecture == "residual" else "hyper"
            if str(getattr(flow_pars, "residual_parameterization", "hyper")).lower() != parameterization:
                raise ValueError(
                    f"FlowPars residual_parameterization must be {parameterization!r} "
                    f"for {self.architecture!r}"
                )
            self.residual = ResidualNormalizingFlows(flow_pars, time=1.0)
            self.mask_mod = self.residual.mask_mod
        else:
            raise ValueError(
                "architecture must be one of smearing, hyper-smearing, residual, hyper-residual"
            )

    def _masks(self, cfgs):
        lattice = tuple(int(n) for n in cfgs.shape[2:-2])
        if len(lattice) != self.D:
            raise ValueError(f"configuration has {len(lattice)} lattice axes, expected {self.D}")
        if any(n % self.mask_mod for n in lattice):
            raise ValueError(
                f"all periodic lattice extents must be divisible by mask_mod={self.mask_mod}"
            )
        coords = torch.meshgrid(
            *(torch.arange(n, device=cfgs.device) for n in lattice), indexing="ij"
        )
        colors = sum(coords)
        return [(color, colors.remainder(self.mask_mod) == color)
                for color in range(self.mask_mod)]

    def _smearing_rho(self, t, cfgs):
        if self.use_rho_functional:
            t = torch.as_tensor(t, dtype=cfgs.real.dtype, device=cfgs.device).reshape(-1)
            if t.numel() == 1:
                return self.rho_functional(t[0]).abs() * self.rho_functional.nstep
            if t.numel() != cfgs.shape[0]:
                raise ValueError(f"t has {t.numel()} values, expected 1 or {cfgs.shape[0]}")
            return torch.vmap(self.rho_functional)(t).abs() * self.rho_functional.nstep
        if self.rho is not None:
            return self.rho.abs()
        t = torch.as_tensor(t, dtype=torch.float64, device=cfgs.device).reshape(-1)
        if t.numel() == 1:
            t = t[0]
        return self.smearing.hyper_smearing(beta=t, delta_beta=torch.ones_like(t))

    def _rho_step(self, rho, step, mu, color, batch_size):
        channel = 2 * mu + (color % 2)
        if rho.dim() == len(self.smearing.rho_shape) + 1:
            return rho[:, step, channel]
        return rho[step, channel]

    def _smearing_generator(self, cfgs, t, mu, color, rho):
        q = cfgs.new_zeros(cfgs.shape[0], *cfgs.shape[2:-2], 3, 3)
        # Every configured substep contributes its instantaneous generator at
        # the same configuration, as required for a vector field.
        for step in range(self.steps):
            rho_step = self._rho_step(rho, step, mu, color, cfgs.shape[0])
            staples = self.smearing.stout_staples(cfgs, mu, rho_step)
            q = q + generate_Q(generate_omega(staples, cfgs[:, mu]), 3)
        return q

    def _residual_generator(self, cfgs, t, mu, color, sites, lin_all, quad_all):
        residual = self.residual
        bs = int(cfgs.shape[0])
        q = cfgs.new_zeros(bs, *cfgs.shape[2:-2], 3, 3)
        mask = torch.zeros((1, self.D, *sites.shape, 1, 1), dtype=cfgs.real.dtype,
                           device=cfgs.device)
        mask[:, mu] = sites.to(dtype=cfgs.real.dtype).view(1, *sites.shape, 1, 1)
        for step in range(self.steps):
            if residual.n_pt:
                values = (1.0 - mask) * cfgs
                for point in range(residual.n_pt):
                    values = residual._frozen_convolution(values, step, point)
            else:
                values = cfgs
            staples = residual._oriented_staples(values, mu)
            features = residual._features(staples, cfgs[:, mu])
            weights = residual._feature_weights(features, lin_all[:, step], quad_all[:, step])
            C = residual._staple_sum(staples, weights, torch.ones(bs, dtype=cfgs.real.dtype,
                                                                   device=cfgs.device))
            q = q + generate_Q(generate_omega(C, cfgs[:, mu]), 3)
        return q

    def _residual_coefficients(self, cfgs, t):
        t = torch.as_tensor(t, dtype=torch.float64, device=cfgs.device).reshape(-1)
        if t.numel() == 1:
            t = t.expand(cfgs.shape[0])
        elif t.numel() != cfgs.shape[0]:
            raise ValueError(f"t has {t.numel()} values, expected 1 or {cfgs.shape[0]}")
        delta = torch.ones_like(t)
        return self.residual._coefficients(
            t, delta, cfgs.shape[0], cfgs.real.dtype, cfgs.device
        )[:2]

    def _generator(self, cfgs, t, mu, color, sites, rho, residual_coeffs):
        if self.architecture in {"smearing", "hyper-smearing"}:
            return self._smearing_generator(cfgs, t, mu, color, rho)
        return self._residual_generator(cfgs, t, mu, color, sites, *residual_coeffs)

    def _field(self, cfgs, t, *, divergence):
        if not torch.is_tensor(cfgs) or not torch.is_complex(cfgs) or cfgs.shape[-2:] != (3, 3):
            raise ValueError("cfgs must be a complex SU(3) link tensor ending in (3, 3)")
        if cfgs.dim() != self.D + 4 or cfgs.shape[1] != self.D:
            raise ValueError(f"cfgs must have shape (B, {self.D}, *lattice, 3, 3)")
        masks = self._masks(cfgs)
        generators = self.generators.to(device=cfgs.device, dtype=cfgs.dtype)
        coeffs = cfgs.real.new_zeros((*cfgs.shape[:-2], 8))
        div = cfgs.real.new_zeros(cfgs.shape[0])
        rho = self._smearing_rho(t, cfgs) if self.architecture in {"smearing", "hyper-smearing"} else None
        residual_coeffs = self._residual_coefficients(cfgs, t) if self.architecture in {
            "residual", "hyper-residual"
        } else None

        for mu in range(self.D):
            for color, sites in masks:
                q_fn = lambda u, mu=mu, color=color, sites=sites: _matrix_to_coordinates(
                    self._generator(u, t, mu, color, sites, rho, residual_coeffs), generators
                )
                q = q_fn(cfgs)
                active = sites.to(dtype=cfgs.real.dtype).unsqueeze(0)
                coeffs[:, mu] = coeffs[:, mu] + q * active.unsqueeze(-1)
                if not divergence:
                    continue

                for a in range(8):
                    tangent = torch.zeros_like(cfgs)
                    delta_u = 1j * torch.einsum("ij,b...jk->b...ik", generators[a], cfgs[:, mu])
                    tangent[:, mu] = delta_u * active.unsqueeze(-1).unsqueeze(-1)
                    _, dcoeff = torch.func.jvp(q_fn, (cfgs,), (tangent,))
                    div = div + (dcoeff[..., a] * active).flatten(1).sum(1)

        return coeffs, div

    def drift(self, cfgs, t):
        """Return Lie-algebra coordinates with shape ``cfgs.shape[:-2] + (8,)``."""
        return self._field(cfgs, t, divergence=False)[0]

    def forward(self, cfgs, t):
        """Return ``(F, div F)`` for the Haar-measure Lie-algebra field."""
        return self._field(cfgs, t, divergence=True)


class DefectHyperSmearingDrift(nn.Module):
    """The infinitesimal generator of the NEO-REX defect coupling layer.

    The hyper-smearing generator acts on the links the discrete DefectCouplingLayer
    smears in its cropped patch; every other link has zero velocity. rho is evaluated
    with unit delta, so the field is a velocity per unit boundary coupling.
    """

    buffer = 2

    def __init__(self, flow_pars):
        super().__init__()
        if int(flow_pars.N) != 3 or int(flow_pars.D) != 4:
            raise ValueError("DefectHyperSmearingDrift supports SU(3) in D=4")
        if int(flow_pars.rho_shape_type) != 3 or not flow_pars.use_hyper_smearing:
            raise ValueError("DefectHyperSmearingDrift requires rho_shape_type=3 hyper-smearing")
        if not flow_pars.hyper_scale_by_delta or flow_pars.hyper_normalize_by_nstep:
            raise ValueError(
                "a velocity per unit parameter requires hyper_scale_by_delta=True "
                "and hyper_normalize_by_nstep=False"
            )
        if flow_pars.defect is None or not flow_pars.small_mask or not flow_pars.smeared_defect_mask:
            raise ValueError("DefectHyperSmearingDrift requires the defect patch masks")
        self.D = 4
        self.defect = flow_pars.defect
        self.steps = int(flow_pars.smearing_steps_per_layer)
        self.smearing = Smearing(flow_pars, time=1.0)
        # Plain attributes: the link-selection cache is keyed on these exact tensors.
        self.small_mask = flow_pars.small_mask[-1]
        self.smeared_defect_mask = flow_pars.smeared_defect_mask[-1]
        self.generators = _su3_generators(dtype=torch.cdouble, device=torch.device("cpu"))

    def _rho(self, t, cfgs):
        t = torch.as_tensor(t, dtype=torch.float64, device=cfgs.device).reshape(-1)
        hyper = self.smearing.hyper_smearing
        if t.numel() == 1:
            return hyper(beta=t[0], delta_beta=1.0), False
        if t.numel() != cfgs.shape[0]:
            raise ValueError(f"t has {t.numel()} values, expected 1 or {cfgs.shape[0]}")
        return hyper(beta=t, delta_beta=torch.ones_like(t)), True

    def _field(self, cfgs, t, *, divergence):
        if not torch.is_tensor(cfgs) or not torch.is_complex(cfgs) or cfgs.shape[-2:] != (3, 3):
            raise ValueError("cfgs must be a complex SU(3) link tensor ending in (3, 3)")
        if cfgs.dim() != self.D + 4 or cfgs.shape[1] != self.D:
            raise ValueError(f"cfgs must have shape (B, {self.D}, *lattice, 3, 3)")
        rho, batched = self._rho(t, cfgs)
        rho = rho.to(device=cfgs.device, dtype=cfgs.real.dtype)
        links = self.smearing._links_for_smearing_mask(self.small_mask, self.smeared_defect_mask)
        small = self.defect.cut_defect(cfgs, buffer=self.buffer)
        cfg_shape = tuple(small.shape)
        flat = sun.flat_links(small, self.D)
        generators = self.generators.to(device=cfgs.device, dtype=cfgs.dtype)
        coeffs = cfgs.real.new_zeros((*flat.shape[:2], 8))
        div = cfgs.real.new_zeros(cfgs.shape[0])

        for m in range(2 * self.D):
            index = links.links[m]
            if index.numel() == 0:
                continue

            def q_fn(value, m=m, index=index):
                U = value.index_select(1, index)
                q = 0
                # Every substep contributes its instantaneous generator at the same configuration.
                for step in range(self.steps):
                    rho_step = rho[:, step, m] if batched else rho[step, m]
                    C = self.smearing._selected_stout_staples(value, cfg_shape, links, m, rho_step)
                    q = q + generate_Q(generate_omega(C, U), 3)
                return _matrix_to_coordinates(q, generators)

            coeffs = coeffs.index_copy(1, index, q_fn(flat))
            if not divergence:
                continue
            # Staples of a parity class never contain its own links, so perturbing the
            # whole class at once leaves only the diagonal derivatives.
            U = flat.index_select(1, index)
            for a in range(8):
                tangent = torch.zeros_like(flat).index_copy(1, index, 1j * (generators[a] @ U))
                _, dq = torch.func.jvp(q_fn, (flat,), (tangent,))
                div = div + dq[..., a].sum(1)

        coeffs = coeffs.reshape(*cfg_shape[:-2], 8)
        full = cfgs.real.new_zeros((*cfgs.shape[:-2], 8))
        return self.defect.embedding(coeffs, full, buffer=self.buffer), div

    def drift(self, cfgs, t):
        """Return Lie-algebra coordinates with shape ``cfgs.shape[:-2] + (8,)``."""
        return self._field(cfgs, t, divergence=False)[0]

    def forward(self, cfgs, t):
        """Return ``(F, div F)`` for the Haar-measure Lie-algebra field."""
        return self._field(cfgs, t, divergence=True)


def make_existing_drift(architecture, flow_pars, rho_init=1e-3):
    """Build the infinitesimal field associated with an existing SNF architecture."""
    if str(architecture).lower() == "defect-hyper-smearing":
        return DefectHyperSmearingDrift(flow_pars)
    return ExistingDrift(architecture, flow_pars, rho_init=rho_init)
