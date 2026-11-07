"""Hasenbusch mass preconditioning (hep-lat/0107019) for the N_f = 2 even-odd Wilson HMC.

Conventions are those of neoqcd.fermions.pseudofermion: D_hat(kappa) = 1_ee - kappa^2 H_eo H_oe on
even sites, M(kappa) = D_hat^dagger D_hat, refresh fields eta complex Gaussian with
<|eta_i|^2> = 1, forces are algebra gradients (tr(F T) = d/dt S(e^{i t T} U)|_0).

Factorisation. With D_hat = D_hat(kappa) (target, light) and D_hat' = D_hat(kappa') at a heavier
quark, kappa' < kappa (plain heavier-mass form, no twisted mass),
    det(D_hat^dagger D_hat) = det(D_hat'^dagger D_hat') * det(A^dagger A),
    A = D_hat D_hat'^-1,   A^dagger A = D_hat'^-dagger D_hat^dagger D_hat D_hat'^-1,
and each factor is written with its own pseudofermion:

- heavy term (`HeavyWilsonTerm`, i.e. `TwoFlavourWilson` at kappa'):
      S_1 = phi_1^dagger (D_hat'^dagger D_hat')^-1 phi_1,        refresh phi_1 = D_hat'^dagger eta_1;
- ratio term (`RatioWilsonTerm`):
      S_2 = phi_2^dagger (A^dagger A)^-1 phi_2 = phi_2^dagger D_hat' M^-1 D_hat'^dagger phi_2
          = chi^dagger M^-1 chi,   chi = D_hat'^dagger phi_2,   M = D_hat^dagger D_hat,
      refresh phi_2 = A^dagger eta_2 = D_hat'^-dagger D_hat^dagger eta_2
                    = D_hat' (D_hat'^dagger D_hat')^-1 D_hat^dagger eta_2   (one CG on the heavy
      normal operator, at `action_tol`), so that S_2(U) = eta_2^dagger eta_2 at the refresh links.
  Integral d phi_2 exp(-S_2) = pi^n det(A^dagger A) = pi^n det M / det M'.

Forces. For even fields a, b let Re[a^dagger dD_hat b] = -kappa^2 Re[(a + Z a)^dagger dH (b + W b)]
with W b = H_oe b and Z a = (H^dagger)_oe a = gamma_5 H gamma_5 a (odd parts): this is the identity
of the pseudofermion module doc (cross terms vanish because dH changes parity), and H does not
depend on kappa, so the same `hopping_derivative` serves both operators.

- heavy term: as `TwoFlavourWilson` at kappa'.
- ratio term: with X = M^-1 chi and Y = D_hat X,
      dS_2 = 2 Re[phi_2^dagger dD_hat' X] - X^dagger dM X
           = 2 Re[phi_2^dagger dD_hat' X] - 2 Re[Y^dagger dD_hat X]
           = 2 Re[(c + Z c)^dagger dH (X + W X)],     c = kappa^2 Y - kappa'^2 phi_2,
  the first piece from D_hat' outside the inverse, the second from M^-1 inside. One CG on M per
  force evaluation (like the single-term force) and one call of `hopping_derivative`.
  Check: kappa' = 0 gives D_hat' = 1 and the single-term force 2 kappa^2 G(Y + Z Y, X + W X).

Cost. The heavy term is cheap (CG on the well-conditioned M') and carries most of the force; the
ratio term needs the light CG but has a small force, so it can sit on a coarse time scale.
"""

import torch

from neoqcd.fermions.dirac import WilsonDirac
from neoqcd.fermions.hmc import Hamiltonian, WilsonGaugeAction
from neoqcd.fermions.pseudofermion import TwoFlavourWilson, hopping_derivative
from neoqcd.fermions.solver import cg

__all__ = ["HeavyWilsonTerm", "RatioWilsonTerm", "HasenbuschHamiltonian", "LEVEL_LAYOUTS"]


def _bview(k, field):
    """A (B,) or (1,) real coupling as a view broadcasting against a (B, lattice, S, N) field."""
    return k.view((-1,) + (1,) * (field.dim() - 1))


class HeavyWilsonTerm(TwoFlavourWilson):
    """The heavy factor det(D_hat'^dagger D_hat'): `TwoFlavourWilson` at kappa_heavy, even-odd.

    Same arguments, state and stats as `TwoFlavourWilson` (the `kappa` attribute is kappa').
    """

    def __init__(self, kappa_heavy, geometry, **kwargs):
        kwargs.setdefault("even_odd", True)
        super().__init__(kappa_heavy, geometry, **kwargs)


class RatioWilsonTerm(TwoFlavourWilson):
    """The ratio factor det(D_hat^dagger D_hat) / det(D_hat'^dagger D_hat') (module doc), even-odd only.

    Args:
        kappa: target hopping parameter (float or (B,) tensor).
        kappa_heavy: heavier-quark hopping parameter kappa' (float or (B,) tensor), kappa' < kappa.
        geometry, time_sign, solver_tol, action_tol, max_iter, reuse_guess: as `TwoFlavourWilson`
            (the light solves of `action` and `force`); the heavy solve of `refresh` runs at
            `action_tol`, so that S_2(U) = eta^dagger eta to that accuracy (the cached old action).

    State: `phi` (phi_2), `X` (last solution of M X = chi, the chronological guess), and the
    refresh cache as in `TwoFlavourWilson`. `stats` as there; `solves` and `cg_iterations` also
    count the heavy refresh solve, which is reported separately in `refresh_solves`,
    `refresh_cg_iterations`.
    """

    def __init__(self, kappa, kappa_heavy, geometry, **kwargs):
        if kwargs.pop("even_odd", True) is not True:
            raise ValueError("RatioWilsonTerm is implemented for the even-odd operator only")
        self.kappa_heavy = kappa_heavy
        super().__init__(kappa, geometry, even_odd=True, **kwargs)

    def reset_stats(self):
        super().reset_stats()
        self.stats["refresh_solves"] = 0
        self.stats["refresh_cg_iterations"] = 0

    # ---- operators -----------------------------------------------------------------------
    def heavy_operator(self, U):
        """D_hat' = D_hat(kappa') at links U."""
        return WilsonDirac(U, self.kappa_heavy, self.geometry, self.time_sign).even_odd()

    def _require_phi(self):
        if self.phi is None:
            raise RuntimeError("the pseudofermion field is not set: call refresh(U, generator) first")
        return self.phi

    def _chi(self, U):
        return self.heavy_operator(U).apply_dagger(self._require_phi())

    def _solve_light(self, op, chi, tol):
        x0 = self.X if (self.reuse_guess and self.X is not None) else None
        res = cg(op.normal, chi, x0=x0, tol=tol, max_iter=self.max_iter)
        self._count(res)
        self.X = res.x
        return res.x

    # ---- ActionTerm interface -------------------------------------------------------------
    def refresh(self, U, generator=None):
        """phi_2 = D_hat' (D_hat'^dagger D_hat')^-1 D_hat^dagger eta (module doc)."""
        light = self.operator(U)
        heavy = self.heavy_operator(U)
        eta = torch.randn(self.geometry.wilson_shape(U.shape[0]), generator=generator,
                          dtype=U.dtype, device=U.device)
        eta = light.even(eta)
        res = cg(heavy.normal, light.apply_dagger(eta), tol=self.action_tol, max_iter=self.max_iter)
        self._count(res)
        self.stats["refresh_solves"] += 1
        self.stats["refresh_cg_iterations"] += int(res.iterations.max().item())
        self.phi = heavy.apply(res.x)
        self.X = None
        self._refresh_links = U
        self._refresh_version = U._version
        self._refresh_action = (eta.conj() * eta).real.reshape(eta.shape[0], -1).sum(1)
        self.stats["refresh_calls"] += 1

    def action(self, U):
        """S_2 = chi^dagger M^-1 chi, chi = D_hat'^dagger phi_2, shape (B,); one light CG at `action_tol`."""
        self.stats["action_calls"] += 1
        chi = self._chi(U)
        X = self._solve_light(self.operator(U), chi, self.action_tol)
        return (chi.conj() * X).real.reshape(X.shape[0], -1).sum(1)

    # action_difference is inherited: the refresh cache holds eta^dagger eta = S_2(U_refresh).

    def force(self, U):
        """Algebra gradient of S_2: 2 G(c + Z c, X + W X), c = kappa^2 Y - kappa'^2 phi_2 (module doc)."""
        self.stats["force_calls"] += 1
        light = self.operator(U)
        heavy = self.heavy_operator(U)
        chi = heavy.apply_dagger(self._require_phi())
        X = self._solve_light(light, chi, self.solver_tol)
        w = light.wilson
        Y = light.apply(X)
        c = _bview(light.kappa ** 2, Y) * Y - _bview(heavy.kappa ** 2, self.phi) * self.phi
        Zc = light.odd(w.gamma5(w.hopping(w.gamma5(c))))  # (H^dagger)_oe c
        WX = light.odd(w.hopping(X))  # H_oe X
        return 2.0 * hopping_derivative(w, c + Zc, X + WX)


# level layouts of HasenbuschHamiltonian: names of the terms on each level, outermost first
LEVEL_LAYOUTS = {
    "nested": (("ratio",), ("heavy",), ("gauge",)),
    "shared": (("ratio", "heavy"), ("gauge",)),
}


class HasenbuschHamiltonian(Hamiltonian):
    """N_f = 2 Wilson with Hasenbusch mass preconditioning plus the Wilson gauge action.

    Terms: `ratio` (RatioWilsonTerm(kappa, kappa_heavy)), `heavy` (HeavyWilsonTerm(kappa_heavy)),
    `gauge` (WilsonGaugeAction(beta)). `layout` fixes the time scales:
        "nested" (default): [ratio] (n_ratio_substeps) > [heavy] (n_heavy_substeps) > [gauge]
                            (n_gauge_substeps), three levels;
        "shared":           [ratio, heavy] (n_ratio_substeps) > [gauge] (n_gauge_substeps);
    or an explicit sequence of (names, n_substeps), e.g. ((("ratio",), 1), (("heavy", "gauge"), 3)).
    n_ratio_substeps sits on the outermost level, so it multiplies the HMC n_steps.

    `fermion_kwargs` go to both fermion terms (solver_tol, action_tol, max_iter, reuse_guess,
    time_sign); `heavy_kwargs` override them for the heavy term only.
    """

    def __init__(self, beta, kappa, kappa_heavy, geometry, n_ratio_substeps=1, n_heavy_substeps=1,
                 n_gauge_substeps=1, layout="nested", plaquette_weights=None, heavy_kwargs=None,
                 **fermion_kwargs):
        self.ratio = RatioWilsonTerm(kappa, kappa_heavy, geometry, **fermion_kwargs)
        hk = dict(fermion_kwargs)
        hk.update(heavy_kwargs or {})
        self.heavy = HeavyWilsonTerm(kappa_heavy, geometry, **hk)
        self.gauge = WilsonGaugeAction(beta, geometry, plaquette_weights)
        terms = {"ratio": self.ratio, "heavy": self.heavy, "gauge": self.gauge}
        if isinstance(layout, str):
            if layout not in LEVEL_LAYOUTS:
                raise ValueError(f"unknown layout {layout!r}: use one of {tuple(LEVEL_LAYOUTS)} or explicit levels")
            names = LEVEL_LAYOUTS[layout]
            subs = (n_ratio_substeps, n_heavy_substeps, n_gauge_substeps) if layout == "nested" else (
                n_ratio_substeps, n_gauge_substeps)
            spec = list(zip(names, subs))
        else:
            spec = [(tuple(n), s) for n, s in layout]
        used = [n for names, _ in spec for n in names]
        if sorted(used) != sorted(terms):
            raise ValueError(f"every term must appear exactly once in the layout, got {used}")
        self.layout = spec
        super().__init__([([terms[n] for n in names], s) for names, s in spec])
