"""Fast path of the DD-HMC (`ddhmc.py`, Luescher hep-lat/0409106): the pseudofermion terms through
the half-lattice operator of `fast.py` (and the fused hop of `fused.py`).

The reference classes of `ddhmc.py` stay the ground truth; test/test_fermions_ddhmc_fast.py checks
every class here against them (and against dense matrices) to the solver tolerance. Conventions
and formulas are those of `ddhmc.py` (module doc) and `fast.py`.

Block term
----------
S_bl = phi^dag (D_bl_hat^dag D_bl_hat)^-1 phi is `FastTwoFlavourWilson` with the operator built on
the block links: `HalfLatticeWilson(U * inblock_mask)` (the cut links set to zero, so both hops
across block faces vanish and D_bl = D_Omega (+) D_Omega*). The global parity is the parity of the
blocks' own even-odd decomposition because the block extents are even. One global CG covers all
blocks (no block-local solver). The operator is cached on the UNMASKED links (identity and
`_version`, as `FastTwoFlavourWilson`), so the masked copy is made once per configuration; the force
is the parent's half-lattice force (`HalfLatticeWilson.hopping_derivative` with the masked links,
zero on the cut links) times the active-link mask. Same heat-bath draws as `BlockTwoFlavour`, same
stats, policy (`on_unconverged`), precisions ("double": lean CG; "mixed": complex64 inner CG) and,
in `FusedBlockTwoFlavour`, the fused hop kernels ("cuda", "cpu", "loop", "auto"): the kernel reads
full links, so the zeroed cut links need no kernel change.

`FastDDHMCHamiltonian` is `DDHMCHamiltonian` with these terms.
"""

from neoqcd.fermions.ddhmc import DDHMCHamiltonian, SchurTwoFlavour
from neoqcd.fermions.fast import FastTwoFlavourWilson, HalfLatticeWilson
from neoqcd.fermions.fused import DEFAULT_LINK_LAYOUT, FusedHalfLatticeWilson, FusedTwoFlavourWilson

__all__ = ["FastBlockTwoFlavour", "FusedBlockTwoFlavour", "FastDDHMCHamiltonian", "FAST_OPTIONS"]

# options of the fast terms (FastTwoFlavourWilson) shared by both DD terms
FAST_OPTIONS = ("precision", "check_every", "inner_tol", "max_outer", "colour_product")


class _BlockOperator:
    """`operator(U)`: the half-lattice operator of the block links U * inblock_mask, cached on the
    unmasked U (identity, `_version`) and the operator key; `force` masked to the active links."""

    def operator(self, U):
        c = self._op_cache
        key = self._operator_key()
        if c is not None and c[0] is U and c[1] == U._version and self._same_key(c[2], key):
            return c[3]
        op = self._half_lattice(U * self.blocks.link_mask(self.blocks.inblock_links, U))
        op._low = None
        self._op_cache = (U, U._version, key, op)
        return op

    def force(self, U):
        return self.blocks.link_mask(self.blocks.active_links, U) * super().force(U)


class FastBlockTwoFlavour(_BlockOperator, FastTwoFlavourWilson):
    """`ddhmc.BlockTwoFlavour` through `HalfLatticeWilson` on the block links (module doc). Extra
    keyword arguments are those of `FastTwoFlavourWilson` (precision, check_every, inner_tol,
    max_outer, colour_product, solver_tol, action_tol, max_iter, reuse_guess, on_unconverged)."""

    def __init__(self, kappa, blocks, time_sign=-1.0, **kwargs):
        self.blocks = blocks
        super().__init__(kappa, blocks.geometry, time_sign=time_sign, **kwargs)

    def _half_lattice(self, links):
        return HalfLatticeWilson(links, self.kappa, self.geometry, self.time_sign, colour_product=self.colour_product)


class FusedBlockTwoFlavour(_BlockOperator, FusedTwoFlavourWilson):
    """`FastBlockTwoFlavour` with `FusedHalfLatticeWilson` (kernel, link_layout, strict as
    `FusedTwoFlavourWilson`)."""

    def __init__(self, kappa, blocks, time_sign=-1.0, **kwargs):
        self.blocks = blocks
        super().__init__(kappa, blocks.geometry, time_sign=time_sign, **kwargs)

    def _half_lattice(self, links):
        return FusedHalfLatticeWilson(links, self.kappa, self.geometry, self.time_sign,
                                      colour_product=self.colour_product, kernel=self.kernel,
                                      link_layout=self.link_layout, strict=self.strict)


class FastDDHMCHamiltonian(DDHMCHamiltonian):
    """`DDHMCHamiltonian` with the fast block term (levels, masks and translations unchanged).

    `fast_kwargs` (keys in FAST_OPTIONS) go to the block term; `kernel` ("off" = the --fast
    operator, else a kernel of `fused.KERNELS`), `link_layout` and `strict` select the fused hop.
    `block_kwargs` / `schur_kwargs` as in `DDHMCHamiltonian` (the Schur term is the reference one).
    """

    def __init__(self, beta, kappa, blocks, n_block_substeps=1, n_gauge_substeps=1, time_sign=-1.0,
                 block_kwargs=None, schur_kwargs=None, fast_kwargs=None, kernel="off",
                 link_layout=DEFAULT_LINK_LAYOUT, strict=False):
        fast_kwargs = dict(fast_kwargs or {})
        unknown = set(fast_kwargs) - set(FAST_OPTIONS)
        if unknown:
            raise ValueError(f"unknown fast options {sorted(unknown)}; expected a subset of {FAST_OPTIONS}")
        self.fast_kwargs = fast_kwargs
        self.kernel = kernel
        self.kernel_kwargs = {} if kernel == "off" else {"kernel": kernel, "link_layout": link_layout,
                                                         "strict": bool(strict)}
        super().__init__(beta, kappa, blocks, n_block_substeps=n_block_substeps, n_gauge_substeps=n_gauge_substeps,
                         time_sign=time_sign, block_kwargs=block_kwargs, schur_kwargs=schur_kwargs)

    def _make_block(self, kappa, blocks, time_sign, kwargs):
        kwargs = dict(self.fast_kwargs, **kwargs)
        if self.kernel == "off":
            return FastBlockTwoFlavour(kappa, blocks, time_sign=time_sign, **kwargs)
        return FusedBlockTwoFlavour(kappa, blocks, time_sign=time_sign, **self.kernel_kwargs, **kwargs)

    def _make_schur(self, kappa, blocks, time_sign, kwargs):
        return SchurTwoFlavour(kappa, blocks, time_sign=time_sign, **kwargs)
