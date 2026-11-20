# Dynamical fermions in neoqcd (`src/neoqcd/fermions/`)

Branch `fermions`, commits `237383c` .. `bcc4757` (the plan, `fermions_plan.md`, is `3c1b17c`).
This page describes the code as it is. The plan holds the motivation and the literature.

## 1. Scope and status

**What exists.**

- Wilson-Dirac operator, N_f = 2 (det D†D, no rooting, c_sw = 0), in D = 2 and D = 4, SU(N).
  Even-odd (Schur) preconditioning, dense-matrix tools, free-field spectrum.
- Staggered (Kogut-Susskind) operator with η phases, even-odd normal operator, free spectrum.
  One even-site pseudofermion gives N_f = 2^{D/2} (4 in 4D, 2 in 2D), with no rooting.
- Batched Krylov solvers (CG, BiCGstab, CGNE) with per-sample convergence.
- Pseudofermion actions, heat-bath refresh and analytic MD forces for Wilson and staggered.
- HMC with su(N) momenta, nested multi-time-scale leapfrog/Omelyan integrators (fermion force
  outside, gauge force inside), and the pure-gauge Wilson action with optional defect weights.
- Exact dense-determinant tools and exact-determinant Metropolis samplers (Wilson, staggered)
  for tiny lattices.
- Hasenbusch mass preconditioning for N_f = 2 Wilson (`hasenbusch.py`): heavy and ratio terms,
  three-level Hamiltonian. On 2D 4×4 it saves no cost; the gain is to be measured on larger lattices.
- Clover term for Wilson (`clover.py`, openQCD convention by default): clover pseudofermion,
  small determinant det A_oo, clover Hamiltonian, exact-determinant Metropolis.
- RHMC for rooted staggered fermions (`rhmc.py`, `rational.py`): multishift CG, Zolotarev and
  Remez rational approximations, N_f = 2 in 4D, also at light masses.
- An opt-in fast path for N_f = 2 Wilson (`fast.py`, `--fast` in `main_hmc.py`, the SNF and both
  REX drivers; `QCDAction(..., fast=True)`): half-lattice even-odd operator, lean and
  mixed-precision CG (§5b).
  (Hasenbusch, clover and RHMC are implemented and reviewed; status in `fermions_plan.md`,
  "Aggiornamento del 13/14 novembre".)
- `QCDAction`: one parametric joint action S(U, φ; λ) with λ ∈ {β, κ, bc} (Wilson) or
  {β, mass, bc} (`fermions="staggered"`). The defect `bc` enters the gauge action only
  (Bonanno et al., arXiv:2404.14151).
- Fermionic NE-MCMC (Jarzynski on the joint (U, φ)) for protocols in β, κ/mass or bc, Wilson or staggered.
- Fermionic SNF: pure-gauge stout/residual layers (trainable) before every HMC block. The S_pf
  gradient comes from a custom autograd Function, with no backpropagation through CG. Wilson
  only for now; staggered SNF with layers is in progress in `snf_layers.py`.
- REX on the gauge-only defect: fermionic HMC replicas and solve-free swaps. There is a
  single-process, device-safe `FermionREX` and a `distributedPT.ParallelTempering` target (tested
  on Gloo). Wilson and staggered.
- Drivers `main/main_hmc.py`, `main/main_snf_fermions.py` and `main/main_rex_fermions.py`, all
  with checkpoint/resume, plus three Leonardo debug jobs.

**Verified only on tiny lattices.** The exact checks run on CPU in complex128 on 2D 4×4 (SU(2),
SU(3)), 2^3×4, 2^4 and 4^4 (the last only for operator-level dense checks). These checks cover
Jarzynski against exact reweighting, HMC against the exact-determinant Metropolis, and REX
against plain HMC. On an A100 the N_f = 2 Wilson HMC (c_sw = 0) is validated against openQCD 2.4.3
at 4^4, β = 1.0, κ = 0.1 and at 8^4, β = 5.6, κ = 0.153 (§5b).

**Not done.**

- DD-HMC in the drivers and the fast path (the algorithm itself is in `ddhmc.py`, §5c).
- Staggered SNF with layers (in progress).
- O-REX and NEO-REX with fermions (out of this plan).
- The 8^4, β = 5.6, κ = 0.1545 point (openQCD reference 0.5687(3) available) and the
  comparison with Grid. The 4^4 and 8^4 κ = 0.153 openQCD comparisons are done (§5b).
- CUDA: `main_hmc.py` (reference and `--fast`) and the SNF/REX drivers (reference) have run on an
  A100; the fast path outside `main_hmc.py` (QCDAction/NE-MCMC/SNF/REX, `--fast` in the SNF and REX
  drivers) is tested on the CPU only.
- A line-of-constant-physics protocol. The β protocols at fixed κ are algorithmic only, since
  the quark mass changes. The β interval (5.45 → 5.6) is open until t0 is measured.

## 2. Conventions

**Links and shifts.**

- `U` has shape (B, D, T, L, ..., L, N, N), complex128. Direction 0 is time, and lattice axis
  μ of a field is tensor axis 1 + μ (`lattice.axis`).
- "x + μ" is `torch.roll(f, +1, axis_μ)` (`lattice.shift(f, mu, 1)`), so in array indices
  f(x + μ) at site n is f[n − 1]. The code's forward direction runs towards decreasing array
  index.
- `U[:, mu][n]` is the link from n to n + μ̂ (array n − 1). This is the reading under which the
  `sun_utils` plaquette is gauge invariant under U_μ(x) → Ω(x) U_μ(x) Ω(x+μ)†
  (`test_fermions_lattice` checks both readings).

**Field shapes.** A Wilson field is (B, T, L, ..., L, S, N) with S = 2^{D/2}. A staggered field is
(B, T, L, ..., L, N). Dense matrices flatten in C order (site, spin, colour), with
index = (site·S + s)·N + c and sites lexicographic in (t, x_1, ...). Even-odd restrictions keep
the selected sites in lexicographic order.

**Wilson operator** (Hasenbusch/Lüscher normalisation, `dirac.py`): D = 1 − κH, κ = 1/(2m_0 + 2D).

    (Hψ)(x) = Σ_μ [ (1 − γ_μ) U_μ(x) ψ(x+μ) + (1 + γ_μ) U_μ(x−μ)† ψ(x−μ) ]

The forward hop uses projector (1 − γ_μ) and the link at x, not daggered. The backward hop uses
(1 + γ_μ) and the dagger of the link at x − μ (array n + 1). In code the colour product is
`psi @ U^T` for U ψ and `psi @ conj(U)` for U† ψ. D† = γ5 D γ5.

**Gamma matrices** (Euclidean, hermitian, index 0 = time; `gamma_matrices`):

- D = 4, chiral representation of openQCD:
  γ_0 = [[0, −1], [−1, 0]] and γ_k = [[0, −iσ_k], [iσ_k, 0]] (2×2 blocks).
  γ5 = γ_0γ_1γ_2γ_3 = diag(1, 1, −1, −1).
- D = 2: γ_0 = σ_1, γ_1 = σ_2 and γ5 = −iγ_0γ_1 = σ_3.

**Momentum space.** Plane waves are ψ(n) = e^{−ip·n} u with n the array index, so
ψ(x+μ) = e^{+ip_μ} ψ(x) and

    D(p) = 1 − κ Σ_μ [(1 − γ_μ) e^{ip_μ} + (1 + γ_μ) e^{−ip_μ}]
         = 1 − 2κ Σ_μ cos p_μ + 2iκ Σ_μ γ_μ sin p_μ,

with p_k = 2πn_k/L and p_0 = (2n_0 + 1)π/T (antiperiodic) or 2πn_0/T (periodic).

**Boundary conditions.** Space is periodic. In time, `time_sign` (−1 antiperiodic, the default,
or +1) multiplies the time links at array index t = 0 (`fermion_boundary_links`). These links
cross the boundary in both hops: the forward hop at t = 0 reads ψ(T−1), and the backward hop at
t = T−1 reads ψ(0) through U_0(t=0)†. The caller's gauge field is never modified.

**Even-odd (Wilson).** In (even, odd) block order D = [[1, −κH_eo], [−κH_oe, 1]], and
D̂ = 1_ee − κ² H_eo H_oe with det D = det D̂. D x = b is solved as D̂ x_e = b_e + κH_eo b_o
(`prepare_rhs`) and x_o = b_o + κH_oe x_e (`reconstruct`). Even-odd fields are full-lattice
tensors that vanish on odd sites. All extents must be even.

**Staggered** (`staggered.py`):

- Phases: η_μ(n) = (−1)^{n_0 + … + n_{μ−1}} in array coordinates (η_0 = 1).
- Operator: D = m + K, with (Kχ)(x) = ½ Σ_μ η_μ(x) [U_μ(x) χ(x+μ) − U_μ(x−μ)† χ(x−μ)].
- K is anti-hermitian and parity-changing, so D†D = m² − K².
- Even-odd: M_ee = m² − K_eo K_oe, hermitian positive definite for m > 0.
  det D = det M_ee and det D†D = (det M_ee)².
- Free spectrum: D†D = m² + Σ sin² p_μ on plane waves.
- Boundary conditions as for Wilson.

**Pseudofermions.**

- Wilson (`pseudofermion.py`):
  - Action: S_pf = φ†(D̂†D̂)^{−1}φ, with φ on even sites (`even_odd=False` gives the full-lattice
    version of the same theory).
  - Normalisation: ∫dφ dφ* e^{−S_pf} = π^n det(D̂†D̂).
  - Refresh: φ = D̂†η, with η complex Gaussian of density ∝ e^{−η†η} (⟨|η_i|²⟩ = 1, `torch.randn`).
    Then S_pf at the refresh links is exactly η†η (cached, no solve) and ⟨S_pf⟩ = n = V/2·S·N.
- Hasenbusch (`hasenbusch.py`, hep-lat/0107019): det D̂†D̂ = det D̂'†D̂' · det A†A with
  D̂' = D̂(κ'), κ' < κ, A = D̂D̂'^{−1}. Heavy term S_1 = φ_1†(D̂'†D̂')^{−1}φ_1; ratio term
  S_2 = χ†M^{−1}χ with χ = D̂'†φ_2, refreshed so that S_2 = η_2†η_2. Formulas and forces in the
  module docstring.
- Staggered (`staggered_pseudofermion.py`):
  - Action: S_pf = φ†M_ee^{−1}φ.
  - Refresh: φ = (D†χ)_e = mχ_e − K_eo χ_o, with χ Gaussian on all sites.
  - S_pf at the refresh links is not χ†χ (D† restricted to even rows is rectangular). It costs
    one solve, cached on first use. In `ParametricStaggeredPseudofermion` the cache is keyed on
    the links' identity and `_version`, so an in-place change of U invalidates it.

**Force convention** (`hmc.py`, `algebra.py`):

- Algebra elements are hermitian traceless X, the group element is exp(iX), and the inner
  product is tr(XY).
- Basis: T^a with tr(T^a T^b) = δ_ab (Gell-Mann/√2, Pauli/√2).
- The force of a term is the hermitian traceless F with tr(F T) = d/dt S(e^{itT}U)|_{t=0} for
  every T (per link).
- Equations of motion: dU/dt = iPU and dP/dt = −F.
- Position step: U ← exp(ihP)U (closed form for SU(2), Cayley-Hamilton for SU(3)), reunitarised
  once at the end of the trajectory.
- Momenta: P = Σ p_a T^a with p_a ~ N(0, 1), so the density is ∝ exp(−½ Σ tr P²) and
  ⟨tr P²⟩ = N² − 1 per link. H = ½ Σ_links tr P² + Σ_terms S.
- Gauge action: S_g = β Σ_x Σ_{μ<ν} [1 − w_μν(x) Re tr U_μν/N] (identical to
  `theory.Wilson_action`). Defect weights: w is the product of c = mask[0] + mask[1]·bc over the
  four links of the plaquette.

**Pseudofermion force.** Let X = M^{−1}φ, Y = D̂X, W = H_oe X and Z = (H†)_oe Y. Then

    F = 2κ² ∇Re[(Y+Z)† H (X+W)]

where ∇ is `hopping_derivative`, the projection onto hermitian traceless of iQ, with
Q = L O1 − O2 L†. Staggered: F = 2∇Re[X†K W] with W = K X.

**Wirtinger gradient for autograd** (`snf_layers.PseudofermionAction`):

- PyTorch stores G = ∂L/∂Re z + i∂L/∂Im z, so dL = Re tr(G†dU).
- Backward returns G = i F U (times grad_out). For dU = iTU dt this gives Re tr(G†dU) = tr(FT) dt.
- G is the tangent part of the gradient. The normal part is irrelevant because the layers map
  exactly into SU(N).
- Inverse: F = P_ht(−i G U†).
- Forward is one CG solve. F is computed from the same solution and saved, so later changes of
  φ or κ do not affect the backward.

**NE-MCMC bookkeeping** (`snf.py`, state x = (U, φ)). For step i:

    jump  dW_i = S(x_{i−1}; λ_i) − S(x_{i−1}; λ_{i−1})            (fixed U and φ)
    layer dW_i^L = S(U'_i, φ_{i−1}; λ_i) − S(x_{i−1}; λ_i) − log J_i  (SNF only, φ fixed)
    block dQ_i = S(x_i; λ_i) − S(x'_i; λ_i)                        (HMC block, heat)
    W + Q = S(x_n; λ_n) − S(x_0; λ_0) − log J

The pseudofermion refresh at the start of every HMC trajectory is a Gibbs step on φ inside the
block, so its change of S counts as heat, not work. Jarzynski,
⟨e^{−W}⟩ = Z(λ_n)/Z(λ_0), holds on the joint (U, φ). Z(λ) is the gauge-fermion partition function
up to a λ-independent π^n.

Jump costs: β is Δβ·S_g(U; β = 1) and bc is gauge only, both without a solve; κ (or the
staggered mass) takes one solve
(S at the old κ is known from the end of the previous block). A zero jump is exactly zero. If x_0 has no φ, φ is drawn by a heat bath at λ_0. The SNF loss is
mean(W). The HMC blocks are under no_grad, so the gradient flows only through the layers.

**REX swap** (`rex.py`). Rungs i < j hold U_a (at bc_i) and U_b (at bc_j). The determinant does
not see bc and cancels configuration by configuration, so

    ΔS = [S_g(U_a; bc_j) − S_g(U_a; bc_i)] + [S_g(U_b; bc_i) − S_g(U_b; bc_j)],   accept with min(1, e^{−ΔS})

This needs no solve and no φ. `action_table` returns the relative table S(U_b; bc_k) − S(U_b; bc_b),
which gives the same swaps as the absolute table. Labels move, configurations do not. The
ladder increases, with rung 0 the most open and the top rung bc = 1. An optional periodic
translation of the bc = 1 replica is exact for antiperiodic fermions as well; it requires a
periodic top rung (guarded). The swap uniforms come from a separate CPU generator
(`swap_generator` or `swap_seed`), so REX works with the fields on any device.

## 3. Module map (`src/neoqcd/fermions/`)

| Module | Public API |
|---|---|
| `_gauge.py` | the only import point from the pure-gauge package (sun_utils, theory, flow, smearing, utils, replica_history) |
| `lattice.py` | `Geometry`, `axis`, `shift`, `parity`, `even_odd_masks`, `fermion_boundary_links`, `gauge_transform_links/field` |
| `algebra.py` | `algebra_basis`, `algebra_components`, `random_algebra`, `trace_product`, `project_hermitian_traceless`, `expi`, `unitarize` |
| `dirac.py` | `gamma_matrices`, `WilsonDirac`, `EvenOddWilsonDirac`, `dense_matrix`, `free_momenta`, `free_wilson_matrix`, `free_wilson_eigenvalues` |
| `solver.py` | `LinearOperator`, `SolverResult`, `cg`, `bicgstab`, `solve` (cgne/bicgstab, even-odd aware), `solve_normal` |
| `staggered.py` | `staggered_phases`, `staggered_epsilon`, `StaggeredDirac`, `StaggeredEvenOdd`, `solve_staggered`, `staggered_dense_matrix`, `free_staggered_(normal_)eigenvalues` |
| `hmc.py` | `ActionTerm`, `WilsonGaugeAction`, `Hamiltonian`, `WilsonGaugeHamiltonian`, `HMC`, `random_momenta`, `kinetic_energy`, `(batched_)defect_plaquette_weights`, `INTEGRATORS`, `OMELYAN_LAMBDA` |
| `pseudofermion.py` | `hopping_derivative`, `TwoFlavourWilson` (action/refresh/force/stats, `on_unconverged`), `TwoFlavourHamiltonian`, `convergence_limit`, `UnconvergedSolveWarning` |
| `fast.py` | `HalfLatticeWilson`, `lean_cg`, `mixed_precision_cg`, `FastTwoFlavourWilson`, `FastTwoFlavourHamiltonian`, `to_half`/`to_full`, `site_tables`, `colour_mul` (§5b) |
| `clover.py` | `CloverWilsonDirac`, `EvenOddCloverDirac`, `TwoFlavourClover`, `CloverDeterminantTerm`, `CloverHamiltonian`, `ExactCloverMetropolis` |
| `rhmc.py`, `rational.py` | `RootedStaggeredPseudofermion`, `RootedStaggeredHamiltonian`, `ExactRootedMetropolis`; rational approximations |
| `hasenbusch.py` | `HeavyWilsonTerm`, `RatioWilsonTerm`, `HasenbuschHamiltonian` (layouts `nested`, `shared` or explicit levels), `LEVEL_LAYOUTS` |
| `staggered_pseudofermion.py` | `staggered_hopping_derivative`, `StaggeredPseudofermion`, `StaggeredHamiltonian` |
| `exact.py` | `log_det_two_flavour`, `assemble_hopping`, `assemble_even_odd`, `ExactMetropolis` |
| `staggered_exact.py` | `assemble_staggered_hopping`, `assemble_staggered_even_odd`, `log_det_staggered`, `ExactStaggeredMetropolis` |
| `action.py` | `QCDAction` (`fermions="wilson"|"staggered"`, `mass`), `ParametricTwoFlavourWilson`, `ParametricStaggeredPseudofermion`, `JointHamiltonian`, `bc_link_couplings`, `bc_plaquette_weights`, `PARAMETER_NAMES`, `STAGGERED_PARAMETER_NAMES`, `FERMION_TYPES` |
| `snf.py` | `FieldState`, `HMCBlock`, `FermionNEMCMC`, `jarzynski_free_energy`, `effective_sample_size`, `OBSERVABLES` |
| `snf_layers.py` | `PseudofermionAction`, `pseudofermion_action`, `joint_action`, `FermionLayerBlock`, `FermionSNF`, `make_gauge_layers` (default ρ = 1e-3) |
| `rex.py` | `swap_action_difference`, `FermionREXTarget`, `distributed_rex_step`, `FermionREX` (`swap_generator`, `swap_seed`) |
| `io.py` | `save_state`/`load_state` (`.pt` + `.json` sidecar), `restore_generator`, `ObservableWriter`, `read_observables`, `write_npz`, `atomic_json_save`, `n_flavours` |

## 4. How to test

Unit tests run on the Mac (CPU, complex128, fixed seeds) in the project environment:

```sh
# whole fermion suite
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .sunenv/bin/python -m unittest discover -s test -t . -p 'test_fermions_*.py'
# one file, with the numbers printed, without the Gloo test
NEOQCD_TEST_VERBOSE=1 NEOQCD_SKIP_GLOO=1 OMP_NUM_THREADS=1 .sunenv/bin/python -m unittest test.test_fermions_rex -v
```

- `NEOQCD_TEST_VERBOSE=1` prints the diagnostics (sigmas, acceptances, CG counts) of the
  statistical tests. It is read by `hmc`, `hmc_fermions`, `staggered_hmc` and `rex`.
- `NEOQCD_SKIP_GLOO=1` skips the three-rank Gloo test in `test_fermions_rex`.

Runtimes are wall times per file on the Mac (one thread), measured on 2026-11-07 before the
Hasenbusch and staggered-action files (their times not yet measured), all passing; the whole suite takes about 5 min, dominated by nemcmc, staggered_hmc, hmc_fermions and hmc.

| File | What it pins | Time |
|---|---|---|
| `lattice` | roll convention (shift reads n−1), plaquette invariant only with the code reading, boundary links flip t = 0 time links only and do not alias U, parity masks | 1s |
| `dirac` | Clifford algebra, γ5 = γ0γ1γ2γ3, literal gamma matrices, gauge covariance, γ5-hermiticity, sparse = dense (2^3×4, 4^4), flattening order, free spectrum, plane-wave momentum sign, batch independence, Schur complement and det, even-odd solve | 5s |
| `solver` | CG/BiCGstab residuals, per-sample κ convergence, b = 0 and tol = 0, conj-view rhs, breakdown, even-odd solve and fewer iterations, solve_normal | 2s |
| `free_correlators` | free pion correlator from the solver against the momentum sum, quark pole mass log(1 + m0), log det against analytic eigenvalues | 3s |
| `staggered` | phases and their algebra, gauge covariance, K anti-hermitian, sparse = dense, free spectrum of D and D†D (detects wrong phases), parity block structure and determinants, CG with per-sample mass | 5s |
| `hmc` | su(N) basis, expi against matrix_exp (gradient at 0), unitarize, momentum normalisation, batched defect weights, gauge action = `Wilson_action`, force against finite differences, reversibility (per level), ΔH ∝ ε², SU(N) preserved, ⟨P⟩ against the HBOR heat bath (SU(2), SU(3)) | 37s |
| `pseudofermion` | dense assembly = sparse, refresh distribution, ⟨e^{−ΔS_pf}⟩ = exact det ratio (4D, 2D; pins N_f = 2), force against finite differences, hopping-derivative bilinear, ⟨force⟩ = −∇log det (pins the normalisation) | 21s |
| `hmc_fermions` | reversibility, ΔH ∝ ε², SU(N) and solver bookkeeping, HMC against `ExactMetropolis` on 2D 4×4 at κ = 0.12 and 0.22 (plaquette, log det, ⟨e^{−ΔH}⟩) | 44s |
| `staggered_hmc` | the same chain for staggered: assembly/log det, refresh, det ratios, force (finite differences, autograd, expectation), reversibility, ΔH ∝ ε², HMC against `ExactStaggeredMetropolis` (m = 0.3, 0.1) | 49s |
| `hasenbusch` | heavy/ratio refresh gives η†η, det ratio against the exact dense det (2D), forces against finite differences and autograd, reversibility, ΔH ∝ ε², HMC against single-term HMC (plaquette, ⟨e^{−ΔH}⟩, force magnitudes, cost at equal acceptance) | – |
| `action` | S after refresh = η†η, β/κ/bc jumps against `Wilson_action`, `DefectMCMCAdapter` and dense det, Hamiltonian parameters | 2s |
| `action_staggered` | parameter names, S after refresh, β/mass/bc jumps, refresh cache after a mass change, Hamiltonian parameters, Jarzynski in mass against exact determinants, β/bc bookkeeping, staggered REX smoke test and shift invariance of the det | – |
| `nemcmc` | Jarzynski in κ (against exact det ratio) and β (against a quenched reference), W + Q bookkeeping, low-acceptance bookkeeping, constant protocol W = 0 | 57s |
| `snf` | autograd Function value and group derivative, gradient through a layer, force saved at forward, refresh-cache version key, layer work, exact Haar log-Jacobian (SU(3), 1e-14), Jarzynski with stout layers, ρ = 0 equals NE-MCMC, 20-step training smoke test | 35s |
| `rex` | local swap ΔS = full action difference, translation keeps the weight, rung plaquette = plain HMC (bc = 0 and 1), history and round trips, three-rung detailed balance, translation needs a periodic top rung, three Gloo ranks with `ParallelTempering` | 30s |
| `main_hmc` | Wilson outputs and bitwise resume, rows past the checkpoint dropped, quenched and staggered runs, argument checks | 3s |
| `main_snf_rex` | NE-MCMC in κ, SNF in bc with training and resume, interrupted SNF resumes identically, residual layers, REX outputs and resume (outputs only, not physics) | 7s |
| `snf_residual` | residual layers: options and conditioning; SU(3) output, dlogJ = dense Haar log-Jacobian (static/hyper × n_pt 0, 1, 2 × mod 2, 4 in 2D, n_pt 1 in 4D 2^4), exact identity at zero step; dS_pf/dθ vs FD and dense solve, nonzero gradient at the identity; bookkeeping (κ, β, bc), identity layers = NE-MCMC, Jarzynski in κ; driver training, resume, architecture check of `--load-flow` | 50s |

## 5. How to run

**Runs never go on the Mac or on the Leonardo login node.** They go to Leonardo jobs: the debug
QOS (`boost_qos_dbg`, 30 min) for tests, pilots and short thermalisations, and production for
anything longer. Sync code via git; submit with `sbatch` from `/leonardo_work/INF26_sft/ecellini/neoqcd`.

**`main/main_hmc.py`** runs B independent HMC chains as one batch.

- Fermions: `--fermions wilson|staggered|none`, with `--kappa` (Wilson) or `--mass` (staggered).
- Lattice and couplings: `--D --T --L --N --beta --time-sign`.
- Integrator: `--tau --n-steps --n-gauge-substeps --integrator omelyan|leapfrog|outer,inner`.
- Solver: `--solver-tol` (force), `--action-tol` (Metropolis), `--max-iter`, `--no-reuse-guess`,
  `--on-unconverged raise|warn|ignore` (Wilson, default raise; convergence policy in §5b).
- Fast path (Wilson, §5b): `--fast`, `--precision double|mixed`, `--colour-product`,
  `--check-every`, `--inner-tol`, `--max-outer` (all rejected without `--fast`; the last two
  need `--precision mixed`).
- Run length: `--batch --start cold|hot --therm --n-traj --meas-every --checkpoint-every`.
- `--exact-logdet` measures the dense log det for dense dimension ≤ `--exact-max-dim`
  (default 3072).
- Control: `--seed --device auto|cpu|cuda --out --resume --overwrite`.
- Outputs in `--out` (default `output/hmc_fermions/<name>/`):
  - `observables.csv`: one row per trajectory × replica, with the plaquette, dH, accepted,
    exp(−dH), log det, solves, CG iterations, timings and `max_residual` (largest true relative
    residual of the trajectory's solves). With `--precision mixed`, `cg_iter` is the real cost:
    complex64 inner applications plus complex128 outer applications (`run_meta.json`
    "cg_iter_definition").
  - `observables.npz`.
  - `run_meta.json`: parameters plus a Gamma-method summary, acceptance, ⟨e^{−ΔH}⟩, seconds per
    trajectory and per solve, and CG iterations per solve.
  - `checkpoint.pt/.json`.
- Resume is bitwise identical on the same device and thread count.

**`main/main_snf_fermions.py`** runs NE-MCMC or SNF on a protocol in one parameter.

- Protocol: `--protocol beta|kappa|bc` with `--initial --final --n-step` (linear) or
  `--protocol-file`.
- Mode: `--mode nemcmc|snf`; for snf, `--layers smearing|smearing_signed|residual|hyper-residual
  --rho-init` (default 1e-3; with `smearing` ρ = 0 cannot train and is refused for training,
  `smearing_signed --rho-init 0` is a trainable identity start) `--train-steps --lr --lr-warmup
  --grad-clip --load-flow`. `param_abs_mean` in training.csv is the mean |parameter| the step ran
  with (before its optimizer step).
- Residual layers (see "Residual layers" below): `--residual-n-pt --residual-mask-mod
  --residual-[no-]include-imag --residual-[no-]quadratic --residual-coeff-init --residual-coeff-max
  --hyper-time-embedding-dim --hyper-hidden-dim --hyper-depth --hyper-activation --layer-sharing
  --residual-condition --residual-delta`; refused with the other layer kinds.
- Solver and fast path: `--solver-tol --action-tol --max-iter --no-reuse-guess --on-unconverged`
  and `--fast --precision --colour-product --check-every --inner-tol --max-outer` (§5b; part of the
  resume key; also in the two REX drivers).
- Sampling: `--batch --eval-samples`, `--n-traj` (HMC trajectories per block), `--n-between`
  (pool trajectories at λ_0 between flow runs).
- Starting pool: `--start-config` (a `main_hmc`/pool checkpoint) and `--therm`.
- Defect: `--defect-shape cube|line --defect-size --defect-time-slice --defect-space-slice`.
- Outputs:
  - `samples.csv`: W, Q, logJ, S0, S_final, solves, ESS per sample.
  - `blocks.csv`: per protocol step.
  - `training.csv`.
  - `run_meta.json`: ΔF = −log⟨e^{−W}⟩ with jackknife error, ESS, solves and CG iterations
    per sample.
  - `flow.pt/.json` and `pool.pt/.json` (resume). `flow.json` records the layer architecture
    (`architecture`: layer kind, residual options except coeff_init, sharing, conditioning modes)
    and the per-step conditioning (c_i, d_i); `--load-flow` refuses a flow whose sidecar records
    another architecture (or has no sidecar). The residual options are also in the resume key and
    in `run_meta.json` (`run.flow_architecture`, `run.layer_conditioning`).

**Residual layers** (`--layers residual|hyper-residual`, `snf_layers.make_gauge_layers(...,
kind, residual={...})`; the pure-gauge `ResidualCouplingLayer`, SU(3) only).

- Kinds, in the pure-gauge spelling of `main_nemcmc_snf.py --nf-architecture`: `residual` has
  static trainable coefficients, one layer per protocol step; `hyper-residual` takes them from a
  hyper-network of the conditioning value, one layer shared by the whole protocol (as the
  pure-gauge `_nf_layer_sharing`; `--layer-sharing per_step|shared` overrides; a per-step
  hyper-network of a constant input is only a reparametrisation of static coefficients).
- Options, with the pure-gauge names, defaults and checks (`snf_layers.residual_options`):
  `n_pt` frozen-link convolutions (default 0), `mask_mod` 2 or 4 (T and L multiples of it when
  n_pt > 0 or mod 4; in D = 2, n_pt = 1 with mod 2 adds no effective dependence and warns),
  imaginary and quadratic trace features (on), `coeff_init` (1e-3), `coeff_max` (tanh bound, 0 =
  off), hyper-network embedding/hidden/depth/activation (8/16/2/silu).
- Conditioning (`snf_layers.layer_conditioning`, `FermionSNF(condition=, delta=)`). The layer
  receives a conditioning value c (the hyper-network input: sin/cos(2^k c) and c) and a step d that
  multiplies its generator (d = 0 is exactly the identity). The pure-gauge convention passes β_i
  and Δβ; for κ (≈ 0.15, Δκ ≈ 1e-3) and bc ∈ [0, 1] that is badly scaled: the Fourier features
  barely change over a κ protocol and the layer is a near-identity (⟨log J⟩ ≈ −2e-3 at the
  default init with Δκ = 0.0075). The default is therefore the same for every protocol parameter:
  `--residual-condition time`, c_i = normalised protocol position s_i/s_n ∈ (0, 1] with
  s_i = Σ_{j≤i} |λ_j − λ_{j−1}| (i/n for a linear protocol); `--residual-delta fraction`,
  d_i = (λ_i − λ_{i−1}) / s_n, the step into λ_i (the jump the layer follows) as a fraction of the
  total variation (1/n for a linear protocol, negative for a decreasing one, exactly 0 for a zero
  step). `parameter` / `step` reproduce the pure-gauge convention (λ_i and the forward difference,
  0 for n = 1). Measured on 2D 4×4, κ 0.18 → 0.21 in 4 steps (d = 1/4): static coefficients
  1e-3 / 0.02 / 0.1 / 0.5 give ⟨log J⟩ ≈ −0.06 / −1.2 / −6 / −37, i.e. the useful range of the
  coefficients is 1e-2-1e-1 per link (intensive: the same per-link scale on larger lattices).
- Identity start and training: `--residual-coeff-init 0` is the exact identity (U′ = U bitwise,
  so the SNF reproduces NE-MCMC) and a regular point: the gradient of the work with respect to
  the linear coefficients (static) or the output layer (hyper) is nonzero there (unit test). The
  convolution weights and the inner hyper layers have zero gradient at that point (they enter at
  second order) and start moving after the first step; there is no dead point like the stout |ρ|.
  Recommended: `--residual-coeff-init 0 --lr 1e-3 --lr-warmup 5` (Adam moves every coefficient by
  ≈ lr per step, against a useful scale of 1e-2-1e-1); for hyper-residual lr ≤ 1e-3 (its output
  coefficients move by several lr per step: every hidden unit contributes). CPU check on 2D 4×4,
  κ 0.18 → 0.21 in 4 steps (NE-MCMC already at ESS 0.77, dissipated work 0.15), 20 Adam steps of
  batch 16 from the identity, paired ⟨W⟩ change on 128 fixed-seed samples: lr = 1e-3 neutral
  (−0.001 ± 0.004 for static and hyper, n_pt 0 or 2); lr = 3e-3 neutral to slightly worse; lr = 1e-2
  worse (static +0.03-0.04 ± 0.02, hyper +0.13-0.20 ± 0.03-0.04, ESS 0.77 → 0.64-0.70). So no gain is visible on
  this toy problem and larger rates overshoot; the gain has to be measured on 4^4 (debug job).

**`main/main_rex_fermions.py`** runs single-process REX.

- Ladder: `--ladder 0,0.5,1` or `--n-rep --bc-min`.
- Run: `--n-chains --steps --therm --swap-parity alternating|even_odd --periodic-translation`.
- Physics, HMC and defect options are those of the SNF driver.
- Outputs:
  - `observables.csv`: per step × rung.
  - `replica_history.npz`: labels, plaquette, trajectory_labels and exp_minus_dH in
    `replica_history` layout.
  - `run_meta.json`: round trips, flux, τ_int on the bottom and top rungs, swap acceptance per
    pair.
  - `checkpoint.*` and `rex_state.pt`.
- The swap uniforms use `FermionREX`'s separate CPU generator (`swap_seed` = seed + 1); the
  former `DeviceREX` workaround is gone.
- Staggered fermions are not yet exposed by the SNF/REX drivers (library only); `main_hmc.py` has them.

**Debug jobs** (one A100, fp64, 30 min). Each resumes automatically if its checkpoint exists:

- `job/debug_hmc_fermions.sh` (`POINTS="w4 w8"`):
  - w4: 4^4 SU(3), β = 1, κ = 0.1, B = 4, 50 + 300 trajectories, exact log det every trajectory.
  - w8: 8^4, β = 5.6, κ = 0.153, 100 + 200 trajectories.
- `job/debug_snf_fermions.sh` (`POINTS="nemcmc snf"`), 4^4, β = 5.6, κ 0.153 → 0.1545 in 8 steps:
  - nemcmc: 100 thermalisation trajectories and 256 samples.
  - snf: starts from the nemcmc pool, 20 Adam steps with stout layers (ρ_init = 1e-3), then 128 samples.
  - Both with `--fast`; output directories end in `_fast` (no old non-fast checkpoint is resumed).
  - residual (optional, `POINTS="nemcmc residual"`): static residual layers with two frozen-link
    convolutions, `--residual-coeff-init 0 --lr 1e-3 --lr-warmup 5`, 20 Adam steps, 128 samples,
    from the nemcmc pool.
- `job/debug_rex_fermions.sh`: 4^4, β = 5.6, κ = 0.153, a 2-link line defect, ladder bc = 0, 0.5, 1,
  4 ladders, DEO swaps with translation, 50 + 200 steps. It starts from the SNF nemcmc pool if
  present. `--fast`, directory ending in `_fast` (as `job/debug_rex_fermions_distributed.sh`).

## 5b. Fast path (`fast.py`; `--fast` in all Wilson drivers; `QCDAction(..., fast=True)`)

**What it is.** `FastTwoFlavourWilson` is `TwoFlavourWilson` (even-odd) with the same heat-bath
draws, action and force to the solver tolerance, computed differently:

- `HalfLatticeWilson`: D̂ = 1 − κ² H_eo H_oe on half-lattice fields (B, V/2, S, N), sites of each
  parity in lexicographic order. Spin projection to half spinors before the neighbour gather,
  links gathered once per configuration with the boundary phase absorbed, colour product
  `matmul`, `broadcast` or `broadcast_sum` (`auto` = `broadcast_sum` on CUDA, `matmul` on CPU).
  complex128 or complex64 (`to(dtype)`; an upcast re-gathers from the complex128 links).
- `lean_cg`: CG with a host convergence check every `check_every` iterations and no per-sample
  masking; returns the true residual and restarts from r = b − A x if a sample's true residual is
  above 10 max(tol, 1e3 ε) at exit (bounded by `max_iter`, stops at a true-residual floor).
- `mixed_precision_cg`: defect correction, complex64 inner CG, complex128 residuals; stops on
  convergence, on stagnation (true residual not halved in an outer step) or when the total inner
  budget `max_iter` is spent.
- The operator is cached per links tensor (`_version`) and per value of κ, time_sign,
  colour product and geometry. Writes through `U.data` or numpy views do not bump `_version`
  and are not supported (as for the η†η cache).

**Measured on an A100** (torch 2.0, complex128 unless stated):

| Quantity | Reference | Fast |
|---|---|---|
| normal operator, site updates/s | 3.1e6 | 2-2.8e7 (half lattice, `broadcast_sum`) |
| full solve | 1 | 6-9× faster |
| HMC at 8^4, per replica | 1 | 7.9× faster (0.65 s per trajectory per replica, 144 CG iterations per solve) |
| complex64 operator | | 1.5-1.65× over complex128 |
| mixed precision solve | | 10-20% over lean double |

The hopping term is now memory-bandwidth bound; the fused kernel below (fermions_plan.md phase P)
addresses that.

**Fused hop kernel** (`fused.py`, opt-in; `main_hmc.py --fast --kernel ...`).

- What it is: `FusedHalfLatticeWilson` is a `HalfLatticeWilson` (same tables, links, force,
  dtype handling) whose hop runs through one custom kernel, one CUDA thread per output (sample,
  site, colour): the spin projection is done on the fly from the 2D neighbour 4-spinors, the
  links are read once, the reconstruction is accumulated in registers and the output is written
  once; the axpy of D̂ = 1 − κ² H_eo H_oe, of `prepare_rhs_half` and of `reconstruct_odd` is
  fused into the second hop. `FusedTwoFlavourWilson` / `FusedTwoFlavourHamiltonian` are the fast
  pseudofermion term and Hamiltonian with this operator (heat bath, action, force, solvers,
  mixed precision unchanged).
- Kernels: `cuda` (the compiled CUDA kernel), `cpu` (the same source built with the host C++
  compiler and run as a loop over the CUDA thread index: tests the index arithmetic without a
  GPU), `loop` (pure torch, explicit only), `auto` (`cuda` on a CUDA device when it builds,
  otherwise the fast operator's own hop, resolved as `fast`: `broadcast_sum` on CUDA, `matmul` on
  CPU). An explicit `cuda`/`cpu` that cannot be built falls back to `fast` with one
  RuntimeWarning (strict=True raises; `main_hmc.py` treats it as an error).
- Link layouts of the compiled kernel: `soa` (default; site index fastest, a copy of the gathered
  links built once per configuration and dtype, coalesced link reads) and `site` (the parent's
  gathered links read in place, no extra memory).
- Build and cache: compiled at first use with plain nvcc (`$CUDA_HOME/bin`, `$CUDA_PATH/bin`,
  PATH) into a shared library with an extern "C" entry point, loaded through ctypes and launched
  on the current torch stream (no torch headers, no ninja, no build lock). Cache directory
  `$NEOQCD_KERNEL_CACHE` (default `~/.cache/neoqcd/kernels`), one file per source hash,
  compiler version and GPU architecture, written under a temporary name and renamed atomically
  (concurrent jobs and killed builds are safe). The `cpu` build uses `$CXX`, c++, g++ or clang++.
- Measured on an A100 (torch 2.0, CUDA 12.1; job 64456003, commit 6858689): the library built
  with nvcc and loaded at the first attempt; all CUDA equality tests passed (largest relative
  deviation 2.55e-16 in complex128). Normal operator D̂†D̂, ms, complex128 / complex64:

| Lattice, batch | `broadcast_sum` | fused `site` | fused `soa` |
|---|---|---|---|
| 8^4, B = 4 | 0.588 / 0.476 | 0.099 / 0.078 | 0.085 / 0.080 |
| 16^4, B = 16 | 53.66 / 32.42 | 3.935 / 1.661 | 2.992 / 1.523 |
| 16^4, B = 64 | 222.2 / 141.8 | 16.23 / 6.544 | 11.99 / 5.979 |

  `soa` is faster than `site` by 25-35% in double precision at 16^4 and by 3-9% in single.
  At 16^4, B = 16: lean CG 66.15 ms per iteration (`broadcast_sum`) vs 5.41 ms (fused `soa`);
  full solve to 1e-10 2256.6 ms (lean CG, double), 219.8 ms (fused, double), 139.7 ms (fused,
  mixed precision). Best throughput 3.5e8 site updates/s in double, 7.0e8 in single. The `loop`
  variant is slower than `broadcast_sum` at small sizes (3.4 ms vs 0.59 ms at 8^4, B = 4) and
  only 1.3× faster at large ones, hence never chosen automatically. (The solve rows of that
  benchmark were labelled `kernel=cuda` only; `bench_fermions.py` now labels every fused row with
  the link layout and also times the solves with `soa`.)
- How to enable: `main_hmc.py --fast --kernel cuda [--link-layout soa|site] [--precision
  mixed]` (also `auto`, `cpu`, `loop`; default `off` = `FastTwoFlavourHamiltonian`). Requested and
  resolved kernel, layout and backend status are in run_meta.json (`run.fused_kernel`), kernel and
  layout in the resume key (older checkpoints count as `--kernel off`). In Python:
  `FusedTwoFlavourHamiltonian(beta, kappa, geometry, kernel="cuda", link_layout="soa", ...)`.
  `job/debug_hmc_fermions.sh` points `w4f`, `w8f` (the `w4`/`w8` validation points with the
  kernel, `_fused` directories) and `w16` (16^4, β = 5.6, κ = 0.153, B = 16).
- Limits: Wilson only (not staggered, not the Hasenbusch terms, not yet the SNF/REX drivers or
  `QCDAction`); compiled for D ∈ {2, 4} and N ∈ {2, 3} (other geometries use `fast`); complex128
  and complex64. Tested on the CPU (`cpu` and `loop` reproduce `--fast` dH to 1e-8 with the same
  accept pattern, double and mixed) and operator-level on the A100; the first HMC validation on
  the GPU (`w4f`, `w8f` against the table below, then `w16`) is still to be run.

**Validation against openQCD 2.4.3** (Wilson N_f = 2, c_sw = 0, antiperiodic quarks; openQCD
plaquette = 3 × ours, converted):

| Point | openQCD | neoqcd |
|---|---|---|
| 4^4, β = 1.0, κ = 0.1 | 0.060765(28) | 0.06084(19) fast (0.06091(19) reference) |
| 8^4, β = 5.6, κ = 0.153 | 0.5661(5) | 0.56609(29) fast (16 replicas, trajectories 121-342) |

**Where it is used.** The reference classes stay the default and the ground truth; the fast path
is opt-in everywhere and tested for equality against them (`test_fermions_fast`,
`test_fermions_fast_action`).

- `main_hmc.py --fast`: `FastTwoFlavourHamiltonian`.
- `QCDAction(..., fast=True, precision=..., colour_product=..., check_every=..., inner_tol=...,
  max_outer=...)` (Wilson only; ValueError for staggered, and for fast-only options without
  `fast=True`): the fermion term is `ParametricFastTwoFlavourWilson` (`action.py`), the fast term
  with the `set_kappa` / `set_phi` / κ-tagged η†η cache of `ParametricTwoFlavourWilson` (one mixin
  for both). `hamiltonian(p)` is the fast two-level Hamiltonian. All three protocols work; a κ
  jump costs one solve (the operator cache is keyed by the value of κ, so it is rebuilt, not
  reused stale). `phi` / `X` are read and written as full-lattice tensors (odd sites zero), so
  `FieldState.phi` round-trips exactly and checkpoints are interchangeable; `set_phi` refuses a
  field with nonzero odd sites.
- NE-MCMC, SNF, REX: `FermionNEMCMC` and `FermionSNF` take the fast `QCDAction` as it is;
  `FermionREX(..., fast=True, precision=...)` passes the options to its `QCDAction`;
  `FermionREXTarget` / `distributed_rex_step` take a fast `QCDAction`.
- SNF layers: `PseudofermionAction` dispatches on the fermion type (`_fermion_kind`: "wilson",
  "wilson_fast", "staggered"; Hasenbusch terms still rejected). For the fast term the action
  Re φ_e†X_e and the force come from the same half-lattice solve
  (`FastTwoFlavourWilson.force_from_solution`, i.e. `HalfLatticeWilson.hopping_derivative`);
  the backward is unchanged (i F U, no CG backpropagation).
- Drivers: `main_snf_fermions.py`, `main_rex_fermions.py`, `main_rex_fermions_distributed.py`
  take `--fast --precision --colour-product --check-every --inner-tol --max-outer` (rules of
  `main_hmc.py`) and `--on-unconverged` (Wilson, reference or fast). The fast settings are in
  run_meta.json (`parameters`; `run.fermion_operator`, resolved `run.colour_product`) and in the
  resume key: a run started without `--fast` is not resumed with it, nor vice versa (the two
  agree to the solver tolerance, not bitwise). Reference runs have unchanged keys, so their old
  checkpoints still resume. Pool / link checkpoints have the same format on both paths.
- Measured on the CPU (2D 4×4 SU(3), tight tolerances): fast and reference runs give the same
  accept/swap decisions and works, actions, swap tables and observables within 1e-8.

**Limits.**

- Cost columns of the SNF/REX drivers (`solves`, `cg_iter`) are the same counters on both paths;
  with `--precision mixed` `cg_iter` counts complex64 inner iterations, not the outer complex128
  applications (only `main_hmc.py` reports the real mixed cost).
- Without per-sample masking every sample of the batch iterates until the slowest has
  converged, so a sample's solution depends on the batch composition at the level of the solver
  tolerance: chains are not bit-reproducible across batch sizes. Resume is bitwise at a fixed
  batch (same device and thread count).

**Convergence policy** (reference and fast terms, and their subclasses). After every solve the
true relative residual is compared with 10 max(tol, 1e3 ε) (the slack of the Hasenbusch heavy
refresh). `on_unconverged="raise"` (default) raises a RuntimeError naming the solve kind
(action/force; "solve" for subclass solves that do not pass their tolerance, then checked
against the looser of solver_tol and action_tol), the residual, the limit, the iterations and the
sample indices; `"warn"` warns once per kind and term; `"ignore"` only counts
(`stats["unconverged_solves"]`). An unconverged action solve would bias the Metropolis step
silently. `main_hmc.py --on-unconverged` sets it; `max_residual` in observables.csv records the
largest residual per trajectory.

## 5c. DD-HMC (`ddhmc.py`, first stage)

Lüscher's domain-decomposition HMC (hep-lat/0409106) for N_f = 2 Wilson fermions. The lattice is
cut into a chessboard of blocks (black Ω, white Ω*), with even block sizes and an even number of
blocks per direction. Cutting the links between blocks gives the block operator
D_bl = D_Ω ⊕ D_Ω* (Dirichlet blocks), and

    det D = det D_Ω · det D_Ω* · det R,   R = P (1 − D_Ω*⁻¹ D_∂Ω* D_Ω⁻¹ D_∂Ω) P   on the boundary set,
    S = φ†(D̂_bl† D̂_bl)⁻¹φ (even-odd in the blocks) + |R⁻¹ φ_R|²,   R⁻¹ψ = P(ψ − D⁻¹ D_∂Ω ψ).

Only the *active* links move in a trajectory: links with both ends in one block and at most one
end on the block's interior boundary (sect. 3.3 and fig. 2 of the paper). Momenta are zero and
forces are masked elsewhere. Active-link fraction in 4D with cubic blocks b⁴, (b−1)(b−2)³/b⁴:

| b | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|
| active fraction | 0.094 | 0.247 | 0.369 | 0.461 | 0.531 |

These match sect. 3.5 of the paper: "a minority of 25%" for 6⁴ blocks and "greater than 50%"
from 12⁴ on. A block size 2 in any direction leaves no active link, and `DDHMC` raises.

**Translations** (checked against the paper on 2026-11-20). Once per update cycle the gauge field is
translated, U(x, μ) → U(x + v, μ) (eq. 3.9). In the code this is `translate(U, v)`, i.e.
U'(n) = U(n − v) in array indices, because "x + μ" is array index n − 1. The blocks and their
colours stay fixed, and the pseudofermions are drawn anew in every trajectory, so only the gauge
field is translated. The paper translates "at the beginning or the end of every update cycle"
(sect. 3.3) and does it at the end (sect. 3.4 (c)). `DDHMC.cycle` / `ddhmc_step` does it at the
beginning, which gives the same chain. The antiperiodic twist stays on the t = 0 time links. The
translation is still an exact symmetry, since moving the twist with the field is the change of
variables ψ → −ψ on the time slices that wrapped around. The choice of v is `DDHMC(translation=...)`:
- `"random"` (default): v uniform on the lattice in every cycle. This is valid by eq. (3.9) alone.
  Appendix C warns that some links are then visited far below average even after hundreds of
  cycles.
- `"luscher"`: appendix C. "A purely random choice of v is followed, in the next few update cycles,
  by translations along a diagonal lattice direction in regular steps." The code takes one uniform
  random v, then n − 1 cycles translated by v = (b₀, …, b₃)/n with n = ½ min{10, b₀, …, b₃}
  (eq. C.2), then a new random v. For 8⁴ blocks this is a random v and then three steps (2, 2, 2, 2)
  (eq. C.1). Each chain has its own random v and the same steps. `n_regular` (default n − 1)
  overrides n. The paper leaves two choices open, and these are ours. When b_μ/n is not an integer
  the paper allows "close-by integer vectors"; we take the nearest integer with halves rounded up,
  the same step in every regular cycle (12⁴ → (2, 2, 2, 2), 16⁴ → (3, 3, 3, 3), the paper's
  8×6²×12 blocks → n = 3, v = (3, 2, 2, 4)). The paper writes D = 4; for other D we use the same
  formula over all directions. `luscher_translation_step(b)` returns (n, v).
- `"odometer"`: ours, not in the paper (this was the code's "luscher" option before 2026-11-20).
  One random v, then the prod(b) − 1 block-relative offsets in odometer order. Over a period every
  link is then active in exactly the active fraction of the cycles.
- `None`: no translation. This is the only option allowed with a defect, and it is not ergodic.

The paper's scheme is balanced only on average. Over a uniform random v every link is active in
each cycle with probability equal to the active fraction. Within one period, though, some links
are never active. Per period of n cycles (exact over v; tests):

| cubic blocks | 4⁴ | 6⁴ | 8⁴ | 10⁴ | 12⁴ |
|---|---|---|---|---|---|
| n, v | 2, (2,2,2,2) | 3, (2,2,2,2) | 4, (2,2,2,2) | 5, (2,2,2,2) | 5, (2,2,2,2) |
| links never active in a period | 13/16 | 1/3 | 3/64 = 4.7% | 0 | 0 |
| variance of the visits (with n random v) | 0.15 (0.17) | 0.34 (0.56) | 0.42 (0.93) | 0.45 (1.24) | 0.56 (1.25) |

No link is missed in a period for cubic blocks from 10⁴ to 20⁴ either. In 2D with 6×6 blocks
(fig. 9: three grids) every link is active in every period. Over 100 cycles with 8⁴ blocks
(mean 36.9 visits) the spread of the visit counts across links is 3.1 against 4.9 with random
vectors, and the least visited link has 26 visits against 20 (8 chains; the spread ratio is
0.60–0.73 over 20 seeds).

**Naming.** The code's labels are colour-mirrored with respect to the paper. The code takes the
Schur complement on Ω*, and its "∂Ω*" (`exterior_boundary`) is the set of white sites with a
black neighbour, which is the paper's ∂Ω. The determinant is the same. There are two exact
deviations from the paper. First, φ_R uses all spin-colour components on the boundary sites
instead of the reduced space of eq. (B.4). Second, the block heat bath is φ = D̂†η, following the
repo convention.

**Validation** (`test_fermions_ddhmc.py`, CPU, complex128; fast suite under a minute):
- geometry and masks;
- the dense factorisation det D = det D_Ω det D_Ω* det R (modulus and phase), on 2D 4×4 and
  8×8 with SU(3) and SU(2), for time_sign = ∓1, and on 4^4 behind `NEOQCD_SLOW`;
- R⁻¹ checked against the dense inverse;
- stochastic determinant ratios for both terms;
- forces checked against finite differences (2D and 4D 8^4 SU(2)) and against autograd on every
  active link (Schur term, block term on the full lattice and even-odd);
- per-chain κ;
- reversibility, frozen inactive links and dH ∝ ε² (2D, plus one 4D 8^4 SU(2) trajectory);
- the Metropolis step: the replayed accept decision, with rejected chains bitwise unchanged;
- `cycle` / `ddhmc_step`: links stay in SU(N), the translation is applied before the trajectory,
  and the stats keys (with `plaquette`) are present;
- translation coverage over all translations, and the `"odometer"` period;
- behind `NEOQCD_SLOW`, DD-HMC with `"random"` and with `"luscher"` translations against plain HMC
  and the exact-determinant Metropolis on 2D 8×8, κ = 0.22 (~20 min).

`test_fermions_ddhmc_translation.py` (16 s) checks the translations against the paper:
- Wilson-operator covariance D[T_s U](σ T_s ψ) = σ T_s(D[U]ψ), with σ = time_sign on the wrapped
  time slices, on 4⁴ SU(3) and 8×4³ SU(2) for time_sign = ±1. Without σ it fails on every
  time-shifted chain at time_sign = −1.
- The dense det D (modulus and phase, built from the links by `exact.py`) and the gauge action
  are unchanged by time and mixed shifts (4⁴ and 8×4³ SU(2)). The twist is felt: the det differs
  between ±1.
- n and v of eqs. (C.1)–(C.2) for a table of block sizes, the active fractions of sect. 3.5, and
  the refusal of decompositions without active links.
- The `next_shifts` sequence (random, then (2,2,2,2) three times for 8⁴), and the restart on a new
  batch size.
- The per-period visiting distribution against the analytic count (3/64 and 1/3 above).
- The 100-cycle balance against random vectors (8 chains).
- 12 `"luscher"` cycles: ⟨e^{−ΔH}⟩ = 1, and inactive links bitwise equal to the translated input.

**Limits.**
- The block solves use a global CG on the block operator, not block-local solvers.
- The Schur solves have no chronological guess.
- There is no Schwarz preconditioner.
- DD-HMC is not wired into the drivers or into the fast path.
- There is no defect. A translation is a symmetry only without a boundary-condition defect.
  `DDHMCHamiltonian` takes no `plaquette_weights`, and `DDHMC.cycle` raises if the gauge term
  carries some. `translation=None` is allowed, but the fixed grid alone is not ergodic.

## 6. Known limitations and TODO

- **Exact comparison and determinant normalisation.** The HMC against `ExactMetropolis` test
  (`hmc_fermions`, κ = 0.22 on 2D 4×4) does not resolve N_f = 2 against N_f = 1 by itself. An HMC
  with S_pf and its force halved passes it at this statistics. The normalisation is pinned
  separately by `StochasticEstimatorTest` and `ForceExpectationTest` in `test_fermions_pseudofermion`.
  At κ = 0.12 the fermion effect is below the errors in 2D.
- **Jarzynski power.** In β the work is gauge-only, so the β test gets its power from a quenched
  reference at > 2σ of the paired jackknife (3σ would need about 4× the statistics). The NE-MCMC
  errors use snapshots 3 trajectories apart on the same chains and may be slightly underestimated.
- **Reversibility with warm start.** With `reuse_guess=True` (the default) CG starts from the
  last solution. The trajectory is then reversible only to the solver tolerance.
  `--no-reuse-guess` starts CG from zero. Even then reversibility is only O(solver tolerance) in
  general: CG iterates stop being smooth in U once orthogonality is lost. The reversibility to
  rounding seen in the tests relies on CG converging in about 20 iterations. The staggered action
  solves in a fixed, warm-started order, so its results do not depend on call order beyond the
  tolerance.
- **Batch cost.** The batched solvers iterate until the slowest sample converges, and
  `cg_iterations` counts that maximum. Per-sample counts are kept separately.
- **Staggered refresh.** It costs one extra solve per trajectory (S_pf at the refresh links is not
  χ†χ). `StaggeredPseudofermion` is N_f = 4 in 4D; N_f = 2 in 4D is the RHMC term of `rhmc.py`.
- **O(V) swap table.** `swap_action_difference` builds each of the n_rungs columns from a sum
  over the whole lattice of (w_old − w_new) Re tr U_p. The terms vanish beyond the plaquettes that
  touch the defect, so this is O(n_rungs·V) work that could be restricted to the defect halo.
  It is cheap compared with a solve, but it scales with V.
- **`rex.py` on CUDA.** `FermionREX` is device-safe (bookkeeping and swap uniforms on the CPU,
  table moved there), but untested on a GPU. The distributed path (`FermionREXTarget`) is tested
  only on Gloo/CPU.
- **Hasenbusch.** On 2D 4×4 the two-term HMC gives no cost saving at equal acceptance; the
  choice of κ' and of the layout is to be measured on 4^4/8^4. Not yet wired into the drivers.
- **CUDA in general.** `main_hmc.py` (reference and `--fast`) and the SNF/REX drivers (reference)
  have run on a GPU (A100); the fast path in the SNF/REX drivers has not yet.
  Generators must be on the device (`torch.Generator(device=U.device)`). `stats["acceptance"]` syncs once per trajectory.
- **Exact tools.** `ExactMetropolis` recomputes a dense determinant per proposal, which is
  practical only up to about 2^3×4 in a unit test. Exact log det on 4^4 even-odd is 1536² and
  feasible. On 8^4 (24576², 9.6 GB) it is planned on A100 but untested.
- **SNF layers.**
  - Only pure-gauge layers; φ is held fixed across a layer and resampled per trajectory, with no
    flow on φ.
  - `FermionLayerBlock` raises if a layer returns a nonzero dQ.
  - Staggered SNF with layers is in progress (`snf_layers.py`); NE-MCMC and REX already take
    staggered fermions.
  - The legacy SU(2) stout log-Jacobian is doubled, as in `Flow._full_logdet`. This is not the
    exact Haar log-Jacobian (relative error ~1e-3 at ρ = 0.01-0.1), so SU(2) stout SNF is only
    approximately Jarzynski-exact. SU(3) is exact to ~1e-14 (unit test).
  - Residual layers: SU(3) and the full periodic lattice only (no cropped defect patch, no
    `residual_mode=class`); 4D mask mod 4 is tested only through the 2D Jacobian checks (a dense
    4^4 Jacobian is too large for a unit test). Up to 14 November the fermionic `--layers residual`
    was the hyper parameterisation with raw (λ_i, forward-difference) conditioning; it is now the
    static one (pure-gauge spelling), and such old flows are refused by `--load-flow`.
  - The stout layer uses |ρ|, whose gradient vanishes at ρ = 0, hence the default ρ = 1e-3.
    `--layers smearing_signed` (`snf_layers.SignedCouplingLayer`, a `Smearing` subclass with a
    signed ρ; the map and its exact log-Jacobian are linear in ρ through the staples, checked
    against the dense Haar Jacobian at negative ρ) starts at the identity with `--rho-init 0`
    and trains; `--lr-warmup k` ramps the learning rate linearly over k steps.
  - Adam's first steps move every parameter by about lr whatever the gradient: with lr = ρ_init
    (the 14 November GPU pilot, lr = ρ_init = 1e-3) step 0 sends each ρ to ≈ 0 or ≈ 2ρ_init.
    `param_abs_mean` in training.csv is the value the step's flow ran with (before its optimizer
    step; before 14 November it was logged after it, which is why the pilot showed 1.5625e-4 =
    5 × 2e-3 / 64 at step 0).
  - The protocol is a list of floats; per-configuration (B,) protocols are accepted by the
    NE-MCMC only.
- **Pool correlation.** The SNF driver's consecutive batches are correlated through the pool's
  autocorrelation (`--n-between`), as in `main_nemcmc_snf.py`.
- **Open questions from the plan.**
  1. The β interval 5.45 → 5.6, to fix after measuring t0.
  2. Whether to build the 2D Schwinger test bed.
  3. Clover is implemented (`clover.py`); when to use it in production is open.
  4. Whether to run the 12^3×24, κ = 0.156 check (about 1 A100 day), after the phase-1 cost
     measurement.
  5. Further work: Hasenbusch on larger lattices, DD-HMC in production (§5c), Grid reference runs, and the
     convention-alignment checklist (κ, plaquette normalisation, τ_int).
