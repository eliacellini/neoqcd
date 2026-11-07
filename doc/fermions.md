# Dynamical fermions in neoqcd (`src/neoqcd/fermions/`)

Branch `fermions`, commits `237383c` .. `ec05820` (the plan, `fermions_plan.md`, is `3c1b17c`).
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
- `QCDAction`: one parametric joint action S(U, φ; λ) with λ ∈ {β, κ, bc}. The defect `bc`
  enters the gauge action only (Bonanno et al., arXiv:2404.14151).
- Fermionic NE-MCMC (Jarzynski on the joint (U, φ)) for protocols in β, κ or bc.
- Fermionic SNF: pure-gauge stout/residual layers (trainable) before every HMC block. The S_pf
  gradient comes from a custom autograd Function, with no backpropagation through CG.
- REX on the gauge-only defect: fermionic HMC replicas and solve-free swaps. There is a
  single-process `FermionREX` and a `distributedPT.ParallelTempering` target (tested on Gloo).
- Drivers `main/main_hmc.py`, `main/main_snf_fermions.py` and `main/main_rex_fermions.py`, all
  with checkpoint/resume, plus three Leonardo debug jobs.

**Verified only on tiny lattices.** The exact checks run on CPU in complex128 on 2D 4×4 (SU(2),
SU(3)), 2^3×4, 2^4 and 4^4 (the last only for operator-level dense checks). These checks cover
Jarzynski against exact reweighting, HMC against the exact-determinant Metropolis, and REX
against plain HMC. No physics number on 4^4 or 8^4 has been produced yet.

**Not done.**

- Hasenbusch mass preconditioning.
- Clover term.
- RHMC: staggered N_f = 2 in 4D, Zolotarev, multishift CG.
- DD-HMC.
- Staggered in `QCDAction`/NE-MCMC/SNF/REX: these are Wilson only.
- O-REX and NEO-REX with fermions (out of this plan).
- The 4^4 (β = 1, κ = 0.1) and 8^4 (β = 5.6, κ = 0.153, 0.1545) benchmarks.
- The comparison with openQCD/Grid.
- Any run on CUDA: the drivers accept `--device cuda` but the CUDA path is untested.
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
- Staggered (`staggered_pseudofermion.py`):
  - Action: S_pf = φ†M_ee^{−1}φ.
  - Refresh: φ = (D†χ)_e = mχ_e − K_eo χ_o, with χ Gaussian on all sites.
  - S_pf at the refresh links is not χ†χ (D† restricted to even rows is rectangular). It costs
    one solve, cached on first use.

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

Jump costs: β is Δβ·S_g(U; β = 1) and bc is gauge only, both without a solve; κ takes one solve
(S at the old κ is known from the end of the previous block). A zero jump is exactly zero. If x_0 has no φ, φ is drawn by a heat bath at λ_0. The SNF loss is
mean(W). The HMC blocks are under no_grad, so the gradient flows only through the layers.

**REX swap** (`rex.py`). Rungs i < j hold U_a (at bc_i) and U_b (at bc_j). The determinant does
not see bc and cancels configuration by configuration, so

    ΔS = [S_g(U_a; bc_j) − S_g(U_a; bc_i)] + [S_g(U_b; bc_i) − S_g(U_b; bc_j)],   accept with min(1, e^{−ΔS})

This needs no solve and no φ. `action_table` returns the relative table S(U_b; bc_k) − S(U_b; bc_b),
which gives the same swaps as the absolute table. Labels move, configurations do not. The
ladder increases, with rung 0 the most open and the top rung bc = 1. An optional periodic
translation of the bc = 1 replica is exact for antiperiodic fermions as well.

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
| `pseudofermion.py` | `hopping_derivative`, `TwoFlavourWilson` (action/refresh/force/stats), `TwoFlavourHamiltonian` |
| `staggered_pseudofermion.py` | `staggered_hopping_derivative`, `StaggeredPseudofermion`, `StaggeredHamiltonian` |
| `exact.py` | `log_det_two_flavour`, `assemble_hopping`, `assemble_even_odd`, `ExactMetropolis` |
| `staggered_exact.py` | `assemble_staggered_hopping`, `assemble_staggered_even_odd`, `log_det_staggered`, `ExactStaggeredMetropolis` |
| `action.py` | `QCDAction`, `ParametricTwoFlavourWilson`, `JointHamiltonian`, `bc_link_couplings`, `bc_plaquette_weights`, `PARAMETER_NAMES` |
| `snf.py` | `FieldState`, `HMCBlock`, `FermionNEMCMC`, `jarzynski_free_energy`, `effective_sample_size`, `OBSERVABLES` |
| `snf_layers.py` | `PseudofermionAction`, `pseudofermion_action`, `joint_action`, `FermionLayerBlock`, `FermionSNF`, `make_gauge_layers` |
| `rex.py` | `swap_action_difference`, `FermionREXTarget`, `distributed_rex_step`, `FermionREX` |
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

Runtimes are wall times per file on the Mac (one thread), measured on 2026-11-07, all passing; the whole suite takes about 5 min, dominated by nemcmc, staggered_hmc, hmc_fermions and hmc.

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
| `action` | S after refresh = η†η, β/κ/bc jumps against `Wilson_action`, `DefectMCMCAdapter` and dense det, Hamiltonian parameters | 2s |
| `nemcmc` | Jarzynski in κ (against exact det ratio) and β (against a quenched reference), W + Q bookkeeping, low-acceptance bookkeeping, constant protocol W = 0 | 57s |
| `snf` | autograd Function value and group derivative, gradient through a layer, force saved at forward, refresh-cache version key, layer work, Jarzynski with stout layers, ρ = 0 equals NE-MCMC, 20-step training smoke test | 35s |
| `rex` | local swap ΔS = full action difference, translation keeps the weight, rung plaquette = plain HMC (bc = 0 and 1), history and round trips, three Gloo ranks with `ParallelTempering` | 30s |
| `main_hmc` | Wilson outputs and bitwise resume, rows past the checkpoint dropped, quenched and staggered runs, argument checks | 3s |
| `main_snf_rex` | NE-MCMC in κ, SNF in bc with training and resume, interrupted SNF resumes identically, residual layers, REX outputs and resume (outputs only, not physics) | 7s |

## 5. How to run

**Runs never go on the Mac or on the Leonardo login node.** They go to Leonardo jobs: the debug
QOS (`boost_qos_dbg`, 30 min) for tests, pilots and short thermalisations, and production for
anything longer. Sync code via git; submit with `sbatch` from `/leonardo_work/INF26_sft/ecellini/neoqcd`.

**`main/main_hmc.py`** runs B independent HMC chains as one batch.

- Fermions: `--fermions wilson|staggered|none`, with `--kappa` (Wilson) or `--mass` (staggered).
- Lattice and couplings: `--D --T --L --N --beta --time-sign`.
- Integrator: `--tau --n-steps --n-gauge-substeps --integrator omelyan|leapfrog|outer,inner`.
- Solver: `--solver-tol` (force), `--action-tol` (Metropolis), `--max-iter`, `--no-reuse-guess`.
- Run length: `--batch --start cold|hot --therm --n-traj --meas-every --checkpoint-every`.
- `--exact-logdet` measures the dense log det for dense dimension ≤ `--exact-max-dim`
  (default 3072).
- Control: `--seed --device auto|cpu|cuda --out --resume --overwrite`.
- Outputs in `--out` (default `output/hmc_fermions/<name>/`):
  - `observables.csv`: one row per trajectory × replica, with the plaquette, dH, accepted,
    exp(−dH), log det, solves, CG iterations and timings.
  - `observables.npz`.
  - `run_meta.json`: parameters plus a Gamma-method summary, acceptance, ⟨e^{−ΔH}⟩, seconds per
    trajectory and per solve, and CG iterations per solve.
  - `checkpoint.pt/.json`.
- Resume is bitwise identical on the same device and thread count.

**`main/main_snf_fermions.py`** runs NE-MCMC or SNF on a protocol in one parameter.

- Protocol: `--protocol beta|kappa|bc` with `--initial --final --n-step` (linear) or
  `--protocol-file`.
- Mode: `--mode nemcmc|snf`; for snf, `--layers smearing|residual --rho-init --train-steps --lr
  --grad-clip --load-flow`.
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
  - `flow.pt/.json` and `pool.pt/.json` (resume).

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
- The swap uniforms use a separate CPU generator (seed + 1).

**Debug jobs** (one A100, fp64, 30 min). Each resumes automatically if its checkpoint exists:

- `job/debug_hmc_fermions.sh` (`POINTS="w4 w8"`):
  - w4: 4^4 SU(3), β = 1, κ = 0.1, B = 4, 50 + 300 trajectories, exact log det every trajectory.
  - w8: 8^4, β = 5.6, κ = 0.153, 100 + 200 trajectories.
- `job/debug_snf_fermions.sh` (`POINTS="nemcmc snf"`), 4^4, β = 5.6, κ 0.153 → 0.1545 in 8 steps:
  - nemcmc: 100 thermalisation trajectories and 256 samples.
  - snf: starts from the nemcmc pool, 20 Adam steps with stout layers, then 128 samples.
- `job/debug_rex_fermions.sh`: 4^4, β = 5.6, κ = 0.153, a 2-link line defect, ladder bc = 0, 0.5, 1,
  4 ladders, DEO swaps with translation, 50 + 200 steps. It starts from the SNF nemcmc pool if
  present.

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
  `--no-reuse-guess` starts CG from zero. The review in progress notes that even then
  reversibility is only O(tol) in general: CG iterates stop being smooth in U once orthogonality
  is lost. The reversibility to rounding seen in the tests relies on CG converging in about 20
  iterations.
- **Batch cost.** The batched solvers iterate until the slowest sample converges, and
  `cg_iterations` counts that maximum. Per-sample counts are kept separately.
- **Staggered refresh.** It costs one extra solve per trajectory (S_pf at the refresh links is not
  χ†χ). Staggered is HMC only, with N_f = 4 in 4D; N_f = 2 in 4D needs RHMC (not done).
- **O(V) swap table.** `swap_action_difference` builds each of the n_rungs columns from a sum
  over the whole lattice of (w_old − w_new) Re tr U_p. The terms vanish beyond the plaquettes that
  touch the defect, so this is O(n_rungs·V) work that could be restricted to the defect halo.
  It is cheap compared with a solve, but it scales with V.
- **`rex.py` on CUDA.**
  - `FermionREX` keeps labels, ladder, counters and history on the CPU. `swap_round` draws its
    uniforms on the CPU and compares them with a device table.
  - The driver works around this with `DeviceREX` (table moved to the CPU, separate CPU swap
    generator).
  - The distributed path (`FermionREXTarget`) is tested only on Gloo/CPU.
- **CUDA in general.** No fermion code has run on a GPU yet. Generators must be on the device
  (`torch.Generator(device=U.device)`). `stats["acceptance"]` syncs once per trajectory.
- **Exact tools.** `ExactMetropolis` recomputes a dense determinant per proposal, which is
  practical only up to about 2^3×4 in a unit test. Exact log det on 4^4 even-odd is 1536² and
  feasible. On 8^4 (24576², 9.6 GB) it is planned on A100 but untested.
- **SNF layers.**
  - Only pure-gauge layers; φ is held fixed across a layer and resampled per trajectory, with no
    flow on φ.
  - `FermionLayerBlock` raises if a layer returns a nonzero dQ.
  - The legacy SU(2) stout log-Jacobian is doubled, as in `Flow._full_logdet`. The review in
    progress measured this as not the exact Haar log-Jacobian (relative error ~1e-3), so SU(2)
    stout SNF is only approximately Jarzynski-exact. SU(3) is exact to ~1e-14.
  - The stout layer without hyper-smearing uses |ρ| (`Smearing._mask_rho`), whose gradient vanishes at ρ = 0. Layers built with ρ = 0 cannot
    train, and that includes `make_gauge_layers` at its committed default and `--rho-init 0.0`
    in `job/debug_snf_fermions.sh`. The uncommitted review fix changes the default to 1e-3.
  - The protocol is a list of floats; per-configuration (B,) protocols are accepted by the
    NE-MCMC only.
- **Pool correlation.** The SNF driver's consecutive batches are correlated through the pool's
  autocorrelation (`--n-between`), as in `main_nemcmc_snf.py`.
- **Open questions from the plan.**
  1. The β interval 5.45 → 5.6, to fix after measuring t0.
  2. Whether to build the 2D Schwinger test bed.
  3. When to add clover.
  4. Whether to run the 12^3×24, κ = 0.156 check (about 1 A100 day), after the phase-1 cost
     measurement.
  5. Further work: Hasenbusch, DD-HMC, RHMC, openQCD/Grid reference runs, and the
     convention-alignment checklist (κ, plaquette normalisation, τ_int).
