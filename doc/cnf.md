# SU(3) continuous fields and REX Path Gradient training

Work lives on branch `cnf`. `main/main_cnf_rex.py` trains a **field**, without
integrating an SNF in the training loop. Evaluation integrates that field and
interleaves the deterministic blocks with HBOR, using the NEO-REX protocol
interface. The existing discrete SNF training defaults are unchanged.

## Field and objectives

`CloverChairLinkDrift` ports the Wilson/rectangle/chair/free-clover geometry
from `multiSU2`. There is no coarse lattice, blocking map, or conditional `y`.
The physical protocol parameter (beta or boundary coupling) remains an input.
The SU(3) basis is `T_a = lambda_a/2`, with `Tr(T_a T_b) = delta_ab/2`:

```
dU/dt = i sum_a F_a(U,t) T_a U
xi = (dS/dt - div F + F . grad S) / V
PG = mean(F_theta . grad xi_frozen)
PINN = V * Var(xi | t)
```

Parameters inside `xi_frozen` are detached, but its dependence on links is
differentiated. PG gradients reach the live field only. PINN is a diagnostic,
computed separately within each replica; it is not the training objective.
At least two chains per replica are required to estimate its variance.

The drift uses traceless Hermitian projections of loop anti-Hermitian parts,
gauge-invariant scalar/Gram features and SU(3) commutator channels. It is a
minimal equivariant extension, not a complete basis of SU(3) invariants:
symmetric `d_abc` channels and the second adjoint projection are not included.
Haar divergence uses eight directional derivatives with the other links held
fixed. On short periodic lattices, paths that reuse their target link are
zeroed to preserve the validity of this diagonal derivative.

The other supported architectures are `smearing`, `hyper-smearing`, `residual`
and `hyper-residual`. Their pre-exponential generators define velocities per
unit physical parameter. Residual frozen-link convolutions and mask mod 4 are
supported; `--rho-functional` selects functional smearing and
`--hyper-smearing-mode per_link` selects anisotropic hyper-smearing.
These fields are trained afresh: integrating them does **not** reproduce the
old finite coupling maps or their discrete Jacobians. The legacy `use_nn`
fixed-time 4D CNN and cropped defect networks are not exposed as CNF fields.
Defect CNFs instead act on the full lattice, with the exact defect action.

## Run

Use the project environment. A small example with real 3D chair loops:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .sunenv/bin/python main/main_cnf_rex.py \
  --architecture clover-chair --D 3 --T 4 --L 4 \
  --parameter-min .5 --parameter-max 1 --replicas 3 --batch-size 2 \
  --hidden-dim 4 --depth 1 --warmup 10 --epochs 10 \
  --output-dir results/cnf_clover
```

Replace `--architecture` to train another field. For a defect ladder add
`--domain defect --D 4 --beta 6 --parameter-min 0 --parameter-max 1` and choose
the defect's size and location. REX uses HBOR and adjacent exchanges with full
cross-actions. The derivative of the defect action is retained at BC=1.

The output contains `metrics.jsonl`, `config.json` and `cnf_checkpoint.pt`.
The checkpoint stores model, optimizer, ladder chains, swap parity and RNG.
`--epochs` is the total requested count on resume. Saved model/sampler settings
are authoritative; other training CLI settings do not override a checkpoint.

```sh
.sunenv/bin/python main/main_cnf_rex.py --checkpoint results/cnf_clover/cnf_checkpoint.pt \
  --epochs 20 --output-dir results/cnf_clover

.sunenv/bin/python main/main_cnf_rex.py --phase evaluate \
  --checkpoint results/cnf_clover/cnf_checkpoint.pt \
  --protocol-steps 4 --integration-steps 4 --eval-steps 10 \
  --output-dir results/cnf_clover/evaluation
```

If a crash left metrics beyond the last checkpoint, resume into a new output
directory. Existing, inconsistent metrics are not silently overwritten.

The existing standalone runner can also load the field:

```sh
.sunenv/bin/python main/main_nemcmc_snf.py --cpu --algorithm snf --phase evaluate \
  --nf-architecture cnf --cnf-checkpoint results/cnf_clover/cnf_checkpoint.pt \
  --D 3 --T 4 --L 4 --batch-size 2 --domain beta \
  --beta-initial .5 --beta-final 1 --protocol-steps 4 --int-meas 1 \
  --cnf-integration-steps 4 --eval-steps 10 --cfg-cache off \
  --output-dir results/cnf_clover/standalone
```

For distributed defect NEO-REX, use the existing `main/main_pt_obc.py` with
`--algorithm neorex --nf-layer-type cnf --cnf-checkpoint PATH
--cnf-integration-steps 4`. Supply matching lattice, beta and defect geometry
and a BC interval contained in the training interval. This selects evaluation;
positive `--train-steps` is rejected because training belongs in the REX field
runner. NEO-REX computes **full** action differences for this field rather than
the cropped-layer halo shortcut. Portable CNF checkpoints are distinct from
the wrapper's `--neorex-load-flow` checkpoints.

## Numerical limits and checks

CF4 stays on SU(3) and integrates the Haar divergence along its stages.
The resulting logdet and inverse have finite-step integration error: they are
not the exact Jacobian and inverse of the discrete numerical map. This matters
for NEO-REX acceptance and SNF weights. Before production, refine the integration
step and check stability of work/observables. `evaluation.json` includes a
logdet refinement test and a forward/backward round-trip test, in addition to
work, ESS and delta F. These diagnostics are not an unbiasedness proof.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .sunenv/bin/python -m unittest \
  test.test_cnf test.test_cnf_adapters test.test_cnf_training \
  test.test_cnf_integration test.test_cnf_neorex \
  test.test_main_nemcmc_snf test.test_neorex_checkpoint -v
```

Tests cover gauge covariance, Wilson-path conventions, numerical divergence,
frozen-residual gradients, CF4 reversibility, all four existing generator
families, defect derivatives at BC=1, REX swap signs and checkpoint integration.
See [cnf_smoke.md](cnf_smoke.md) for the CPU pilot; it is not a production benchmark.
