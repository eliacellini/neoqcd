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
Its time encoder sees the parameter rescaled to [0, 1] over the trained interval
(`time_normalization: unit_interval` in new checkpoints; older ones keep the raw
parameter). Haar divergence uses eight directional derivatives with the other links held
fixed. On short periodic lattices, paths that reuse their target link are
zeroed to preserve the validity of this diagonal derivative.

The other supported architectures are `smearing`, `hyper-smearing`, `residual`
and `hyper-residual`. Their pre-exponential generators define velocities per
unit physical parameter. Residual frozen-link convolutions and mask mod 4 are
supported; `--rho-functional` selects functional smearing and
`--hyper-smearing-mode per_link` selects anisotropic hyper-smearing.
These fields are trained afresh: integrating them does **not** reproduce the
old finite coupling maps or their discrete Jacobians. The legacy `use_nn`
fixed-time 4D CNN is not exposed as a CNF field.

`defect-hyper-smearing` is the infinitesimal generator of the NEO-REX defect
coupling layer (`DefectCouplingLayer` with hyper-smearing): the same cropped
patch (buffer 2), link selection, staples and `HyperSmearing(bc)` network, with
rho evaluated at unit delta, i.e. per unit boundary coupling. Links outside the
smeared patch have zero velocity. To first order in the step it equals the
finite layer, so the trained network can be evaluated with that layer and its
exact logdet (see below). It needs D=4, `T >= 4`, `L >= defect-size + 4` and a
two-site buffer around the defect; `--hyper-smearing-mode` may be `shared`,
`per_link` or `class`.

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

In an O-REX/NEO-REX swap the heatbath runs at both ends of the protocol, and a
chain moving down the ladder applies the inverse of the step its partner takes
going up (`inverse` of each layer: fixed-point iteration for smearing and
residual layers, backward integration for a CNF). The two protocols are then
each other's time reversal, as detailed balance of the swap requires; for a
CNF the inverse, like the logdet, carries the CF4 integration error.

### Evaluating a defect field with the smearing layer

A `defect-hyper-smearing` field can instead be evaluated with the finite
smearing layer, which has an exact logdet, either in NEO-REX or in the
standalone SNF. Copy its network into the runner's checkpoint, passing the
evaluation flags after `--`.

NEO-REX (`main_pt_obc.py`):

```sh
FLAGS=(--cpu --algorithm neorex --D 4 --T 4 --L 6 --N 3 --beta 6
  --defect-size 1 --time-slice 2 --space-slice 2 --parameter-min 0 --parameter-max 1
  --nf-layer-type smearing --use-hyper-smearing --hyper-smearing-mode per_link
  --hyper-hidden-dim 16 --hyper-depth 2 --hyper-scale-by-delta --hyper-no-normalize-by-nstep
  --neorex-protocol-steps 2 --neorex-heatbath-steps-per-step 1 --batch-size 16)
.sunenv/bin/python main/convert_cnf_to_smearing_flow.py neorex results/cnf_defect/cnf_checkpoint.pt \
  results/cnf_defect/neorex_flow.pt -- "${FLAGS[@]}"
.sunenv/bin/torchrun --nnodes 1 --nproc_per_node 4 --master_addr 127.0.0.1 --master_port 29500 \
  main/main_pt_obc.py "${FLAGS[@]}" --neorex-load-flow results/cnf_defect/neorex_flow.pt \
  --neorex-eval-flow --output-dir results/cnf_defect/neorex
```

Standalone SNF (`main_nemcmc_snf.py`), from BC=0 to BC=1:

```sh
FLAGS=(--algorithm snf --domain defect --cpu --D 4 --T 4 --L 6 --beta 6
  --bc-initial 0 --bc-final 1 --defect-size 1 --time-slice 2 --space-slice 2
  --nf-architecture hyper-smearing --hyper-smearing-mode per_link
  --hyper-hidden-dim 16 --hyper-depth 2 --hyper-no-normalize-by-nstep
  --protocol-steps 6 --orsteps 4 --updates-per-layer 1 --prior-mcmc-steps 1 --batch-size 64)
.sunenv/bin/python main/convert_cnf_to_smearing_flow.py snf results/cnf_defect/cnf_checkpoint.pt \
  results/cnf_defect/snf_checkpoint.pt -- "${FLAGS[@]}"
.sunenv/bin/python main/main_nemcmc_snf.py "${FLAGS[@]}" --phase evaluate \
  --checkpoint results/cnf_defect/snf_checkpoint.pt --eval-samples 1024 \
  --output-dir results/cnf_defect/snf
```

`--hyper-scale-by-delta` (the SNF default) and `--hyper-no-normalize-by-nstep`
keep the trained convention: rho per unit BC, scaled by each protocol step.
Geometry, beta, hyper mode and network shape must match the field; the
converter refuses a mismatch. Each protocol step applies rho(bc) * delta_bc with
the parities updated in sequence: a finite discretization of the field, exact
as a map, so step size affects efficiency but not the unbiasedness of the
weights or acceptance.

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
