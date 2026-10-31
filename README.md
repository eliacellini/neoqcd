# neoqcd

Continuous SU(3) CloverChair fields, REX Path Gradient training and PINN
diagnostics are described in [doc/cnf.md](doc/cnf.md). Training avoids SNF
integration; evaluation reconstructs the stochastic flow with CF4.

Residual flows support learned gauge-covariant convolutions of frozen links on
the full lattice (`main/main_nemcmc_snf.py`, `--domain beta`). Select
`--nf-architecture residual` for static coefficients or `hyper-residual` for
beta-conditioned coefficients, then choose:

| Variant | Options |
| --- | --- |
| Original plaquette residual (default) | `--residual-n-pt 0 --residual-mask-mod 2` |
| One frozen-link convolution | `--residual-n-pt 1 --residual-mask-mod 2` |
| Two frozen-link convolutions | `--residual-n-pt 2 --residual-mask-mod 2` |
| Sparser mask and extended paths | `--residual-n-pt 2 --residual-mask-mod 4` |

The same options are available as `FlowPars(residual_n_pt=..., residual_mask_mod=...)`.
Each convolution combines a link-like field with learned weights times its
forward/backward staples. Its initial field has the active links set to zero;
the original active link is used only when closing the final loops. This keeps
the local SU(3) Jacobian valid while extending the paths used by the residual.
The final number of trace features is unchanged.
In two dimensions, a single convolution with mod 2 adds no effective dependence;
use at least two convolutions or mod 4 there.

Extended variants require periodic lattice extents divisible by the mask modulus.
Mod 4 uses 16 directional/color substeps in four dimensions, versus 8 for mod 2.
MCMC retains its original checkerboard masks. The cropped defect adapters do
not support these extensions yet because their halo is fixed.

Default model weights and checkpoint metadata remain compatible with the original
residual. Extended checkpoints record the convolution depth and mask modulus;
use matching options when loading them.

Run CPU smoke tests in the project environment (no simulation):

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .sunenv/bin/python -m unittest \
  test.test_residual_normalizing_flows test.test_residual_frozen_convolution -v
```

The whole suite (including a three-rank Gloo NEO-REX run) is discovered with:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .sunenv/bin/python -m unittest discover -s test -t .
```

## Replica history of REX runs

`main/main_pt_obc.py` records, after every step, the ladder rung of each chain slot and its
plaquette (`replica_history.npz`, keys `labels` and `plaquette` of shape steps x ranks x batch,
rewritten every `--log-every` steps; `--no-save-replica-history` disables it). `run_meta.json`
then reports the round trips bc = 0 -> 1 -> 0 completed by the chain slots, their mean duration in
REX steps, the per-rung flux fraction (share of walkers that last visited bc = 0) and the
Gamma-method `tau_int` of the plaquette on the bottom and top rungs. `neoqcd.replica_history`
holds the analysis; `main/summarize_neorex_validation.py` prints the round trips per wall-clock
hour and the seconds per independent plaquette sample (`s/step x tau_int`) as cost measures.

## Local updates along the O-REX/NEO-REX protocols

Between swaps every replica is updated with one HB and `--orsteps` OR sweeps over the whole
lattice. Along the protocols `main/main_pt_obc.py` can instead follow the O-REX paper:
`--protocol-patch-halfsize h` restricts the HB+OR sweeps to a box around the defect
(`Defect.protocol_patch_mask`; h = 0 is the smallest box holding every link of the plaquettes
whose weight depends on bc, each unit adds one site in every direction) and
`--protocol-symmetric-updates` applies each sweep or its reverse (OR sweeps, then HB, blocks and
SU(2) subgroups backwards) with probability 1/2, which makes every protocol kernel self-adjoint.
Both default off; the swap work and acceptance remain exact in either case.

The REX step is dispatch-bound on the GPU (about 120k kernel launches per step at 8^4, batch 64).
`--cuda-graphs` captures and replays CUDA graphs of the HBOR sweeps (equilibrium and protocol)
and of the defect smearing layer's map and inverse; `test/test_cuda_graphs.py` (GPU only,
`job/debug_gpu_tests.sh`) compares replay with eager execution. A failed capture leaves the
CUDA random generator unusable, so the captured code must be sync-free by construction; with
NCCL, export `NCCL_ASYNC_ERROR_HANDLING=0` (job/debug_neorex_eval.sh does it when the flag is
passed), because NCCL's cleanup thread polling its events during a capture aborts the process.
The replays gain on the smearing layer (map 22 to 5 ms, inverse 83 to 17 ms on an A100) but not
on the HBOR sweep, which is bound by GPU kernel latency rather than by dispatch. `--profile-steps N` writes a torch.profiler table of N
whole steps, split by phase, to `step_profile.txt` in the output directory.

## Classic SU(3) smearing Jacobian

Classic SU(3) smearing now uses the exact full-Haar log determinant. The old
runner multiplied its stored log determinant by two. The Jacobian is now
computed with the corrected structured formula and stable coefficients, and
its full-Haar log determinant is used once; the legacy extra factor of two has
been removed. Checkpoints trained with the old
classic-smearing convention are rejected; retrain them with the current code
or use a residual-flow checkpoint. This change does not alter the SU(2)
smearing convention or the residual-flow Jacobian. See the
[upstream integration inventory](doc/inventario_integrazione_runtime.md) for
the adapted changes and compatibility notes.

## Standalone NE-MCMC/SNF runs

`main/main_nemcmc_snf.py` supports full, progressive, and blockwise SNF
training. Blockwise training is the default; `--grad-acc N` accumulates N
prior-chain samples per optimizer step. Progressive training advances one
protocol block at a time; set `--epochs-per-block` to control its duration.
Full training differentiates through the complete stochastic flow and uses
more memory.

Use `--protocol-file PATH` to supply exactly `--protocol-steps` target values,
one per line. The last value must equal `--beta-final` (beta domain) or
`--bc-final` (defect domain); the starting value still comes from the
corresponding `*-initial` option. Evaluation writes intermediate work, heat,
log-Jacobian, action, plain plaquette, and plaquette-squared values to
`observables.csv`; `--int-meas` must divide the number of protocol steps.
After evaluation, rank 0 writes `gamma_analysis.csv` with Gamma-method errors,
error-of-error, integrated autocorrelation time, and window for every
intermediate measurement. It includes work, work variance, ΔF, ESS fraction,
heat, log-Jacobian, action, plain plaquette/plaquette², and their
work-reweighted values. The `work_sem` fields in `summary.json` and `metrics.csv`
remain pooled IID standard errors; `summary.json` also reports
`work_sem_gamma_final` and the Gamma report status. Fewer than five trajectories
per sample stream are reported as `insufficient_statistics` with errors and
autocorrelation fields left empty.

Reanalyze an existing output table with the same Gamma engine:

```sh
.sunenv/bin/python -m neoqcd.analysis results/run/observables.csv --skip 10 --S 2
```

This rewrites the sibling `gamma_analysis.csv` atomically; `--nmax` limits the
number of trajectories after `--skip`.

Training/evaluation checkpoints include model and optimizer state, the current
prior chain, random-number state, and output row counts. Resume an interrupted
run in the same output directory with the same run settings and a requested
total that is at least the saved progress:

```sh
.sunenv/bin/python main/main_nemcmc_snf.py --resume --output-dir results/run ...
```

SNF averages gradients across ranks before checking them. An optimizer update
with missing or non-finite gradients is skipped without changing its Adam or
scheduler state; the default limit stops training after five such consecutive
updates for the same optimizer. Set `--max-consecutive-nan-grad-updates N` to
change the positive limit. Per-optimizer statuses and streaks are recorded in
`metrics.csv`, run metadata, summaries, and resumable checkpoints; skipped
block contributions remain separate diagnostics.

`--load-prior --checkpoint TRAIN/snf_checkpoint.pt` starts a new evaluation
output while continuing that checkpoint's rank-local prior chains and random
streams. Exact resume uses `--resume`; it requires the same training/evaluation
semantics and distributed layout. `--transfer-rho-functional` can transfer the
shared time-dependent smearing network across protocol lengths when its
architecture and tensor shapes match; training transfer also requires
`--reset-optimizer`.

NEO-REX flow `.pt` checkpoints use a self-contained format 2 with a signature
of the settings that change the map; mismatches are rejected before weights
are loaded. Changing `nstep` is allowed only when
`--hyper-no-normalize-by-nstep` is set. Legacy hyper-smearing or residual-flow
checkpoints without a verifiable signature must be re-saved with their original
configuration; do not infer missing settings. Classic-smearing checkpoints
with the old Jacobian convention must be retrained.

Defect runs can add nested MCMC boxes with `--hierarchical-levels N`, one count
per level via `--hierarchical-updates U1 ... UN`, and optional decreasing box
buffers via `--hierarchical-buffers B1 ... BN`. When omitted, buffers default
to `N+1, ..., 2`. Buffers must fit around the defect inside the lattice.
Production defect job scripts accept the same options through the
`HIERARCHICAL_LEVELS`, `HIERARCHICAL_UPDATES`, and `HIERARCHICAL_BUFFERS`
environment variables. Set `CONTINUE_PRIOR=0` in an SNF job to start its
evaluation chain from the configured cache instead of the training checkpoint.

The initial thermalized batch cache remains separate from the persistent chain
stored in resumable checkpoints. `--prior-hot-start` initializes that thermal
chain with exact Haar-random links.
