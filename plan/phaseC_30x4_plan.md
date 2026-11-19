# Phase C: frozen-regime study at 30^4, beta 6.4

Plan of 2026-11-18. Phase C of `neorex_paper_plan.md`, postponed on 2/11 and revived on 17/11 after the
fused kernels. One volume only: one large volume is enough for an SU(3) paper; the continuum limit is
decided later.

**Design in one line:** test 5 patches with O-REX at one acceptance, with the model's optimal
parameters; take the best patch; then one-factor scans in n_rep and n_step for O-REX and NEO-REX, and in
n_rep for REX. Every run measures tau_int(Q^2) and is long enough for the standard tau.

## Lattice and references
- SU(3) Wilson, beta 6.4, L/a = 30, defect L_d = 3 (time slice 15, space slice 13).
  a/sqrt(8 t0) = 0.109, L ~ 1.55 fm. This is the coarsest lattice of the SNF paper (arXiv:2610.20747).
- Standard algorithm (1 HB + 4 OR): tau_int(Q^2) ~ 575 updates (SNF paper).
- SNF at the same point: tau_int(Q^2) = 1.3 samples, n_step 200, n_between 100, ESS 0.40 (SNF paper).
- Thermodynamic length from the timing job 64705096 (rough): Lambda = 21.7 +- 0.2 (16^4, beta 6.0,
  L_d 3: 20.4).
- Work: O-REX n8 K8 h1, <W> = 1.371 +- 0.029, A = 0.40 (64705096, 30 steps); at 16^4 it was 1.31. The
  ratio 1.05 is close to (21.7/20.4)^2 = 1.13.

## Measured cost (main 0b66eab, `--fused-kernels --protocol-box --cuda-graphs`, batch 16 per GPU)
- Equilibrium sweep (1 HB + 4 OR): 0.76 s; bookkeeping 0.07 s per step.
- Protocol: 10.4 ms per protocol step on the box (O-REX K8: 0.90 s/step).
- Smearing-class flow: 12.5 ms more per protocol step (NEO-REX K8: 0.99 s/step).
- Q (60 Wilson cooling steps + clover charge): 0.42 s per chain; 1-3.5 s per measurement on the
  periodic replica (`--topology-replicas top`), set by how its chains are spread over the GPUs.
- Maximum batch 24 per GPU at 30^4; 20 is safe.

## Methods and metrics
- Methods: REX (PT on the defect coupling); O-REX; NEO-REX smearing class, with a flow trained at 30^4,
  beta 6.4, L_d 3.
- Primary observable: tau_int(Q^2) of the periodic replica, in REX steps.
- Cost per effective sample of Q^2, in two forms:
  1. in updates: C x 2 tau_int(Q^2) / B, with C the cost per REX step of `neorex_paper_plan.md` (units
     of one equilibrium update of one replica) and v = V_patch / V fixed by the patch;
  2. in GPU time: t_step x n_rep x 2 tau_int(Q^2) / B.
  B is the number of independent chains on the periodic rung, i.e. the batch.
- The round trip is not a proxy for tau_int(Q^2).
- Lambda of REX comes from the PG training: its swap acceptances per edge give 2 erfc^-1(A) per link.
  The Lambda of NEO-REX (from the PINN) is Roberto's.

## Model and centre of the scans
- REX on the optimal ladder: A = erfc(Lambda / (2 (n_rep - 1))).
- O-REX and NEO-REX: <W> = k_h V_def / ((n_rep - 1)^2 n_step^alpha), A = erfc(sqrt(<W>) / 2). At 16^4
  alpha = 0.901 (O-REX) and 0.839 (class), and the (n_rep - 1)^2 factorization holds (n4, n6, n8).
- Cost per REX step: C = n + 2 (n-1) [(K-1) v + K v_NF]: the protocol updates the patch at the K - 1
  interior points only (merge of interior-trex, 19/11; before, at the K + 1 points with both ends, which
  left the acceptance unchanged). O-REX at K = 1 is REX, C = n. At fixed acceptance the O-REX minimum
  is at K* = alpha (1 - 2v) / (2 v (2 - alpha)) ~ alpha / (4 (1 - alpha/2) v) for v << 1, independent
  of A. The numbers below come from the K + 1 model: at these K they change by at most a few percent
  (K* ~ 30 -> 29.5 in GPU time; n4 K50 costs 2% less).
- O-REX:
  - in updates, v = V_patch / V = 0.0017 (h = 1): K* ~ 240, n_rep 2-3;
  - in GPU time, v = 10.4 ms / 0.76 s = 0.0135 (the box kernels are launch-bound, 8x V_patch / V):
    K* ~ 30, n_rep 4-5;
  - the minimum is flat: **n4 K50 at A = 40%** is the GPU-time optimum (5.4 GPU-s per step) and 16%
    above the optimum in updates (n3 K125).
- NEO-REX class: the flow dominates the protocol cost, K* ~ 20 in updates and ~12 in GPU time:
  **n4 K18 at A = 40%**. Its W at 30^4 is predicted from the 16^4 class/O-REX ratio (0.51 at n8 K8)
  until the flow of C1 is trained; the NEO-REX K below are recomputed then.

## Prerequisites
- [x] Measurement of Q: Wilson cooling (60 steps, as in the SNF paper) and clover charge, kernels and
      torch (main b6ade22): `--topology-every`, `--cooling-steps`, `--topology-replicas top|all`.
- [x] Resume of interrupted runs (main 0b66eab): `--save-resume`, `--resume`, `--resume-dir`
      (default `$SCRATCH/neoqcd_resume`); bitwise equal to an uninterrupted run on GPU.
- [x] PG training with `--fused-kernels` validated (debug job 64772900, 18/11): 173 epochs against the
      reference 63254793 with the same learning-rate curve; PINN, loss, swap acceptance per edge and
      gradient norm agree within noise; 3.9 s per epoch against 9.1 (sweep 0.94 s against 6.08).
- [ ] NEO-REX flow at 30^4, beta 6.4 (C1).
- Roberto pulls main into his own directory for the runs; the Leonardo production checkout is at main
  c0e2389 (18/11).

## Run settings (all runs)
- **Length:** ~60k steps. One REX step is one standard update (1 HB + 4 OR) of every replica, so on the
  periodic replica tau_int(Q^2) <= ~575 steps, and 100 x 575 steps are enough in every case.
- **Q every 10 steps** on the periodic replica (+10-35% of the step); resolves tau >= ~50. A run with
  tau < 50 (a gain > 10 on the standard) gets a short dedicated run with Q every 2 steps.
- **24-hour jobs with `--save-resume`;** runs longer than that continue with `--resume`.
- **Early stop:** the Q history is written along the run (`--replica-history-save-minutes`); a run that
  already has >= 100 tau can be stopped.
- **No short runs without tau:** A, W, k_h, alpha and Lambda come from the same runs.
- **Predictions registered before the runs** (acceptances, as in phase A: `plan/submitted_*.txt`).
- With B = 16 chains, 1,000 effective samples need N ~ 125 tau steps: 6k steps if tau ~ 50, 72k if
  tau ~ 575.

## Stages
- **C1. Thermalization and PG training** at 30^4, beta 6.4, on a ladder from Lambda = 21.7. Gives Lambda
  (REX acceptances and ladder) and the NEO-REX flow. Runs in parallel with C2, which needs no flow.
- **C2. Patch test, O-REX,** n4 K50, h = 0-4. Choose h* by the cost per effective sample (acceptance,
  decorrelation of Q and cost of the patch update together). Gives the first tau; W and the K of C3-C6
  are recalibrated with it.
- **C3. O-REX in n_step,** n4, h*: K = 20, 32, 83 (with K50 of C2: A = 20, 30, 40, 50%). Gives alpha at
  30^4.
- **C4. O-REX in n_rep,** K50, h*: n_rep = 3, 6, 8 (with n4: A = 20, 40, 61, 72%).
- **C5. NEO-REX in n_step,** n4, h*, flow of C1: K = 7, 11, 18, 30 (A = 20, 30, 40, 50%).
- **C6. NEO-REX in n_rep,** K18, h*: n_rep = 3, 6, 8 (with n4: A = 21, 40, 61, 72%).
- **C7. REX in n_rep:** n_rep = 12, 16, 20, 24 (A = 16, 31, 42, 50% for Lambda = 21.7, refined after
  C1); even n_rep on whole nodes.

C3-C7 start together once C1 and C2 are done. Optional: O-REX n3 K125 (the optimum in updates,
A = 40%); NEO-REX with the neighbour of h* at the centre (the best patch could change with the flow).

## Runs
Worst case: 60k steps, Q every 10 steps (x1.2). Predicted A from the model above; s/step for h = 1
(h >= 2 a little more: larger box). Node-hours counted per GPU (4 GPUs per node).

| stage | method | n_rep | K | h | A pred | s/step | wall h | GPUs | node-h |
|---|---|---|---|---|---|---|---|---|---|
| C2 | O-REX | 4 | 50 | 0 | 0.40 | 1.35 | 27 | 4 | 27 |
| C2 | O-REX | 4 | 50 | 1 | 0.40 | 1.35 | 27 | 4 | 27 |
| C2 | O-REX | 4 | 50 | 2 | 0.40 | 1.35 | 27 | 4 | 27 |
| C2 | O-REX | 4 | 50 | 3 | 0.40 | 1.35 | 27 | 4 | 27 |
| C2 | O-REX | 4 | 50 | 4 | 0.40 | 1.35 | 27 | 4 | 27 |
| C3 | O-REX | 4 | 20 | h* | 0.20 | 1.04 | 21 | 4 | 21 |
| C3 | O-REX | 4 | 32 | h* | 0.30 | 1.16 | 23 | 4 | 23 |
| C3 | O-REX | 4 | 83 | h* | 0.50 | 1.70 | 34 | 4 | 34 |
| C4 | O-REX | 3 | 50 | h* | 0.20 | 1.35 | 27 | 3 | 20 |
| C4 | O-REX | 6 | 50 | h* | 0.61 | 1.35 | 27 | 6 | 41 |
| C4 | O-REX | 8 | 50 | h* | 0.72 | 1.35 | 27 | 8 | 54 |
| C5 | NEO-REX | 4 | 7 | h* | 0.21 | 0.99 | 20 | 4 | 20 |
| C5 | NEO-REX | 4 | 11 | h* | 0.30 | 1.08 | 22 | 4 | 22 |
| C5 | NEO-REX | 4 | 18 | h* | 0.40 | 1.24 | 25 | 4 | 25 |
| C5 | NEO-REX | 4 | 30 | h* | 0.50 | 1.52 | 30 | 4 | 30 |
| C6 | NEO-REX | 3 | 18 | h* | 0.21 | 1.24 | 25 | 3 | 19 |
| C6 | NEO-REX | 6 | 18 | h* | 0.61 | 1.24 | 25 | 6 | 37 |
| C6 | NEO-REX | 8 | 18 | h* | 0.72 | 1.24 | 25 | 8 | 50 |
| C7 | REX | 12 | - | - | 0.16 | 0.83 | 17 | 12 | 50 |
| C7 | REX | 16 | - | - | 0.31 | 0.83 | 17 | 16 | 67 |
| C7 | REX | 20 | - | - | 0.42 | 0.83 | 17 | 20 | 83 |
| C7 | REX | 24 | - | - | 0.50 | 0.83 | 17 | 24 | 100 |

Layout: one rung per GPU; n_rep = 3 and 6 run with 3 GPUs per node; REX uses even n_rep on whole nodes
(12: 3 nodes, 16: 4, 20: 5, 24: 6).

## Budget (worst case)

| stage | runs | node-h |
|---|---|---|
| C1 thermalization + PG training | 1-2 | 10-20 (timed after 64706032) |
| C2 O-REX patch | 5 | 135 |
| C3-C4 O-REX scans | 6 | 193 |
| C5-C6 NEO-REX scans | 7 | 203 |
| C7 REX | 4 | 300 |
| **total** | **23-24** | **~850** |

~850 node-hours = ~27,000 local hours (32 per node-hour), ~3% of the allocation; less if runs are
stopped early. All of it in the production queue; the debug queue only for tests.

## Decisions (user, 17-18/11)
- tau_int(Q^2) is measured directly; the round trip is not a proxy.
- Lambda comes from the training; v is fixed by the patch geometry, not measured.
- The patch is chosen with O-REX at one acceptance and reused for NEO-REX.
- NEO-REX is part of the plan (scans as for O-REX); the NEO-REX Lambda from the PINN is Roberto's.
- No short runs without tau; every run sized for the standard tau (~60k steps).
- One rung per GPU; target acceptances 20, 30, 40, 50%.
- REX with even n_rep on whole nodes: 12 instead of 13.
- Both cost metrics (updates and GPU time).

## Out of scope for now
Continuum limit (other lattice spacings: later), residual layer, clover-chair.
