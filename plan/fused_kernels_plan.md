# Fused CUDA kernels for HBOR and smearing (branch `fused-kernels`)

Agreed with the user on 2026-11-17. Branch from main 3eecc1b. The branch `fermions` is left alone:
it will be aligned by its own agents after this branch is merged into main.

## Source
snf_SUN, branch `perf-main` at a3bfc7b (`snf_code/fused.py`, `snf_code/fused_smearing.py`): plain
CUDA C compiled with nvcc into a shared library, called through ctypes with raw device pointers on
the current torch stream; the same source compiled with the host C++ compiler is a CPU emulation, so
the equality tests run on a laptop. Timings there (one A100, probably batch 8): HBOR update (1 HB + 4
OR) 8^4-16^4 121-451 ms -> 2.7-16.6 ms; smearing layer without gradients, 16^4, batch 8, 238 -> 9.3
ms; forward + backward 965 -> 131 ms.

Files here keep the snf_SUN names, so that they can be diffed file by file:
`src/neoqcd/fused.py` (build and load machinery + HBOR kernel), `src/neoqcd/fused_smearing.py`,
`test/test_fused.py`, `test/test_fused_smearing.py`.

## Adaptations
HBOR:
1. per-chain defect parameter (O-REX batch): link factors kappa of shape (B, D, V), not (D, V);
2. reverse sweep of the symmetric updates: OR sweeps then HB, blocks backwards, SU(2) subgroups in
   the order 2, 1, 0;
3. protocol patch: the kernel reads one list of sites per mask (snf_SUN uses it for the hierarchical
   boxes); check the in-place condition on the patch masks;
4. re-unitarization of the updated links as in `HBOR._sweep`; same random numbers
   (B, 18, D, T, L, L, L); CUDA graphs not needed on the fused path (~40 launches per sweep).
Smearing (NEO-REX smearing class):
5. signed rho (snf_SUN applies |rho| in Python before the kernel);
6. rho from the hypernetwork, one per link and staple: pass it as the per-site, per-staple field;
7. NEW: inverse kernel (fixed point, as `Smearing.inverse` with fixed iterations; staples frozen per
   mask, iterations in registers) and its log det;
8. later, optional: fused backward for the PG training (gradient per site of rho, then autograd to
   the hypernetwork).
Not covered: the residual layer (no kernel in snf_SUN).

## Phases
0. Baseline profiles at 16^4 on main 3eecc1b (debug queue): O-REX and NEO-REX smearing class in the
   production configuration of phase A, and a few epochs of PG class training.
1. Fused HBOR behind `--fused-kernels` (off by default; torch fallback with a warning; torch in grad
   mode). Tests: equality with the torch HBOR to ~1e-12 per link in the CPU emulation (Mac) and on
   CUDA (Leonardo): whole lattice and patch, with and without defect, scalar and per-chain parameter,
   forward and reverse. Validation: O-REX 8^4 physics against 64624468 (Crooks included), 16^4
   profile against 64624470 (4.04 s/step) and the phase-0 baseline.
2. Fused smearing forward + log det and inverse for NEO-REX smearing class. Tests: forward, inverse,
   log det equal to torch to ~1e-14, inverse(forward) = identity. Validation: NEO-REX 8^4 physics
   (acceptance ~0.62, work ~0.54 at K=2) and 16^4 timing.
3. New profile; decide what is left (swap step, bookkeeping, residual layer, training backward, CUDA
   graphs, torch 2.2 of cineca-ai/4.3.0).
4. Optional: PG training with the fused backward.

## Rules
- One commit per completed step on this branch, pushed (Leonardo pulls from GitHub). No merge into
  main without the user's word.
- Unit tests only on the Mac. Everything else on Leonardo's debug queue, from a separate worktree with
  `--chdir=<worktree>` and `PYTHONPATH=<worktree>/src` (the venv's editable install points at the
  production checkout's src). Production checkout untouched until the merge.
- Fused and torch paths are statistically, not bitwise, equivalent.

## Results (2026-11-17, head 15684db)
All on Leonardo (A100, cineca-ai 4.1.1, torch 2.0), 16^4, defect size 3, 8 ranks, batch 64,
K = 8, h = 1, the production configuration of phase A; s/step at steady state.

| | main 3eecc1b | fused-kernels | gain |
|---|---|---|---|
| O-REX | 5.27 | 0.394 | 13x |
| NEO-REX smearing class | 6.13 | 0.528 | 11.6x |

How the time came down (NEO-REX smearing class, s/step):
- 6.13: main.
- 1.58: fused HBOR (phase 1). The equilibrium sweep went 3.49 -> 0.215.
- 1.26: fused stout step and inverse (phase 2).
- 1.13: S_halo without host syncs (3f360a3).
- 0.74: protocol box (1215a72, `--protocol-box`).
- 0.61: fewer syncs and S_halo from gather tables (92946f3, 7a28c7f).
- 0.53: fused plaquette for the replica history (15684db).

Phase 3, what was added:
- `--protocol-box` (utils.ProtocolBox). The protocol of a swap runs on a box around the defect: the
  patch with a one-site frame, the cut of the flow and the halo of the action change, which is
  6 x 9^3 sites at 16^4. The box is cut once per swap and written back into the accepted chains.
  With the same random numbers it equals the whole-lattice protocol to 1e-12 (test/test_protocol_box.py).
- `fused.plaquette` (snf_plaquette in the HBOR library). It is used for the replica-history plaquette
  with `--fused-kernels`.
- Host syncs removed:
  - S_halo bounding boxes;
  - HBOR branch choice;
  - down.any() per protocol step;
  - the CPU zero in DeferredChecks.resolve.
  S_halo now evaluates all halo plaquettes from gather tables.
- The smearing graph captures the map and its inverse together, so no capture lands mid-run.
- NEO-REX evaluation calls the flow without DDP; load_flow now broadcasts the buffers too, which it
  did not before.
- `--replica-history-save-minutes` (default 10). The history was recompressed every log step, a cost
  that grew linearly with the run: +50% per step at 3000 steps.
- `--profile-python`: cProfile of the profiled steps; record_function labels for the parts of a swap.
- Fixes found on the way:
  - Wilson_action computed float32 link factors for an integer defect mask;
  - DefectBox cut the defect mask with the indexing of a configuration;
  - plaquette_SUN rolled the whole configuration tensor.

Validation (fused + box against torch on main, same seeds):
- O-REX 8^4 K = 2, 3000 steps: work z = -0.10; Crooks chi2 5.1/7.
- NEO-REX smearing class 8^4 K = 2, 1000 steps: every pair agrees; Crooks max 1.5 sigma.
- 16^4 runs:
  - O-REX: links bit-identical along the phase 3 commits.
  - NEO-REX: the work changes at ~1e-11 from the summation order of S_halo; acceptance, labels and
    plaquettes are identical.
- GPU tests: 44 OK.

Left out (diminishing returns):
- the CUDA graph of the protocol MCMC on the box (O-REX, ~-0.05 s/step);
- resolving the smearing graph checks once per swap instead of 32 times. NEO-REX ~-0.1 s/step; a
  failed check must then repeat the swap with the same random numbers.

The equilibrium sweep (0.215 s, the fused HBOR kernel on the whole lattice, near memory bandwidth)
is now 40-55% of the step. Production use: add `--fused-kernels --protocol-box --cuda-graphs`.
