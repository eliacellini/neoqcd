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
