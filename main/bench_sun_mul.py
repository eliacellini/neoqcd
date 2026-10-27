"""Time 3x3 complex link products on GPU: forward and backward of a staple, per formulation.

A staple is U_a U_b^dagger U_c^dagger, as in sun_utils.pstaple. Each formulation computes the
same product; the backward is that of a scalar loss Re sum(conj(C) * staple), as in the PG loss.
"""

import argparse
import json

import torch


def mul_select(U, V):
    # The current CUDA branch of sun_utils.SUN_mul.
    W = U[..., :, 0, None] * V[..., None, 0, :]
    for j in range(1, U.shape[-1]):
        W = torch.addcmul(W, U[..., :, j, None], V[..., None, j, :])
    return W


def mul_unbind(U, V):
    # Same products and order; the columns and rows are taken with one unbind each,
    # whose backward is one stack instead of a zero-filled tensor per index.
    cols, rows = U.unbind(-1), V.unbind(-2)
    W = cols[0].unsqueeze(-1) * rows[0].unsqueeze(-2)
    for j in range(1, U.shape[-1]):
        W = torch.addcmul(W, cols[j].unsqueeze(-1), rows[j].unsqueeze(-2))
    return W


def mul_broadcast(U, V):
    return (U.unsqueeze(-1) * V.unsqueeze(-3)).sum(-2)


def mul_matmul(U, V):
    return torch.matmul(U, V)


VARIANTS = dict(select=mul_select, unbind=mul_unbind, broadcast=mul_broadcast, matmul=mul_matmul)


def dagger(U):
    return U.conj().transpose(-2, -1)


def staple(mul, A, B, C):
    return mul(mul(A, dagger(B)), dagger(C))


def time_variant(mul, A, B, C, G, repeats):
    def step():
        A.grad = B.grad = C.grad = None
        W = staple(mul, A, B, C)
        (W * G.conj()).real.sum().backward()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats()
    start.record()
    for _ in range(repeats):
        step()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats, torch.cuda.max_memory_allocated() / 2**30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, nargs="+", default=[8, 64])
    parser.add_argument("--L", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    torch.manual_seed(0)
    for L in args.L:
        for batch in args.batch:
            shape = (batch, L, L, L, L, 3, 3)
            if batch * L**4 * 9 * 16 > 2**31:  # keep each tensor below 2 GB
                continue
            A, B, C, G = (torch.randn(shape, dtype=torch.cdouble, device="cuda") for _ in range(4))
            for X in (A, B, C):
                X.requires_grad_(True)
            reference = staple(mul_matmul, A, B, C).detach()
            for name, mul in VARIANTS.items():
                error = (staple(mul, A, B, C).detach() - reference).abs().max().item()
                ms, memory = time_variant(mul, A, B, C, G, args.repeats)
                print(json.dumps(dict(L=L, batch=batch, variant=name, ms=round(ms, 3),
                                      peak_gb=round(memory, 2), max_error=error)), flush=True)
            del A, B, C, G, reference
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
