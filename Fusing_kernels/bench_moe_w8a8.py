"""
Benchmark + correctness harness for the raw-math MoE reference
(`moe_w8a8_reference.fused_moe_reference`).

WHY NOT COMPARE AGAINST THE .cu KERNEL HERE
-------------------------------------------
`moe_w8a8.cu` uses WGMMA / TMA / setmaxnreg, which only exist on sm_90 (Hopper).
This machine is an RTX 2050 (sm_86), so the kernel cannot launch here at all, and
the repo has no host/binding to call it through. What we validate + time instead
is the full-precision math itself:

  1. CORRECTNESS: the readable loop reference vs. a fully *vectorized* (batched)
     implementation of the identical math. No fp8 is involved, so these should
     agree to floating-point tolerance (~1e-5). Agreement proves the loop version
     computes the right thing; disagreement means a real bug.

  2. SPEED: time both. The loop version is the oracle (clear, slow); the batched
     version shows what the same math costs when expressed for the GPU. This is
     the meaningful speed comparison available without Hopper.

If you move to a Hopper GPU: bind the .cu and set CUDA_KERNEL to a callable with
the same signature as `run_reference`; the script will then also compare + time it.
"""

from __future__ import annotations

import argparse
import time

import torch

from moe_w8a8_reference import fused_moe_reference, silu
from moe_unfused import moe_unfused, split_fused_weights

# Set to callable(x, w1, w2, topk_ids, topk_weights, scaling_factor) -> [M, d]
# to also benchmark a real kernel (e.g. a bound Hopper .cu). None on non-Hopper.
CUDA_KERNEL = None


# --------------------------------------------------------------------- inputs
def make_inputs(M, d, I, E, top_k, device, dtype=torch.float32, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(M, d, generator=g, device=device, dtype=dtype)
    w1 = torch.randn(E, 2 * I, d, generator=g, device=device, dtype=dtype) * (d ** -0.5)
    w2 = torch.randn(E, d, I, generator=g, device=device, dtype=dtype) * (I ** -0.5)
    topk_ids = torch.randint(0, E, (M, top_k), generator=g, device=device)
    topk_weights = torch.softmax(
        torch.randn(M, top_k, generator=g, device=device, dtype=dtype), dim=1
    )
    return dict(x=x, w1=w1, w2=w2, topk_ids=topk_ids, topk_weights=topk_weights)


def run_reference(inp, scaling_factor=1.0):
    """The readable loop-over-(token, expert) oracle (fused gate+up matmul)."""
    return fused_moe_reference(scaling_factor=scaling_factor, **inp)


def run_unfused(inp, scaling_factor=1.0):
    """Non-fused MoE: separate gate/up/down projections, per-expert grouped GEMMs.

    Splits the fused w1/w2 into gate_proj/up_proj/down_proj so it is numerically
    identical to the fused path, then dispatches tokens expert-by-expert.
    """
    gate_proj, up_proj, down_proj = split_fused_weights(inp["w1"], inp["w2"])
    return moe_unfused(
        inp["x"], gate_proj, up_proj, down_proj,
        inp["topk_ids"], inp["topk_weights"], scaling_factor=scaling_factor,
    )


# ------------------------------------------------------ vectorized same-math
def run_batched(inp, scaling_factor=1.0):
    """Identical math, expressed without Python loops.

    Flatten (token, slot) into one batch of routed rows, gather each row's expert
    weights, do the two GEMMs as batched matmuls, then scatter-add the top_k
    contributions back into the output. Same result as run_reference, GPU-shaped.
    """
    x = inp["x"]; w1 = inp["w1"]; w2 = inp["w2"]
    topk_ids = inp["topk_ids"]; topk_weights = inp["topk_weights"]

    M, d = x.shape
    E, two_I, _ = w1.shape
    I = two_I // 2
    top_k = topk_ids.shape[1]

    # One row per (token, slot). [M*top_k, d]
    rows = x.repeat_interleave(top_k, dim=0)                     # [M*top_k, d]
    exp = topk_ids.reshape(-1)                                   # [M*top_k]
    wts = topk_weights.reshape(-1)                               # [M*top_k]

    W1 = w1[exp]                                                 # [M*top_k, 2*I, d]
    W2 = w2[exp]                                                 # [M*top_k, d,   I]

    # GEMM 1: (rows[:,1,d] @ W1^T[:,d,2I]) -> [M*top_k, 2*I]
    up_full = torch.bmm(rows.unsqueeze(1), W1.transpose(1, 2)).squeeze(1)
    gate, up = up_full[:, :I], up_full[:, I:]
    h = silu(gate) * up                                         # [M*top_k, I]

    # GEMM 2: (h[:,1,I] @ W2^T[:,I,d]) -> [M*top_k, d]
    down = torch.bmm(h.unsqueeze(1), W2.transpose(1, 2)).squeeze(1)
    down = down * (wts.unsqueeze(1) * scaling_factor)           # [M*top_k, d]

    # scatter-add each slot back to its token
    out = torch.zeros(M, d, dtype=x.dtype, device=x.device)
    token_of_row = torch.arange(M, device=x.device).repeat_interleave(top_k)
    out.index_add_(0, token_of_row, down)
    return out


# ------------------------------------------------------------------- timing
def timed(fn, *, warmup, iters, cuda):
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms/iter


def rel_err(a, b):
    denom = b.abs().amax().clamp_min(1e-12)
    return ((a - b).abs().amax() / denom).item()


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--I", type=int, default=256)
    ap.add_argument("--E", type=int, default=4)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--scaling_factor", type=float, default=1.0)
    args = ap.parse_args()

    dev = torch.device(args.device)
    cuda = dev.type == "cuda"
    if cuda:
        print(f"GPU: {torch.cuda.get_device_name(dev)}  (torch {torch.__version__}, CUDA {torch.version.cuda})")
    print(f"Config: M={args.M} d={args.d} I={args.I} E={args.E} top_k={args.top_k} on {dev}\n")

    inp = make_inputs(args.M, args.d, args.I, args.E, args.top_k, dev)

    # ---- correctness: loop reference vs vectorized same-math --------------
    out_ref = run_reference(inp, args.scaling_factor)
    out_bat = run_batched(inp, args.scaling_factor)
    out_unf = run_unfused(inp, args.scaling_factor)

    err_bat = rel_err(out_bat, out_ref)
    err_unf = rel_err(out_unf, out_ref)
    finite = bool(
        torch.isfinite(out_ref).all()
        and torch.isfinite(out_bat).all()
        and torch.isfinite(out_unf).all()
    )
    ok = finite and err_bat < 1e-4 and err_unf < 1e-4
    print("CORRECTNESS  (vs fused loop reference; pure fp32, should match)")
    print(f"  vectorized vs reference : {err_bat:.3e}")
    print(f"  unfused    vs reference : {err_unf:.3e}")
    print(f"  all finite              : {finite}")
    print(f"  verdict                 : {'PASS' if ok else 'FAIL'}  (threshold 1e-4)\n")

    if CUDA_KERNEL is not None:
        out_cuda = CUDA_KERNEL(scaling_factor=args.scaling_factor, **inp)
        print("CORRECTNESS  (CUDA kernel vs reference)")
        print(f"  max rel err : {rel_err(out_cuda, out_ref):.3e}\n")

    # ---- speed ------------------------------------------------------------
    t_ref = timed(lambda: run_reference(inp, args.scaling_factor),
                  warmup=args.warmup, iters=args.iters, cuda=cuda)
    t_bat = timed(lambda: run_batched(inp, args.scaling_factor),
                  warmup=args.warmup, iters=args.iters, cuda=cuda)
    t_unf = timed(lambda: run_unfused(inp, args.scaling_factor),
                  warmup=args.warmup, iters=args.iters, cuda=cuda)
    print("SPEED")
    print(f"  fused loop reference    : {t_ref:10.3f} ms/iter")
    print(f"  fused vectorized        : {t_bat:10.3f} ms/iter")
    print(f"  unfused (per-expert)     : {t_unf:10.3f} ms/iter")
    print(f"  unfused vs fused-vec     : {t_unf / t_bat:10.2f}x  (>1 = unfused slower)")
    if CUDA_KERNEL is not None:
        t_cuda = timed(lambda: CUDA_KERNEL(scaling_factor=args.scaling_factor, **inp),
                       warmup=args.warmup, iters=args.iters, cuda=cuda)
        print(f"  CUDA kernel    : {t_cuda:10.3f} ms/iter")
    else:
        print("  CUDA kernel    : n/a (needs sm_90 Hopper; this GPU is sm_86)")


if __name__ == "__main__":
    main()
