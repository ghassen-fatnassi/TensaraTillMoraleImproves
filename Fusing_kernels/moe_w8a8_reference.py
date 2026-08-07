"""
Raw-math PyTorch reference for `moe_w8a8.cu`
(`fused_moe_w8a8_wgmma_up_down_acc_kernel`).

This is the *pure math* the kernel computes, in full precision. Everything in the
.cu that is there only to go fast on Hopper tensor cores has been removed, because
none of it changes the result -- it only changes speed/accuracy trade-offs:

    FP8 (e4m3) storage + block/per-token scales -> just a fast, lossy way to do
        the same matmuls on tensor cores. Mathematically it approximates the
        full-precision matmul; here we simply do the full-precision matmul.
    TMA / WGMMA / mbarriers / warp specialization / swizzle / ldmatrix / stmatrix
        -> data movement, tiling and pipelining. No effect on the result.

What's left is a textbook SwiGLU Mixture-of-Experts feed-forward block with a
top-k accumulate. For each routed token and each of its top-k experts:

    up_full = x @ w1[e]^T            # [2*I]   (gate and up, concatenated)
    gate, up = up_full[:I], up_full[I:]
    h       = silu(gate) * up        # [I]     SwiGLU
    down    = h @ w2[e]^T            # [d]
    out[token] += topk_weight * scaling_factor * down

--------------------------------------------------------------------------------
Shapes (names as in the kernel)
--------------------------------------------------------------------------------
    E   = num_experts
    d   = model hidden size          (== K in the kernel; x is [M, d])
    I   = expert intermediate size   (== N/2; the gate/up projection makes 2*I)

    x   : [M, d]        token activations
    w1  : [E, 2*I, d]   fused gate+up weight per expert  (kernel's "w")
    w2  : [E, d,   I]   down weight per expert           (kernel's "w2")
    out : [M, d]

The kernel does the final accumulate as a scatter-add (reduce) because the same
output row is written by all top_k experts; here that is just `+=`.

Gate/up layout: the kernel produces f_acc[..][0]=gate, f_acc[..][1]=up, i.e.
w1 = concat(gate_rows, up_rows) along the output dim. We assume w1[:, :I] = gate,
w1[:, I:] = up. If your checkpoint interleaves them differently, reorder w1's rows.

This file is written to be read and to serve as a correctness oracle -- it loops
over tokens x experts in the clearest way, not the fastest.
"""

from __future__ import annotations

import torch


def silu(x: torch.Tensor) -> torch.Tensor:
    # swiglu_mul in the kernel: x / (1 + exp(-x)) == x * sigmoid(x) == silu(x).
    return x * torch.sigmoid(x)


def fused_moe_reference(
    x: torch.Tensor,              # [M, d]      token activations
    w1: torch.Tensor,             # [E, 2*I, d] fused gate+up weight per expert
    w2: torch.Tensor,             # [E, d, I]   down weight per expert
    topk_ids: torch.Tensor,       # [M, top_k]  expert chosen for each (token, slot)
    topk_weights: torch.Tensor,   # [M, top_k]  routing weight for each (token, slot)
    scaling_factor: float = 1.0,
) -> torch.Tensor:
    """Full-precision SwiGLU MoE FFN. Returns out [M, d].

    Per (token, top-k slot):
        1. up_full = x @ w1[e]^T            -> [2*I]
        2. gate, up = up_full[:I], up_full[I:]
        3. h    = silu(gate) * up           -> [I]      (SwiGLU)
        4. down = h @ w2[e]^T               -> [d]
        5. out[token] += topk_weight * scaling_factor * down
    """
    M, d = x.shape
    E, two_I, d_w = w1.shape
    assert d_w == d, f"w1 last dim {d_w} != x hidden {d}"
    I = two_I // 2
    top_k = topk_ids.shape[1]

    out = torch.zeros(M, d, dtype=x.dtype, device=x.device)

    for token in range(M):
        a = x[token]                                  # [d]
        for slot in range(top_k):
            e = int(topk_ids[token, slot])
            weight = float(topk_weights[token, slot])

            # ---- GEMM 1: gate+up projection ------------------------------
            up_full = a @ w1[e].T                     # [2*I]
            gate, up = up_full[:I], up_full[I:]       # each [I]

            # ---- SwiGLU --------------------------------------------------
            h = silu(gate) * up                       # [I]

            # ---- GEMM 2: down projection ---------------------------------
            down = h @ w2[e].T                        # [d]

            # ---- route-weight & accumulate over top_k --------------------
            out[token] += (weight * scaling_factor) * down

    return out


# --------------------------------------------------------------- tiny self-test
if __name__ == "__main__":
    torch.manual_seed(0)

    M = 4          # tokens
    d = 256        # hidden size
    I = 256        # intermediate size
    E = 3          # experts
    top_k = 2

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randn(M, d, device=dev)
    w1 = torch.randn(E, 2 * I, d, device=dev) * (d ** -0.5)
    w2 = torch.randn(E, d, I, device=dev) * (I ** -0.5)

    topk_ids = torch.randint(0, E, (M, top_k), device=dev)
    topk_weights = torch.softmax(torch.randn(M, top_k, device=dev), dim=1)

    out = fused_moe_reference(x, w1, w2, topk_ids, topk_weights, scaling_factor=1.0)
    print("out shape:", tuple(out.shape), "dtype:", out.dtype)
    print("out[0, :6]:", out[0, :6].tolist())
    print("finite:", bool(torch.isfinite(out).all()))
