"""
Non-fused, "textbook" PyTorch MoE FFN -- the canonical way you'd write this with
standard building blocks, for comparison against the fused reference in
`moe_w8a8_reference.py`.

Same math, different expression. Two axes of "un-fusing":

  1. WEIGHTS: gate and up are kept as *separate* projections (gate_proj, up_proj,
     down_proj), the way an nn.Module MoE expert is normally written -- instead of
     the fused `w1 = concat(gate, up)` single matmul the reference uses. We derive
     them from the same w1/w2 so the two implementations are numerically identical:
         gate_proj[e] = w1[e][:I]      up_proj[e] = w1[e][I:]      down_proj[e]=w2[e]

  2. DISPATCH: instead of looping token-by-token, group tokens by the expert they
     were routed to and run one dense matmul per expert (the standard grouped-GEMM
     MoE dispatch). This is what real non-fused MoE code does (HF / vLLM style):
     scatter tokens to experts, compute, scatter results back weighted by the
     routing weight.

The point of having this next to the fused reference: it's a second, independently
written path to the *same* result, so it doubles as a correctness cross-check, and
it shows the speed cost of expressing MoE as separate ops + per-expert grouping
rather than one batched routed computation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def split_fused_weights(w1: torch.Tensor, w2: torch.Tensor):
    """Turn the fused (w1, w2) into separate gate/up/down projections.

    w1: [E, 2*I, d]  ->  gate_proj [E, I, d], up_proj [E, I, d]
    w2: [E, d, I]    ->  down_proj [E, d, I]
    """
    E, two_I, d = w1.shape
    I = two_I // 2
    gate_proj = w1[:, :I, :].contiguous()   # [E, I, d]
    up_proj = w1[:, I:, :].contiguous()     # [E, I, d]
    down_proj = w2.contiguous()             # [E, d, I]
    return gate_proj, up_proj, down_proj


def moe_unfused(
    x: torch.Tensor,              # [M, d]
    gate_proj: torch.Tensor,      # [E, I, d]
    up_proj: torch.Tensor,        # [E, I, d]
    down_proj: torch.Tensor,      # [E, d, I]
    topk_ids: torch.Tensor,       # [M, top_k]
    topk_weights: torch.Tensor,   # [M, top_k]
    scaling_factor: float = 1.0,
) -> torch.Tensor:
    """Non-fused MoE FFN via per-expert grouped GEMMs.

    For each expert e:
        rows = all (token, slot) pairs routed to e
        g    = X_rows @ gate_proj[e]^T          (separate gate projection)
        u    = X_rows @ up_proj[e]^T            (separate up projection)
        h    = silu(g) * u
        y    = h @ down_proj[e]^T
        out[token] += weight * scaling_factor * y     (scatter-add)
    """
    M, d = x.shape
    E = gate_proj.shape[0]
    top_k = topk_ids.shape[1]

    # Flatten routing to one entry per (token, slot).
    flat_expert = topk_ids.reshape(-1)                                   # [M*top_k]
    flat_weight = topk_weights.reshape(-1)                               # [M*top_k]
    flat_token = torch.arange(M, device=x.device).repeat_interleave(top_k)  # [M*top_k]

    out = torch.zeros(M, d, dtype=x.dtype, device=x.device)

    for e in range(E):
        sel = (flat_expert == e).nonzero(as_tuple=True)[0]   # rows routed to expert e
        if sel.numel() == 0:
            continue
        tok = flat_token[sel]                # which token each selected row belongs to
        w = flat_weight[sel].unsqueeze(1)    # [n_e, 1] routing weight

        rows = x[tok]                        # [n_e, d]  gather activations
        # Separate gate & up projections (this is the "un-fused" part).
        g = rows @ gate_proj[e].T            # [n_e, I]
        u = rows @ up_proj[e].T              # [n_e, I]
        h = F.silu(g) * u                    # [n_e, I]
        y = h @ down_proj[e].T               # [n_e, d]
        y = y * (w * scaling_factor)

        out.index_add_(0, tok, y)            # scatter-add back to tokens
    return out


if __name__ == "__main__":
    # Standalone check against the fused reference.
    import torch as _t
    from moe_w8a8_reference import fused_moe_reference

    _t.manual_seed(0)
    M, d, I, E, top_k = 8, 256, 256, 3, 2
    dev = "cuda" if _t.cuda.is_available() else "cpu"

    x = _t.randn(M, d, device=dev)
    w1 = _t.randn(E, 2 * I, d, device=dev) * (d ** -0.5)
    w2 = _t.randn(E, d, I, device=dev) * (I ** -0.5)
    topk_ids = _t.randint(0, E, (M, top_k), device=dev)
    topk_weights = _t.softmax(_t.randn(M, top_k, device=dev), dim=1)

    gate_proj, up_proj, down_proj = split_fused_weights(w1, w2)
    a = fused_moe_reference(x, w1, w2, topk_ids, topk_weights)
    b = moe_unfused(x, gate_proj, up_proj, down_proj, topk_ids, topk_weights)
    print("max abs diff fused vs unfused:", (a - b).abs().max().item())
