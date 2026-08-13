import torch
import torch.nn as nn
import torch.functional as F

H     = 512        # attention out size
NE    = 32         # MoE experts
TK    = 4          # top-K
MI    = 256        # expert intermediate size
SI    = 256        # shared expert intermediate size

class SharedExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(H, SI, bias=False)
        self.up_proj   = nn.Linear(H, SI, bias=False)
        self.down_proj = nn.Linear(SI, H, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(NE, 2*MI, H))
        self.down_proj    = nn.Parameter(torch.empty(NE, H, MI))


class MoEFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts            = Experts()
        self.gate               = nn.Linear(H, NE, bias=False)
        self.shared_expert      = SharedExpert()
        self.shared_expert_gate = nn.Linear(H, 1, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        xf = x.reshape(-1, H)
        N  = xf.shape[0]

        w, idx = torch.topk(self.gate(xf), TK, -1)
        w = F.softmax(w, -1).to(x.dtype)

        flat_idx = idx.reshape(-1)
        flat_w   = w.reshape(-1)
        token_rep = xf.unsqueeze(1).expand(N, TK, H).reshape(N * TK, H)

        sort_order    = torch.argsort(flat_idx, stable=True)
        sorted_idx    = flat_idx[sort_order]
        sorted_tokens = token_rep[sort_order] #[5,2,8,12,1,3,4,6,7,9,...]
        sorted_weights = flat_w[sort_order]

        expert_counts  = torch.bincount(sorted_idx, minlength=NE)
        print(f"Expert counts: {expert_counts}")
        expert_offsets = torch.cat([torch.zeros(1, device=x.device, dtype=torch.long),
                                    expert_counts.cumsum(0)[:-1]])
        print(f"Expert offsets: {expert_offsets}") #basically it's a prefix sum over expert count
        sorted_out = torch.zeros(N * TK, H, device=x.device, dtype=x.dtype)
        for e in range(NE):
            cnt = expert_counts[e].item()
            if cnt == 0:
                continue
            start = expert_offsets[e].item()
            xt = sorted_tokens[start:start+cnt]
            gw, uw = self.experts.gate_up_proj[e].chunk(2, 0)
            h = F.silu(xt @ gw.t()) * (xt @ uw.t())
            h = h @ self.experts.down_proj[e].t()
            sorted_out[start:start+cnt] = sorted_weights[start:start+cnt].unsqueeze(-1) * h

        unsort_order = torch.argsort(sort_order, stable=True)
        out = sorted_out[unsort_order].reshape(N, TK, H).sum(dim=1)
        sg  = torch.sigmoid(self.shared_expert_gate(xf))
        return (out + sg * self.shared_expert(xf)).view(B, T, H)
