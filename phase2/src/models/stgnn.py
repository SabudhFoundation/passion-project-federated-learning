"""
STGNN encoder + swappable prediction head.

This is the Phase 1 architecture (GAT -> GCN -> GRU -> LayerNorm -> head)
re-expressed with dense adjacency instead of PyTorch Geometric.

Why dense instead of PyG
------------------------
Two reasons, both deliberate:

1. Portability. PyG's GATConv/GCNConv rely on scatter/gather ops that have no
   TFLite Micro equivalent. Board item 2b wants TinyML. Dense matmul-based graph
   ops are exportable; PyG's sparse ops are not. Writing it dense now means the
   quantization work later is a conversion, not a rewrite.
2. It runs anywhere. No torch-scatter / torch-sparse build wall for teammates.

At 435 PeMS sensors a dense 435x435 adjacency is ~190K floats. Fine. If the
sensor count grows past a few thousand, revisit.

Heads
-----
  "fc"     -> Linear -> SiLU -> Dropout -> Linear   (Phase 1 STGAT+GCN winner)
  "taylor" -> TaylorKAN                             (Phase 2 proposal)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .taylor_kan import TaylorKAN


class DenseGCN(nn.Module):
    """Graph convolution with a precomputed normalised adjacency."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features)

    def forward(self, x, adj_norm):
        # x: (B, N, F_in), adj_norm: (N, N)
        return self.lin(torch.einsum("ij,bjf->bif", adj_norm, x))


class DenseGAT(nn.Module):
    """Single-head graph attention, masked to the road-network adjacency."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, 1, out_features))
        self.att_dst = nn.Parameter(torch.empty(1, 1, out_features))
        self.dropout = dropout
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x, adj_mask):
        h = self.lin(x)                       # (B, N, F_out)
        a_src = (h * self.att_src).sum(-1)    # (B, N)
        a_dst = (h * self.att_dst).sum(-1)    # (B, N)
        # e[b,i,j] = LeakyReLU(a_src[i] + a_dst[j])
        e = F.leaky_relu(a_src.unsqueeze(2) + a_dst.unsqueeze(1), 0.2)
        e = e.masked_fill(adj_mask.unsqueeze(0) == 0, float("-inf"))
        alpha = torch.softmax(e, dim=-1)
        alpha = F.dropout(alpha, self.dropout, self.training)
        return torch.bmm(alpha, h)


class STGNN(nn.Module):
    """GAT -> GCN -> GRU -> LayerNorm -> head."""

    def __init__(
        self,
        n_nodes: int,
        in_features: int = 1,
        gnn_hidden: int = 32,
        gru_hidden: int = 64,
        gru_layers: int = 2,
        gru_dropout: float = 0.2,
        horizon: int = 5,
        head: str = "fc",
        taylor_order: int = 2,
        kan_hidden=(64,),
        fc_dropout: float = 0.1,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.horizon = horizon
        self.head_type = head

        self.gat = DenseGAT(in_features, gnn_hidden)
        self.gcn = DenseGCN(gnn_hidden, gnn_hidden)
        self.gru = nn.GRU(
            input_size=gnn_hidden,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(gru_hidden)

        if head == "taylor":
            self.head = TaylorKAN([gru_hidden, *kan_hidden, horizon], order=taylor_order)
        elif head == "fc":
            h = kan_hidden[0] if kan_hidden else gru_hidden
            self.head = nn.Sequential(
                nn.Linear(gru_hidden, h),
                nn.SiLU(),
                nn.Dropout(fc_dropout),
                nn.Linear(h, horizon),
            )
        else:
            raise ValueError(f"unknown head: {head!r}")

    def forward(self, x, adj_mask, adj_norm):
        """x: (B, T, N, F_in) -> (B, N, horizon)"""
        B, T, N, Fin = x.shape

        # Spatial encoding, shared across timesteps.
        xs = x.reshape(B * T, N, Fin)
        hs = F.elu(self.gat(xs, adj_mask))
        hs = F.elu(self.gcn(hs, adj_norm))
        C = hs.shape[-1]

        # Temporal encoding, per node.
        hs = hs.reshape(B, T, N, C).permute(0, 2, 1, 3).reshape(B * N, T, C)
        out, _ = self.gru(hs)
        z = self.norm(out[:, -1, :])          # last timestep, (B*N, gru_hidden)

        y = self.head(z)                      # (B*N, horizon)
        return y.reshape(B, N, self.horizon)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def head_params(self) -> int:
        return sum(p.numel() for p in self.head.parameters() if p.requires_grad)


def normalise_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """Symmetric normalisation with self-loops: D^-1/2 (A + I) D^-1/2."""
    a = adj + torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
    deg = a.sum(1)
    dinv = deg.pow(-0.5)
    dinv[torch.isinf(dinv)] = 0.0
    return dinv.unsqueeze(1) * a * dinv.unsqueeze(0)
