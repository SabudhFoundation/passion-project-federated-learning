"""Taylor-expansion KAN layer. Uses a learnable truncated Taylor series on
each edge instead of a B-spline basis, which needs fewer params per edge."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Analytic counts live in a torch-free module so the parameter comparison can be
# run without the ML stack installed. Re-exported here for convenience.
from .kan_params import bspline_kan_params, taylor_kan_params  # noqa: F401

class TaylorKANLayer(nn.Module):
    """KAN layer with a learnable Taylor-expansion basis on each edge."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        order: int = 2,
        use_base: bool = True,
        squash: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.order = order
        self.use_base = use_base
        self.squash = squash

        # Taylor coefficients: one (order+1)-vector per (in, out) edge.
        self.coef = nn.Parameter(torch.empty(in_features, out_features, order + 1))

        # Per-input expansion centre m_i. Cheap (in_features params) and lets the
        # layer place its expansion point where the data actually lives.
        self.centre = nn.Parameter(torch.zeros(in_features))

        # Optional SiLU base path, same idea as efficient-kan's base_weight.
        # Keeps a linear escape hatch so the Taylor part only learns the residual.
        if use_base:
            self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        else:
            self.register_parameter("base_weight", None)

        # Precomputed 1/k!, buffer, not a parameter.
        inv_fact = torch.tensor([1.0 / math.factorial(k) for k in range(order + 1)])
        self.register_buffer("inv_factorial", inv_fact)

        self.reset_parameters()

    def reset_parameters(self):
        # Scale down with order: high-order terms start near-silent so the layer
        # behaves like a linear map at init and grows curvature as it trains.
        with torch.no_grad():
            std = 1.0 / math.sqrt(self.in_features)
            for k in range(self.order + 1):
                self.coef[:, :, k].normal_(0.0, std / (k + 1))
        if self.base_weight is not None:
            nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x = x.reshape(-1, self.in_features)  # (B, in)

        u = torch.tanh(x) if self.squash else x
        u = u - self.centre  # (B, in)

        # powers[..., k] = u^k / k!
        powers = torch.stack([u.pow(k) for k in range(self.order + 1)], dim=-1)
        powers = powers * self.inv_factorial  # (B, in, order+1)

        # contract over inputs and orders -> (B, out)
        out = torch.einsum("bik,iok->bo", powers, self.coef)

        if self.base_weight is not None:
            out = out + F.linear(F.silu(x), self.base_weight)

        return out.reshape(*orig_shape[:-1], self.out_features)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, order={self.order}"

class TaylorKAN(nn.Module):
    """Stack of TaylorKANLayers, drop-in replacement for the efficient-kan head."""

    def __init__(self, layers_hidden, order: int = 2, use_base: bool = True):
        super().__init__()
        self.layers = nn.ModuleList(
            TaylorKANLayer(i, o, order=order, use_base=use_base)
            for i, o in zip(layers_hidden[:-1], layers_hidden[1:])
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

# Parameter accounting lives in kan_params.py (torch-free) and is re-exported
# at the top of this module.
