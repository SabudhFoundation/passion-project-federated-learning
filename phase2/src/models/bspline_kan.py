"""B-spline KAN head, same basis as efficient-kan (the Phase 1 head).

Written out here so the runner has no extra install step. Each edge gets
(grid_size + spline_order) spline coefficients plus a SiLU base weight.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BSplineKANLayer(nn.Module):
    """KAN layer with a learnable B-spline on each (in, out) edge."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 8,
        spline_order: int = 3,
        grid_range=(-1.0, 1.0),
        scale_noise: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Uniform knots, extended by spline_order on each side.
        h = (grid_range[1] - grid_range[0]) / grid_size
        knots = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32)
        grid = knots * h + grid_range[0]
        self.register_buffer("grid", grid.expand(in_features, -1).contiguous())

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        self.scale_noise = scale_noise
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        with torch.no_grad():
            # small random splines, so the layer starts close to its base path
            self.spline_weight.uniform_(-1.0, 1.0)
            self.spline_weight.mul_(self.scale_noise / self.grid_size)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Cox-de Boor recursion. x: (B, in) -> (B, in, grid_size + spline_order)"""
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)])
            right = (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:-k])
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_out = F.linear(F.silu(x), self.base_weight)
        # tanh keeps inputs inside the knot range so no edge goes flat
        splines = self.b_splines(torch.tanh(x))
        spline_out = torch.einsum("bik,oik->bo", splines, self.spline_weight)

        out = base_out + spline_out
        return out.reshape(*orig_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"grid={self.grid_size}, order={self.spline_order}")


class BSplineKAN(nn.Module):
    """Stack of BSplineKANLayers, same interface as TaylorKAN."""

    def __init__(self, layers_hidden, grid_size: int = 8, spline_order: int = 3):
        super().__init__()
        self.layers = nn.ModuleList(
            BSplineKANLayer(i, o, grid_size=grid_size, spline_order=spline_order)
            for i, o in zip(layers_hidden[:-1], layers_hidden[1:])
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
