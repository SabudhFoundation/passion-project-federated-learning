"""
Analytic parameter counts for KAN prediction heads.

Deliberately dependency-free (no torch) so the parameter-budget comparison can be
run and cited anywhere — including in a meeting on a machine without the ML stack
installed. taylor_kan.py imports from here; nothing here imports from it.
"""


def bspline_kan_params(layers_hidden, grid_size: int = 8, spline_order: int = 3) -> int:
    """Parameter count for efficient-kan, the Phase 1 configuration.

    Per (in, out) edge: (grid_size + spline_order) spline coefficients + 1 base weight.
    """
    total = 0
    for i, o in zip(layers_hidden[:-1], layers_hidden[1:]):
        total += i * o * (grid_size + spline_order)  # spline coefficients
        total += i * o                               # base_weight
    return total


def taylor_kan_params(layers_hidden, order: int = 2, use_base: bool = True) -> int:
    """Parameter count for TaylorKAN.

    Per (in, out) edge: (order + 1) Taylor coefficients + 1 base weight.
    Plus one expansion centre per input feature.
    """
    total = 0
    for i, o in zip(layers_hidden[:-1], layers_hidden[1:]):
        total += i * o * (order + 1)  # Taylor coefficients
        total += i                    # centres
        if use_base:
            total += i * o            # base_weight
    return total
