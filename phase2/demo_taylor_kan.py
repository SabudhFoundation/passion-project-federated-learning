"""
Taylor-KAN vs B-spline KAN — parameter budget.

Board item 3. Run:  python demo_taylor_kan.py

Uses the exact Phase 1 KAN configuration from the repo README:
    hidden layers [128, 128, 64], grid_size 8, spline_order 3
feeding from GRU hidden 128 into a 5-step forecast horizon.

The analytic counts need only Python. If torch is installed the script also
instantiates the real layer and checks the analytic count against the actual
parameter tensors, plus a quick numerical-stability check.
"""

import sys

sys.path.insert(0, ".")

from src.models.kan_params import bspline_kan_params, taylor_kan_params

# Phase 1 config, straight from the README.
GRU_HIDDEN = 128
KAN_HIDDEN = [128, 128, 64]
HORIZON = 5
GRID_SIZE = 8
SPLINE_ORDER = 3

LAYERS = [GRU_HIDDEN, *KAN_HIDDEN, HORIZON]

# Phase 1's winning FC head, for reference: Linear -> SiLU -> Dropout -> Linear
FC_HIDDEN = 128
FC_PARAMS = (GRU_HIDDEN * FC_HIDDEN + FC_HIDDEN) + (FC_HIDDEN * HORIZON + HORIZON)


def fmt(n: int) -> str:
    return f"{n:,}"


def kb(n: int, bytes_per: int = 4) -> float:
    return n * bytes_per / 1024


def main():
    print()
    print("=" * 74)
    print("PREDICTION HEAD PARAMETER BUDGET".center(74))
    print(f"layers {LAYERS}".center(74))
    print("=" * 74)
    print()

    bspline = bspline_kan_params(LAYERS, GRID_SIZE, SPLINE_ORDER)

    rows = [("B-spline KAN (Phase 1)", bspline, f"grid={GRID_SIZE}, order={SPLINE_ORDER}")]
    for order in (1, 2, 3):
        rows.append(
            (f"Taylor-KAN (order {order})", taylor_kan_params(LAYERS, order), f"{order + 1} coeff/edge")
        )
    rows.append(("FC head (Phase 1 winner)", FC_PARAMS, "Linear-SiLU-Linear"))

    print(f"{'Head':<28}{'Params':>12}{'fp32':>10}{'int8':>9}   {'vs B-spline':>12}")
    print("-" * 74)
    for name, n, note in rows:
        ratio = f"{bspline / n:.2f}x smaller" if n < bspline else ("baseline" if n == bspline else f"{n / bspline:.2f}x larger")
        print(f"{name:<28}{fmt(n):>12}{kb(n):>9.0f}K{kb(n, 1):>8.0f}K   {ratio:>12}")

    print()
    print("Per-edge coefficient cost")
    print("-" * 74)
    print(f"  B-spline    : grid_size + spline_order + base = {GRID_SIZE} + {SPLINE_ORDER} + 1 = {GRID_SIZE + SPLINE_ORDER + 1}")
    for order in (1, 2, 3):
        print(f"  Taylor p={order}  : (p+1) + base = {order + 1} + 1 = {order + 2}")

    t2 = taylor_kan_params(LAYERS, 2)
    print()
    print("=" * 74)
    print("HEADLINE")
    print("=" * 74)
    print(f"  Order-2 Taylor-KAN cuts the head from {fmt(bspline)} to {fmt(t2)} params")
    print(f"  = {bspline / t2:.2f}x smaller, {fmt(bspline - t2)} parameters removed")
    print(f"  Per communication round, per client, that is {kb(bspline - t2):.0f} KB less uplink.")
    print(f"  Across 4 clients x 100 rounds: {kb(bspline - t2) * 4 * 100 / 1024:.1f} MB less traffic.")
    print()
    print("  Phase 1 attributed STGAT+GCN's win over GNN-GRU-KAN to 'lower parameter")
    print("  complexity' and 'reduced sensitivity to client-side model divergence'.")
    print()
    print("  BE HONEST ABOUT WHAT THIS DOES AND DOESN'T CLOSE:")
    print(f"    B-spline KAN {fmt(bspline):>10}   =  {bspline / FC_PARAMS:5.1f}x the FC head")
    print(f"    Taylor  p=2  {fmt(t2):>10}   =  {t2 / FC_PARAMS:5.1f}x the FC head")
    print(f"    Taylor  p=1  {fmt(taylor_kan_params(LAYERS, 1)):>10}   =  "
          f"{taylor_kan_params(LAYERS, 1) / FC_PARAMS:5.1f}x the FC head")
    print(f"    FC head      {fmt(FC_PARAMS):>10}   =    1.0x  (Phase 1 winner)")
    print()
    print("  Taylor-KAN removes ~2/3 of the KAN head's parameters, but it is still")
    print("  roughly an order of magnitude heavier than the FC head. So this alone")
    print("  is NOT guaranteed to overturn Phase 1's result.")
    print()
    print("  The open question is whether the remaining gap still costs accuracy")
    print("  under FedAvg, or whether KAN's learnable activations start paying for")
    print("  themselves once the parameter count drops. Shrinking the KAN hidden")
    print("  layers from [128,128,64] is the other lever, and it is untested.")
    print("  That is the Phase 2 experiment — an open question, not a foregone win.")
    print("=" * 74)
    print()

    _torch_check()


def _torch_check():
    try:
        import torch
        from src.models.taylor_kan import TaylorKAN
    except ImportError:
        print("(torch not installed — analytic counts only.")
        print(" `pip install torch` to also verify against real tensors.)")
        print()
        return

    print("Verifying analytic counts against instantiated layers")
    print("-" * 74)
    ok = True
    for order in (1, 2, 3):
        model = TaylorKAN(LAYERS, order=order)
        actual = model.n_params()
        predicted = taylor_kan_params(LAYERS, order)
        match = "OK" if actual == predicted else "MISMATCH"
        ok &= actual == predicted
        print(f"  order {order}: analytic {fmt(predicted):>10}   actual {fmt(actual):>10}   {match}")

    # Stability: Taylor bases blow up without input squashing. Confirm they don't.
    print()
    print("Numerical stability (tanh squashing active)")
    print("-" * 74)
    model = TaylorKAN(LAYERS, order=2)
    for scale in (1.0, 10.0, 100.0):
        x = torch.randn(64, GRU_HIDDEN) * scale
        with torch.no_grad():
            y = model(x)
        finite = torch.isfinite(y).all().item()
        print(f"  input scale {scale:>6.0f}: output absmax {y.abs().max():>10.3f}   finite={finite}")
        ok &= finite

    print()
    print("  RESULT:", "all checks passed" if ok else "CHECKS FAILED")
    print()


if __name__ == "__main__":
    main()
