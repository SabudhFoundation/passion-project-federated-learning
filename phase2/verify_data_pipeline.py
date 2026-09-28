"""Check the data pipeline on synthetic districts."""

import subprocess
import sys
from pathlib import Path

import numpy as np

# Make synthetic data first.
subprocess.run([sys.executable, "scripts/make_synthetic_districts.py"], check=True)

sys.path.insert(0, ".")
from src.data.district_data import prepare_all_districts  # noqa: E402
from src.data.windowing import HORIZON, LOOKBACK  # noqa: E402

dirs = sorted(str(p) for p in Path("data_synth").glob("district_*"))
clients = prepare_all_districts(dirs)

print("\n" + "=" * 68)
print("DATA PIPELINE VERIFICATION".center(68))
print("=" * 68)

all_ok = True

def check(label, cond):
    global all_ok
    all_ok = all_ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")

for c in clients:
    N, F = c["n_nodes"], c["n_features"]
    Xtr, Ytr = c["train"]
    Xva, Yva = c["val"]
    Xte, Yte = c["test"]

    print(f"\nDistrict '{c['name']}'  (N={N} nodes, F={F} features)")
    print(f"  train X {Xtr.shape}  Y {Ytr.shape}")
    print(f"  val   X {Xva.shape}  Y {Yva.shape}")
    print(f"  test  X {Xte.shape}  Y {Yte.shape}")

    # 1. shape contract
    check("X is (num, lookback, N, F)", Xtr.shape[1:] == (LOOKBACK, N, F))
    check("Y is (num, N, horizon)", Ytr.shape[1:] == (N, HORIZON))

    # 2. chronological split roughly 70/15/15 (by windowed sample count)
    total = Xtr.shape[0] + Xva.shape[0] + Xte.shape[0]
    frac_tr = Xtr.shape[0] / total
    check(f"train fraction ~0.70 (got {frac_tr:.2f})", 0.6 < frac_tr < 0.78)

    # 3. no leakage: scaled train flow ~ mean 0 / std 1; val/test NOT re-centred
    tr_flow = Xtr[:, :, :, 0]
    check(f"train flow standardised (mean {tr_flow.mean():+.2f})", abs(tr_flow.mean()) < 0.25)
    va_flow = Xva[:, :, :, 0]
    # val inherits train scaling, so its mean is generally != 0. If it were exactly
    # 0 that would mean the scaler saw val data (leakage).
    check("val flow uses train scaling (not re-centred)", abs(va_flow.mean()) > 1e-3)

    # 4. inverse transform round-trips
    s = c["scaler"]
    sample = tr_flow[:50]
    recovered = s.inverse_transform(sample)
    re_scaled = s.transform(recovered)
    check("scaler inverse round-trips", np.allclose(sample, re_scaled, atol=1e-5))

    # 5. adjacency shapes
    check("adj_mask is (N,N)", c["adj_mask"].shape == (N, N))
    check("adj_norm is (N,N)", c["adj_norm"].shape == (N, N))

# different node counts across districts
ncounts = [c["n_nodes"] for c in clients]
print(f"\nNode counts across districts: {ncounts}")
check("districts have differing node counts (non-IID sizes)", len(set(ncounts)) > 1)

print("\n" + "=" * 68)
print(("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED").center(68))
print("=" * 68)
sys.exit(0 if all_ok else 1)
