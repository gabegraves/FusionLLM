# src/labeling.py
import os
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.dirname(__file__))
DESC_FILE = os.path.join(BASE, "data", "descriptors.csv")
OUT_FILE = os.path.join(BASE, "data", "synthetic_labels.csv")

THERMAL_SCORE = {
    "W": 0.6, "Cu": 1.0, "Fe": 0.4, "Cr": 0.3, "Mo": 0.5, "Ta": 0.2,
    "Nb": 0.25, "V": 0.2, "Ti": 0.35, "Al": 0.45, "Si": 0.3, "Mn": 0.25
}
DUCTILITY_BONUS = {
    "Fe": 1.2, "Al": 1.1, "Cu": 1.0, "Ti": 0.9, "W": 0.6, "Cr": 0.7,
    "Mo": 0.7, "Ta": 0.6, "Nb": 0.8, "V": 0.75, "Si": 0.5, "Mn": 0.7
}

def compute_thermal(row, allowed):
    s = 0.0
    for el in allowed:
        frac = row.get(f"frac_{el}", 0.0)
        s += frac * THERMAL_SCORE.get(el, 0.3)
    return float(s)

def compute_ductility(row, allowed):
    bonus = 0.0
    for el in allowed:
        bonus += row.get(f"frac_{el}", 0.0) * DUCTILITY_BONUS.get(el, 0.8)
    std_r = row.get("std_atomic_radius", 1.0)
    val = bonus / (1.0 + std_r*0.01)
    return float(val)

def main():
    df = pd.read_csv(DESC_FILE)
    import json
    with open(os.path.join(BASE, "data", "elements_list.json"), "r") as f:
        allowed = json.load(f)

    # Precompute per-element physics metrics via Henry's code
    from src.physics_adapter import compute_element_metrics
    per_element = {}
    for el in allowed:
        metrics = compute_element_metrics(el)
        per_element[el] = (metrics["pct_transmuted"], metrics["pct_radioactive"])

    def _aggregate_physics(row):
        t_sum = 0.0
        r_sum = 0.0
        for el in allowed:
            frac = float(row.get(f"frac_{el}", 0.0))
            t_el, r_el = per_element.get(el, (0.0, 0.0))
            t_sum += frac * float(t_el)
            r_sum += frac * float(r_el)
        return pd.Series({
            "pct_transmuted": float(t_sum),
            "pct_radioactive": float(r_sum),
            # For backward compatibility (temporary): map activation_proxy to pct_radioactive
            "activation_proxy": float(r_sum),
        })

    out = df.copy()
    agg = out.apply(_aggregate_physics, axis=1)
    for col in ["pct_transmuted", "pct_radioactive", "activation_proxy"]:
        out[col] = agg[col]
    out["thermal_proxy"] = out.apply(lambda r: compute_thermal(r, allowed), axis=1)
    out["ductility_proxy"] = out.apply(lambda r: compute_ductility(r, allowed), axis=1)

    out.to_csv(OUT_FILE, index=False)
    print("Wrote", OUT_FILE)

if __name__ == "__main__":
    main()
