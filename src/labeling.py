# src/labeling.py
import os
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.dirname(__file__))
DESC_FILE = os.path.join(BASE, "data", "descriptors.csv")
OUT_FILE = os.path.join(BASE, "data", "synthetic_labels.csv")

ACTIVATION_SCORE = {
    "W": 1.0, "Cu": 3.0, "Fe": 2.0, "Cr": 2.0, "Mo": 1.0, "Ta": 1.5,
    "Nb": 1.2, "V": 1.6, "Ti": 1.1, "Al": 1.0, "Si": 1.0, "Mn": 1.8
}
THERMAL_SCORE = {
    "W": 0.6, "Cu": 1.0, "Fe": 0.4, "Cr": 0.3, "Mo": 0.5, "Ta": 0.2,
    "Nb": 0.25, "V": 0.2, "Ti": 0.35, "Al": 0.45, "Si": 0.3, "Mn": 0.25
}
DUCTILITY_BONUS = {
    "Fe": 1.2, "Al": 1.1, "Cu": 1.0, "Ti": 0.9, "W": 0.6, "Cr": 0.7,
    "Mo": 0.7, "Ta": 0.6, "Nb": 0.8, "V": 0.75, "Si": 0.5, "Mn": 0.7
}

def compute_activation(row, allowed):
    s = 0.0
    for el in allowed:
        frac = row.get(f"frac_{el}", 0.0)
        s += frac * ACTIVATION_SCORE.get(el, 1.0)
    return float(s)

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

    out = df.copy()
    out["activation_proxy"] = out.apply(lambda r: compute_activation(r, allowed), axis=1)
    out["thermal_proxy"] = out.apply(lambda r: compute_thermal(r, allowed), axis=1)
    out["ductility_proxy"] = out.apply(lambda r: compute_ductility(r, allowed), axis=1)

    out.to_csv(OUT_FILE, index=False)
    print("Wrote", OUT_FILE)

if __name__ == "__main__":
    main()
