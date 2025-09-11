# src/compute_descriptors.py
import os
import pandas as pd
from src.descriptors import comp_features

BASE = os.path.dirname(os.path.dirname(__file__))
INFILE = os.path.join(BASE, "data", "materials_seed.csv")
OUTFILE = os.path.join(BASE, "data", "descriptors.csv")

def row_to_features(row):
    formula = row["formula"]
    try:
        feats = comp_features(formula)
    except Exception as e:
        print(f"Error parsing {formula}: {e}")
        raise
    # flatten fraction vector into frac_<EL> columns
    frac_vec = feats["frac_vector"]
    out = {
        "id": row["id"],
        "formula": formula,
        "avg_Z": feats["avg_Z"],
        "avg_chi": feats["avg_chi"],
        "avg_mass": feats["avg_mass"],
        "std_atomic_radius": feats["std_atomic_radius"],
        "n_elements": feats["n_elements"],
        "contains_Cu": feats["contains_Cu"],
        "contains_W": feats["contains_W"],
    }
    # read allowed elements in same order
    import json
    el_file = os.path.join(BASE, "data", "elements_list.json")
    with open(el_file, "r") as f:
        allowed = json.load(f)
    for i, el in enumerate(allowed):
        out[f"frac_{el}"] = float(frac_vec[i])
    return out

def main():
    df = pd.read_csv(INFILE)
    rows = []
    for _, r in df.iterrows():
        rows.append(row_to_features(r))
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTFILE, index=False)
    print("Wrote", OUTFILE)

if __name__ == "__main__":
    main()
