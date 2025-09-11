#creates roughly 100 compositioons with 4 or less elements and saves this in materials_seed.csv
import json, os, csv, random
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
EL_FILE = os.path.join(BASE_DIR, "data", "elements_list.json")
OUT_FILE = os.path.join(BASE_DIR, "data", "materials_seed.csv")

with open(EL_FILE, "r") as f:
    ALLOWED = json.load(f)

def random_comp(max_parts=4, min_frac=0.05):
    k = random.randint(2, max_parts)  # number of elements in alloy
    picks = random.sample(ALLOWED, k)
    raw = np.random.dirichlet([1.0]*k)
    # threshold tiny fractions
    raw = np.maximum(raw, min_frac)
    raw = raw / np.sum(raw)
    return {el: float(raw[i]) for i, el in enumerate(picks)}

def formula_from_dict(d):
    # produce readable formula like W0.6-Cu0.4
    parts = [f"{el}{round(frac,3)}" for el, frac in d.items()]
    return "-".join(parts)

# produce N compositions
N = 100
rows = []
for i in range(N):
    comp = random_comp()
    formula = formula_from_dict(comp)
    row = {"id": i+1, "formula": formula}
    for el in ALLOWED:
        row[f"frac_{el}"] = round(comp.get(el, 0.0), 6)
    rows.append(row)

# write CSV
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, "w", newline="") as csvfile:
    fieldnames = ["id", "formula"] + [f"frac_{el}" for el in ALLOWED]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print("Wrote", OUT_FILE)
