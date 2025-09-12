#!/usr/bin/env python3
import random, json, pandas as pd, math
from src.surrogate import load_models, predict_with_uncertainty

# config
N_CANDIDATES = 2000   # number of random candidates to try
MAX_ELEMS = 4         # up to 4 elements per alloy
ALLOWED = json.load(open("data/elements_list.json"))

def random_comp(allowed, max_elems=4):
    k = random.randint(1, max_elems)
    els = random.sample(allowed, k)
    # random fractions
    fracs = [random.random() for _ in els]
    s = sum(fracs)
    fracs = [f/s for f in fracs]
    comp = "-".join(f"{el}{round(fr,6)}" for el,fr in zip(els,fracs))
    return comp

def score_comp(comp):
    # try both styles: dict or formula string allowed by your predict fn
    try:
        pred = predict_with_uncertainty(comp)
    except Exception:
        # try parse to dict: e.g. "W0.6-Cu0.4" -> {"W":0.6,"Cu":0.4}
        d={}
        for part in comp.split("-"):
            for i,ch in enumerate(part):
                if ch.isdigit() or ch=='.':
                    el = part[:i]
                    frac = float(part[i:])
                    d[el]=frac
                    break
        pred = predict_with_uncertainty(d)
    # scoring: adjust weights to your preference
    # Prefer lower radioactivity/transmutation, higher thermal/ductility
    # Backward-compatible: activation == radioactive
    score = -0.6*pred["radioactive"]["mean"] - 0.2*pred.get("transmuted", pred["radioactive"]).get("mean", 0)
    score += 0.9*pred["thermal"]["mean"] + 0.6*pred["ductility"]["mean"]
    score -= 0.4*(pred["radioactive"]["std"] + pred["thermal"]["std"] + pred["ductility"]["std"])
    return pred, score

def main():
    load_models()  # ensure models loaded once
    rows=[]
    for i in range(N_CANDIDATES):
        comp = random_comp(ALLOWED, MAX_ELEMS)
        pred, score = score_comp(comp)
        rows.append({
            "formula": comp,
            "score": score,
            # Maintain activation_* for downstream compatibility (radioactive)
            "activation_mean": pred["activation"]["mean"],
            "activation_std": pred["activation"]["std"],
            # Explicit physics-informed metrics
            "radioactive_mean": pred["radioactive"]["mean"],
            "radioactive_std": pred["radioactive"]["std"],
            "transmuted_mean": pred.get("transmuted", {}).get("mean", None),
            "transmuted_std": pred.get("transmuted", {}).get("std", None),
            # Other proxies
            "thermal_mean": pred["thermal"]["mean"],
            "thermal_std": pred["thermal"]["std"],
            "ductility_mean": pred["ductility"]["mean"],
            "ductility_std": pred["ductility"]["std"]
        })
        if (i+1) % 200 == 0:
            print(f"Generated {i+1}/{N_CANDIDATES}")
    df = pd.DataFrame(rows)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df.to_csv("data/ga_top_candidates.csv", index=False)
    print("Wrote data/ga_top_candidates.csv")
    print(df.head(20).to_string(index=False))
if __name__ == "__main__":
    main()
