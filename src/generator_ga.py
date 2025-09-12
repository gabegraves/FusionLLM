# src/generator_ga.py
"""
GA generator for constrained alloy search.
Usage:
  PYTHONPATH=. python -m src.generator_ga --pop 50 --gens 30 --top 10 --seed 42
"""
import os, json, random, argparse
from copy import deepcopy
BASE = os.path.dirname(os.path.dirname(__file__))

# local surrogate utils
from src.surrogate import load_models, predict_with_uncertainty

def load_allowed_elements():
    with open(os.path.join(BASE, "data", "elements_list.json"), "r") as f:
        return json.load(f)

def random_composition(allowed, max_elems=4):
    k = random.randint(1, max_elems)
    elems = random.sample(allowed, k)
    fracs = [random.random() for _ in elems]
    s = sum(fracs)
    return {e: f/s for e,f in zip(elems, fracs)}

def comp_to_formula(comp):
    return "-".join([f"{el}{round(frac,6)}" for el,frac in sorted(comp.items())])

def mutate(comp, allowed):
    comp = deepcopy(comp)
    for el in list(comp.keys()):
        if random.random() < 0.25:
            comp[el] = max(1e-4, comp[el]*(0.5+random.random()))
    if random.random() < 0.12:
        keys=list(comp.keys())
        if keys:
            rem=random.choice(keys); comp.pop(rem)
            choices=[e for e in allowed if e not in comp]
            if choices: comp[random.choice(choices)] = random.random()
    s=sum(comp.values()); return {k:v/s for k,v in comp.items()} if s>0 else random_composition(allowed)

def crossover(a,b):
    elems=set(a.keys())|set(b.keys()); comp={}
    for el in elems: comp[el]=a.get(el,0)*0.5 + b.get(el,0)*0.5
    comp = dict(sorted(comp.items(), key=lambda x: x[1], reverse=True)[:4])
    s=sum(comp.values()); return {k:v/s for k,v in comp.items()}

def score_from_pred(pred):
    a=pred["activation"]["mean"]; t=pred["thermal"]["mean"]; d=pred["ductility"]["mean"]
    uncert = pred["activation"]["std"] + pred["thermal"]["std"] + pred["ductility"]["std"]
    return -0.6*a + 0.9*t + 0.6*d - 0.3*uncert

def run_ga(pop=50, gens=30, top_k=10, seed=42):
    random.seed(seed)
    allowed = load_allowed_elements()
    # ensure models loaded
    load_models()
    # init population
    population = [ random_composition(allowed) for _ in range(pop) ]
    evaluated = []
    for comp in population:
        pred = predict_with_uncertainty(comp)
        evaluated.append((comp, score_from_pred(pred), pred))
    # evolve
    for g in range(gens):
        evaluated.sort(key=lambda x: x[1], reverse=True)
        elites = evaluated[: max(2, pop//10)]
        children = list(elites)
        while len(children) < pop:
            a = random.choice(evaluated)[0]
            b = random.choice(evaluated)[0]
            child = crossover(a,b)
            child = mutate(child, allowed)
            pr = predict_with_uncertainty(child)
            children.append((child, score_from_pred(pr), pr))
        evaluated = children
        best = evaluated[0]
        print(f"Gen {g+1}/{gens} best score: {best[1]:.4f} formula: {comp_to_formula(best[0])}")
    # write top results
    evaluated.sort(key=lambda x: x[1], reverse=True)
    top = evaluated[:top_k]
    import pandas as pd
    rows = []
    for comp, sc, pr in top:
        rows.append({
            "formula": comp_to_formula(comp),
            "score": sc,
            "activation_mean": pr["activation"]["mean"],
            "activation_std": pr["activation"]["std"],
            "thermal_mean": pr["thermal"]["mean"],
            "thermal_std": pr["thermal"]["std"],
            "ductility_mean": pr["ductility"]["mean"],
            "ductility_std": pr["ductility"]["std"],
            "comp_dict": comp
        })
    outp = os.path.join(BASE, "data", "ga_top_candidates.csv")
    pd.DataFrame(rows).to_csv(outp, index=False)
    print(f"Saved top {top_k} candidates to {outp}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--pop", type=int, default=50)
    p.add_argument("--gens", type=int, default=30)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_ga(pop=args.pop, gens=args.gens, top_k=args.top, seed=args.seed)
