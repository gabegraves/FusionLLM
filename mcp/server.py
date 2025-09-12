import os
import sys
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Ensure project root is importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.physics_adapter import compute_element_metrics
from src.descriptors import comp_features
from src.surrogate import load_models, predict_with_uncertainty
import numpy as np
from src.descriptors import parse_formula


app = FastAPI(title="FusionLLM MCP", version="0.1.0")


class Composition(BaseModel):
    composition: Dict[str, float] = Field(..., description="Element fractions summing to ~1.0")


class ElementMetricsReq(BaseModel):
    symbol: str
    energy_eV: float = 14.1e6
    flux_n_cm2_s: float = 1e13
    time_s: float = 60 * 60
    Natoms: float = 1e23


class ScoreReq(Composition):
    weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Override weights for scoring. Keys: radioactive, transmuted, thermal, ductility",
    )


class ProposeReq(BaseModel):
    n_candidates: int = 100
    max_elems: int = 4
    allowed_elements: Optional[List[str]] = None
    seed: Optional[int] = None


class ParetoReq(BaseModel):
    n_samples: int = 2000
    max_elems: int = 4
    allowed_elements: Optional[List[str]] = None
    seed: Optional[int] = None


class RangeSpec(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None
    num: Optional[int] = None
    values: Optional[List[float]] = None


class SweepReq(BaseModel):
    composition: Dict[str, float]
    flux: RangeSpec = Field(default_factory=lambda: RangeSpec(min=1e12, max=1e14, num=5))
    time_s: RangeSpec = Field(default_factory=lambda: RangeSpec(min=3600, max=86400, num=5))
    energy_eV: RangeSpec = Field(default_factory=lambda: RangeSpec(values=[14.1e6]))


class ProtocolReq(BaseModel):
    composition: Dict[str, float]
    tolerance_at_pct: float = 0.05
    irradiation: Dict[str, float] = Field(
        default_factory=lambda: {
            "flux_n_cm2_s": 1e13,
            "time_s": 60 * 60,
            "energy_eV": 14.1e6,
            "temperature_C": 400.0,
        }
    )


class ParetoSaveReq(BaseModel):
    items: List[str]  # formula strings like "W0.6-Cu0.4"
    target_csv: Optional[str] = None  # defaults to data/ga_top_candidates.csv
    mode: str = "upsert"  # or "append"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/compute_element_metrics")
def route_compute_element_metrics(req: ElementMetricsReq):
    try:
        out = compute_element_metrics(
            req.symbol,
            energy_eV=req.energy_eV,
            flux_n_cm2_s=req.flux_n_cm2_s,
            time_s=req.time_s,
            Natoms=req.Natoms,
        )
        return out
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/evaluate_alloy")
def route_evaluate_alloy(req: Composition):
    # Compute per-element metrics and aggregate by fractions
    comp = req.composition
    t_sum = 0.0
    r_sum = 0.0
    for el, frac in comp.items():
        m = compute_element_metrics(el)
        t_sum += float(frac) * float(m["pct_transmuted"])
        r_sum += float(frac) * float(m["pct_radioactive"])
    return {"pct_transmuted": t_sum, "pct_radioactive": r_sum}


@app.post("/evaluate_alloy_breakdown")
def route_evaluate_alloy_breakdown(req: Composition):
    """Return per-element physics metrics and the aggregated result.

    Useful for explainability and ranking contributions by element.
    """
    comp = req.composition
    per_el = {}
    t_sum = 0.0
    r_sum = 0.0
    for el, frac in comp.items():
        m = compute_element_metrics(el)
        per_el[el] = {
            "frac": float(frac),
            "pct_transmuted": float(m["pct_transmuted"]),
            "pct_radioactive": float(m["pct_radioactive"]),
            "weighted_transmuted": float(frac) * float(m["pct_transmuted"]),
            "weighted_radioactive": float(frac) * float(m["pct_radioactive"]),
        }
        t_sum += per_el[el]["weighted_transmuted"]
        r_sum += per_el[el]["weighted_radioactive"]
    return {"per_element": per_el, "aggregate": {"pct_transmuted": t_sum, "pct_radioactive": r_sum}}


@app.post("/compute_descriptors")
def route_compute_descriptors(req: Composition):
    feats = comp_features({k: float(v) for k, v in req.composition.items()})

    def to_py(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: to_py(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_py(x) for x in obj]
        return obj

    return to_py(feats)


@app.post("/predict_surrogates")
def route_predict_surrogates(req: Composition):
    try:
        load_models()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Models not loaded: {e}")
    out = predict_with_uncertainty(req.composition)
    return out


@app.post("/score_candidate")
def route_score_candidate(req: ScoreReq):
    try:
        load_models()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Models not loaded: {e}")
    pred = predict_with_uncertainty(req.composition)
    w = req.weights or {}
    radioactive_w = float(w.get("radioactive", -0.6))
    transmuted_w = float(w.get("transmuted", -0.2))
    thermal_w = float(w.get("thermal", 0.9))
    ductility_w = float(w.get("ductility", 0.6))
    score = (
        radioactive_w * pred["radioactive"]["mean"]
        + transmuted_w * pred.get("transmuted", pred["radioactive"])["mean"]
        + thermal_w * pred["thermal"]["mean"]
        + ductility_w * pred["ductility"]["mean"]
    )
    return {"score": score, "pred": pred}


@app.post("/propose_candidates")
def route_propose_candidates(req: ProposeReq):
    import random

    allowed = req.allowed_elements
    if not allowed:
        # Default to elements list if available
        try:
            import json

            with open(os.path.join(ROOT, "data", "elements_list.json"), "r") as f:
                allowed = json.load(f)
        except Exception:
            allowed = ["W", "Cu", "Fe", "Cr", "Mo", "Ta", "Nb", "V", "Ti", "Al", "Si", "Mn"]

    if req.seed is not None:
        random.seed(req.seed)

    try:
        load_models()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Models not loaded: {e}")

    def rand_comp():
        k = random.randint(1, max(1, req.max_elems))
        els = random.sample(allowed, k)
        frs = [random.random() for _ in els]
        s = sum(frs)
        frs = [f / s for f in frs]
        return {el: fr for el, fr in zip(els, frs)}

    results = []
    for _ in range(req.n_candidates):
        comp = rand_comp()
        pred = predict_with_uncertainty(comp)
        score = (
            -0.6 * pred["radioactive"]["mean"]
            - 0.2 * pred.get("transmuted", pred["radioactive"])["mean"]
            + 0.9 * pred["thermal"]["mean"]
            + 0.6 * pred["ductility"]["mean"]
            - 0.4 * (pred["radioactive"]["std"] + pred["thermal"]["std"] + pred["ductility"]["std"])
        )
        results.append({"composition": comp, "score": score, "pred": pred})

    results.sort(key=lambda r: r["score"], reverse=True)
    return {"top": results[: min(50, len(results))]}


def _non_dominated_front(points: List[List[float]]):
    """Return indices of the Pareto non-dominated front for minimize-all objectives.
    points: list of vectors.
    """
    idxs = []
    for i, pi in enumerate(points):
        dominated = False
        for j, pj in enumerate(points):
            if j == i:
                continue
            # pj dominates pi if pj <= pi for all and < for at least one
            if all(pj[k] <= pi[k] for k in range(len(pi))) and any(pj[k] < pi[k] for k in range(len(pi))):
                dominated = True
                break
        if not dominated:
            idxs.append(i)
    return idxs


@app.post("/pareto_candidates")
def route_pareto_candidates(req: ParetoReq):
    import random

    allowed = req.allowed_elements
    if not allowed:
        try:
            import json

            with open(os.path.join(ROOT, "data", "elements_list.json"), "r") as f:
                allowed = json.load(f)
        except Exception:
            allowed = ["W", "Cu", "Fe", "Cr", "Mo", "Ta", "Nb", "V", "Ti", "Al", "Si", "Mn"]

    if req.seed is not None:
        random.seed(req.seed)
        np.random.seed(req.seed)

    try:
        load_models()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Models not loaded: {e}")

    def rand_comp():
        k = random.randint(1, max(1, req.max_elems))
        els = random.sample(allowed, k)
        frs = [random.random() for _ in els]
        s = sum(frs)
        frs = [f / s for f in frs]
        return {el: fr for el, fr in zip(els, frs)}

    comps = [rand_comp() for _ in range(req.n_samples)]
    preds = [predict_with_uncertainty(c) for c in comps]

    # objectives: minimize radioactive, transmuted; maximize thermal, ductility -> minimize negatives
    objs = [
        [
            p["radioactive"]["mean"],
            p.get("transmuted", p["radioactive"])["mean"],
            -p["thermal"]["mean"],
            -p["ductility"]["mean"],
        ]
        for p in preds
    ]
    front_idx = _non_dominated_front(objs)
    out = [
        {
            "composition": comps[i],
            "pred": preds[i],
            "objectives": {
                "radioactive": objs[i][0],
                "transmuted": objs[i][1],
                "neg_thermal": objs[i][2],
                "neg_ductility": objs[i][3],
            },
        }
        for i in front_idx
    ]
    # sort front by a simple scalarization to make viewing easier
    out.sort(key=lambda r: (r["objectives"]["radioactive"], r["objectives"]["transmuted"], r["pred"]["thermal"]["mean"]), reverse=False)
    return {"front": out}


def _expand_range(r: RangeSpec) -> List[float]:
    if r.values is not None:
        return list(r.values)
    if r.min is None or r.max is None:
        return []
    n = int(r.num or 5)
    if n <= 1:
        return [float(r.min)]
    return list(np.linspace(float(r.min), float(r.max), n))


@app.post("/sweep_physics")
def route_sweep_physics(req: SweepReq):
    comp = req.composition
    flux_vals = _expand_range(req.flux)
    time_vals = _expand_range(req.time_s)
    energy_vals = _expand_range(req.energy_eV)
    if not flux_vals:
        flux_vals = [1e13]
    if not time_vals:
        time_vals = [60 * 60]
    if not energy_vals:
        energy_vals = [14.1e6]

    # Precompute per-element metrics for each param triple and aggregate for alloy
    records = []
    for E in energy_vals:
        for F in flux_vals:
            for T in time_vals:
                t_sum = 0.0
                r_sum = 0.0
                for el, frac in comp.items():
                    m = compute_element_metrics(el, energy_eV=E, flux_n_cm2_s=F, time_s=T)
                    t_sum += float(frac) * float(m["pct_transmuted"])
                    r_sum += float(frac) * float(m["pct_radioactive"])
                records.append({
                    "energy_eV": E,
                    "flux_n_cm2_s": F,
                    "time_s": T,
                    "pct_transmuted": t_sum,
                    "pct_radioactive": r_sum,
                })
    return {"energy": energy_vals, "flux": flux_vals, "time_s": time_vals, "records": records}


@app.post("/generate_protocol")
def route_generate_protocol(req: ProtocolReq):
    try:
        load_models()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Models not loaded: {e}")
    pred = predict_with_uncertainty(req.composition)

    # Build JSON protocol
    comp_pct = {el: 100.0 * float(fr) for el, fr in req.composition.items()}
    protocol_json = {
        "composition_tolerances": [
            {"element": el, "target_at_pct": round(val, 6), "tolerance_at_pct": req.tolerance_at_pct}
            for el, val in comp_pct.items()
        ],
        "instruments": {
            "radioactivity": "HPGe gamma spectrometer (lead shielding, collimation)",
            "transmutation": ["ICP-MS", "EPMA/WDS"],
            "mechanical": ["Vickers micro-hardness (100 g)", "Miniature tensile tester (500 N)", "3-point bend"],
            "thermal": ["Laser Flash Apparatus (LFA)", "DSC"],
            "microstructure": ["SEM/EDS", "TEM"],
        },
        "acceptance_criteria": [
            {"metric": "gamma_activity_Bq_per_g_1y_Eg>100keV", "threshold": "<=1000"},
            {"metric": "max_delta_at_pct_primary", "threshold": "<=0.5"},
            {"metric": "k_400C_W_per_mK", "threshold": ">=100"},
            {"metric": "uniform_elongation_pct", "threshold": ">=5"},
            {"metric": "void_swelling_pct", "threshold": "<=1"},
        ],
        "irradiation": req.irradiation,
        "predicted": pred,
    }

    # Markdown rendering using protocol_json
    md_lines = []
    md_lines.append("## Reason")
    md_lines.append("- Low predicted activation and transmutation compared with peers.")
    md_lines.append("- Favorable thermal and ductility proxies for heat removal and toughness.")
    md_lines.append("- Alloy fractions chosen for performance balance and manufacturability.")
    md_lines.append("")
    md_lines.append("## Experiment Protocol")
    md_lines.append("### 1. Sample Preparation")
    md_lines.append("- Composition targets and tolerances (at.%):")
    for item in protocol_json["composition_tolerances"]:
        md_lines.append(f"  - {item['element']}: {item['target_at_pct']} ± {item['tolerance_at_pct']}")
    md_lines.append("- Purity: ≥ 5N feedstock. Arc-melt 20 g ingots, re-melt ×5 under Ar. Hot roll to 1 mm.")
    md_lines.append("- Specimens: Mini-tensile (n=3), Disks Ø3×0.5 mm (n=3), Coupons 10×10×1 mm (n=3).")
    md_lines.append("")
    md_lines.append("### 2. Irradiation Conditions")
    irr = req.irradiation
    md_lines.append(f"- Energy: {irr.get('energy_eV', 14.1e6)} eV; Flux: {irr.get('flux_n_cm2_s', 1e13)} n/cm²/s; Time: {irr.get('time_s',3600)} s; Temp: {irr.get('temperature_C',400)} °C.")
    md_lines.append("- Atmosphere: Helium or high vacuum. Dosimetry: Fe/Ni/Nb/Au foils.")
    md_lines.append("")
    md_lines.append("### 3. PIE Instrumentation")
    for k, v in protocol_json["instruments"].items():
        if isinstance(v, list):
            md_lines.append(f"- {k.capitalize()}: " + ", ".join(v))
        else:
            md_lines.append(f"- {k.capitalize()}: {v}")
    md_lines.append("")
    md_lines.append("### 4. Calibration & Controls")
    md_lines.append("- HPGe: Calibrate with Co-60, Eu-152, Ba-133. Materials: matrix-matched standards.")
    md_lines.append("- Controls: Un-irradiated alloy set and reference material (e.g., EUROFER97).")
    md_lines.append("")
    md_lines.append("### 5. Acceptance Criteria")
    md_lines.append("| Metric | Threshold |")
    md_lines.append("|---|---|")
    for ac in protocol_json["acceptance_criteria"]:
        md_lines.append(f"| {ac['metric']} | {ac['threshold']} |")
    md_lines.append("")
    md_lines.append("### 6. Reporting Template")
    md_lines.append("- Figures: gamma spectra, stress–strain, diffusivity vs T, SEM/TEM micrographs.")
    md_lines.append("- Tables: composition (ICP‑MS), mechanical/thermal stats (mean±std), activity by isotope.")

    markdown = "\n".join(md_lines)
    return {"markdown": markdown, "protocol": protocol_json}


@app.post("/pareto_save")
def route_pareto_save(req: ParetoSaveReq):
    import pandas as pd

    # Determine output CSV path
    target_csv = req.target_csv or os.path.join(ROOT, "data", "ga_top_candidates.csv")
    if not os.path.isabs(target_csv):
        target_csv = os.path.join(ROOT, target_csv)

    try:
        load_models()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Models not loaded: {e}")

    rows = []
    for formula in req.items:
        try:
            comp = parse_formula(formula)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Parse error for '{formula}': {e}")
        pred = predict_with_uncertainty(comp)
        # compute score (mirror generate_and_score)
        score = (
            -0.6 * pred["radioactive"]["mean"]
            - 0.2 * pred.get("transmuted", pred["radioactive"])["mean"]
            + 0.9 * pred["thermal"]["mean"]
            + 0.6 * pred["ductility"]["mean"]
            - 0.4 * (pred["radioactive"]["std"] + pred["thermal"]["std"] + pred["ductility"]["std"])
        )
        rows.append({
            "formula": formula,
            "score": score,
            # compatibility (activation == radioactive)
            "activation_mean": pred["radioactive"]["mean"],
            "activation_std": pred["radioactive"]["std"],
            # explicit
            "radioactive_mean": pred["radioactive"]["mean"],
            "radioactive_std": pred["radioactive"]["std"],
            "transmuted_mean": pred.get("transmuted", {}).get("mean", None),
            "transmuted_std": pred.get("transmuted", {}).get("std", None),
            "thermal_mean": pred["thermal"]["mean"],
            "thermal_std": pred["thermal"]["std"],
            "ductility_mean": pred["ductility"]["mean"],
            "ductility_std": pred["ductility"]["std"],
        })

    new_df = pd.DataFrame(rows)
    if os.path.exists(target_csv):
        try:
            cur = pd.read_csv(target_csv)
        except Exception:
            cur = pd.DataFrame()
    else:
        cur = pd.DataFrame()

    if req.mode == "append" or cur.empty:
        out = pd.concat([cur, new_df], ignore_index=True)
    else:  # upsert by formula
        if "formula" in cur.columns:
            cur = cur[~cur["formula"].isin(new_df["formula"])].copy()
        out = pd.concat([cur, new_df], ignore_index=True)

    os.makedirs(os.path.dirname(target_csv), exist_ok=True)
    out.to_csv(target_csv, index=False)
    return {"saved": len(new_df), "path": target_csv}
