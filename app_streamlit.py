import os
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import requests
import google.generativeai as genai

from src.descriptors import parse_formula
from src.physics_adapter import compute_element_metrics
from src.surrogate import load_models, predict_with_uncertainty


st.set_page_config(page_title="Extreme Materials Candidate Explorer", layout="wide")
st.title("Extreme Materials Candidate Explorer")
st.caption("Physics-informed activation (NuDat3/ENSDF + ENDF) + surrogate predictions.")


# -------- Data loading --------
DATA_FILE = Path("data/top10_with_explanations.csv")
ALT_FILES = [Path("data/top10_candidates_for_demo.csv"), Path("data/ga_top_candidates.csv")]
if not DATA_FILE.exists():
    for p in ALT_FILES:
        if p.exists():
            DATA_FILE = p
            break


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # normalize column names
    def choose(cols: List[str]):
        # Return the first matching column name from the provided list,
        # also accepting common merge suffixes like _x/_y
        for base in cols:
            if base in df.columns:
                return base
            if f"{base}_x" in df.columns:
                return f"{base}_x"
            if f"{base}_y" in df.columns:
                return f"{base}_y"
        return None

    colmap = {}
    colmap["formula"] = choose(["formula", "Formula"]) or "formula"
    colmap["score"] = choose(["score", "Score"]) or "score"
    # activation alias to radioactive if present
    colmap["activation_mean"] = choose(["activation_mean", "act_mean", "activation", "radioactive_mean"]) or "activation_mean"
    colmap["activation_std"] = choose(["activation_std", "act_std", "radioactive_std"]) or "activation_std"
    # optional explicit physics
    trm, trs = choose(["transmuted_mean"]), choose(["transmuted_std"]) 
    ram, ras = choose(["radioactive_mean"]), choose(["radioactive_std"]) 
    if trm: colmap["transmuted_mean"] = trm
    if trs: colmap["transmuted_std"] = trs
    if ram: colmap["radioactive_mean"] = ram
    if ras: colmap["radioactive_std"] = ras
    # other proxies
    colmap["thermal_mean"] = choose(["thermal_mean", "th_mean", "thermal"]) or "thermal_mean"
    colmap["thermal_std"] = choose(["thermal_std", "th_std"]) or "thermal_std"
    colmap["ductility_mean"] = choose(["ductility_mean", "du_mean", "ductility"]) or "ductility_mean"
    colmap["ductility_std"] = choose(["ductility_std", "du_std"]) or "ductility_std"
    # explanation
    colmap["reason"] = choose(["reason", "explanation"]) or "reason"
    colmap["experiment"] = choose(["experiment", "experiment_suggestion"]) or "experiment"

    # rename in dataframe for convenience
    rename = {v: k for k, v in colmap.items() if v in df.columns}
    df = df.rename(columns=rename)
    # ensure required columns exist
    # ensure required numeric columns exist
    for k in [
        "formula",
        "score",
        "activation_mean",
        "activation_std",
        "transmuted_mean",
        "transmuted_std",
        "thermal_mean",
        "thermal_std",
        "ductility_mean",
        "ductility_std",
    ]:
        if k not in df.columns:
            df[k] = pd.NA

    # coerce numeric columns to numeric (strings like 'None' -> NaN)
    num_cols = [
        "score",
        "activation_mean","activation_std",
        "radioactive_mean","radioactive_std",
        "transmuted_mean","transmuted_std",
        "thermal_mean","thermal_std",
        "ductility_mean","ductility_std",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # if activation_* missing but radioactive_* present, alias
    if "activation_mean" in df.columns and "radioactive_mean" in df.columns:
        df["activation_mean"] = df["activation_mean"].fillna(df["radioactive_mean"])
    if "activation_std" in df.columns and "radioactive_std" in df.columns:
        df["activation_std"] = df["activation_std"].fillna(df["radioactive_std"]) 

    # fallback: predict missing columns with surrogates
    try:
        from src.surrogate import load_models, predict_with_uncertainty  # local import to avoid heavy import cost
        load_models()
        need_cols = ["activation_mean","activation_std","transmuted_mean","transmuted_std","thermal_mean","thermal_std","ductility_mean","ductility_std"]
        missing_mask = df[need_cols].isna().any(axis=1)
        for idx, row in df[missing_mask].iterrows():
            comp = row.get("formula")
            if not isinstance(comp, str) or not comp:
                continue
            try:
                pred = predict_with_uncertainty(comp)
                df.at[idx, "activation_mean"] = df.at[idx, "activation_mean"] if pd.notna(df.at[idx, "activation_mean"]) else pred["radioactive"]["mean"]
                df.at[idx, "activation_std"] = df.at[idx, "activation_std"] if pd.notna(df.at[idx, "activation_std"]) else pred["radioactive"]["std"]
                df.at[idx, "transmuted_mean"] = df.at[idx, "transmuted_mean"] if pd.notna(df.at[idx, "transmuted_mean"]) else pred.get("transmuted", pred["radioactive"])["mean"]
                df.at[idx, "transmuted_std"] = df.at[idx, "transmuted_std"] if pd.notna(df.at[idx, "transmuted_std"]) else pred.get("transmuted", pred["radioactive"])["std"]
                df.at[idx, "thermal_mean"] = df.at[idx, "thermal_mean"] if pd.notna(df.at[idx, "thermal_mean"]) else pred["thermal"]["mean"]
                df.at[idx, "thermal_std"] = df.at[idx, "thermal_std"] if pd.notna(df.at[idx, "thermal_std"]) else pred["thermal"]["std"]
                df.at[idx, "ductility_mean"] = df.at[idx, "ductility_mean"] if pd.notna(df.at[idx, "ductility_mean"]) else pred["ductility"]["mean"]
                df.at[idx, "ductility_std"] = df.at[idx, "ductility_std"] if pd.notna(df.at[idx, "ductility_std"]) else pred["ductility"]["std"]
            except Exception:
                pass
    except Exception:
        # if models not available, keep as-is
        pass
    return df


df = load_data(DATA_FILE)
if df.empty:
    st.warning("No demo CSV found. Expected data/top10_with_explanations.csv (or alternatives).")
else:
    # Sidebar filters/export
    st.sidebar.header("Filters / Export")
    def _slider_vals(series: pd.Series | None, fb_min: float, fb_max: float, fb_def: float, default_quantile: float | None = None):
        try:
            if series is not None and pd.api.types.is_numeric_dtype(series):
                ser = pd.to_numeric(series, errors="coerce")
                if ser.dropna().shape[0] > 0:
                    mn = float(ser.min(skipna=True))
                    mx = float(ser.max(skipna=True))
                    if default_quantile is not None:
                        dv = float(ser.quantile(default_quantile))
                    else:
                        dv = float((mn + mx) / 2.0)
                    if not (mn < mx):
                        margin = max(1e-9, abs(mn) * 0.1 if abs(mn) > 0 else 1e-3)
                        return mn - margin, mx + margin, dv
                    return mn, mx, dv
        except Exception:
            pass
        return fb_min, fb_max, fb_def

    s_min, s_max, s_def = _slider_vals(df.get("score"), 0.0, 1.0, 0.0, default_quantile=0.1)
    min_score = st.sidebar.slider("Min score", s_min, s_max, s_def)

    a_min, a_max, a_def = _slider_vals(df.get("activation_mean"), 0.0, 1.0, 0.0, default_quantile=0.9)
    max_activation = st.sidebar.slider("Max radioactive mean (%)", a_min, a_max, a_def)
    st.sidebar.download_button("Download CSV (shown)", f.to_csv(index=False), "top10_with_explanations.csv", "text/csv")
    st.sidebar.header("MCP Server")
    use_mcp = st.sidebar.checkbox("Use MCP server", value=False)
    mcp_url = st.sidebar.text_input("MCP URL", value=os.environ.get("MCP_URL", "http://localhost:8000"))
    # LLM API key
    st.sidebar.header("LLM (Gemini 2.5 Flash)")
    if "gemini_api_key" not in st.session_state:
        st.session_state["gemini_api_key"] = os.environ.get("GEMINI_API_KEY", "")
    api_in = st.sidebar.text_input("GEMINI_API_KEY", value=st.session_state["gemini_api_key"], type="password")
    if api_in != st.session_state["gemini_api_key"]:
        st.session_state["gemini_api_key"] = api_in
        if api_in:
            os.environ["GEMINI_API_KEY"] = api_in

    # Filtered view
    f = df.copy()
    if pd.api.types.is_numeric_dtype(f["score"]):
        f = f[f["score"] >= min_score]
    if pd.api.types.is_numeric_dtype(f["activation_mean"]):
        f = f[f["activation_mean"] <= max_activation]
    f = f.reset_index(drop=True)

    st.subheader(f"Top candidates (showing {len(f)} rows)")
    cols_show = [c for c in [
        "formula","score",
        "activation_mean","activation_std",
        "radioactive_mean","radioactive_std",
        "transmuted_mean","transmuted_std",
        "thermal_mean","thermal_std","ductility_mean","ductility_std"
    ] if c in f.columns]
    st.dataframe(f[cols_show], height=280)

    if len(f) == 0:
        st.info("No matching rows. Relax filters.")
    else:
        sel = st.selectbox("Select candidate to inspect", f["formula"].tolist())
        row = f[f["formula"] == sel].iloc[0]
        left, right = st.columns([2, 1])
        with left:
            st.markdown("### Candidate summary")
            st.write("**Formula:**", row["formula"])
            if pd.notna(row.get("score")):
                st.write("**Score:**", f"{row['score']:.3f}")
            if pd.notna(row.get("activation_mean")):
                st.write("**Radioactive (%):**", f"{row['activation_mean']:.3f} ± {row['activation_std']:.3f}")
            if pd.notna(row.get("transmuted_mean")):
                st.write("**Transmutation (%):**", f"{row['transmuted_mean']:.3f} ± {row['transmuted_std']:.3f}")
            if pd.notna(row.get("thermal_mean")):
                st.write("**Thermal proxy:**", f"{row['thermal_mean']:.3f} ± {row['thermal_std']:.3f}")
            if pd.notna(row.get("ductility_mean")):
                st.write("**Ductility proxy:**", f"{row['ductility_mean']:.3f} ± {row['ductility_std']:.3f}")

            # per-element physics breakdown
            try:
                comp = parse_formula(str(row["formula"]))
            except Exception:
                comp = {}
            if comp:
                per_rows = []
                t_sum = 0.0
                r_sum = 0.0
                if use_mcp:
                    try:
                        resp = requests.post(f"{mcp_url}/evaluate_alloy_breakdown", json={"composition": comp}, timeout=60)
                        resp.raise_for_status()
                        data = resp.json()
                        for el, info in data["per_element"].items():
                            per_rows.append({
                                "element": el,
                                "fraction": info.get("frac"),
                                "pct_transmuted": info.get("pct_transmuted"),
                                "pct_radioactive": info.get("pct_radioactive"),
                                "weighted_transmuted": info.get("weighted_transmuted"),
                                "weighted_radioactive": info.get("weighted_radioactive"),
                            })
                    except Exception as e:
                        st.warning(f"MCP breakdown failed: {e}")
                else:
                    for el, frac in comp.items():
                        m = compute_element_metrics(el)
                        wt = float(frac) * float(m["pct_transmuted"]) 
                        wr = float(frac) * float(m["pct_radioactive"]) 
                        t_sum += wt
                        r_sum += wr
                        per_rows.append({
                            "element": el,
                            "fraction": float(frac),
                            "pct_transmuted": float(m["pct_transmuted"]),
                            "pct_radioactive": float(m["pct_radioactive"]),
                            "weighted_transmuted": wt,
                            "weighted_radioactive": wr,
                        })
                st.markdown("#### Physics breakdown (per element)")
                st.dataframe(pd.DataFrame(per_rows))

            st.markdown("### LLM reason & suggested quick experiment")
            reason = row.get("reason") if "reason" in row else row.get("explanation")
            experiment = row.get("experiment")
            reason_text = str(reason) if pd.notna(reason) and str(reason).strip().lower() != "nan" else ""
            exp_text = str(experiment) if pd.notna(experiment) and str(experiment).strip().lower() != "nan" else ""
            if reason_text:
                st.markdown(reason_text)
            if exp_text:
                st.markdown("**Suggested quick experiment**\n\n" + exp_text)

            # LLM generation button
            if st.session_state.get("gemini_api_key"):
                if st.button("Generate/upscale explanation with Gemini"):
                    try:
                        genai.configure(api_key=st.session_state["gemini_api_key"])
                        model = genai.GenerativeModel("gemini-2.5-flash")
                        prompt = f"""
You are an experimental materials scientist. Given the alloy candidate and metrics below, output ONLY Markdown with these sections and formatting:

## Reason
- 2–4 concise bullet points justifying suitability for fusion cladding (focus on low activation, ductility, thermal performance).

## Experiment Protocol
### 1. Sample Preparation
- Composition and tolerances (at.%); purity; melt/rolling schedule; number of remelts; specimen geometries and counts.

### 2. Irradiation Conditions
- Facility type; neutron spectrum; target fluence and flux; temperature setpoint; atmosphere; dosimetry plan.

### 3. Post‑Irradiation Examination (PIE)
- Radioactivity (HPGe); Transmutation (ICP‑MS/EPMA); Mechanical (hardness + miniature tensile); Thermal (LFA/DSC); Microstructure (SEM/TEM).

### 4. Calibration & Controls
- Specific calibration references and control samples (un‑irradiated and reference material).

### 5. Acceptance Criteria
| Metric | Threshold | Rationale |
|---|---:|---|
| Total specific gamma activity (Bq/g, Eγ>100 keV, 1‑year decay) | ≤ 1,000 | Waste minimization |
| Max Δ at.% for primaries | ≤ 0.5 | Limited transmutation |
| Thermal conductivity @ 400°C | ≥ 100 W/mK | Heat removal |
| Uniform elongation | ≥ 5% | Non‑brittle behavior |
| Void swelling | ≤ 1% | Microstructural stability |

### 6. Reporting Template
- Bullet list of required figures/tables (spectra, stress‑strain, diffusivity, TEM images) and CSV column names to export.

Context
- Formula: {row['formula']}
- Score: {row.get('score')}
- Radioactive % (mean±std): {row.get('activation_mean')} ± {row.get('activation_std')}
- Transmutation % (mean±std): {row.get('transmuted_mean')} ± {row.get('transmuted_std')}
- Thermal proxy (mean±std): {row.get('thermal_mean')} ± {row.get('thermal_std')}
- Ductility proxy (mean±std): {row.get('ductility_mean')} ± {row.get('ductility_std')}
"""
                        # Clean up mojibake artifacts in prompt
                        prompt = (
                            prompt
                            .replace('A�', '±')
                            .replace('�?`', '-')
                            .replace('�?"', '-')
                            .replace('I"', 'delta')
                            .replace('A~', '~')
                            .replace('A-','x')
                            .replace('�', '')
                        )
                        resp = model.generate_content(prompt)
                        txt = resp.text.strip() if hasattr(resp, 'text') else str(resp)
                        st.markdown(txt)
                    except Exception as e:
                        st.warning(f"LLM generation failed: {e}")

        with right:
            st.markdown("### Quick actions")
            st.download_button("Download candidate CSV", f[f['formula']==sel].to_csv(index=False), f"{sel}_candidate.csv", "text/csv")
            st.download_button("Download top3 for slides", f.head(3).to_csv(index=False), "top3_for_slides.csv", "text/csv")
            # Follow-up chat
            st.markdown("### Follow-up chat")
            if "chat_history" not in st.session_state:
                st.session_state["chat_history"] = []
            for msg in st.session_state["chat_history"]:
                role = "You" if msg.get("role") == "user" else "Gemini"
                st.write(f"{role}: {msg.get('parts',[{}])[0].get('text','')}")
            question = st.text_input("Ask about this candidate")
            if st.session_state.get("gemini_api_key") and question:
                if st.button("Send"):
                    try:
                        genai.configure(api_key=st.session_state["gemini_api_key"])
                        model = genai.GenerativeModel("gemini-2.5-flash")
                        system = f"We are discussing alloy {row['formula']} with properties: score={row.get('score')}, radioactive={row.get('activation_mean')}±{row.get('activation_std')}%, transmutation={row.get('transmuted_mean')}±{row.get('transmuted_std')}%. Provide technical, actionable answers."
                        # sanitize any stray encoding artifacts
                        system = system.replace('A�', '±')
                        hist = st.session_state["chat_history"] + [
                            {"role": "user", "parts": [{"text": system}]},
                            {"role": "user", "parts": [{"text": question}]},
                        ]
                        conv = model.start_chat(history=hist)
                        ans = conv.send_message(question)
                        reply = ans.text if hasattr(ans, 'text') else str(ans)
                        st.session_state["chat_history"] += [
                            {"role": "user", "parts": [{"text": question}]},
                            {"role": "model", "parts": [{"text": reply}]},
                        ]
                        try:
                            st.rerun()
                        except Exception:
                            st.experimental_rerun()
                    except Exception as e:
                        st.warning(f"Chat failed: {e}")


# -------- Advanced Tools Tabs --------
st.markdown("## Advanced Tools")
tab_pareto, tab_sweep, tab_protocol = st.tabs(["Pareto Explorer", "Physics Sweep", "Protocol Builder"])


def _non_dominated_front(points: List[List[float]]):
    idxs = []
    for i, pi in enumerate(points):
        dominated = False
        for j, pj in enumerate(points):
            if j == i:
                continue
            if all(pj[k] <= pi[k] for k in range(len(pi))) and any(pj[k] < pi[k] for k in range(len(pi))):
                dominated = True
                break
        if not dominated:
            idxs.append(i)
    return idxs


with tab_pareto:
    st.markdown("Explore the multi-objective front: minimize radioactivity/transmutation; maximize thermal/ductility.")
    n_samples = st.number_input("Random samples", min_value=100, max_value=10000, value=1500, step=100)
    colA, colB, colC = st.columns(3)
    with colA:
        max_elems = st.slider("Max elements per alloy", 1, 5, 4)
    with colB:
        min_elems = st.slider("Min elements per alloy", 1, 5, 2)
    with colC:
        dirichlet_alpha = st.number_input("Dirichlet α (mixing)", value=2.0, min_value=0.5, step=0.5)
    try:
        allowed = json.load(open("data/elements_list.json"))
    except Exception:
        allowed = ["W","Cu","Fe","Cr","Mo","Ta","Nb","V","Ti","Al","Si","Mn"]
    chosen = st.multiselect("Allowed elements", options=allowed, default=allowed)
    seed = st.number_input("Seed (optional)", value=0, step=1)
    if st.button("Compute Pareto Front"):
        np.random.seed(int(seed) if seed else None)
        import random
        if seed:
            random.seed(int(seed))
        records = []
        if use_mcp:
            try:
                payload = {
                    "n_samples": int(n_samples),
                    "max_elems": int(max_elems),
                    "allowed_elements": chosen,
                    "seed": int(seed) if seed else None,
                }
                resp = requests.post(f"{mcp_url}/pareto_candidates", json=payload, timeout=120)
                resp.raise_for_status()
                front = resp.json().get("front", [])
                for item in front:
                    comp = item.get("composition", {})
                    # format to formula string
                    formula = "-".join(f"{k}{round(float(v),4)}" for k,v in comp.items())
                    p = item.get("pred", {})
                    records.append({
                        "formula": formula,
                        "radioactive": p.get("radioactive", {}).get("mean"),
                        "transmuted": p.get("transmuted", p.get("radioactive", {})).get("mean"),
                        "thermal": p.get("thermal", {}).get("mean"),
                        "ductility": p.get("ductility", {}).get("mean"),
                    })
            except Exception as e:
                st.error(f"MCP pareto failed: {e}")
        else:
            try:
                load_models()
            except Exception as e:
                st.error(f"Models not loaded: {e}. Run: python -m src.surrogate")
            def rand_comp() -> Dict[str, float]:
                k = random.randint(int(min_elems), max(1, int(max_elems)))
                els = random.sample(chosen, k)
                # Dirichlet for better mixing
                alpha = [float(dirichlet_alpha)] * k
                frs = list(np.random.dirichlet(alpha))
                return {el: fr for el, fr in zip(els, frs)}
            comps = [rand_comp() for _ in range(int(n_samples))]
            preds = [predict_with_uncertainty(c) for c in comps]
            objs = [[p["radioactive"]["mean"], p.get("transmuted", p["radioactive"]).get("mean", 0.0), -p["thermal"]["mean"], -p["ductility"]["mean"]] for p in preds]
            front_idx = _non_dominated_front(objs)
            for i in front_idx:
                records.append({
                    "formula": "-".join(f"{k}{round(v,4)}" for k,v in comps[i].items()),
                    "radioactive": preds[i]["radioactive"]["mean"],
                    "transmuted": preds[i].get("transmuted", preds[i]["radioactive"]).get("mean", 0.0),
                    "thermal": preds[i]["thermal"]["mean"],
                    "ductility": preds[i]["ductility"]["mean"],
                })
        # Deduplicate by formula and sort
        df_front = pd.DataFrame(records).drop_duplicates(subset=["formula"]).sort_values(["radioactive","transmuted","thermal"], ascending=[True, True, False])
        st.dataframe(df_front, height=280)
        st.download_button("Download Pareto CSV", df_front.to_csv(index=False), "pareto_front.csv", "text/csv")
        if len(df_front) > 0:
            chart = alt.Chart(df_front).mark_circle().encode(
                x=alt.X("radioactive:Q", title="Radioactive (%)"),
                y=alt.Y("transmuted:Q", title="Transmutation (%)"),
                color=alt.Color("thermal:Q", scale=alt.Scale(scheme="viridis")),
                size=alt.Size("ductility:Q"),
                tooltip=["formula","radioactive","transmuted","thermal","ductility"],
            )
            st.altair_chart(chart, use_container_width=True)
        # Save selected
        if len(records) > 0:
            sel_formulas = st.multiselect("Select formulas to save to main dataset", [r["formula"] for r in records])
            if st.button("Save Selected") and sel_formulas:
                if use_mcp:
                    try:
                        payload = {"items": sel_formulas}
                        resp = requests.post(f"{mcp_url}/pareto_save", json=payload, timeout=60)
                        resp.raise_for_status()
                        st.success(f"Saved {len(sel_formulas)} to dataset")
                    except Exception as e:
                        st.error(f"Pareto save failed: {e}")
                else:
                    # local save using surrogates
                    try:
                        load_models()
                        rows = []
                        for formula in sel_formulas:
                            comp = parse_formula(formula)
                            pred = predict_with_uncertainty(comp)
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
                                "activation_mean": pred["radioactive"]["mean"],
                                "activation_std": pred["radioactive"]["std"],
                                "radioactive_mean": pred["radioactive"]["mean"],
                                "radioactive_std": pred["radioactive"]["std"],
                                "transmuted_mean": pred.get("transmuted", {}).get("mean", None),
                                "transmuted_std": pred.get("transmuted", {}).get("std", None),
                                "thermal_mean": pred["thermal"]["mean"],
                                "thermal_std": pred["thermal"]["std"],
                                "ductility_mean": pred["ductility"]["mean"],
                                "ductility_std": pred["ductility"]["std"],
                            })
                        target_csv = os.path.join("data", "ga_top_candidates.csv")
                        if os.path.exists(target_csv):
                            cur = pd.read_csv(target_csv)
                        else:
                            cur = pd.DataFrame()
                        new_df = pd.DataFrame(rows)
                        if not cur.empty and "formula" in cur.columns:
                            cur = cur[~cur["formula"].isin(new_df["formula"])].copy()
                        out = pd.concat([cur, new_df], ignore_index=True)
                        out.to_csv(target_csv, index=False)
                        st.success(f"Saved {len(sel_formulas)} to {target_csv}")
                    except Exception as e:
                        st.error(f"Local save failed: {e}")


with tab_sweep:
    st.markdown("Run physics sweeps vs flux/time/energy for a given composition.")
    default_formula = df["formula"].iloc[0] if not df.empty else "W0.6-Cu0.4"
    form = st.text_input("Composition formula", value=str(default_formula))
    col1, col2, col3 = st.columns(3)
    with col1:
        flux_min = st.number_input("Flux min (n/cm^2/s)", value=1e12, format="%e")
        flux_max = st.number_input("Flux max (n/cm^2/s)", value=1e14, format="%e")
        flux_n = st.number_input("Flux steps", value=4, step=1)
    with col2:
        time_min = st.number_input("Time min (s)", value=3600.0)
        time_max = st.number_input("Time max (s)", value=86400.0)
        time_n = st.number_input("Time steps", value=4, step=1)
    with col3:
        energy = st.text_input("Energies (eV, comma-separated)", value="14100000")
    if st.button("Run Sweep"):
        try:
            comp = parse_formula(form)
        except Exception as e:
            st.error(f"Parse error: {e}")
            comp = None
        if comp:
            flux_vals = list(np.linspace(float(flux_min), float(flux_max), int(flux_n)))
            time_vals = list(np.linspace(float(time_min), float(time_max), int(time_n)))
            try:
                energy_vals = [float(x.strip()) for x in energy.split(',') if x.strip()]
            except Exception:
                energy_vals = [14.1e6]
            if use_mcp:
                payload = {
                    "composition": comp,
                    "flux": {"values": flux_vals},
                    "time_s": {"values": time_vals},
                    "energy_eV": {"values": energy_vals},
                }
                try:
                    resp = requests.post(f"{mcp_url}/sweep_physics", json=payload, timeout=120)
                    resp.raise_for_status()
                    data = resp.json()
                    df_sweep = pd.DataFrame(data.get("records", []))
                except Exception as e:
                    st.error(f"MCP sweep failed: {e}")
                    df_sweep = pd.DataFrame()
            else:
                rows = []
                for E in energy_vals:
                    for F in flux_vals:
                        for T in time_vals:
                            t_sum = 0.0
                            r_sum = 0.0
                            for el, frac in comp.items():
                                m = compute_element_metrics(el, energy_eV=E, flux_n_cm2_s=F, time_s=T)
                                t_sum += float(frac) * float(m["pct_transmuted"]) 
                                r_sum += float(frac) * float(m["pct_radioactive"]) 
                            rows.append({"energy_eV":E, "flux":F, "time_s":T, "pct_transmuted":t_sum, "pct_radioactive":r_sum})
                df_sweep = pd.DataFrame(rows)
            st.dataframe(df_sweep, height=280)
            # basic charts for first energy
            for E in energy_vals:
                sub = df_sweep[df_sweep["energy_eV"] == E]
                if len(sub) == 0:
                    continue
                st.markdown(f"#### Heatmaps @ energy {E:.2e} eV")
                for metric in ["pct_radioactive", "pct_transmuted"]:
                    chart = alt.Chart(sub).mark_rect().encode(
                        x=alt.X("flux:Q", title="Flux (n/cm^2/s)"),
                        y=alt.Y("time_s:Q", title="Time (s)"),
                        color=alt.Color(f"{metric}:Q", scale=alt.Scale(scheme="inferno")),
                        tooltip=["flux","time_s",metric]
                    )
                    st.altair_chart(chart, use_container_width=True)


with tab_protocol:
    st.markdown("Generate a structured protocol (Markdown + JSON) for a given composition.")
    default_formula = df["formula"].iloc[0] if not df.empty else "W0.6-Cu0.4"
    form = st.text_input("Composition formula (for protocol)", value=str(default_formula), key="proto_formula")
    tol = st.number_input("Tolerance (at.%)", min_value=0.01, max_value=1.0, value=0.05, step=0.01)
    irr_col1, irr_col2 = st.columns(2)
    with irr_col1:
        irr_flux = st.number_input("Flux (n/cm^2/s)", value=1e13, format="%e")
        irr_time = st.number_input("Time (s)", value=3600.0)
    with irr_col2:
        irr_energy = st.number_input("Energy (eV)", value=14.1e6)
        irr_temp = st.number_input("Temperature (°C)", value=400.0)
    if st.button("Generate Protocol"):
        try:
            comp = parse_formula(form)
        except Exception as e:
            st.error(f"Parse error: {e}")
            comp = None
        if comp:
            if use_mcp:
                payload = {
                    "composition": comp,
                    "tolerance_at_pct": float(tol),
                    "irradiation": {
                        "flux_n_cm2_s": float(irr_flux),
                        "time_s": float(irr_time),
                        "energy_eV": float(irr_energy),
                        "temperature_C": float(irr_temp),
                    },
                }
                try:
                    resp = requests.post(f"{mcp_url}/generate_protocol", json=payload, timeout=60)
                    resp.raise_for_status()
                    data = resp.json()
                    protocol_json = data.get("protocol", {})
                    md = data.get("markdown", "")
                except Exception as e:
                    st.error(f"MCP protocol failed: {e}")
                    protocol_json = {}
                    md = ""
            else:
                comp_pct = {el: 100.0 * float(fr) for el, fr in comp.items()}
                protocol_json = {
                    "composition_tolerances": [
                        {"element": el, "target_at_pct": round(val, 6), "tolerance_at_pct": float(tol)}
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
                    "irradiation": {
                        "flux_n_cm2_s": float(irr_flux),
                        "time_s": float(irr_time),
                        "energy_eV": float(irr_energy),
                        "temperature_C": float(irr_temp),
                    }
                }
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
            irr = protocol_json["irradiation"]
            md_lines.append(f"- Energy: {irr['energy_eV']} eV; Flux: {irr['flux_n_cm2_s']} n/cm^2/s; Time: {irr['time_s']} s; Temp: {irr['temperature_C']} °C.")
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
            md = "\n".join(md_lines)
            # Clean up mojibake artifacts in rendered Markdown
            md = (
                md
                .replace('A�', '±')
                .replace('�?`', '-')
                .replace('�?"', '-')
                .replace('A~', '~')
                .replace('A-', 'x')
                .replace('�', '')
            )
            st.markdown(md)
            st.download_button("Download protocol.json", json.dumps(protocol_json, indent=2), "protocol.json", "application/json")
            st.download_button("Download protocol.md", md, "protocol.md", "text/markdown")
