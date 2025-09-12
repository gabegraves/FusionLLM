import streamlit as st
import pandas as pd, textwrap
from pathlib import Path

st.set_page_config(page_title="Extreme Materials — Candidate Explorer", layout="wide")
st.title("Extreme Materials — Candidate Explorer")
st.markdown("**Demo:** surrogate predictions + LLM explanations for fusion-first-wall candidate alloys.")

DATA_FILE = Path("data/top10_with_explanations.csv")
ALT_FILES = [Path("data/top10_candidates_for_demo.csv"), Path("data/ga_top_candidates.csv")]
if not DATA_FILE.exists():
    for p in ALT_FILES:
        if p.exists():
            DATA_FILE = p
            break

if not DATA_FILE.exists():
    st.warning(f"No demo CSV found. Expected data/top10_with_explanations.csv (or alternatives).")
    st.stop()

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize column names (handle _x/_y suffixes or alt names)
    colmap = {}
    def choose(cols):
        for c in cols:
            if c in df.columns: return c
        return None
    colmap["formula"] = choose(["formula", "Formula"])
    colmap["score"] = choose(["score", "Score"])
    colmap["activation_mean"] = choose(["activation_mean","activation_mean_x","activation_mean_y","act_mean","activation"])
    colmap["activation_std"] = choose(["activation_std","activation_std_x","activation_std_y","act_std"])
    colmap["thermal_mean"] = choose(["thermal_mean","thermal_mean_x","thermal_mean_y","th_mean","thermal"])
    colmap["thermal_std"] = choose(["thermal_std","thermal_std_x","thermal_std_y","th_std"])
    colmap["ductility_mean"] = choose(["ductility_mean","ductility_mean_x","ductility_mean_y","du_mean","ductility"])
    colmap["ductility_std"] = choose(["ductility_std","ductility_std_x","ductility_std_y","du_std"])
    # explanation fields
    colmap["reason"] = choose(["reason","explanation"])
    colmap["experiment"] = choose(["experiment","experiment_suggestion"])

    # rename in dataframe for convenience
    rename = {v:k for k,v in colmap.items() if v is not None}
    df = df.rename(columns=rename)
    # ensure formula exists
    if "formula" not in df.columns:
        st.error("Could not find a 'formula' column in the CSV. Columns found: " + ", ".join(df.columns))
        st.stop()
    # fill missing numeric cols with NaNs
    for k in ["score","activation_mean","activation_std","thermal_mean","thermal_std","ductility_mean","ductility_std"]:
        if k not in df.columns:
            df[k] = pd.NA
    return df

df = load_data(DATA_FILE)

# Sidebar filters
st.sidebar.header("Filters / Export")
min_score = st.sidebar.slider("Min score", float(df["score"].min(skipna=True) if pd.api.types.is_numeric_dtype(df["score"]) else 0.0),
                                          float(df["score"].max(skipna=True) if pd.api.types.is_numeric_dtype(df["score"]) else 1.0),
                                          float(df["score"].quantile(0.1) if pd.api.types.is_numeric_dtype(df["score"]) else 0.0))
max_activation = st.sidebar.slider("Max activation mean", float(df["activation_mean"].min(skipna=True) if pd.api.types.is_numeric_dtype(df["activation_mean"]) else 10.0),
                                                    float(df["activation_mean"].max(skipna=True) if pd.api.types.is_numeric_dtype(df["activation_mean"]) else 0.0),
                                                    float(df["activation_mean"].quantile(0.9) if pd.api.types.is_numeric_dtype(df["activation_mean"]) else 10.0))
st.sidebar.download_button("Download CSV (shown)", df.to_csv(index=False), "top10_with_explanations.csv", "text/csv")

# filtering (tolerant to missing numeric data)
f = df.copy()
if pd.api.types.is_numeric_dtype(f["score"]):
    f = f[f["score"] >= min_score]
if pd.api.types.is_numeric_dtype(f["activation_mean"]):
    f = f[f["activation_mean"] <= max_activation]
f = f.reset_index(drop=True)

st.subheader(f"Top candidates (showing {len(f)} rows)")
cols_show = [c for c in ["formula","score","activation_mean","activation_std","thermal_mean","thermal_std","ductility_mean","ductility_std"] if c in f.columns]
st.dataframe(f[cols_show], height=300)

if len(f) == 0:
    st.info("No matching rows. Relax filters.")
else:
    sel = st.selectbox("Select candidate to inspect", f["formula"].tolist())
    row = f[f["formula"] == sel].iloc[0]
    left, right = st.columns([2,1])
    with left:
        st.markdown("### Candidate summary")
        st.write("**Formula:**", row["formula"])
        if pd.notna(row.get("score")): st.write("**Score:**", f"{row['score']:.3f}")
        if pd.notna(row.get("activation_mean")):
            st.write("**Activation:**", f"{row['activation_mean']:.3f} ± {row['activation_std']:.3f}")
            st.write("**Thermal proxy:**", f"{row['thermal_mean']:.3f} ± {row['thermal_std']:.3f}")
            st.write("**Ductility proxy:**", f"{row['ductility_mean']:.3f} ± {row['ductility_std']:.3f}")
        st.markdown("### LLM reason & suggested quick experiment")
        reason = row.get("reason") or row.get("explanation") or ""
        experiment = row.get("experiment") or ""
        if reason:
            st.write(textwrap.fill(str(reason), 300))
        if experiment:
            st.info("Suggested quick experiment: " + str(experiment))
    with right:
        st.markdown("### Quick actions")
        st.download_button("Download candidate CSV", f[f['formula']==sel].to_csv(index=False), f"{sel}_candidate.csv", "text/csv")
        st.download_button("Download top3 for slides", f.head(3).to_csv(index=False), "top3_for_slides.csv", "text/csv")

st.markdown("---")
st.caption("Built by Extreme Materials — demo UI for rapid materials triage (fusion).")
