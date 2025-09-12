# app_streamlit.py
import streamlit as st
import pandas as pd
import textwrap

st.set_page_config(page_title="Extreme Materials — Top Candidates", layout="wide")

st.title("Extreme Materials — Candidate Explorer")
st.markdown("**Fast triage for fusion first-wall alloys —** surrogate scores + LLM explanations (demo).")

@st.cache_data
def load_data():
    # prefer cleaned merge file
    for fn in ("data/top10_with_explanations_clean.csv", "data/top10_with_explanations.csv"):
        try:
            df = pd.read_csv(fn)
            return df.fillna("")
        except FileNotFoundError:
            continue
    return pd.DataFrame()

df = load_data()
if df.empty:
    st.warning("No demo CSV found. Run the generator & explainer scripts first.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters & Export")
min_score = st.sidebar.slider("Min score", float(df["score"].min()), float(df["score"].max()), float(df["score"].quantile(0.1)))
max_activation = st.sidebar.slider("Max activation mean", float(df["activation_mean_x"].min()), float(df["activation_mean_x"].max()), float(df["activation_mean_x"].quantile(0.9)))
st.sidebar.write("Download full tables:")
st.sidebar.download_button("Download top10 CSV", df.to_csv(index=False), "top10_with_explanations.csv", "text/csv")

# Main table
st.subheader("Top candidates (filtered)")
# adapt column names if your file uses _x suffixes (from previous merge)
act_col = "activation_mean_x" if "activation_mean_x" in df.columns else "activation_mean"
therm_col = "thermal_mean_x" if "thermal_mean_x" in df.columns else "thermal_mean"
duct_col = "ductility_mean_x" if "ductility_mean_x" in df.columns else "ductility_mean"
act_std_col = "activation_std_x" if "activation_std_x" in df.columns else "activation_std"
therm_std_col = "thermal_std_x" if "thermal_std_x" in df.columns else "thermal_std"
duct_std_col = "ductility_std_x" if "ductility_std_x" in df.columns else "ductility_std"

filtered = df[(df["score"] >= min_score) & (df[act_col] <= max_activation)].reset_index(drop=True)
st.dataframe(filtered[[ "formula","score", act_col, act_std_col, therm_col, therm_std_col, duct_col, duct_std_col ]].rename(columns={
    act_col: "activation_mean", act_std_col: "activation_std",
    therm_col: "thermal_mean", therm_std_col: "thermal_std",
    duct_col: "ductility_mean", duct_std_col: "ductility_std"
}), height=300)

# Detailed view for selection
sel = st.selectbox("Select candidate to inspect", filtered["formula"].tolist())
row = filtered[filtered["formula"] == sel].iloc[0]

st.markdown("### Candidate summary")
cols = st.columns([2,1])
with cols[0]:
    st.write("**Formula:**", row["formula"])
    st.write("**Score:**", f"{row['score']:.3f}")
    st.write("**Activation:**", f"{row[act_col]:.3f} ± {row[act_std_col]:.3f}")
    st.write("**Thermal proxy:**", f"{row[therm_col]:.3f} ± {row[therm_std_col]:.3f}")
    st.write("**Ductility proxy:**", f"{row[duct_col]:.3f} ± {row[duct_std_col]:.3f}")
with cols[1]:
    st.download_button("Download candidate CSV", filtered[filtered["formula"]==sel].to_csv(index=False), f"{sel}_candidate.csv", "text/csv")
    st.download_button("Download top3 CSV", filtered.head(3).to_csv(index=False), "top3_for_slides.csv", "text/csv")

st.markdown("### LLM reasoning & quick experiment")
# pick available text columns (try 'reason' then 'explanation' then 'raw')
reason = row.get("reason") or row.get("explanation") or row.get("raw") or ""
experiment = row.get("experiment") or ""

if reason:
    st.write(textwrap.fill(str(reason), 300))
else:
    st.write("_No reason available_")

if experiment:
    st.info("Suggested quick experiment: " + str(experiment))
else:
    st.info("Suggested quick experiment: (none extracted) — consider running explainer LLM with explicit 'experiment' instruction.")

st.markdown("---")
st.caption("Built by Extreme Materials — demo UI for fast screening. Export candidates to prepare DFT/experiments.")
