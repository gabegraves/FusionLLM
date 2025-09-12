# app_streamlit.py
import streamlit as st
import pandas as pd
import textwrap

st.set_page_config(page_title="Extreme Materials — Top Candidates", layout="wide")

st.title("Extreme Materials — Candidate Explorer")
st.markdown("**Fast triage for fusion first-wall alloys —** surrogate scores + LLM explanations (demo).")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/top10_with_explanations.csv")
    except FileNotFoundError:
        df = pd.DataFrame()
    return df

df = load_data()
if df.empty:
    st.warning("No demo CSV found. Run the generator & explainer scripts first.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters & Export")
min_score = st.sidebar.slider("Min score", float(df["score"].min()), float(df["score"].max()), float(df["score"].quantile(0.1)))
max_activation = st.sidebar.slider("Max activation mean", float(df["activation_mean"].min()), float(df["activation_mean"].max()), float(df["activation_mean"].quantile(0.9)))
st.sidebar.write("Download full tables:")
st.sidebar.download_button("Download top10 CSV", df.to_csv(index=False), "top10_with_explanations.csv", "text/csv")

# Main table
st.subheader("Top candidates (filtered)")
filtered = df[(df["score"] >= min_score) & (df["activation_mean"] <= max_activation)].reset_index(drop=True)
st.dataframe(filtered[["formula","score","activation_mean","activation_std","thermal_mean","thermal_std","ductility_mean","ductility_std"]], height=300)

# Detailed view for selection
sel = st.selectbox("Select candidate to inspect", filtered["formula"].tolist())
row = filtered[filtered["formula"] == sel].iloc[0]

st.markdown("### Candidate summary")
cols = st.columns([2,1])
with cols[0]:
    st.write("**Formula:**", row["formula"])
    st.write("**Score:**", f"{row['score']:.3f}")
    st.write("**Activation:**", f"{row['activation_mean']:.3f} ± {row['activation_std']:.3f}")
    st.write("**Thermal proxy:**", f"{row['thermal_mean']:.3f} ± {row['thermal_std']:.3f}")
    st.write("**Ductility proxy:**", f"{row['ductility_mean']:.3f} ± {row['ductility_std']:.3f}")
with cols[1]:
    st.download_button("Download candidate CSV", filtered[filtered["formula"]==sel].to_csv(index=False), f"{sel}_candidate.csv", "text/csv")

st.markdown("### LLM reasoning & quick experiment")
reason = row.get("reason") or row.get("explanation") or ""
experiment = row.get("experiment") or ""
if reason:
    st.write(textwrap.fill(str(reason), 200))
if experiment:
    st.info("Suggested quick experiment: " + str(experiment))

st.markdown("---")
st.caption("Built by Extreme Materials — demo UI for fast screening. Export candidates to prepare DFT/experiments.")

