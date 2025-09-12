# scripts/clean_and_merge_explanations.py
import pandas as pd
import os

def normalize_formula(f):
    if pd.isna(f):
        return ""
    return str(f).replace(" ", "").replace(",", "").replace('"','').strip()

IN_EXPL = "data/llm_explanations_top10.csv"
IN_CANDS = "data/ga_top_candidates.csv"
OUT = "data/top10_with_explanations_clean.csv"

if not os.path.exists(IN_EXPL):
    raise SystemExit(f"Missing {IN_EXPL}")
if not os.path.exists(IN_CANDS):
    raise SystemExit(f"Missing {IN_CANDS}")

expl = pd.read_csv(IN_EXPL, dtype=str).fillna("")
expl['formula'] = expl['formula'].apply(normalize_formula)

# clean text fields
for col in ['reason','experiment','raw']:
    if col in expl.columns:
        expl[col] = expl[col].astype(str).str.replace('"""', '"').str.replace('\n',' ').str.strip()

cands = pd.read_csv(IN_CANDS, dtype=str).fillna("")
cands['formula'] = cands['formula'].apply(normalize_formula)

merged = cands.merge(expl, on="formula", how="left")

# report how many missing explanations after merge
missing = merged['reason'].isna().sum() if 'reason' in merged.columns else len(merged)
print(f"Rows: {len(merged)}  missing reasons: {missing}")

merged.to_csv(OUT, index=False)
print("Wrote", OUT)

