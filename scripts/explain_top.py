#!/usr/bin/env python3
"""
Generate one-line LLM explanations for top candidates.

Usage:
  python scripts/explain_top.py

Prefers data/ga_top_candidates.csv, falls back to data/top20_seed_scored.csv.
Outputs: data/llm_explanations_top10.csv
"""
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL = "google/flan-t5-small"
OUT_CSV = "data/llm_explanations_top10.csv"

def load_candidates():
    if os.path.exists("data/ga_top_candidates.csv"):
        df = pd.read_csv("data/ga_top_candidates.csv")
    elif os.path.exists("data/top20_seed_scored.csv"):
        df = pd.read_csv("data/top20_seed_scored.csv")
    else:
        raise FileNotFoundError("No candidate CSV found. Run scoring or GA first.")
    return df.head(10)

def make_prompt(row):
    return (
        f"One concise sentence: why might the alloy {row['formula']} score "
        f"activation={row['activation_mean']:.2f}±{row['activation_std']:.2f}, "
        f"thermal={row['thermal_mean']:.2f}±{row['thermal_std']:.2f}, "
        f"ductility={row['ductility_mean']:.2f}±{row['ductility_std']:.2f}? "
        "Mention one plausible physical reason and one quick experiment to check it."
    )

def main():
    df = load_candidates()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}; model: {MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(device)
    model.eval()

    outputs = []
    for _, r in df.iterrows():
        # normalize column names if needed
        row = r.copy()
        for alt in [("activation_mean","act_mean"), ("activation_std","act_std"),
                    ("thermal_mean","th_mean"), ("thermal_std","th_std"),
                    ("ductility_mean","du_mean"), ("ductility_std","du_std")]:
            if (alt[0] not in row or pd.isna(row.get(alt[0], float("nan")))) and alt[1] in row:
                row[alt[0]] = row[alt[1]]
        prompt = make_prompt(row)
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
        out = model.generate(**inputs, max_new_tokens=64, num_beams=4, temperature=0.2)
        text = tok.decode(out[0], skip_special_tokens=True)
        outputs.append({"formula": row["formula"], "explanation": text})
        print(row["formula"], "->", text)

    pd.DataFrame(outputs).to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")

if __name__ == "__main__":
    main()
