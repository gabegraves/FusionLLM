#!/usr/bin/env python3
"""
Improved JSON explainer for top candidates (fusion-aware few-shot, deterministic).
Outputs:
 - data/llm_explanations_top10.json
 - data/llm_explanations_top10.csv
"""
import os, json, pandas as pd, torch, re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL = "google/flan-t5-small"
OUT_JSON = "data/llm_explanations_top10.json"
OUT_CSV  = "data/llm_explanations_top10.csv"

def load_candidates():
    if os.path.exists("data/ga_top_candidates.csv"):
        df = pd.read_csv("data/ga_top_candidates.csv")
    elif os.path.exists("data/top20_seed_scored.csv"):
        df = pd.read_csv("data/top20_seed_scored.csv")
    else:
        raise FileNotFoundError("Run GA or scoring first.")
    return df.head(10)

FEWSHOT = r'''
Example 1
Input: alloy Al0.80-Mo0.20 (activation=1.03±0.05, thermal=0.45±0.03, ductility=0.80±0.04)
Output JSON:
{"reason":"High Al fraction tends to reduce neutron activation and supports ductility; Mo increases thermal conductivity.","experiment":"Measure laser-flash thermal diffusivity and perform a short irradiation proxy (ion-implantation) followed by hardness/EDX to check retention."}

Example 2
Input: alloy W0.90-Ta0.10 (activation=1.20±0.08, thermal=0.60±0.03, ductility=0.55±0.05)
Output JSON:
{"reason":"Dominant W increases thermal capacity but raises activation and lowers ductility; Ta improves high-temperature strength.","experiment":"Run microhardness, small-scale thermal conductivity and a high-temperature creep/indentation test; consider short PKA simulation for damage proxies."}
'''

def make_prompt(row):
    a_mean = float(row.get("activation_mean", row.get("act_mean", 0.0)))
    a_std  = float(row.get("activation_std",  row.get("act_std",  0.0)))
    t_mean = float(row.get("thermal_mean",    row.get("th_mean",   0.0)))
    t_std  = float(row.get("thermal_std",     row.get("th_std",    0.0)))
    d_mean = float(row.get("ductility_mean",  row.get("du_mean",   0.0)))
    d_std  = float(row.get("ductility_std",   row.get("du_std",    0.0)))
    info = f"activation={a_mean:.2f}±{a_std:.2f}, thermal={t_mean:.2f}±{t_std:.2f}, ductility={d_mean:.2f}±{d_std:.2f}"
    prompt = f"""{FEWSHOT}

Input: alloy {row['formula']} ({info})
Output JSON:
"""
    return prompt

def extract_json(text):
    # find first { ... } block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None, text
    jtxt = m.group(0)
    try:
        obj = json.loads(jtxt)
        # clean string fields
        for k,v in obj.items():
            if isinstance(v, str):
                v = v.strip()
                if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                    v = v[1:-1]
                v = v.replace('""','"')
                obj[k] = v
        return obj, text
    except Exception:
        return None, text

def main():
    df = load_candidates()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, "Model:", MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(device)
    model.eval()

    results = []
    for _, r in df.iterrows():
        pr = r.copy()
        prompt = make_prompt(pr)
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        out = model.generate(**inputs, max_new_tokens=140, num_beams=4, do_sample=False)
        text = tok.decode(out[0], skip_special_tokens=True).strip()
        parsed, raw = extract_json(text)
        if parsed is None:
            parsed = {"reason": text, "experiment": "", "raw": raw}
        entry = {
            "formula": pr["formula"],
            "activation_mean": pr.get("activation_mean", pr.get("act_mean", None)),
            "activation_std":  pr.get("activation_std", pr.get("act_std", None)),
            "thermal_mean":    pr.get("thermal_mean", pr.get("th_mean", None)),
            "thermal_std":     pr.get("thermal_std", pr.get("th_std", None)),
            "ductility_mean":  pr.get("ductility_mean", pr.get("du_mean", None)),
            "ductility_std":   pr.get("ductility_std", pr.get("du_std", None)),
            "reason": parsed.get("reason",""),
            "experiment": parsed.get("experiment",""),
            "raw": raw
        }
        results.append(entry)
        print(pr["formula"], "->", entry["reason"])

    pd.DataFrame(results).to_json(OUT_JSON, orient="records", indent=2)
    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print("Wrote", OUT_JSON, "and", OUT_CSV)

if __name__ == '__main__':
    main()
