FusionLLM – Physics‑Informed Extreme Materials Explorer

Overview
- Physics‑informed exploration and ranking of fusion cladding candidate alloys using:
  - NuDat3/ENSDF + ENDF nuclear data (via Henry Chance’s Physics code) for activation/transmutation.
  - Lightweight surrogates for thermal/ductility proxies and uncertainty.
  - Streamlit UI for triage, LLM‑assisted rationale, and protocol generation.
  - MCP (FastAPI) service exposing domain tools for agents/automation.

Features
- Physics adapter: per‑element transmutation and radioactivity from ENDF + ENSDF, aggregated by alloy fractions.
- Surrogates: prediction + uncertainty for transmuted, radioactive, thermal, ductility.
- Candidate Explorer: table, detailed metrics, per‑element breakdown, Gemini rationale + follow‑up chat.
- Pareto Explorer: multi‑objective front (min radioactive/transmuted; max thermal/ductility) and CSV export. Save selected to dataset.
- Physics Sweep: heatmaps versus flux/time/energy.
- Protocol Builder: export Markdown + JSON (composition tolerances, instruments, acceptance criteria, irradiation).
- MCP endpoints for external agents; Streamlit can call MCP instead of local functions.

Quick Start
- Create env: `python -m venv .venv && .venv\Scripts\activate && python -m pip install --upgrade pip`
- Install: `pip install -r requirements.txt`
- Build labels: `python -m src.labeling`
- Train surrogates: `python -m src.surrogate`
- Generate candidates (optional): `python -m scripts.generate_and_score`
- Run UI: `streamlit run app_streamlit.py`

MCP Service (optional)
- Start: `python -m mcp` (uses env var `MCP_PORT` or defaults to 8000)
- Endpoints (FastAPI):
  - POST `/compute_element_metrics` { symbol, energy_eV?, flux_n_cm2_s?, time_s?, Natoms? }
  - POST `/evaluate_alloy` { composition }
  - POST `/evaluate_alloy_breakdown` { composition } (per‑element contributions + aggregate)
  - POST `/compute_descriptors` { composition }
  - POST `/predict_surrogates` { composition }
  - POST `/score_candidate` { composition, weights? }
  - POST `/propose_candidates` { n_candidates?, max_elems?, allowed_elements?, seed? }
  - POST `/pareto_candidates` { n_samples?, max_elems?, allowed_elements?, seed? }
  - POST `/sweep_physics` { composition, flux{...}, time_s{...}, energy_eV{...} }
  - POST `/generate_protocol` { composition, tolerance_at_pct?, irradiation{...} } → markdown + JSON
  - POST `/pareto_save` { items: [formula strings], target_csv?, mode? } → upsert to dataset

LLM Setup (Gemini)
- Provide API key via env `GEMINI_API_KEY` or paste it in the Streamlit sidebar under “LLM (Gemini 2.5 Flash)”.

Physics Data & Credits (Henry Chance)
- This project integrates and builds on Henry Chance’s nuclear physics workflow (HR‑Chance/LLM_Hackathon/Physics):
  - Data from ENDF nuclear data files (version VIII.1) and NuDat3/ENSDF half‑life data (IAEA LiveChart API).
  - Two primary scripts in the original project:
    - Command‑line input: `finalCalc_API_CMD_input.py` (e.g., `python3 finalCalc_API_CMD_input.py Zn`).
    - Alloy activation: `alloyCalc.py` to compute activation for alloy claddings.
  - Prereqs (original README): Python ≥ 3.6; packages: `mendeleev`, `numpy`, `matplotlib`, `endf`; stdlib: `urllib.request`, `csv`, `re`, `warnings`.
  - ENDF files in `./ENDF/neutrons-version.VIII.1/` named like `n-<Z>_<Element>_<A>.endf` (e.g., `n-030_Zn_064.endf`).
  - `isotopes_data.py` with isotope abundance data; internet for IAEA LiveChart API.
  - Typical parameters: 14.1 MeV, flux 1e13 n/cm²/s, 3600 s, Natoms 1e23.
  - Outputs: Percentage Transmuted and Product Activity (Bq).

Repo Layout
- `src/physics_adapter.py` – per‑element metrics from ENDF/ENSDF; optional API path for activity (Bq).
- `src/labeling.py` – builds `data/synthetic_labels.csv` with physics‑informed targets.
- `src/surrogate.py` – trains models and provides `predict_with_uncertainty`.
- `scripts/generate_and_score.py` – random candidate generation and scoring.
- `mcp/server.py`, `mcp/__main__.py` – MCP (FastAPI) endpoints.
- `app_streamlit.py` – Streamlit UI with Candidate Explorer, Pareto, Sweep, Protocol.

Install Requirements
- `pip install -r requirements.txt` (key packages: numpy, pandas, scikit‑learn, joblib, pymatgen, transformers, gradio, matplotlib, seaborn, endf, mendeleev, streamlit, google‑generativeai, altair, requests, scikit‑image)

License
- Copyright © contributors.
- Portions adapted from Henry Chance’s LLM_Hackathon/Physics; credit above. Respect the licenses of any vendored data/code.

Contact
- Questions or suggestions: open an issue or PR. For the original Physics code, contact Henry Chance (hchance@utexas.edu).

