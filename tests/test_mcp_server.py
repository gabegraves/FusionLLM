import os
import json
import tempfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mcp.server import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_compute_element_metrics_ok(client):
    r = client.post(
        "/compute_element_metrics",
        json={"symbol": "W", "energy_eV": 14.1e6, "flux_n_cm2_s": 1e13, "time_s": 3600, "Natoms": 1e23},
    )
    assert r.status_code == 200
    data = r.json()
    assert set(data.keys()) == {"pct_transmuted", "pct_radioactive"}
    assert data["pct_transmuted"] >= 0
    assert data["pct_radioactive"] >= 0


def test_compute_element_metrics_bad_symbol(client):
    r = client.post(
        "/compute_element_metrics",
        json={"symbol": "ZZ", "energy_eV": 14.1e6, "flux_n_cm2_s": 1e13, "time_s": 3600, "Natoms": 1e23},
    )
    assert r.status_code == 400


def test_evaluate_alloy_and_breakdown(client):
    comp = {"W": 0.6, "Cu": 0.4}
    r1 = client.post("/evaluate_alloy", json={"composition": comp})
    assert r1.status_code == 200
    agg = r1.json()
    assert set(agg.keys()) == {"pct_transmuted", "pct_radioactive"}

    r2 = client.post("/evaluate_alloy_breakdown", json={"composition": comp})
    assert r2.status_code == 200
    data = r2.json()
    per = data["per_element"]
    # aggregation should match weighted sum of per-element
    t_sum = sum(per[e]["weighted_transmuted"] for e in per)
    r_sum = sum(per[e]["weighted_radioactive"] for e in per)
    assert pytest.approx(t_sum, rel=1e-6) == agg["pct_transmuted"]
    assert pytest.approx(r_sum, rel=1e-6) == agg["pct_radioactive"]


def test_compute_descriptors(client):
    comp = {"W": 0.6, "Cu": 0.4}
    r = client.post("/compute_descriptors", json={"composition": comp})
    assert r.status_code == 200
    feats = r.json()
    for k in ["avg_Z", "avg_chi", "avg_mass", "std_atomic_radius", "n_elements", "contains_Cu", "contains_W", "frac_vector"]:
        assert k in feats
    assert isinstance(feats["frac_vector"], list)
    assert len(feats["frac_vector"]) > 0
    assert pytest.approx(sum(feats["frac_vector"]), rel=1e-6) == 1.0


def test_predict_surrogates_and_score(client):
    comp = {"W": 0.6, "Cu": 0.4}
    r = client.post("/predict_surrogates", json={"composition": comp})
    assert r.status_code == 200
    pred = r.json()
    for k in ["radioactive", "transmuted", "thermal", "ductility"]:
        assert k in pred
        assert set(pred[k].keys()) == {"mean", "std"}

    r2 = client.post("/score_candidate", json={"composition": comp})
    assert r2.status_code == 200
    out = r2.json()
    assert "score" in out and "pred" in out
    # recompute expected score from returned pred
    pr = out["pred"]
    score_expected = (
        -0.6 * pr["radioactive"]["mean"]
        - 0.2 * pr.get("transmuted", pr["radioactive"])["mean"]
        + 0.9 * pr["thermal"]["mean"]
        + 0.6 * pr["ductility"]["mean"]
    )
    assert pytest.approx(score_expected, rel=1e-10) == out["score"]


def test_propose_candidates_and_pareto(client):
    # Small sizes for quick test
    r = client.post("/propose_candidates", json={"n_candidates": 10, "max_elems": 3, "seed": 42})
    assert r.status_code == 200
    data = r.json()
    assert "top" in data
    top = data["top"]
    assert len(top) > 0
    first = json.dumps(top[0])

    # Same seed should produce identical top[0]
    r2 = client.post("/propose_candidates", json={"n_candidates": 10, "max_elems": 3, "seed": 42})
    assert r2.status_code == 200
    assert first == json.dumps(r2.json()["top"][0])

    # Pareto endpoint
    r3 = client.post("/pareto_candidates", json={"n_samples": 64, "max_elems": 3, "seed": 1})
    assert r3.status_code == 200
    front = r3.json().get("front", [])
    assert isinstance(front, list)
    assert len(front) > 0
    item = front[0]
    for k in ["composition", "pred", "objectives"]:
        assert k in item


def test_sweep_physics(client):
    comp = {"W": 0.6, "Cu": 0.4}
    r = client.post(
        "/sweep_physics",
        json={
            "composition": comp,
            "flux": {"min": 1e12, "max": 1e12, "num": 1},
            "time_s": {"values": [3600, 7200]},
            "energy_eV": {"values": [14.1e6]},
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["records"]) == 2 * 1 * 1
    for rec in data["records"]:
        assert set(["energy_eV", "flux_n_cm2_s", "time_s", "pct_transmuted", "pct_radioactive"]).issubset(rec)


def test_generate_protocol(client):
    comp = {"W": 0.6, "Cu": 0.4}
    r = client.post(
        "/generate_protocol",
        json={"composition": comp, "tolerance_at_pct": 0.05, "irradiation": {"flux_n_cm2_s": 1e13, "time_s": 3600, "energy_eV": 14.1e6, "temperature_C": 400.0}},
    )
    assert r.status_code == 200
    data = r.json()
    assert "markdown" in data and data["markdown"]
    proto = data["protocol"]
    assert "composition_tolerances" in proto and len(proto["composition_tolerances"]) == 2
    assert "instruments" in proto
    assert "acceptance_criteria" in proto


def test_pareto_save_tmpfile(client):
    items = ["W0.6-Cu0.4", "W0.5-Cu0.5"]
    with tempfile.TemporaryDirectory() as td:
        target = os.path.join(td, "out.csv")
        r = client.post(
            "/pareto_save",
            json={"items": items, "target_csv": target, "mode": "upsert"},
        )
        assert r.status_code == 200
        info = r.json()
        assert info["saved"] == len(items)
        assert os.path.exists(info["path"]) and os.path.getsize(info["path"]) > 0

