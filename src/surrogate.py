
"""
Train lightweight surrogate regressors for the toy proxies and provide a
predict_with_uncertainty(comp_dict) API.

Usage (train):
    python -m src.surrogate

Usage (imported):
    from src.surrogate import load_models, predict_with_uncertainty
    load_models()
    predict_with_uncertainty({"W":0.6,"Cu":0.4})
"""
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

BASE = os.path.dirname(os.path.dirname(__file__))
DATA_FILE = os.path.join(BASE, "data", "synthetic_labels.csv")
MODELS_DIR = os.path.join(BASE, "models")
FEATURES_FILE = os.path.join(MODELS_DIR, "feature_columns.json")

MODEL_ACT = os.path.join(MODELS_DIR, "rf_activation.pkl")
MODEL_TH = os.path.join(MODELS_DIR, "rf_thermal.pkl")
MODEL_DU = os.path.join(MODELS_DIR, "rf_ductility.pkl")

_ACT_MODEL = None
_TH_MODEL = None
_DU_MODEL = None
_FEATURE_COLUMNS = None

def _ensure_models_dir():
    os.makedirs(MODELS_DIR, exist_ok=True)

def _select_feature_columns(df):
    exclude = {"id", "formula", "activation_proxy", "thermal_proxy", "ductility_proxy"}
    return [c for c in df.columns if c not in exclude]

def train_surrogates(data_csv=DATA_FILE, test_size=0.15, random_state=42, n_estimators=200):
    _ensure_models_dir()
    df = pd.read_csv(data_csv)
    feature_columns = _select_feature_columns(df)
    X = df[feature_columns].values
    y_act = df["activation_proxy"].values
    y_th = df["thermal_proxy"].values
    y_du = df["ductility_proxy"].values

    X_train, X_test, ya_train, ya_test, yt_train, yt_test, yd_train, yd_test = train_test_split(
        X, y_act, y_th, y_du, test_size=test_size, random_state=random_state
    )

    def _train_save(y_train, y_test, out_path):
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        joblib.dump(model, out_path)
        return model, mae, r2

    print("Training activation model...")
    m_act, mae_act, r2_act = _train_save(ya_train, ya_test, MODEL_ACT)
    print(f"Activation model: MAE={mae_act:.4f}, R2={r2_act:.3f}")

    print("Training thermal model...")
    m_th, mae_th, r2_th = _train_save(yt_train, yt_test, MODEL_TH)
    print(f"Thermal model: MAE={mae_th:.4f}, R2={r2_th:.3f}")

    print("Training ductility model...")
    m_du, mae_du, r2_du = _train_save(yd_train, yd_test, MODEL_DU)
    print(f"Ductility model: MAE={mae_du:.4f}, R2={r2_du:.3f}")

    with open(FEATURES_FILE, "w") as f:
        json.dump(feature_columns, f, indent=2)

    print("Saved models to", MODELS_DIR)
    return {
        "feature_columns": feature_columns,
        "metrics": {
            "activation": {"mae": mae_act, "r2": r2_act},
            "thermal": {"mae": mae_th, "r2": r2_th},
            "ductility": {"mae": mae_du, "r2": r2_du},
        }
    }

def load_models():
    global _ACT_MODEL, _TH_MODEL, _DU_MODEL, _FEATURE_COLUMNS
    if _ACT_MODEL is not None:
        return
    if not os.path.exists(MODEL_ACT):
        raise FileNotFoundError("Models not found. Run train_surrogates first.")
    _ACT_MODEL = joblib.load(MODEL_ACT)
    _TH_MODEL = joblib.load(MODEL_TH)
    _DU_MODEL = joblib.load(MODEL_DU)
    with open(FEATURES_FILE, "r") as f:
        _FEATURE_COLUMNS = json.load(f)
    print("Loaded models and feature columns.")

def _vectorize_comp(comp_dict):
    from src.descriptors import comp_features, parse_formula
    if isinstance(comp_dict, str):
        compd = parse_formula(comp_dict)
    elif isinstance(comp_dict, dict):
        compd = {k: float(v) for k, v in comp_dict.items()}
    else:
        raise ValueError("comp_dict must be formula string or dict")

    feats = comp_features(compd)
    import json, os
    with open(os.path.join(BASE, "data", "elements_list.json"), "r") as f:
        allowed = json.load(f)

    flat = {
        "avg_Z": feats["avg_Z"],
        "avg_chi": feats["avg_chi"],
        "avg_mass": feats["avg_mass"],
        "std_atomic_radius": feats["std_atomic_radius"],
        "n_elements": feats["n_elements"],
        "contains_Cu": feats["contains_Cu"],
        "contains_W": feats["contains_W"],
    }
    for i, el in enumerate(allowed):
        flat[f"frac_{el}"] = float(feats["frac_vector"][i])

    vec = [flat.get(col, 0.0) for col in _FEATURE_COLUMNS]
    return np.array(vec).reshape(1, -1)

def _rf_predict_with_tree_stats(model, X):
    all_preds = np.stack([est.predict(X) for est in model.estimators_], axis=0)
    mean = np.mean(all_preds, axis=0)
    std = np.std(all_preds, axis=0)
    return mean, std

def predict_with_uncertainty(comp):
    if _ACT_MODEL is None:
        load_models()
    X = _vectorize_comp(comp)
    a_mean, a_std = _rf_predict_with_tree_stats(_ACT_MODEL, X)
    t_mean, t_std = _rf_predict_with_tree_stats(_TH_MODEL, X)
    d_mean, d_std = _rf_predict_with_tree_stats(_DU_MODEL, X)
    return {
        "activation": {"mean": float(a_mean[0]), "std": float(a_std[0])},
        "thermal": {"mean": float(t_mean[0]), "std": float(t_std[0])},
        "ductility": {"mean": float(d_mean[0]), "std": float(d_std[0])}
    }

if __name__ == "__main__":
    print("Training surrogates from", DATA_FILE)
    out = train_surrogates()
    print("Metrics:", out["metrics"])
    print("Feature columns saved to", FEATURES_FILE)
