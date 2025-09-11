# src/descriptors.py
import re
import json
import os
import numpy as np

# Try to import pymatgen for element properties; if not available use fallback table
try:
    from pymatgen.core import Element
    PYMATGEN_AVAILABLE = True
except Exception:
    PYMATGEN_AVAILABLE = False

# Fallback element properties (Z, electronegativity (Pauling), atomic_mass, covalent_radius (pm))
# Values are approximate for the allowed fusion-friendly set.
ELEMENT_PROPS = {
    "W":  {"Z":74, "X":2.36, "mass":183.84, "radius":135},
    "Cu": {"Z":29, "X":1.90, "mass":63.546, "radius":132},
    "Fe": {"Z":26, "X":1.83, "mass":55.845, "radius":126},
    "Cr": {"Z":24, "X":1.66, "mass":51.996, "radius":128},
    "Mo": {"Z":42, "X":2.16, "mass":95.95, "radius":139},
    "Ta": {"Z":73, "X":1.5, "mass":180.95, "radius":146},
    "Nb": {"Z":41, "X":1.6, "mass":92.906, "radius":146},
    "V":  {"Z":23, "X":1.63, "mass":50.942, "radius":125},
    "Ti": {"Z":22, "X":1.54, "mass":47.867, "radius":136},
    "Al": {"Z":13, "X":1.61, "mass":26.981, "radius":118},
    "Si": {"Z":14, "X":1.90, "mass":28.085, "radius":111},
    "Mn": {"Z":25, "X":1.55, "mass":54.938, "radius":127},
}

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
EL_FILE = os.path.join(BASE_DIR, "data", "elements_list.json")
if os.path.exists(EL_FILE):
    with open(EL_FILE, "r") as f:
        ALLOWED_ELEMENTS = json.load(f)
else:
    ALLOWED_ELEMENTS = list(ELEMENT_PROPS.keys())

# --- helpers ----------------------------------------------------------------
_formula_re = re.compile(r'([A-Z][a-z]?)([0-9]*\.?[0-9]*)')

def parse_formula(formula):
    """
    Parse formats like:
      - "W0.6-Cu0.4"
      - "W0.6Cu0.4"
      - "W:0.6,Cu:0.4"
    Returns dict {element: fraction} normalized to sum 1.0
    """
    if isinstance(formula, dict):
        comp = {k: float(v) for k, v in formula.items()}
        s = sum(comp.values())
        if s == 0:
            raise ValueError("Empty composition")
        return {k: v/s for k,v in comp.items()}

    # normalize separators
    norm = formula.replace("-", "").replace(":", "").replace(",", "")
    matches = _formula_re.findall(norm)
    if not matches:
        raise ValueError(f"Can't parse formula: {formula}")
    comp = {}
    for el, val in matches:
        if val == "":
            comp[el] = 1.0
        else:
            comp[el] = float(val)
    # normalize
    total = sum(comp.values())
    if total <= 0:
        raise ValueError("Composition sums to zero")
    return {k: v/total for k, v in comp.items()}

def _element_props(el):
    if PYMATGEN_AVAILABLE:
        E = Element(el)
        return {
            "Z": E.Z,
            "X": E.X if E.X is not None else ELEMENT_PROPS.get(el, {}).get("X", 0.0),
            "mass": E.atomic_mass if E.atomic_mass is not None else ELEMENT_PROPS.get(el, {}).get("mass", 0.0),
            "radius": ELEMENT_PROPS.get(el, {}).get("radius", 0.0),
        }
    else:
        return ELEMENT_PROPS.get(el, {"Z":0, "X":0.0, "mass":0.0, "radius":0.0})

def comp_to_vector(comp_dict, allowed_list=ALLOWED_ELEMENTS):
    """
    Turn comp dict -> fraction vector array ordered by allowed_list
    """
    return np.array([comp_dict.get(e, 0.0) for e in allowed_list], dtype=float)

def comp_features(comp):
    """
    Accepts formula string or comp_dict.
    Returns a dict of features:
      - frac_vector (list)
      - avg_Z, avg_chi, avg_mass, std_radius, n_elements, contains_refractory, contains_Cu
    """
    if isinstance(comp, str) or isinstance(comp, (dict,)):
        compd = parse_formula(comp) if isinstance(comp, str) else comp
    else:
        raise ValueError("comp must be formula string or dict")

    fracs = comp_to_vector(compd, ALLOWED_ELEMENTS)
    elems_present = [e for e, f in zip(ALLOWED_ELEMENTS, fracs) if f > 0]
    if len(elems_present) == 0:
        raise ValueError("No allowed elements present in composition")

    # aggregate properties
    weighted_Z = 0.0
    weighted_X = 0.0
    weighted_mass = 0.0
    radii = []
    for el, frac in compd.items():
        props = _element_props(el)
        weighted_Z += frac * props.get("Z", 0.0)
        weighted_X += frac * props.get("X", 0.0)
        weighted_mass += frac * props.get("mass", 0.0)
        radii.append(props.get("radius", 0.0))

    std_rad = float(np.std([_element_props(e).get("radius", 0.0) for e in compd.keys()]))
    features = {
        "frac_vector": fracs,
        "avg_Z": float(weighted_Z),
        "avg_chi": float(weighted_X),
        "avg_mass": float(weighted_mass),
        "std_atomic_radius": std_rad,
        "n_elements": int(len(elems_present)),
        "contains_Cu": int(compd.get("Cu", 0.0) > 0.0),
        "contains_W": int(compd.get("W", 0.0) > 0.0),
    }
    return features

# Quick __main__ for manual test
if __name__ == "__main__":
    print("PYMATGEN_AVAILABLE =", PYMATGEN_AVAILABLE)
    print(comp_features("W0.6-Cu0.4"))
