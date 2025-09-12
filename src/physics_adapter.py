import os
import sys
import warnings
from typing import Dict, Tuple

import numpy as np


BASE = os.path.dirname(os.path.dirname(__file__))
PHYSICS_DIR = os.path.join(BASE, "external", "LLM_Hackathon", "Physics")
ENDF_DIR = os.path.join(PHYSICS_DIR, "ENDF", "neutrons-version.VIII.1")
ENSDF_DIR = os.path.join(PHYSICS_DIR, "ENSDF")


def _ensure_import_paths():
    if PHYSICS_DIR not in sys.path:
        sys.path.insert(0, PHYSICS_DIR)


def _parse_abundance_to_fraction(val) -> float:
    """Robustly parse abundance entries like '48.63 %' to a 0..1 fraction.
    Returns 0.0 on failure.
    """
    if val is None:
        return 0.0
    s = str(val).strip()
    # strip parenthetical details if present, e.g. '48.63 % (approx)'
    # conservative: remove anything after first space that looks like a unit/sign
    s = s.replace('%', '').strip()
    # keep only first token that looks numeric
    token = s.split()[0] if s else s
    try:
        return float(token) / 100.0
    except Exception:
        return 0.0


def _extract_half_life_seconds(ensdf_file_path: str, mass_number: int, isotope_symbol: str):
    """Minimal ENSDF ground-state half-life extraction in seconds.
    Returns 'Stable' string if stable, or a float seconds, or 'Not found'.
    """
    unit_conversions = {
        'Y': 365.25 * 24 * 3600,
        'D': 24 * 3600,
        'H': 3600,
        'M': 60,
        'S': 1,
        'MS': 1e-3,
        'US': 1e-6,
        'NS': 1e-9,
        'PS': 1e-12,
        'FS': 1e-15,
    }

    mass_str = f"{mass_number:3}"
    iso_str = f"{isotope_symbol.upper():2}"
    in_section = False

    try:
        with open(ensdf_file_path, 'r') as fh:
            for line in fh:
                if line.startswith(mass_str + iso_str) and 'ADOPTED LEVELS' in line:
                    in_section = True
                    continue
                if in_section and line.startswith(mass_str + iso_str) and line[6:8].strip() == 'L':
                    # ground state level has energy 0
                    if line[8:17].strip() in ['0', '0.0']:
                        parts = line[8:].split()
                        if len(parts) >= 3:
                            half_life = parts[2]
                            if half_life == 'STABLE':
                                return 'Stable'
                            unit = parts[3] if len(parts) > 3 else 'S'
                            try:
                                value = float(half_life)
                                return value * unit_conversions.get(unit.upper(), 1)
                            except Exception:
                                return 'Not found'
                        break
                elif in_section and line.startswith(mass_str) and 'ADOPTED LEVELS' in line:
                    break
    except FileNotFoundError:
        return 'Not found'
    return 'Not found'


def _endf_file_path(Z: int, symbol: str, A: int) -> str:
    return os.path.join(ENDF_DIR, f"n-{Z:03d}_{symbol}_{A:03d}.endf")


def compute_element_metrics(
    symbol: str,
    energy_eV: float = 14.1e6,
    flux_n_cm2_s: float = 1e13,
    time_s: float = 60 * 60,
    Natoms: float = 1e23,
) -> Dict[str, float]:
    """Return Henry-style physics metrics for a pure element target.

    Outputs:
    - pct_transmuted: percentage of atoms transmuted over time_s.
    - pct_radioactive: percentage of atoms transmuted that are radioactive.
    """
    _ensure_import_paths()
    # Lazy imports to avoid import-time failures when not needed.
    try:
        from mendeleev import element as _element
        import endf  # type: ignore
        from isotopes_data import isotopes_data  # provided by external repo
    except Exception as e:
        raise RuntimeError(
            "Missing physics dependencies. Please install 'endf' and 'mendeleev' and ensure external/LLM_Hackathon/Physics is present."
        ) from e

    barns_to_cm2 = 1e-24
    elem = _element(symbol)
    Z = int(elem.atomic_number)

    try:
        data = isotopes_data[symbol]
    except KeyError as e:
        raise ValueError(f"Element {symbol} not found in isotopes_data") from e

    Ntransmuted = 0.0
    Nrad = 0.0

    for isot in data.get('isotopes', []):
        A = int(isot['mass_number'])
        abund_frac = _parse_abundance_to_fraction(isot.get('abundance'))
        if abund_frac <= 0:
            continue
        endf_path = _endf_file_path(Z, symbol, A)
        try:
            mat = endf.Material(endf_path)
            # MT=102 (n,gamma)
            xsAb = mat.section_data[3, 102]['sigma']
        except Exception:
            # Skip if data missing or parse failed.
            continue

        n_iso = Natoms * abund_frac
        try:
            sigma = xsAb(energy_eV)
        except Exception:
            # If interpolation fails, skip this isotope.
            continue
        dN = n_iso * (1.0 - np.exp(-sigma * barns_to_cm2 * flux_n_cm2_s * time_s))
        Ntransmuted += float(dN)

        daughter_A = A + 1
        ensdf_file = os.path.join(ENSDF_DIR, f"ensdf.{daughter_A:03d}")
        hl = _extract_half_life_seconds(ensdf_file, daughter_A, symbol)
        if hl not in ('Stable', 'Not found'):
            Nrad += float(dN)

    pct_trans = (Ntransmuted / Natoms) * 100.0 if Natoms > 0 else 0.0
    pct_rad = (Nrad / Natoms) * 100.0 if Natoms > 0 else 0.0
    return {"pct_transmuted": pct_trans, "pct_radioactive": pct_rad}


def compute_element_metrics_api(
    symbol: str,
    energy_eV: float = 14.1e6,
    flux_n_cm2_s: float = 1e13,
    time_s: float = 60 * 60,
    Natoms: float = 1e23,
):
    """Henry's API-based variant: also computes Product Activity (Bq) using IAEA LiveChart API.

    Returns: { pct_transmuted, pct_radioactive, activity_bq }
    """
    _ensure_import_paths()
    try:
        from mendeleev import element as _element
        import endf  # type: ignore
        from isotopes_data import isotopes_data
        import urllib.request, io, csv
    except Exception as e:
        raise RuntimeError("Missing dependencies for physics API path.") from e

    barns_to_cm2 = 1e-24
    elem = _element(symbol)
    Z = int(elem.atomic_number)
    try:
        data = isotopes_data[symbol]
    except KeyError as e:
        raise ValueError(f"Element {symbol} not found in isotopes_data") from e

    def _get_half_life_seconds(sym: str, A: int):
        nuclide = f"{A}{sym.lower()}"
        url = f"https://nds.iaea.org/relnsd/v1/data?fields=ground_states&nuclides={nuclide}"
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                content = response.read().decode('utf-8')
        except Exception:
            return None
        csv_reader = csv.DictReader(io.StringIO(content))
        rec = next(csv_reader, None)
        if not rec:
            return None
        hl = rec.get('half_life', 'N/A')
        unit = rec.get('unit_hl', 'S')
        if hl == 'STABLE':
            return 'Stable'
        try:
            val = float(hl)
        except Exception:
            return None
        conv = {
            'Y': 31557600,
            'D': 86400,
            'H': 3600,
            'M': 60,
            'S': 1,
            'MS': 1e-3,
            'US': 1e-6,
        }.get(unit.upper(), 1)
        return val * conv

    Ntransmuted = 0.0
    Nrad = 0.0
    Activity = 0.0
    for isot in data.get('isotopes', []):
        A = int(isot['mass_number'])
        abund_frac = _parse_abundance_to_fraction(isot.get('abundance'))
        if abund_frac <= 0:
            continue
        endf_path = _endf_file_path(Z, symbol, A)
        try:
            mat = endf.Material(endf_path)
            xsAb = mat.section_data[3, 102]['sigma']
            sigma = xsAb(energy_eV)
        except Exception:
            continue
        n_iso = Natoms * abund_frac
        dN = n_iso * (1.0 - np.exp(-sigma * barns_to_cm2 * flux_n_cm2_s * time_s))
        Ntransmuted += float(dN)
        daughter_A = A + 1
        hl = _get_half_life_seconds(symbol, daughter_A)
        if hl not in ('Stable', None):
            Nrad += float(dN)
            decay_const = np.log(2) / float(hl)
            Activity += float(dN) * decay_const

    pct_trans = (Ntransmuted / Natoms) * 100.0 if Natoms > 0 else 0.0
    pct_rad = (Nrad / Natoms) * 100.0 if Natoms > 0 else 0.0
    return {"pct_transmuted": pct_trans, "pct_radioactive": pct_rad, "activity_bq": Activity}


def aggregate_alloy_metrics(comp: Dict[str, float], element_metrics: Dict[str, Tuple[float, float]]):
    """Weighted aggregation for an alloy: sum over elements of frac_i * metric_i.

    comp: mapping element symbol -> fraction (sums to 1).
    element_metrics: mapping element symbol -> (pct_transmuted, pct_radioactive).
    """
    t_sum = 0.0
    r_sum = 0.0
    for el, frac in comp.items():
        m = element_metrics.get(el)
        if not m:
            continue
        t_sum += float(frac) * float(m[0])
        r_sum += float(frac) * float(m[1])
    return {"pct_transmuted": t_sum, "pct_radioactive": r_sum}
