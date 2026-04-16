import importlib

from pathlib import Path
import sys
import json
import io
import builtins
import os
import pathlib
from unittest.mock import patch

import numpy as np
from typing import Union

from collections import Counter

import copy

from ..models.toroidal import ToroidalModel
from ..methods.method import BaseMethod


# # Paths
# root = Path(__file__).resolve().parents[1]
# submodule_path = root / "methods" / "sc-data-functions"
# aswo_config_path = Path(__file__).resolve().parents[2] / "config.json"
# submodule_config_path = submodule_path / "config.json"

# # Make sure submodule is importable (for its internal absolute imports like `from functions_general import ...`)
# if str(submodule_path) not in sys.path:
#     sys.path.insert(0, str(submodule_path))

# # ---------- load & deep-merge configs ----------
# def deep_merge(base: dict, override: dict) -> dict:
#     out = base.copy()
#     for k, v in override.items():
#         if k in out and isinstance(out[k], dict) and isinstance(v, dict):
#             out[k] = deep_merge(out[k], v)
#         else:
#             out[k] = v
#     return out

# with open(aswo_config_path) as f:
#     aswo_cfg = json.load(f)
# with open(submodule_config_path) as f:
#     sub_cfg = json.load(f)

# merged_cfg = deep_merge(sub_cfg, aswo_cfg)

# # ---------- patch file reads for config.json ----------
# _original_open = builtins.open

# def _fake_open(path, *args, **kwargs):
#     # Normalize to string
#     p = str(path)
#     # Only intercept the submodule's config.json, not your own
#     if p.endswith("config.json") and str(submodule_path) in p:
#         return io.StringIO(json.dumps(merged_cfg))
#     return _original_open(path, *args, **kwargs)

# def _fake_path_open(self, *args, **kwargs):
#     # Delegate Path.open to the same interceptor
#     return _fake_open(self, *args, **kwargs)

# # ---------- optional env overrides (in case load_path checks env) ----------
# # Set both uppercase and lowercase just in case
# os.environ.setdefault("KERNELS_PATH", aswo_cfg.get("kernels_path", ""))
# os.environ.setdefault("kernels_path", aswo_cfg.get("kernels_path", ""))

# # ---------- import while patches are active ----------
# with patch("builtins.open", _fake_open), patch.object(pathlib.Path, "open", _fake_path_open):
#     # Load the package __init__
#     spec = importlib.util.spec_from_file_location("sc_data_functions", submodule_path / "__init__.py")
#     sc_data_functions = importlib.util.module_from_spec(spec)
#     sys.modules["sc_data_functions"] = sc_data_functions
#     spec.loader.exec_module(sc_data_functions)

#     # IMPORTANT: import submodules that read config **inside the patch**
#     import importlib as _il
#     _il.import_module("sc_data_functions.data_frame_transforms")

# # --- Import functions from the now-patched submodule ---
# from sc_data_functions.position_frame_transforms import (
#     HEE_to_HEEQ
# )

def _norm360(angle_deg: Union[float, np.ndarray]):
    """Normalize angle(s) to [0, 360) degrees."""
    return np.asarray(angle_deg) % 360

def check_flux_rope_type(iparams):
    t_factor = float(iparams["t_factor"]["default_value"])
    inc_in = float(iparams["cme_inclination"]["default_value"])

    rhparam = abs(t_factor)
    lhparam = -rhparam
    handedness = "RH" if t_factor > 0 else "LH"

    inc_in = _norm360(inc_in)

    # Keep your canonical inclination pair
    if 90.0 < inc_in < 270.0:
        inc1 = inc_in
        inc2 = inc_in + 180.0
    else:
        inc2 = inc_in
        inc1 = inc_in + 180.0

    inc1 = _norm360(inc1)
    inc2 = _norm360(inc2)

    tol = 1.0
    boundaries = np.array([45.0, 135.0, 225.0, 315.0])

    # Intermediate / ambiguous near any quadrant boundary
    if np.any(np.abs(boundaries - inc1) < tol) or np.any(np.abs(boundaries - inc2) < tol):
        raise TypeError("Intermediate type not supported")

    def classify_from_handedness_and_inc(handedness, inc):
        inc = _norm360(inc)

        # High inclination: classify by S/N
        if 45.0 < inc < 135.0:
            axis_dir = "S"
            high_inc = True
            suffix = "-1"
        elif 225.0 < inc < 315.0:
            axis_dir = "N"
            high_inc = True
            suffix = "-2"

        # Low inclination: classify by W/E
        elif 135.0 < inc < 225.0:
            axis_dir = "W"
            high_inc = False
            suffix = "-1"
        else:
            axis_dir = "E"
            high_inc = False
            suffix = "-2"

        mapping = {
            ("RH", "W"): "righthanded - SWN",
            ("RH", "S"): "righthanded - ESW",
            ("RH", "E"): "righthanded - NES",
            ("RH", "N"): "righthanded - WNE",
            ("LH", "W"): "lefthanded - NWS",
            ("LH", "S"): "lefthanded - WSE",
            ("LH", "E"): "lefthanded - SEN",
            ("LH", "N"): "lefthanded - ENW",
        }

        return {
            "flux_rope_type": f"{handedness}{suffix}",
            "high_inc_flag": high_inc,
            "axis_direction": axis_dir,
            "message": mapping[(handedness, axis_dir)],
        }

    # classify the original input directly from its actual inclination
    original = classify_from_handedness_and_inc(handedness, inc_in)

    def make_variant(t_value, inc_value):
        p = copy.deepcopy(iparams)
        p["t_factor"]["default_value"] = t_value
        p["cme_inclination"]["default_value"] = inc_value
        return p

    paramlist = [
        make_variant(rhparam, inc1),  # RH with inc1
        make_variant(rhparam, inc2),  # RH with inc2
        make_variant(lhparam, inc1),  # LH with inc1
        make_variant(lhparam, inc2),  # LH with inc2
    ]

    info = {
        "flux_rope_type": original["flux_rope_type"],
        "handedness": handedness,
        "inc_in": inc_in,
        "inc1": inc1,
        "inc2": inc2,
        "high_inc_flag": original["high_inc_flag"],
        "axis_direction": original["axis_direction"],
        "message": original["message"],
    }

    return paramlist, original["high_inc_flag"], info


def _axis_direction_from_inc(inc_deg, tol=1.0):
    """
    Map inclination angle to the dominant axis direction.

    Returns
    -------
    axis : str
        "E", "W", "N", "S", or "intermediate"
    high_inc : bool
        True for N/S cases, False for E/W cases
    """
    inc = _norm360(inc_deg)

    boundaries = np.array([45.0, 135.0, 225.0, 315.0])
    if np.any(np.abs(inc - boundaries) < tol):
        return "intermediate", None

    if 45.0 < inc < 135.0:
        return "S", True
    elif 135.0 < inc < 225.0:
        return "W", False
    elif 225.0 < inc < 315.0:
        return "N", True
    else:
        return "E", False


def _flux_rope_name(handedness, inc_deg, tol=1.0):
    """
    Returns
    -------
    name : str
        Full flux rope type name
    high_inc : bool or None
    axis : str
    """
    axis, high_inc = _axis_direction_from_inc(inc_deg, tol=tol)

    if axis == "intermediate":
        return "intermediate", None, axis

    mapping = {
        ("RH", "W"): "righthanded - SWN",
        ("RH", "S"): "righthanded - ESW",
        ("RH", "E"): "righthanded - NES",
        ("RH", "N"): "righthanded - WNE",
        ("LH", "W"): "lefthanded - NWS",
        ("LH", "S"): "lefthanded - WSE",
        ("LH", "E"): "lefthanded - SEN",
        ("LH", "N"): "lefthanded - ENW",
    }

    return mapping[(handedness, axis)], high_inc, axis


def make_flux_rope_type_models(model_object, tol=1.0):
    ip = model_object.iparams_arr
    t_factors = ip[:, 8].astype(float)
    inc_ins = ip[:, 3].astype(float)

    rhparams = np.abs(t_factors)
    lhparams = -np.abs(t_factors)

    handednesses = np.where(t_factors > 0, "RH", "LH")
    inc_in = _norm360(inc_ins)

    # Keep your two canonical inclination choices
    in_90_270 = (inc_in > 90.0) & (inc_in < 270.0)

    inc1 = np.where(in_90_270, inc_in, inc_in + 180.0)
    inc2 = np.where(in_90_270, inc_in + 180.0, inc_in)

    inc1 = _norm360(inc1)
    inc2 = _norm360(inc2)

    # Detect intermediate boundary cases
    intermediate_mask = np.array([
        _axis_direction_from_inc(inc, tol=tol)[0] == "intermediate"
        for inc in inc_in
    ])

    if np.any(intermediate_mask):
        print(f"Number of intermediate cases found: {np.sum(intermediate_mask)}.")

    # Classify original models correctly from actual handedness + actual inclination
    classified = [
        _flux_rope_name(handedness, inc, tol=tol)
        for handedness, inc in zip(handednesses, inc_in)
    ]

    flux_rope_types = [x[0] for x in classified]
    high_inc_flags = np.array([x[1] for x in classified], dtype=object)
    axis_directions = [x[2] for x in classified]

    valid_flux_rope_types = [t for t in flux_rope_types if t != "intermediate"]

    # Keep your old consistency checks
    if len(set(valid_flux_rope_types)) > 1:
        type_counts = Counter(valid_flux_rope_types)
        counts_str = ", ".join(f"{k}: {v}" for k, v in type_counts.items())

        # Check whether only high/low inclination differs
        handedness_axis_pairs = [
            (h, a) for h, a in zip(handednesses, axis_directions) if a != "intermediate"
        ]

        if len(set(handedness_axis_pairs)) > 1:
            raise ValueError(
                f"Multiple flux rope types found: {counts_str}. "
                f"They differ in handedness and/or axis direction, not just boundary ambiguity."
            )
        else:
            print(
                f"Multiple flux rope types found, but they only differ by boundary ambiguity. "
                f"Counts: {counts_str}"
            )

            # Most common actual type
            flux_rope_type = Counter(valid_flux_rope_types).most_common(1)[0][0]

            valid_high_inc = [f for f in high_inc_flags if f is not None]
            high_inc_flag = Counter(valid_high_inc).most_common(1)[0][0] if valid_high_inc else None

    elif len(valid_flux_rope_types) == 1:
        print(f"All models have the same flux rope type: {valid_flux_rope_types[0]}")
        flux_rope_type = valid_flux_rope_types[0]

        valid_high_inc = [f for f in high_inc_flags if f is not None]
        high_inc_flag = valid_high_inc[0] if valid_high_inc else None

    else:
        raise ValueError("All models are intermediate cases; no unique flux rope type can be assigned.")

    def make_model_variant(t_values, inc_values):
        model_variant = copy.deepcopy(model_object)
        iparams_arr = model_variant.iparams_arr.copy()
        iparams_arr[:, 8] = t_values
        iparams_arr[:, 3] = inc_values
        model_variant.overwrite(iparams_arr=iparams_arr)
        return model_variant

    model_variants = [
        make_model_variant(rhparams, inc1),  # RH with inc1
        make_model_variant(rhparams, inc2),  # RH with inc2
        make_model_variant(lhparams, inc1),  # LH with inc1
        make_model_variant(lhparams, inc2),  # LH with inc2
    ]

    # Classify each variant from its actual handedness + actual inclination
    # We assume all ensemble members within a given variant represent the same FR family.
    variant_inputs = [
        ("RH", inc1[0]),
        ("RH", inc2[0]),
        ("LH", inc1[0]),
        ("LH", inc2[0]),
    ]

    model_flux_rope_types = [
        _flux_rope_name(h, inc, tol=tol)[0]
        for h, inc in variant_inputs
    ]

    info = {
        "flux_rope_type": flux_rope_type,
        "handedness": handednesses[0],   # assuming original models are consistent
        "high_inc_flag": high_inc_flag,
        "message": flux_rope_type,
    }

    return model_variants, high_inc_flags, info, model_flux_rope_types


def model_from_iparams(iparams, t_launch):
    model_kwargs = {
        "ensemble_size": int(1),
        "iparams": iparams
    }
    
    model_obj = ToroidalModel(model_kwargs, t_launch)
    model_obj.generator()

    return model_obj

def model_from_file(fit_file):

    return BaseMethod(fit_file)


def shift_first_valid_to_top(arr):
    # arr shape: (time, series)
    out = np.full_like(arr, np.nan)

    # True where values exist
    valid = ~np.isnan(arr)

    # first valid index per column
    first_idx = np.argmax(valid, axis=0)

    # columns that contain at least one valid value
    has_valid = valid.any(axis=0)

    for j in np.where(has_valid)[0]:
        start = first_idx[j]
        data = arr[start:, j]
        n = len(data)
        out[:n, j] = data

    return out