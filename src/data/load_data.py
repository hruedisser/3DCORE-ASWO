import json
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

import importlib.util
import sys
import io
import os
import builtins
import pathlib
from pathlib import Path
from unittest.mock import patch

###############################################################
####### Patch numpy.rec to avoid import errors in the submodule
###############################################################
import sys, types, numpy

# Create fake numpy.rec module
numpy_rec = types.ModuleType("numpy.rec")

# Assign the real numpy.recarray class to it (works since numpy.recarray still exists)
numpy_rec.recarray = numpy.recarray

# Register fake module so pickle finds it
sys.modules["numpy.rec"] = numpy_rec
###############################################################



# Paths
root = Path(__file__).resolve().parents[1]
submodule_path = root / "methods" / "sc-data-functions"
aswo_config_path = Path(__file__).resolve().parents[2] / "config.json"
submodule_config_path = submodule_path / "config.json"

# Make sure submodule is importable (for its internal absolute imports like `from functions_general import ...`)
if str(submodule_path) not in sys.path:
    sys.path.insert(0, str(submodule_path))

# ---------- load & deep-merge configs ----------
def deep_merge(base: dict, override: dict) -> dict:
    out = base.copy()
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

with open(aswo_config_path) as f:
    aswo_cfg = json.load(f)
with open(submodule_config_path) as f:
    sub_cfg = json.load(f)

merged_cfg = deep_merge(sub_cfg, aswo_cfg)

# ---------- patch file reads for config.json ----------
_original_open = builtins.open

def _fake_open(path, *args, **kwargs):
    # Normalize to string
    p = str(path)
    # Only intercept the submodule's config.json, not your own
    if p.endswith("config.json") and str(submodule_path) in p:
        return io.StringIO(json.dumps(merged_cfg))
    return _original_open(path, *args, **kwargs)

def _fake_path_open(self, *args, **kwargs):
    # Delegate Path.open to the same interceptor
    return _fake_open(self, *args, **kwargs)

# ---------- optional env overrides (in case load_path checks env) ----------
# Set both uppercase and lowercase just in case
os.environ.setdefault("KERNELS_PATH", aswo_cfg.get("kernels_path", ""))
os.environ.setdefault("kernels_path", aswo_cfg.get("kernels_path", ""))

# ---------- import while patches are active ----------
with patch("builtins.open", _fake_open), patch.object(pathlib.Path, "open", _fake_path_open):
    # Load the package __init__
    spec = importlib.util.spec_from_file_location("sc_data_functions", submodule_path / "__init__.py")
    sc_data_functions = importlib.util.module_from_spec(spec)
    sys.modules["sc_data_functions"] = sc_data_functions
    spec.loader.exec_module(sc_data_functions)

    # IMPORTANT: import submodules that read config **inside the patch**
    import importlib as _il
    _il.import_module("sc_data_functions.data_frame_transforms")

# --- Import functions from the now-patched submodule ---
from sc_data_functions.data_frame_transforms import (
    GSM_to_GSE_mag_components,
    GSE_to_HEEQ_mag_components,
    HEEQ_to_RTN_mag_components,
    HEEQ_to_GSE_mag_components,
    GSE_to_GSM_mag_components

)

# === Load file_names from JSON config ===
def load_file_names(config_file=Path(__file__).resolve().parents[2] /'config.json'):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config['file_names']

# Load file_names once globally
file_names = load_file_names()
print(f"File names loaded")

# === Load data_path from JSON config ===
def load_data_path(config_file=Path(__file__).resolve().parents[2] /'config.json'):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config['data_path']

# Load data_path once globally
data_path = load_data_path()
print(f"Data path loaded: {data_path}")



def load_bepi(data_begin, data_end):
    file_name = file_names['bepi']
    b_data, pos_data, t_data, body_data, v_data = get_data_from_file_name(file_name, data_begin, data_end)
    return b_data, pos_data, t_data, body_data, v_data

def load_maven(data_begin, data_end):
    file_name = file_names['maven']
    b_data, pos_data, t_data, body_data, v_data = get_data_from_file_name(file_name, data_begin, data_end)
    return b_data, pos_data, t_data, body_data, v_data
    

def load_messenger(data_begin, data_end):
    file_name = file_names['messenger']
    b_data, pos_data, t_data, body_data, v_data = get_data_from_file_name(file_name, data_begin, data_end)
    return b_data, pos_data, t_data, body_data, v_data
    

def load_msl(data_begin, data_end):
    file_name = file_names['msl']
    b_data, pos_data, t_data, body_data, v_data = get_data_from_file_name(file_name, data_begin, data_end)
    return b_data, pos_data, t_data, body_data, v_data
    

def load_noaa_archive(data_begin, data_end):
    file_name = file_names['noaa_archive']
    b_data, pos_data, t_data, body_data, v_data = get_data_from_file_name(file_name, data_begin, data_end)
    return b_data, pos_data, t_data, body_data, v_data
    

def load_noaa_rtsw(data_begin, data_end):
    file_name = file_names['noaa_rtsw']
    b_data, pos_data, t_data, body_data, v_data = get_data_from_file_name(file_name, data_begin, data_end)
    return b_data, pos_data, t_data, body_data, v_data
    

def load_solo(data_begin, data_end):
    file_name = file_names['solo']
    b_data, pos_data, t_data, body_data, v_data = get_data_from_file_name(file_name, data_begin, data_end)
    return b_data, pos_data, t_data, body_data, v_data
    

def load_stereo_a(data_begin, data_end):
    file_name = file_names['stereo_a']
    b_data, pos_data, t_data, body_data, v_data = get_data_from_file_name(file_name, data_begin, data_end)
    return b_data, pos_data, t_data, body_data, v_data
    

def load_stereo_a_beacon(data_begin, data_end):
    file_name = file_names['stereo_a_beacon']
    b_data, pos_data, t_data, body_data, v_data = get_data_from_file_name(file_name, data_begin, data_end)
    return b_data, pos_data, t_data, body_data, v_data
    

def load_stereo_b(data_begin, data_end):
    file_name = file_names['stereo_b']
    b_data, pos_data, t_data, body_data, v_data = get_data_from_file_name(file_name, data_begin, data_end)
    return b_data, pos_data, t_data, body_data, v_data
    

def load_ulysses(data_begin, data_end):
    file_name = file_names['ulysses']
    b_data, pos_data, t_data, body_data, v_data = get_data_from_file_name(file_name, data_begin, data_end)
    return b_data, pos_data, t_data, body_data, v_data
    

def load_vex(data_begin, data_end):
    file_name = file_names['vex']
    b_data, pos_data, t_data, body_data, v_data = get_data_from_file_name(file_name, data_begin, data_end)
    return b_data, pos_data, t_data, body_data, v_data
    

def load_wind(data_begin, data_end):
    file_name = file_names['wind']
    b_data, pos_data, t_data, body_data, v_data = get_data_from_file_name(file_name, data_begin, data_end)
    return b_data, pos_data, t_data, body_data, v_data

def load_from_df(dataframe, reference_frame = "GSM"):

    if reference_frame == "GSM":
        df_gsm = dataframe
        times = df_gsm['time']

        df_gse = df_gsm.copy()
        bx_gse, by_gse, bz_gse = GSM_to_GSE_mag_components(df_gsm["bx"], df_gsm["by"], df_gsm["bz"], times)
        df_gse['bx'] = bx_gse
        df_gse['by'] = by_gse
        df_gse['bz'] = bz_gse
        print(f"Converted GSM to GSE data")

        df_heeq = df_gse.copy()
        bx_heeq, by_heeq, bz_heeq = GSE_to_HEEQ_mag_components(df_gse["bx"], df_gse["by"], df_gse["bz"], times)
        df_heeq['bx'] = bx_heeq
        df_heeq['by'] = by_heeq
        df_heeq['bz'] = bz_heeq
        print(f"Converted GSM to HEEQ data")

        df_rtn = df_heeq.copy()
        bx_rtn, by_rtn, bz_rtn = HEEQ_to_RTN_mag_components(df_heeq["bx"], df_heeq["by"], df_heeq["bz"],df_heeq["x"], df_heeq["y"], df_heeq["z"])
        df_rtn['bx'] = bx_rtn
        df_rtn['by'] = by_rtn
        df_rtn['bz'] = bz_rtn
        print(f"Converted HEEQ to RTN data")

    else:
        raise ValueError(f"Reference frame {reference_frame} not supported for load_from_df.")



    b_data = {}

    b_data["RTN"] = np.column_stack(
        (
            df_rtn['bx'],
            df_rtn['by'],
            df_rtn['bz']
        )
    )
    
    b_data["HEEQ"] = np.column_stack(
        (
            df_heeq['bx'],
            df_heeq['by'],
            df_heeq['bz']
        )
    )

    b_data["GSM"] = np.column_stack(
        (
            df_gsm['bx'],
            df_gsm['by'],
            df_gsm['bz']
        )
    )

    t_data = pd.to_datetime(df_rtn['time'], utc=False).dt.tz_localize(None).to_list()

    pos_data = np.column_stack(
        (
            df_heeq['x'],
            df_heeq['y'],
            df_heeq['z']
        )
    )

    # check if pos_data is in AU or km (if all positions are < 50, assume AU)
    if np.all(np.abs(pos_data) > 50):
        pos_data = pos_data / 1.495978707E8  # convert km to AU

    v_data = df_heeq["vt"] if "vt" in df_heeq.columns else None

    # check if there are NaNs in the position data
    if np.isnan(pos_data).any():
        raise Warning("Position data contains NaNs. Fitting might not be possible.")
    
    return b_data, pos_data, t_data, v_data

def HEEQ_to_RTN_wrapper(df_heeq):
    df = df_heeq.copy()
    bx, by, bz = HEEQ_to_RTN_mag_components(
        df["bx"].values,
        df["by"].values,
        df["bz"].values,
        df["x"].values,
        df["y"].values,
        df["z"].values,
    )
    df["bx"], df["by"], df["bz"] = bx, by, bz
    return df


def HEEQ_to_GSE_wrapper(df_heeq):
    df = df_heeq.copy()
    bx, by, bz = HEEQ_to_GSE_mag_components(
        df["bx"].values,
        df["by"].values,
        df["bz"].values,
        pd.to_datetime(df["time"]).to_list(),
    )
    df["bx"], df["by"], df["bz"] = bx, by, bz
    return df


def GSE_to_HEEQ_wrapper(df_gse):
    df = df_gse.copy()
    bx, by, bz = GSE_to_HEEQ_mag_components(
        df["bx"].values,
        df["by"].values,
        df["bz"].values,
        pd.to_datetime(df["time"]).to_list(),
    )
    df["bx"], df["by"], df["bz"] = bx, by, bz
    return df


def GSE_to_GSM_wrapper(df_gse):
    df = df_gse.copy()
    bx, by, bz = GSE_to_GSM_mag_components(
        df["bx"].values,
        df["by"].values,
        df["bz"].values,
        pd.to_datetime(df["time"]).to_list(),
    )
    df["bx"], df["by"], df["bz"] = bx, by, bz
    return df
    

def get_data_from_file_name(file_name, data_begin, data_end, delta = 60):

    positions_name = file_names['positions']
    print(f"Loading positions data from {positions_name}")

    positions_data = pickle.load(open(Path(data_path, positions_name), "rb"))

    planets = [
        "earth",
        "mercury",
        "venus",
        "mars",
    ]

    spacecraft = [
        "psp",
        "solo",
        "sta",
        "stb",
        "bepi",
        "wind"
    ]

    body_data = {}
    for obj in planets + spacecraft:
        obj_df = pd.DataFrame(positions_data[obj])


        obj_df['time'] = pd.to_datetime(obj_df['time'], unit='D')

        # round to nearest 10 minutes
        obj_df['time'] = obj_df['time'].dt.round('10min')
        obj_df.index = obj_df["time"]

        # Resample to 10-minute intervals and interpolate missing values with maximum gap of 20 minutes

        body_data[obj] = obj_df[(obj_df['time'] >= data_begin - pd.Timedelta(days=delta)) & (obj_df['time'] <= data_end + pd.Timedelta(days=delta))]

    
    def _replace_coord(name, target):
        for coord in ["rtn", "heeq", "gse", "gsm", "sceq"]:
            if coord in name:
                return name.replace(coord, target)
        return None

    def _load_df(fname, label):
        df = pickle.load(open(Path(data_path, fname), "rb"))
        df = pd.DataFrame(df[0])
        df = df[(df["time"] >= data_begin) & (df["time"] <= data_end)]
        if df.empty:
            raise ValueError(
                f"{label} data is empty after filtering for the given date range "
                f"{data_begin} to {data_end}. Please check the data file."
            )
        print(f"Loaded {label} data from {fname}")
        return df
    

    # infer related filenames
    heeq_file = _replace_coord(file_name, "heeq")
    rtn_file  = _replace_coord(file_name, "rtn")
    gse_file  = _replace_coord(file_name, "gse")
    gsm_file  = _replace_coord(file_name, "gsm")

    if "heeq" in file_name:
        heeq_file = file_name
    elif "rtn" in file_name or "sceq" in file_name:
        rtn_file = file_name
    elif "gse" in file_name:
        gse_file = file_name
    elif "gsm" in file_name:
        gsm_file = file_name

    file_map = {
        "HEEQ": heeq_file,
        "RTN": rtn_file,
        "GSE": gse_file,
        "GSM": gsm_file,
    }

    # check existence
    for key, fname in file_map.items():
        if fname is None or not Path(data_path, fname).exists():
            if fname is not None:
                print(f"File {fname} not found in data path {data_path}, will convert {key} instead of loading.")
            file_map[key] = None

    if all(v is None for v in file_map.values()):
        raise FileNotFoundError(
            "No files found. Make sure at least one file exists and contains one of: "
            "(gse, gsm, heeq, rtn)."
        )

    print(f"Loading data from {data_path}")

    df_heeq = _load_df(file_map["HEEQ"], "HEEQ") if file_map["HEEQ"] is not None else None
    df_rtn  = _load_df(file_map["RTN"], "RTN")   if file_map["RTN"] is not None else None
    df_gse  = _load_df(file_map["GSE"], "GSE")   if file_map["GSE"] is not None else None
    df_gsm  = _load_df(file_map["GSM"], "GSM")   if file_map["GSM"] is not None else None

    
    # ------------------------------------------------------------------
    # Most efficient conversion strategy:
    #
    # HEEQ -> RTN
    # HEEQ -> GSE -> GSM
    #
    # GSE  -> HEEQ -> RTN
    # GSE  -> GSM
    #
    # RTN  -> HEEQ -> GSE -> GSM
    #
    # GSM  -> GSE -> HEEQ -> RTN
    # ------------------------------------------------------------------

    if df_heeq is not None:
        if df_rtn is None:
            df_rtn = HEEQ_to_RTN_wrapper(df_heeq)
            print(f"Converted HEEQ to RTN data")
        if df_gse is None:
            df_gse = HEEQ_to_GSE_wrapper(df_heeq)
            print(f"Converted HEEQ to GSE data")
        if df_gsm is None:
            df_gsm = GSE_to_GSM_wrapper(df_gse)
            print(f"Converted GSE to GSM data")
    
    elif df_gse is not None:
        if df_heeq is None:
            df_heeq = GSE_to_HEEQ_wrapper(df_gse)
            print(f"Converted GSE to HEEQ data")
        if df_rtn is None:
            df_rtn = HEEQ_to_RTN_wrapper(df_heeq)
            print(f"Converted HEEQ to RTN data")
        if df_gsm is None:
            df_gsm = GSE_to_GSM_wrapper(df_gse)
            print(f"Converted GSE to GSM data")
    
    elif df_rtn is not None:
        if df_heeq is None:
            df_heeq = HEEQ_to_RTN_wrapper(df_rtn)
            print(f"Converted RTN to HEEQ data")
        if df_gse is None:
            df_gse = HEEQ_to_GSE_wrapper(df_heeq)
            print(f"Converted HEEQ to GSE data")
        if df_gsm is None:
            df_gsm = GSE_to_GSM_wrapper(df_gse)
            print(f"Converted GSE to GSM data")
    
    elif df_gsm is not None:
        if df_gse is None:
            df_gse = GSE_to_GSM_wrapper(df_gsm)
            print(f"Converted GSM to GSE data")
        if df_heeq is None:
            df_heeq = GSE_to_HEEQ_wrapper(df_gse)
            print(f"Converted GSE to HEEQ data")
        if df_rtn is None:
            df_rtn = HEEQ_to_RTN_wrapper(df_heeq)
            print(f"Converted HEEQ to RTN data")


    df_rtn  = df_rtn[(df_rtn["time"] >= data_begin) & (df_rtn["time"] <= data_end)]
    df_heeq = df_heeq[(df_heeq["time"] >= data_begin) & (df_heeq["time"] <= data_end)]
    df_gse  = df_gse[(df_gse["time"] >= data_begin) & (df_gse["time"] <= data_end)]
    df_gsm  = df_gsm[(df_gsm["time"] >= data_begin) & (df_gsm["time"] <= data_end)]

    b_data = {
        "RTN": np.column_stack((df_rtn["bx"], df_rtn["by"], df_rtn["bz"])),
        "HEEQ": np.column_stack((df_heeq["bx"], df_heeq["by"], df_heeq["bz"])),
        "GSE": np.column_stack((df_gse["bx"], df_gse["by"], df_gse["bz"])),
        "GSM": np.column_stack((df_gsm["bx"], df_gsm["by"], df_gsm["bz"])),
    }
    
    t_data = pd.to_datetime(df_rtn['time']).to_list()

    pos_data = np.column_stack(
        (
            df_heeq["x"],
            df_heeq["y"],
            df_heeq["z"]
        )
    )

    # check if pos_data is in AU or km (if all positions are < 50, assume AU)
    if np.all(np.abs(pos_data) > 50):
        pos_data = pos_data / 1.495978707E8  # convert km to AU

    v_data = df_heeq["vt"] if "vt" in df_heeq.columns else None

    # check if there are NaNs in the position data
    if np.isnan(pos_data).any():
        print("Position data contains NaNs. Fitting might not be possible.")
    
    return b_data, pos_data, t_data, body_data, v_data


def exported_HEEQ_to_RTN_mag_components(bx_heeq, by_heeq, bz_heeq, x_heeq, y_heeq, z_heeq):
    return HEEQ_to_RTN_mag_components(bx_heeq, by_heeq, bz_heeq, x_heeq, y_heeq, z_heeq)