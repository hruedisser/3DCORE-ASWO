import json
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

from ..methods.conversions.data_frame_transforms import (
    HEEQ_to_RTN,
    RTN_to_GSM,
    GSM_to_HEEQ,
    RTN_to_HEEQ,
    GSE_to_GSM,
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


    if "rtn" in file_name:
        rtn_file = file_name
        heeq_file = file_name.replace("rtn", "heeq")
        gsm_file = file_name.replace("rtn", "gsm")
    if "sceq" in file_name:
        rtn_file = file_name
        heeq_file = file_name.replace("sceq", "heeq")
        gsm_file = file_name.replace("sceq", "gsm")
    if "heeq" in file_name:
        heeq_file = file_name
        rtn_file = file_name.replace("heeq", "rtn")
        gsm_file = file_name.replace("heeq", "gsm")
    if "gsm" in file_name:
        gsm_file = file_name
        heeq_file = file_name.replace("gsm", "heeq")
        rtn_file = file_name.replace("gsm", "rtn")
    if "gse" in file_name:
        gsm_file = file_name.replace("gse", "gsm")
        heeq_file = file_name.replace("gse", "heeq")
        rtn_file = file_name.replace("gse", "rtn")

    # check if files exist
    for f in [rtn_file, heeq_file, gsm_file]:
        if not Path(data_path, f).exists():
            print(f"File {f} not found in data path {data_path}, converting instead of loading.")
            
            # set the according file to None so it gets converted later
            if "rtn" in f:
                rtn_file = None
            if "heeq" in f:
                heeq_file = None
            if "gsm" in f:
                gsm_file = None
    
    print(f"Loading data from {data_path}")

    if gsm_file is not None:
        df_gsm = pickle.load(open(Path(data_path, gsm_file), "rb"))
        df_gsm = pd.DataFrame(df_gsm[0])
        print(f"Loaded GSM data from {gsm_file}")

        df_gsm = df_gsm[(df_gsm['time'] >= data_begin) & (df_gsm['time'] <= data_end)]

        if heeq_file is not None:
            df_heeq = pickle.load(open(Path(data_path, heeq_file), "rb"))
            df_heeq = pd.DataFrame(df_heeq[0])
            print(f"Loaded HEEQ data from {heeq_file}")
        else:
            df_heeq = GSM_to_HEEQ(df_gsm)
            print(f"Converted GSM to HEEQ data")
        
        if rtn_file is not None:
            df_rtn = pickle.load(open(Path(data_path, rtn_file), "rb"))
            df_rtn = pd.DataFrame(df_rtn[0])
            print(f"Loaded RTN data from {rtn_file}")
        else:
            df_rtn = HEEQ_to_RTN(df_heeq)
            print(f"Converted HEEQ to RTN data")

    elif heeq_file is not None:
        df_heeq = pickle.load(open(Path(data_path, heeq_file), "rb"))
        df_heeq = pd.DataFrame(df_heeq[0])
        print(f"Loaded HEEQ data from {heeq_file}")

        df_heeq = df_heeq[(df_heeq['time'] >= data_begin) & (df_heeq['time'] <= data_end)]
        
        if rtn_file is not None:
            df_rtn = pickle.load(open(Path(data_path, rtn_file), "rb"))
            df_rtn = pd.DataFrame(df_rtn[0])
            print(f"Loaded RTN data from {rtn_file}")
        else:
            df_rtn = HEEQ_to_RTN(df_heeq)
            print(f"Converted HEEQ to RTN data")
        
        if gsm_file is not None:
            df_gsm = pickle.load(open(Path(data_path, gsm_file), "rb"))
            df_gsm = pd.DataFrame(df_gsm[0])
            print(f"Loaded GSM data from {gsm_file}")
        else:
            df_gsm = RTN_to_GSM(df_rtn)
            print(f"Converted RTN to GSM data")
    
    elif rtn_file is not None:
        df_rtn = pickle.load(open(Path(data_path, rtn_file), "rb"))
        df_rtn = pd.DataFrame(df_rtn[0])
        print(f"Loaded RTN data from {rtn_file}")

        df_rtn = df_rtn[(df_rtn['time'] >= data_begin) & (df_rtn['time'] <= data_end)]
        
        if heeq_file is not None:
            df_heeq = pickle.load(open(Path(data_path, heeq_file), "rb"))
            df_heeq = pd.DataFrame(df_heeq[0])
            print(f"Loaded HEEQ data from {heeq_file}")
        else:
            df_heeq = RTN_to_HEEQ(df_rtn)
            print(f"Converted RTN to HEEQ data")

        if gsm_file is not None:
            df_gsm = pickle.load(open(Path(data_path, gsm_file), "rb"))
            df_gsm = pd.DataFrame(df_gsm[0])
            print(f"Loaded GSM data from {gsm_file}")
        else:
            df_gsm = RTN_to_GSM(df_rtn)
            print(f"Converted RTN to GSM data")
    
    else:
        df_gse = pickle.load(open(Path(data_path, file_name), "rb"))
        df_gse = pd.DataFrame(df_gse[0])
        print(f"Loaded GSE data from {file_name}")

        df_gse = df_gse[(df_gse['time'] >= data_begin) & (df_gse['time'] <= data_end)]

        df_gsm = GSE_to_GSM(df_gse)
        print(f"Converted GSE to GSM data")

        df_heeq = GSM_to_HEEQ(df_gsm)
        print(f"Converted GSM to HEEQ data")

        df_rtn = HEEQ_to_RTN(df_heeq)
        print(f"Converted HEEQ to RTN data")


    df_rtn = df_rtn[(df_rtn['time'] >= data_begin) & (df_rtn['time'] <= data_end)]
    df_heeq = df_heeq[(df_heeq['time'] >= data_begin) & (df_heeq['time'] <= data_end)]
    df_gsm = df_gsm[(df_gsm['time'] >= data_begin) & (df_gsm['time'] <= data_end)]

    b_data = {}

    b_data["RTN"] = np.column_stack(
        (
            df_rtn["bx"],
            df_rtn["by"],
            df_rtn["bz"]
        )
    )
    
    b_data["HEEQ"] = np.column_stack(
        (
            df_heeq["bx"],
            df_heeq["by"],
            df_heeq["bz"]
        )
    )

    b_data["GSM"] = np.column_stack(
        (
            df_gsm["bx"],
            df_gsm["by"],
            df_gsm["bz"]
        )
    )

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
    
    return b_data, pos_data, t_data, body_data, v_data

        
        
        

