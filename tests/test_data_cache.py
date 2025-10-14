from pathlib import Path
from src.data.data_cache import get_data_cache

idds = {
    "NOAA_ARCHIVE": "ICME_NOAA_ARCHIVE_CUSTOM_20210827_01",
    "STEREO_A": "ICME_STEREO_A_MOESTL_20240324_01", # this is the ID of the event we want to analyze
    "SOLO": "ICME_SOLO_MOESTL_20240323_01",
    "BEPI": 'ICME_BEPI_MOESTL_20240324_01'
}

for idd_key, idd_val in idds.items():
    
    try:
        print(f"Trying example for {idd_key}")

        data_cache = get_data_cache(idd_val, mean_hours=24) # mean_hours is the number of hours before the event to calculate the mean solar wind speed

        print(f"Success for {idd_key}")
    
    except Exception as e:
        print(f"Failed for {idd_key} with error: {e}")
        continue

    