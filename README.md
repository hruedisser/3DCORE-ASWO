# 3DCORE-ASWO

**3DCORE-ASWO** is the version of the 3DCORE model used at the [Austrian Space Weather Office (ASWO)](https://helioforecast.space) at GeoSphere Austria.
It is based on the original open-source implementation [py3DCORE](https://github.com/ajefweiss/py3DCORE) by Andreas J. Weiss and collaborators, and has been adapted.


⸻

## 📦 Installation

Clone the repository including its submodules:

```
git clone --recurse-submodules https://github.com/hruedisser/3DCORE-ASWO.git
```

Potentially update the submodules

```
git submodule update --remote
```

Then, set up the Python environment using conda:

```
conda env create -f environment.yml
conda activate 3dcore-aswo
```


⸻

## ⚙️ Configuration

Before running the model, make sure to **create the config.json file in the top level of the repository**.
It should have the following format

```
{
  "kernels_path": "/Volumes/SSDPortable/data/kernels",
  "data_path": "/Volumes/SSDPortable/data",
  "file_names": {
    "bepi": "bepi_ob_2019_now_rtn.p",
    "maven": "maven_2014_2018_removed_smoothed.p",
    "messenger": "messenger_2007_2015_sceq.p",
    "msl": "msl_2012_2019_rad.p",
    "noaa_archive": "rtsw_realtime_archive_gsm.p",
    "noaa_rtsw": "noaa_rtsw_last_35files_now.p",
    "solo": "solo_2020_now_rtn.p",
    "stereo_a": "stereoa_2007_now_rtn.p",
    "stereo_a_beacon": "stereoa_beacon_rtn_last_300days_now.p",
    "stereo_b": "stereob_2007_2014_rtn.p",
    "ulysses": "ulysses_1990_2009_rtn.p",
    "vex": "vex_2007_2014_sceq_removed.p",
    "wind": "wind_1995_now_gse.p",
    
    "positions": "positions_all_HEEQ_10min_rad_ed.p"
  },
  "ace_path": "/Volumes/SSDPortable/data/ACE",
  "aditya_path": "/Volumes/SSDPortable/data/aditya",
  "bepi_path": "/Volumes/SSDPortable/data/aditya",
  "juice_path": "/Volumes/SSDPortable/data/juice",
  "juno_path": "/Volumes/SSDPortable/data/juno",
  "mes_path": "/Volumes/SSDPortable/data/mes",
  "dscovr_path": "/Volumes/SSDPortable/data/dscovr",
  "psp_path": "/Volumes/SSDPortable/data/psp",
  "solo_path": "/Volumes/SSDPortable/data/solo",
  "stereoa_path": "/Volumes/SSDPortable/data/stereoa",
  "stereob_path": "/Volumes/SSDPortable/data/stereob",
  "themis_path": "/Volumes/SSDPortable/data/themis",
  "ulysses_path": "/Volumes/SSDPortable/data/ulysses",
  "vex_path": "/Volumes/SSDPortable/data/vex",
  "wind_path": "/Volumes/SSDPortable/data/wind/",
  "rtsw_path": "/Volumes/SSDPortable/data/rtsw"
}
```

This configuration file contains all **paths to local data directories and resources**, such as spacecraft data and SPICE kernels. These paths must be chosen to match your local setup.

If you do not have the files necessary for the spacecraft you want to use data from, you can download them [here](https://figshare.com/articles/dataset/In_Situ_Data_for_https_github_com_hruedisser_3DCORE-ASWO/30343477).

The kernels needed for the coordinate transformations are available [here](https://figshare.com/articles/dataset/Kernels_for_https_github_com_hruedisser_sc-data-functions/30343831?file=58687090).

💡 **Note**: In a future release, it will be possible to automatically download required data and kernels from a central storage, but this functionality is not yet available.


⸻

## 🧩 Dependencies

This repository includes the submodule
➡️ [sc-data-functions](https://github.com/hruedisser/sc-data-functions)
which provides shared data handling functions for spacecraft data access, preprocessing, and analysis. The submodule was originally forked from [@eedavies](https://github.com/ee-davies/) at [sc-data-functions](https://github.com/ee-davies/sc-data-functions).

⸻