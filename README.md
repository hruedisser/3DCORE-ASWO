# 3DCORE-ASWO

**3DCORE-ASWO** is the version of the 3DCORE model used at the [Austrian Space Weather Office (ASWO)](https://helioforecast.space) at GeoSphere Austria.
It is based on the original open-source implementation [py3DCORE](https://github.com/ajefweiss/py3DCORE) by Andreas J. Weiss and collaborators, and has been adapted.


⸻

## 📦 Installation

Clone the repository including its submodules:

```
git clone --recurse-submodules https://github.com/hruedisser/3DCORE-ASWO.git
```

Then, set up the Python environment using conda:

```
conda env create -f environment.yml
conda activate 3dcore-aswo
```


⸻

## 🧩 Dependencies

This repository includes the submodule
➡️ [sc-data-functions](https://github.com/hruedisser/sc-data-functions)
which provides shared data handling functions for spacecraft data access, preprocessing, and analysis.

⸻