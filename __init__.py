###############################################################
####### Patch numpy.rec to avoid import errors in the submodule
###############################################################
import sys, types, numpy, json, io, os, builtins, pathlib, importlib, pkgutil
from pathlib import Path
from unittest.mock import patch

# Create fake numpy.rec module (needed by some older pickled structures)
numpy_rec = types.ModuleType("numpy.rec")
numpy_rec.recarray = numpy.recarray
sys.modules["numpy.rec"] = numpy_rec

###############################################################
####### Paths and configuration setup
###############################################################
# This file is in: src/3DCORE-ASWO/__init__.py
# Inner submodule is in: src/3DCORE-ASWO/src/methods/sc-data-functions
root = Path(__file__).resolve().parents[1]  # points to 3DCORE-ASWO/src
submodule_path = root / "methods" / "sc-data-functions"

# Config files
aswo_config_path = Path(__file__).resolve().parent / "config.json"
submodule_config_path = submodule_path / "config.json"

# Ensure submodule is importable
if str(submodule_path) not in sys.path:
    sys.path.insert(0, str(submodule_path))

###############################################################
####### Deep-merge helper for configs
###############################################################
def deep_merge(base: dict, override: dict) -> dict:
    out = base.copy()
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

# Load and merge configs
try:
    with open(aswo_config_path) as f:
        aswo_cfg = json.load(f)
except FileNotFoundError:
    aswo_cfg = {}
try:
    with open(submodule_config_path) as f:
        sub_cfg = json.load(f)
except FileNotFoundError:
    sub_cfg = {}

merged_cfg = deep_merge(sub_cfg, aswo_cfg)

###############################################################
####### Patch open() so submodule sees merged config
###############################################################
_original_open = builtins.open

def _fake_open(path, *args, **kwargs):
    p = str(path)
    # Only replace submodule's config.json, not others
    if p.endswith("config.json") and str(submodule_path) in p:
        return io.StringIO(json.dumps(merged_cfg))
    return _original_open(path, *args, **kwargs)

def _fake_path_open(self, *args, **kwargs):
    return _fake_open(self, *args, **kwargs)

# Optional environment overrides
os.environ.setdefault("KERNELS_PATH", aswo_cfg.get("kernels_path", ""))
os.environ.setdefault("kernels_path", aswo_cfg.get("kernels_path", ""))

###############################################################
####### Load sc-data-functions with patches active
###############################################################
with patch("builtins.open", _fake_open), patch.object(pathlib.Path, "open", _fake_path_open):
    # Load the subpackage itself
    spec = importlib.util.spec_from_file_location("sc_data_functions", submodule_path / "__init__.py")
    sc_data_functions = importlib.util.module_from_spec(spec)
    sys.modules["sc_data_functions"] = sc_data_functions
    spec.loader.exec_module(sc_data_functions)

    # Dynamically import all modules under sc-data-functions
    __all__ = []
    for modinfo in pkgutil.iter_modules([str(submodule_path)]):
        # Skip __init__.py
        if modinfo.name.startswith("__"):
            continue
        try:
            module = importlib.import_module(modinfo.name)
            globals()[modinfo.name] = module
            __all__.append(modinfo.name)
        except Exception as e:
            print(f"[WARN] Could not import {modinfo.name}: {e}")