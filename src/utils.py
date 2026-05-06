"""
track2_eval/src/utils.py
========================
Shared helpers: config loading, filter loading, pickle I/O, debug subsampling,
skipped-model logging.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "alpha": 1.0,
    "n_folds": 5,
    "epsilon": 0.2,
    "batch_size": 128,
    "max_layers_per_pass": 2,
    "visual_encoder_only": True,
    "fr_thresh": 100,
    "top_percentile": 0.15,
    "visual_thresholds": {
        "C_0005": 0.005,
        "C_001": 0.01,
        "C_002": 0.02,
        "C_005": 0.05,
    },
    "debug_frames": 0,
    "debug_neurons": 0,
}


def load_config(yaml_path: str) -> dict:
    """
    Load a YAML config file and fill missing keys with defaults.

    All path values in the config are resolved relative to the project root
    (parent of track2_eval/).  The resolved root is stored as cfg['_root'].
    """
    yaml_path = Path(yaml_path).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Fill defaults for any missing key
    for k, v in _DEFAULTS.items():
        if k not in cfg:
            cfg[k] = v

    # Resolve root: two levels up from track2_eval/configs/
    # yaml lives at <root>/track2_eval/configs/<name>.yaml
    cfg["_root"] = yaml_path.parent.parent.parent

    return cfg


def resolve_path(cfg: dict, rel_or_abs: str) -> Path:
    """
    Resolve a path that may be relative (to project root) or absolute.
    """
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return cfg["_root"] / p


# ---------------------------------------------------------------------------
# Filter file
# ---------------------------------------------------------------------------

def load_filter(filter_path: Optional[str]) -> Optional[dict]:
    """
    Load filter_raw.pkl and validate its structure.

    Returns None if filter_path is None or the string "null" / empty.
    This indicates pre-filtered sessions where no post-hoc masking is needed.

    Expected schema:
        {mouse_stem: {"R_visual": float32[N], "FR": float32[N], ...}}

    The file may also have a top-level 'per_mouse' wrapper (from the CLI script).
    Both layouts are normalised to a flat {mouse: {...}} dict.
    """
    if filter_path is None or str(filter_path).strip().lower() in ("null", "none", ""):
        print("  filter_file: None — sessions are pre-filtered; all neurons treated as passing.")
        return None

    p = Path(filter_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Filter file not found: {p}\n"
            "Re-run Track2_filtering_regression.ipynb Cell 2 to regenerate it."
        )

    with open(p, "rb") as f:
        raw = pickle.load(f)

    # Unwrap 'per_mouse' wrapper if present
    if isinstance(raw, dict) and "per_mouse" in raw and isinstance(raw["per_mouse"], dict):
        data = raw["per_mouse"]
    elif isinstance(raw, dict):
        # Check: every value must itself be a dict with 'R_visual'
        # (could be top-level mouse dict OR a metadata+per_mouse layout)
        if all(isinstance(v, dict) and "R_visual" in v for v in raw.values()):
            data = raw
        else:
            raise ValueError(
                f"Unexpected filter file layout in {p}. "
                "Expected either {{mouse: {{R_visual: ...}}}} or {{per_mouse: {{mouse: {{R_visual: ...}}}}}}."
            )
    else:
        raise ValueError(f"Cannot parse filter file {p}: root is {type(raw)}")

    # Validate at least one mouse entry has R_visual
    for mouse, block in data.items():
        if "R_visual" not in block:
            raise KeyError(f"Filter block for mouse '{mouse}' is missing 'R_visual'.")
        if "FR" not in block:
            raise KeyError(f"Filter block for mouse '{mouse}' is missing 'FR'.")
        break  # only check first entry; trust rest are consistent

    print(f"  Loaded filter: {p.name}  |  mice: {sorted(data.keys())}")
    return data


# ---------------------------------------------------------------------------
# Pickle I/O
# ---------------------------------------------------------------------------

def save_pkl(obj: object, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(path: Union[str, Path]) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PKL file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Debug subsampling
# ---------------------------------------------------------------------------

def subsample_bundle(bundle: dict, n_frames: int, n_neurons: int) -> dict:
    """
    Trim a session bundle to the first n_frames frames and first n_neurons neurons.
    Pass n_frames=0 or n_neurons=0 to skip that dimension.

    Modifies a copy of the dict; original is untouched.
    """
    b = dict(bundle)

    if n_frames and n_frames > 0:
        T_avail = b["frames"].shape[0]
        n_frames = min(n_frames, T_avail)
        b["frames"] = b["frames"][:n_frames]
        b["spikes"] = b["spikes"][:, :n_frames]
        b["T"] = n_frames

    if n_neurons and n_neurons > 0:
        N_avail = b["spikes"].shape[0]
        n_neurons = min(n_neurons, N_avail)
        b["spikes"] = b["spikes"][:n_neurons, :]
        b["N"] = n_neurons

    return b


# ---------------------------------------------------------------------------
# Skipped-model log
# ---------------------------------------------------------------------------

def make_skipped_log(skipped: List[Dict], out_path: Union[str, Path]) -> None:
    """Save a list of skipped-model dicts to CSV."""
    import pandas as pd
    if not skipped:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(skipped).to_csv(out_path, index=False)
    print(f"  Logged {len(skipped)} skipped models → {out_path}")
