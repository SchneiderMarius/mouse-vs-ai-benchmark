"""
track2_eval/src/data_loading.py
================================
Load preprocessed MouseAI session bundles (.npz) and apply the canonical
blackout mask.

Canonical source: Track2_avoid_kernel_death2.ipynb Cell 14 loading block.

Critical invariants (DO NOT CHANGE):
  - Blackout mask: mask = d["blackouts"] == 0  (0 = valid frame, 1 = excluded)
  - Frame indexing: frames = d["frames"].astype("float32")[mask[0, :]]
  - Spike indexing: spikes = d["spikes"].astype("float32")[:, mask[0, :]]
  - FR stored as np.sum(spikes, axis=1) — total counts, NOT Hz
"""

from pathlib import Path

import numpy as np


def load_session_bundle(path: str) -> dict:
    """
    Load one preprocessed session bundle and apply the blackout mask.

    Parameters
    ----------
    path : str
        Path to a .npz file with keys: frames, spikes, blackouts, time, cellarea.

    Returns
    -------
    dict with keys:
        frames      (T_valid, 1, 86, 155)  float32
        spikes      (N, T_valid)            float32
        mouse_name  str                     stem of the file path
        T           int                     number of valid frames
        N           int                     number of neurons
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Session bundle not found: {p}")

    d = np.load(p, allow_pickle=False)

    # --- Canonical blackout masking from Cell 14 ---
    # blackouts shape: (1, T_full); 0 = valid, 1 = excluded
    mask = d["blackouts"] == 0                          # (1, T_full) bool
    frames = d["frames"].astype("float32")[mask[0, :]]  # (T_valid, 1, H, W)
    spikes = d["spikes"].astype("float32")[:, mask[0, :]]  # (N, T_valid)

    T, N = frames.shape[0], spikes.shape[0]
    mouse_name = p.stem

    n_blackout = int((~mask[0, :]).sum())
    print(
        f"  Loaded {mouse_name}: T_valid={T}  N={N}  "
        f"blackout_frames={n_blackout} ({100*n_blackout/(T+n_blackout):.1f}%)"
    )

    return {
        "frames":     frames,
        "spikes":     spikes,
        "mouse_name": mouse_name,
        "T":          T,
        "N":          N,
        "path":       str(p),
    }


def load_all_bundles(session_paths: list) -> list:
    """
    Load all session bundles in order, returning a list of bundle dicts.

    Parameters
    ----------
    session_paths : list of str
        Ordered list of .npz paths.

    Returns
    -------
    list of dicts (same schema as load_session_bundle).
    """
    bundles = []
    for path in session_paths:
        b = load_session_bundle(path)
        bundles.append(b)
    return bundles
