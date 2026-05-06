"""
track2_eval/src/regression_readout.py
=======================================
SRP + Ridge CV scoring loop.

Helper functions (compute_random_projection_dim, contiguous_folds, cor_in_time,
smoothing_with_np_conv) are lifted VERBATIM from
Track2_avoid_kernel_death2.ipynb Cell 14.

run_srp_ridge_cv() wraps the Cell 14 main scoring loop into a single callable
with config-driven parameters and optional debug-mode subsampling.

Critical invariants (DO NOT CHANGE):
  - SRP random_state = 42 + fold_index
  - Ridge alpha = cfg['alpha'] = 1.0
  - Smoothing applied to BOTH Y_hat and Y_val before cor_in_time()
  - smoothing size = int(2000/100) = 20
  - scores stored as nanmean across folds: shape [L, N] float32
  - FR stored as np.sum(spikes, axis=1).astype(float32)  (total counts)
"""

import gc
import os
from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.random_projection import SparseRandomProjection

from .activation_extraction import (
    expose_all_layer_outputs,
    expose_layer_subset,
    get_visual_encoder_layers_only,
    keep_layer,
    run_and_collect_subset,
)
from .utils import subsample_bundle


# ---------------------------------------------------------------------------
# Helper functions — verbatim from Cell 14
# ---------------------------------------------------------------------------

def compute_random_projection_dim(N_train: int, epsilon: float = 0.2) -> int:
    return int(np.ceil((4 * np.log(max(N_train, 2))) / (epsilon ** 2)))


def contiguous_folds(T: int, n_folds: int) -> list:
    block = T // n_folds
    splits = []
    for f in range(n_folds):
        start = f * block
        end = (f + 1) * block if f < n_folds - 1 else T
        val_idx = np.arange(start, end)
        if start > 0:
            train_idx = np.concatenate([np.arange(0, start), np.arange(end, T)])
        else:
            train_idx = np.arange(end, T)
        splits.append((train_idx, val_idx))
    return splits


def cor_in_time(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
    pred = pred.copy()
    label = label.copy()
    pred -= pred.mean(axis=0, keepdims=True)
    label -= label.mean(axis=0, keepdims=True)
    num = np.sum(pred * label, axis=0)
    den = np.sqrt(np.sum(pred ** 2, axis=0) * np.sum(label ** 2, axis=0)) + 1e-9
    r = num / den
    r[np.isnan(r)] = 0
    return r.astype(np.float32)


def smoothing_with_np_conv(nsp: np.ndarray, size: int = int(2000 / 100)) -> np.ndarray:
    np_conv_res = []
    for i in range(nsp.shape[1]):
        np_conv_res.append(
            np.convolve(nsp[:, i], np.ones(size) / size, mode="same")
        )
    return np.transpose(np.array(np_conv_res))


# ---------------------------------------------------------------------------
# Main scoring loop
# ---------------------------------------------------------------------------

def run_srp_ridge_cv(
    model_path: str,
    session_bundles: list,
    cfg: dict,
    debug: bool = False,
) -> Optional[Dict]:
    """
    Run SRP + Ridge contiguous CV scoring for one ONNX model over all session
    bundles.

    Implements Track2_avoid_kernel_death2.ipynb Cell 14 main loop exactly.

    Parameters
    ----------
    model_path : str
        Path to the ONNX model file.
    session_bundles : list of dict
        Each dict from data_loading.load_session_bundle().
    cfg : dict
        Config loaded by utils.load_config(). Relevant keys:
            alpha, n_folds, epsilon, batch_size, max_layers_per_pass,
            visual_encoder_only, debug_frames, debug_neurons.
    debug : bool
        If True, subsample each bundle to cfg['debug_frames'] frames and
        cfg['debug_neurons'] neurons before scoring.

    Returns
    -------
    dict  — result_dict matching the PKL schema from Cell 14, or
    None  — if the model was skipped (too many layers, no visual encoder layers,
            path missing, shape mismatch).
    """
    # --- Config ---
    ALPHA              = float(cfg.get("alpha", 1.0))
    N_FOLDS            = int(cfg.get("n_folds", 5))
    EPSILON            = float(cfg.get("epsilon", 0.2))
    BATCH              = int(cfg.get("batch_size", 128))
    MAX_LAYERS_PER_PASS = int(cfg.get("max_layers_per_pass", 2))
    VISUAL_ONLY        = bool(cfg.get("visual_encoder_only", True))
    DEBUG_FRAMES       = int(cfg.get("debug_frames", 0)) if debug else 0
    DEBUG_NEURONS      = int(cfg.get("debug_neurons", 0)) if debug else 0

    model_path = str(model_path)
    model_name_parts = os.path.splitext(os.path.basename(model_path))[0]
    # Derive model_name as "{id}_{owner}" from the PKL output directory convention
    # The caller sets this — here we just build from the path if needed.
    # We store whatever the caller passed as model_path stem.
    model_name = model_name_parts  # overridden by caller in evaluate_*.py

    if not os.path.exists(model_path):
        print(f"  SKIP: model path does not exist — {model_path}")
        return None

    # --- Expose all valid layers ---
    try:
        session, all_layers_raw = expose_all_layer_outputs(model_path)
    except Exception as e:
        print(f"  SKIP: expose_all_layer_outputs failed — {e}")
        return None

    all_layers = [n for n in all_layers_raw if keep_layer(n)]
    del session
    gc.collect()

    # --- Restrict to visual encoder ---
    if VISUAL_ONLY:
        LAYERS, fusion_node_name = get_visual_encoder_layers_only(model_path, all_layers)
    else:
        LAYERS = all_layers
        fusion_node_name = None

    print(f"  {len(all_layers)} valid layers total  |  "
          f"{len(LAYERS)} visual-encoder layers used  |  "
          f"fusion node: {fusion_node_name}")

    if len(LAYERS) > 200:
        print(f"  SKIP: too many layers ({len(LAYERS)}).")
        return None

    if len(LAYERS) == 0:
        print(f"  SKIP: no usable visual encoder layers found.")
        return None

    # --- Build result container ---
    result_dict: dict = {
        "model":            model_name,
        "proj_type":        "srp",
        "alpha_fixed":      ALPHA,
        "epsilon":          EPSILON,
        "n_folds":          N_FOLDS,
        "layers":           LAYERS,
        "fusion_node_name": fusion_node_name,
        "timestamp":        np.datetime64("now"),
        "per_mouse":        {},
    }

    # --- Score each session bundle ---
    for bundle in session_bundles:
        mouse = bundle["mouse_name"]
        print(f"\n  Mouse: {mouse}")

        # Optional debug subsampling
        b = subsample_bundle(bundle, DEBUG_FRAMES, DEBUG_NEURONS) if debug else bundle

        frames = b["frames"]    # (T, 1, H, W) float32
        spikes = b["spikes"]    # (N, T) float32
        T, N = b["T"], b["N"]

        folds = contiguous_folds(T, N_FOLDS)
        scores_cv = np.zeros((N_FOLDS, len(LAYERS), N), dtype=np.float32)
        srp_dims  = np.zeros(N_FOLDS, dtype=np.int32)

        flag = 0  # shape-mismatch flag

        # --- Layer chunks ---
        for start in range(0, len(LAYERS), MAX_LAYERS_PER_PASS):
            chunk_layers = LAYERS[start : start + MAX_LAYERS_PER_PASS]
            print(
                f"    layers {start}–{start + len(chunk_layers) - 1} / {len(LAYERS)}",
                end="  ",
                flush=True,
            )

            try:
                sess = expose_layer_subset(model_path, chunk_layers)
                acts_chunk = run_and_collect_subset(sess, chunk_layers, frames, batch=BATCH)
            except Exception as e:
                print(f"\n  SKIP (layer chunk error): {e}")
                flag = 1
                gc.collect()
                break

            # Shape sanity check
            if acts_chunk[chunk_layers[0]].shape[0] != T:
                print(
                    f"\n  SKIP: activation T={acts_chunk[chunk_layers[0]].shape[0]} "
                    f"!= frames T={T}"
                )
                flag = 1
                del sess, acts_chunk
                gc.collect()
                break

            # --- CV folds ---
            for f, (tr_idx, va_idx) in enumerate(folds):
                srp_dim = compute_random_projection_dim(len(tr_idx), epsilon=EPSILON)
                srp_dims[f] = srp_dim

                for li_local, layer in enumerate(chunk_layers):
                    li_global = start + li_local
                    A = acts_chunk[layer]

                    srp = SparseRandomProjection(
                        n_components=srp_dim,
                        dense_output=False,
                        random_state=42 + f,   # INVARIANT: must match original
                    )
                    A_tr = srp.fit_transform(A[tr_idx])
                    A_va = srp.transform(A[va_idx])

                    Y_tr = spikes[:, tr_idx].T   # (T_tr, N)
                    Y_va = spikes[:, va_idx].T   # (T_va, N)

                    clf = Ridge(alpha=ALPHA)
                    clf.fit(A_tr, Y_tr)
                    Y_hat = clf.predict(A_va)    # (T_va, N)

                    # INVARIANT: smooth BOTH before correlation
                    scores_cv[f, li_global, :] = cor_in_time(
                        smoothing_with_np_conv(Y_hat),
                        smoothing_with_np_conv(Y_va),
                    )

            del sess, acts_chunk
            gc.collect()

        if flag == 1:
            return None

        scores_mean = np.nanmean(scores_cv, axis=0)  # (L, N)

        result_dict["per_mouse"][mouse] = {
            "scores_cv":    scores_cv,
            "scores":       scores_mean,
            "FR":           np.sum(spikes, axis=1).astype(np.float32),  # total counts
            "neuron_count": int(N),
            "srp_dims":     srp_dims.tolist(),
        }

        mean_per_layer = scores_mean.mean(axis=1)
        best_idx = int(np.argmax(mean_per_layer))
        print(
            f"\n    best R={mean_per_layer.max():.3f} at layer {best_idx}  "
            f"({LAYERS[best_idx]})"
        )
        gc.collect()

    return result_dict
