"""
track2_eval/src/metrics.py
============================
Score aggregation: from per-model PKL files to the final leaderboard CSV.

compute_model_score() is lifted VERBATIM from
Track2_avoid_kernel_death2.ipynb Cell 19 (compute_scores function).
aggregate_scores() wraps it for directory-level batch processing.

Critical invariants (DO NOT CHANGE):
  - FR threshold: cfg['fr_thresh'] = 100 (total spike count, not Hz)
  - Best-layer selection: nanargmax(nanmean(scores[:, mask_A], axis=1))
    where mask_A = FR >= fr_thresh  (from FR-filtered neurons ONLY)
  - Score B: top max(5, ceil(top_percentile × n_FR_neurons)) by regression R
    (NOT by R_visual — it is the top regression-score neurons)
  - Score C: mask_A & (R_visual >= threshold) — BOTH FR and R_visual required
  - All score criteria use the SAME best-layer index chosen for Score A
  - Filter file must be filter_raw.pkl (not pca variant)
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .utils import load_pkl, save_pkl


# ---------------------------------------------------------------------------
# Per-model score computation (verbatim from Cell 19 compute_scores)
# ---------------------------------------------------------------------------

def compute_model_score(
    result_dict: dict,
    filter_data: dict,
    cfg: dict,
) -> Optional[Dict]:
    """
    Compute Track 2 scores for one model from its PKL result dict.

    Reproduces Cell 19 compute_scores() exactly.

    Parameters
    ----------
    result_dict : dict
        PKL loaded from regression_readout.run_srp_ridge_cv() output.
    filter_data : dict
        Loaded from filter_raw.pkl: {mouse_stem: {"R_visual": ..., "FR": ...}}.
    cfg : dict
        Config dict.  Relevant keys: fr_thresh, top_percentile, visual_thresholds.

    Returns
    -------
    dict with columns matching track2_scores_visual_new.csv, or None on failure.
    """
    VISUAL_THRESHOLDS = cfg.get(
        "visual_thresholds",
        {"C_0005": 0.005, "C_001": 0.01, "C_002": 0.02, "C_005": 0.05},
    )
    FR_THRESH     = float(cfg.get("fr_thresh", 100))
    TOP_PERCENTILE = float(cfg.get("top_percentile", 0.15))

    model_name = result_dict["model"]
    per_mouse  = result_dict["per_mouse"]

    # Storage for pooled neuron scores
    A_vals = []
    B_vals = []
    C_vals = {key: [] for key in VISUAL_THRESHOLDS}
    failures = []

    for mouse, mblock in per_mouse.items():

        scores = np.asarray(mblock["scores"])   # [L, N]
        FR     = np.asarray(mblock["FR"])        # [N]
        n_neurons = len(FR)

        if filter_data is None:
            # Pre-filtered sessions: every neuron already passed FR >= 100
            # and R_visual >= 0.005 by construction (applied before scoring).
            # Use a dummy R_visual = 1.0 so any visual threshold is satisfied.
            mask_A   = np.ones(n_neurons, dtype=bool)
            R_visual = np.ones(n_neurons, dtype=np.float64)
        else:
            if mouse not in filter_data:
                failures.append(f"missing_visual_filter:{mouse}")
                continue

            R_visual = np.asarray(filter_data[mouse]["R_visual"])  # [N_total]

            # Debug-mode safety: PKL may have been subsampled to first n_neurons;
            # align R_visual to the same first-N neurons.
            if len(R_visual) != n_neurons:
                if len(R_visual) > n_neurons:
                    R_visual = R_visual[:n_neurons]
                else:
                    failures.append(f"filter_shorter_than_pkl:{mouse}:"
                                     f"filter={len(R_visual)} pkl={n_neurons}")
                    continue

            # ---- Mask A: FR >= threshold ----
            mask_A = FR >= FR_THRESH
        if mask_A.sum() == 0:
            failures.append(f"no_FR_neurons:{mouse}")
            continue

        # Best layer chosen from FR-filtered neurons only
        mean_layer = np.nanmean(scores[:, mask_A], axis=1)   # [L]
        if np.all(np.isnan(mean_layer)):
            failures.append(f"all_nan_scores:{mouse}")
            continue
        best_idx = int(np.nanargmax(mean_layer))

        R_best_all = scores[best_idx, :]    # [N] — all neurons, best layer
        R_best_FR  = R_best_all[mask_A]     # [n_A] — FR-filtered only

        # ---- Score A ----
        A_vals.extend(R_best_FR.tolist())

        # ---- Score B: top TOP_PERCENTILE of regression scores in FR set ----
        n_top = max(5, int(np.ceil(TOP_PERCENTILE * len(R_best_FR))))
        B_vals.extend(np.sort(R_best_FR)[-n_top:].tolist())

        # ---- Score C variants ----
        for cname, thr in VISUAL_THRESHOLDS.items():
            mask_C = mask_A & (R_visual >= thr)
            R_C = R_best_all[mask_C]
            if len(R_C) > 0:
                C_vals[cname].extend(R_C.tolist())

    if len(A_vals) == 0:
        failures.append("no_valid_neurons")
        return None

    out: dict = {
        "model":          model_name,
        "id":             model_name.split("_")[0],
        "score_A_FR":     float(np.nanmean(A_vals)),
        "score_B_top15":  float(np.nanmean(B_vals)) if B_vals else np.nan,
        "n_A":            len(A_vals),
        "n_B":            len(B_vals),
    }
    for cname, vals in C_vals.items():
        out[f"score_{cname}"] = float(np.nanmean(vals)) if vals else np.nan
        out[f"n_{cname}"]     = len(vals)

    return out


# ---------------------------------------------------------------------------
# Directory-level aggregation
# ---------------------------------------------------------------------------

def aggregate_scores(
    results_dir: Union[str, Path],
    filter_file: str,
    cfg: dict,
    out_csv: Union[str, Path],
) -> pd.DataFrame:
    """
    Scan results_dir for *.pkl files, score each one, and save a CSV.

    Skips files that fail to parse or produce no valid neurons.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing one .pkl per model (output of regression_readout).
    filter_file : str
        Path to filter_raw.pkl.
    cfg : dict
        Config dict.
    out_csv : str or Path
        Where to write the output CSV.

    Returns
    -------
    pd.DataFrame  sorted by score_A_FR descending (matches original CSV order).
    """
    from .utils import load_filter

    results_dir = Path(results_dir)
    out_csv     = Path(out_csv)

    pkl_files = sorted(results_dir.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(
            f"No .pkl result files found in {results_dir}.\n"
            "Run evaluate_track2.py first to generate them."
        )
    print(f"  Aggregating {len(pkl_files)} PKL files from {results_dir}")

    filter_data = load_filter(filter_file)  # None if filter_file is None/null

    records  = []
    failures = []

    for pkl_path in pkl_files:
        try:
            result_dict = load_pkl(pkl_path)
        except Exception as e:
            failures.append({"model": pkl_path.stem, "reason": f"load_error: {e}"})
            continue

        row = compute_model_score(result_dict, filter_data, cfg)
        if row is None:
            failures.append({"model": pkl_path.stem, "reason": "no_valid_neurons"})
        else:
            records.append(row)

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("score_A_FR", ascending=False).reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"\n  Scored {len(records)} models, {len(failures)} failures.")
    if failures:
        print("  Failures:")
        for fa in failures:
            print(f"    {fa['model']} — {fa['reason']}")
    print(f"  Saved → {out_csv}")

    return df


# ---------------------------------------------------------------------------
# Quick comparison helper (used by entry points with --compare_csv)
# ---------------------------------------------------------------------------

def compare_to_reference(
    new_csv: Union[str, Path],
    ref_csv: Union[str, Path],
    tol: float = 1e-4,
    verbose: bool = True,
    compare_cols: Optional[List] = None,
) -> bool:
    """
    Compare new scores to the reference CSV.

    Parameters
    ----------
    compare_cols : list of str, optional
        Score column names to check.  Default: ["score_C_0005"].
        Pass None to use the default, or supply a list for custom columns.

    Returns True if all checked scores match within tolerance, False otherwise.
    Prints a full diff table.
    """
    if compare_cols is None:
        compare_cols = ["score_C_0005"]

    new_csv = Path(new_csv)
    ref_csv = Path(ref_csv)

    if not new_csv.exists():
        print(f"ERROR: new_csv not found: {new_csv}")
        return False
    if not ref_csv.exists():
        print(f"ERROR: ref_csv not found: {ref_csv}")
        return False

    new_df = pd.read_csv(new_csv)
    ref_df = pd.read_csv(ref_csv)

    merged = new_df.merge(ref_df, on="model", suffixes=("_new", "_ref"))
    if merged.empty:
        print(
            "WARNING: No overlapping model IDs between new and reference CSV.\n"
            "  Check that the same PKL files were used for aggregation."
        )
        return False

    # Filter to requested columns, warn if any are missing
    available = [c for c in ref_df.columns if c.startswith("score_")]
    score_cols = [c for c in compare_cols if c in available]
    missing_req = [c for c in compare_cols if c not in available]
    if missing_req:
        print(f"  WARNING: requested column(s) not in reference CSV: {missing_req}")
        print(f"  Available score columns: {available}")
    if not score_cols:
        print("  ERROR: no valid comparison columns — aborting comparison.")
        return False

    diff_rows = []
    nan_rows  = []   # rows where new=NaN but ref has a real value
    any_fail  = False

    for col in score_cols:
        col_new = f"{col}_new"
        col_ref = f"{col}_ref"
        if col_new not in merged.columns:
            print(f"  WARNING: {col} not in new CSV — skipping.")
            continue

        # Flag rows where new=NaN but reference is a real number
        nan_mask = merged[col_new].isna() & merged[col_ref].notna()
        for _, row in merged[nan_mask].iterrows():
            nan_rows.append({
                "model":   row["model"],
                "column":  col,
                "ref":     row[col_ref],
                "new":     "NaN",
                "|delta|": float("inf"),
            })
            any_fail = True

        # Flag rows that exceed tolerance (ignoring NaN pairs)
        delta = (merged[col_new] - merged[col_ref]).abs()
        for _, row in merged[(delta > tol) & ~nan_mask].iterrows():
            diff_rows.append({
                "model":   row["model"],
                "column":  col,
                "ref":     row[col_ref],
                "new":     row[col_new],
                "delta":   row[col_new] - row[col_ref],
                "|delta|": abs(row[col_new] - row[col_ref]),
            })
            any_fail = True

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Score comparison  (tol={tol})")
        print(f"  Column(s) checked: {score_cols}")
        print(f"  Models compared  : {len(merged)}  |  ref: {ref_csv.name}")
        print(f"{'='*60}")

        if nan_rows:
            print(f"\n  NOTE: {len(nan_rows)} new value(s) are NaN where reference is non-NaN:")
            nan_df = pd.DataFrame(nan_rows)
            print(nan_df[["model", "column", "ref", "new"]].to_string(index=False))

        if diff_rows:
            diff_df = pd.DataFrame(diff_rows).sort_values("|delta|", ascending=False)
            print(diff_df.to_string(index=False))
            print(
                f"\n  FAIL: {len(diff_rows)} score(s) exceeded tolerance "
                f"(max |delta| = {diff_df['|delta|'].max():.2e})"
            )
        elif nan_rows:
            print(f"\n  FAIL: {len(nan_rows)} score(s) are NaN in new but non-NaN in reference.")
            print("  This usually means too few neurons passed the thresholds (debug mode?).")
        else:
            n_checked = len(score_cols) * len(merged)
            print(f"  PASS: all {n_checked} values ({len(score_cols)} col x {len(merged)} models) within tol={tol}")

    return not any_fail
