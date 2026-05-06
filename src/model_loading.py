"""
track2_eval/src/model_loading.py
=================================
Load model metadata from the SQLite submission database or from a directory of
baseline ONNX models.

Canonical sources:
  - load_models_from_db : Track2_avoid_kernel_death2.ipynb Cell 11
  - load_baseline_models: Track2_avoid_kernel_death2.ipynb Cell 16
"""

import os
import sqlite3
from pathlib import Path
from typing import List, Optional

import pandas as pd


def load_models_from_db(
    db_path: str,
    leaderboard_path: Optional[str] = None,
    id_filter: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load model metadata from the submission database.

    Replicates Cell 11: SELECT id, owner, filepath FROM submission_database,
    optionally cross-referencing the leaderboard CSV.

    Parameters
    ----------
    db_path : str
        Path to submission_database.db (SQLite).
    leaderboard_path : str, optional
        Path to leaderboard.csv.  If provided, adds a 'score' column from the
        leaderboard (best score per id across all submissions).
    id_filter : list of str, optional
        If given, keep only rows whose 'id' is in this list.

    Returns
    -------
    pd.DataFrame with columns: id, owner, filepath  (+ score if leaderboard given).
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Submission database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT id, owner, filepath FROM submission_database", conn
    )
    conn.close()

    df["id"] = df["id"].astype(str)

    if leaderboard_path is not None:
        lb_path = Path(leaderboard_path)
        if lb_path.exists():
            lb = pd.read_csv(lb_path)
            lb["id"] = lb["submission"].str.split("_").str[0]
            lb["score"] = pd.to_numeric(lb["score"], errors="coerce")
            best_score = lb.groupby("id")["score"].max().reset_index()
            df = df.merge(best_score, on="id", how="left")
        else:
            print(f"  WARNING: leaderboard not found at {lb_path}, skipping score join.")

    if id_filter is not None:
        id_filter_set = set(str(x) for x in id_filter)
        df = df[df["id"].isin(id_filter_set)].copy()
        print(f"  Filtered to {len(df)} model(s) from id_filter list.")

    print(f"  Loaded {len(df)} model record(s) from DB: {db_path.name}")
    return df.reset_index(drop=True)


def load_baseline_models(model_root: str) -> pd.DataFrame:
    """
    Scan model_root for Baseline_* subdirectories and build a model DataFrame.

    Replicates Cell 16: finds each Baseline_* folder, takes the first *.onnx
    file found inside it.

    Parameters
    ----------
    model_root : str
        Root directory containing Baseline_* subdirs (e.g. D:/model_data).

    Returns
    -------
    pd.DataFrame with columns: id (folder name), owner ('baseline'), filepath.
    """
    model_root = Path(model_root)
    if not model_root.exists():
        raise FileNotFoundError(f"model_root not found: {model_root}")

    records = []
    for folder in sorted(model_root.glob("Baseline_*")):
        if not folder.is_dir():
            continue
        onnx_files = sorted(folder.rglob("*.onnx"))
        if not onnx_files:
            print(f"  WARNING: No ONNX file in {folder} — skipping.")
            continue
        if len(onnx_files) > 1:
            print(f"  WARNING: Multiple ONNX files in {folder}; using first.")
        records.append({
            "id":       folder.name,
            "owner":    "baseline",
            "filepath": str(onnx_files[0]),
        })

    df = pd.DataFrame(records).reset_index(drop=True)
    print(f"  Found {len(df)} baseline model(s) in {model_root}")
    return df


def split_available(df: pd.DataFrame) -> tuple:
    """
    Partition df into (available, missing) based on whether filepath exists.

    Returns
    -------
    (available_df, missing_df) — both share the same columns as df.
    """
    exists_mask = df["filepath"].apply(lambda p: os.path.exists(str(p)))
    available = df[exists_mask].copy().reset_index(drop=True)
    missing   = df[~exists_mask].copy().reset_index(drop=True)

    if len(missing):
        print(f"  {len(missing)} model path(s) not found on disk (will be skipped):")
        for _, row in missing.iterrows():
            print(f"    {row.get('id','')}_{row.get('owner','')} → {row['filepath']}")

    print(f"  Available: {len(available)} / {len(df)} models")
    return available, missing
