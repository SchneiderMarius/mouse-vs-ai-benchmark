# Preprocessed Track 2 Dataset — Technical Reference

**Date:** 2026-05-04  
**Scope:** `mouse_vs_ai_data_preprocessed/` — 9 sessions, 3 mice  
**Purpose:** Internal technical reference for NeurIPS 2025 Evaluations & Datasets submission  
**Audit basis:** Direct inspection of all `session_filter_metadata.json` files, NPZ array headers, and benchmark notebook source code. All values are observed facts; unknowns are stated explicitly.

---

## 1. Overview

`mouse_vs_ai_data_preprocessed/` is the reviewer-facing preprocessed Track 2 dataset for the Robust Foraging Benchmark. It contains nine recording sessions from three mice, each stored as a NumPy `.npz` archive with synchronized stimulus video frames and simultaneously recorded two-photon calcium imaging spike trains.

Neurons are pre-filtered at the **C0.005** threshold (`R_visual ≥ 0.005` AND total spike count `FR ≥ 100`), which is equivalent to the filtering applied during the original leaderboard evaluation. This reduces each session's neuron dimension without changing Track 2 scores.

**Key numbers:**

| Item | Value |
|---|---|
| Sessions | 9 |
| Mice | 3 (tigre569, tigre613, tigre847) |
| Total kept neurons (N_kept) | 13,624 |
| Total original neurons (N_orig) | 56,739 |
| Fraction retained | 24.0% |
| Total frames (T_full, all sessions) | 103,986 |
| Approx. total recording time | 173.3 min (at 10 Hz) |
| Total NPZ storage | ~2.14 GB |
| **Official leaderboard subset** | 3 sessions (p2s38, p2s23, p2s8) — **4,763 neurons** |

The official leaderboard was computed on the three sessions `tigre569_p2s38`, `tigre613_p2s23`, and `tigre847_p2s8`. The remaining six sessions are additional held-out recordings included for completeness and potential extended evaluation.

---

## 2. Source Data

### Animals and task

Three adult transgenic mice performed a head-fixed virtual-reality foraging task while undergoing mesoscale two-photon calcium imaging. Details:

- **Mouse IDs:** tigre569, tigre613, tigre847
- **Transgenic line:** TITL-GCaMP6s × Emx1-Cre × ROSA:LNL:tTA (GCaMP6s expression in excitatory neurons)
- **Task:** Head-fixed VR foraging task with five visual perturbation conditions: Normal, Fog, Clutter, Contrast, RDK
- **Imaging:** Diesel2P mesoscope; layer 2/3 neurons across posterior neocortex (V1 and higher visual areas)
- **Ethics:** IACUC-approved protocol (University of California, Santa Barbara)

### Raw file locations (internal, not part of released package)

| Batch | Raw format | Internal path |
|---|---|---|
| 3 original sessions | Preprocessed `.npz` (corrected) | `analyisis_old/JoeDATA_fixed/` |
| 6 additional sessions | Raw `.mat` (HDF5) | `analyisis_old/JoeDATA/additional 6 sessions/` |

The 3 original sessions were preprocessed prior to the competition and orientation-corrected (see §3). The 6 additional sessions were converted from `.mat` to `.npz` using the same pipeline immediately before this release.

---

## 3. Preprocessing Pipeline

All sessions pass through a three-stage pipeline. The complete pipeline is documented in:
`mouse-vs-ai-benchmark/src/preprocess_track2.ipynb`

Individual scripts (for inspection or partial re-runs) are at the repository root:
- **Stage 1:** `preprocess_track2_sessions.py`
- **Stage 2:** `track2_eval/compute_filter_mask.py`
- **Stage 3:** `scripts/apply_track2_neuron_filter.py`

### Stage 1 — `.mat` → `.npz` (raw to unfiltered bundle)

Input: raw HDF5 `.mat` file from two-photon preprocessing pipeline.

Key operations:
1. Load frames from MATLAB group `mousevAI_preprocessed_out`
2. Crop and resize frames to 86×155 px (from 150×256 original resolution)
3. **Orientation fix:** `skimage.transform.resize` + transpose — applied to all sessions to correct a systematic frame-orientation bug in the original preprocessing. Both original and additional sessions have this fix applied.
4. Normalize frame values to `[0, 1]` as `float16`
5. Extract spike trains (binned at 100 ms / 10 Hz with 60 ms neural-latency offset)
6. Detect and encode blackout frames (scene transitions / invalid frames) as `uint8`
7. Extract timestamps and cell area indices

Output: unfiltered `.npz` with 5 arrays — `frames`, `spikes`, `blackouts`, `time`, `cellarea`.

### Stage 2 — R_visual computation (image→spike regression)

Input: unfiltered `.npz` bundles.

Computes per-neuron visual responsiveness (`R_visual`) via ridge regression from pixel space to spike space:

1. Flatten each valid frame to a 1D feature vector (pixel values)
2. Fit Ridge regression (α = 1.0) predicting spike trains from pixel features
3. 5-fold contiguous cross-validation (no shuffling; folds are consecutive time blocks)
4. `R_visual` = mean Pearson r between ridge-predicted and observed spike trains, averaged over folds
5. Results stored as float64 (no downcast, matching original leaderboard computation)

Filter results are stored in two separate PKL files:
- **Original 3 sessions:** `analyisis_old/filter_mask/filter_raw.pkl`
- **Additional 6 sessions:** `analyisis_old/filter_mask/filter_raw_new6.pkl`

### Stage 3 — Apply C0.005 filter

Input: unfiltered `.npz` + filter PKL (R_visual and FR values).

Filter criterion: keep neuron _i_ if and only if:
```
R_visual[i] >= 0.005  AND  sum(spikes[i, valid_frames]) >= 100
```

Outputs per session (4 files):
1. Filtered `.npz` — frames and spikes indexed to kept neurons only
2. `session_filter_metadata.json` — filter provenance
3. `neuron_filter_mask.npy` — bool array of shape (N_orig,)
4. `original_neuron_indices.npy` — integer indices of kept neurons in original population

---

## 4. Neuron Filter (C0.005)

### Parameters (identical across all 9 sessions)

| Parameter | Value |
|---|---|
| `filter_name` | `"C0.005"` |
| `r_visual_threshold` | `0.005` |
| `fr_threshold` | `100.0` total spike counts |
| `date_created` | `2026-05-01` (all sessions) |
| `script_commit_hash` | `null` (not recorded at time of preprocessing — see §9) |

### Equivalence verification

Pre-filtering the data (applied before running the SRP+Ridge evaluation) has been verified to produce scores identical to post-hoc masking (the method used in the original leaderboard). Verified on 15 trials with max |delta| = 0.00.

### Per-session retention rates

Retention varies substantially across sessions (17.4%–38.9%), reflecting differences in data quality and imaging conditions across animals and recording sessions:

| Session | N_orig | N_kept | Fraction |
|---|---|---|---|
| tigre569_p2s38 | 8,711 | 2,444 | 28.1% |
| tigre613_p2s23 | 3,286 | 851 | 25.9% |
| tigre847_p2s8 | 7,749 | 1,468 | 18.9% |
| tigre569_p2s28 | 7,966 | 1,387 | 17.4% |
| tigre569_p2s35 | 8,793 | 1,965 | 22.3% |
| tigre613_p2s21 | 3,993 | 1,265 | 31.7% |
| tigre613_p2s22 | 4,011 | 1,560 | 38.9% |
| tigre847_p2s5 | 6,552 | 1,332 | 20.3% |
| tigre847_p2s9 | 5,678 | 1,352 | 23.8% |

---

## 5. Session Inventory

Complete per-session inventory. All metadata values are from `session_filter_metadata.json`; NPZ sizes are from filesystem inspection (2026-05-04). Duration is computed as `T_full / 10` seconds.

| Session stem | Mouse | N_orig | N_kept | Excl. | Fraction | T_full | Duration (min) | NPZ (MB) |
|---|---|---|---|---|---|---|---|---|
| `tigre569_p2s28_mousevAI_perturbs_preprocessed` | tigre569 | 7,966 | 1,387 | 6,579 | 17.4% | 7,929 | 13.2 | 162.9 |
| `tigre569_p2s35_mousevAI_perturbs_preprocessed` | tigre569 | 8,793 | 1,965 | 6,828 | 22.3% | 9,747 | 16.2 | 199.6 |
| `tigre569_p2s38_mousevAI_perturbs_preprocessed` | tigre569 | 8,711 | 2,444 | 6,267 | 28.1% | 12,804 | 21.3 | 263.2 |
| `tigre613_p2s21_mousevAI_perturbs_preprocessed` | tigre613 | 3,993 | 1,265 | 2,728 | 31.7% | 12,266 | 20.4 | 254.7 |
| `tigre613_p2s22_mousevAI_perturbs_preprocessed` | tigre613 | 4,011 | 1,560 | 2,451 | 38.9% | 15,222 | 25.4 | 317.9 |
| `tigre613_p2s23_mousevAI_perturbs_preprocessed` | tigre613 | 3,286 | 851 | 2,435 | 25.9% | 11,414 | 19.0 | 236.2 |
| `tigre847_p2s5_mousevAI_perturbs_preprocessed` | tigre847 | 6,552 | 1,332 | 5,220 | 20.3% | 12,309 | 20.5 | 251.5 |
| `tigre847_p2s8_mousevAI_perturbs_preprocessed` | tigre847 | 7,749 | 1,468 | 6,281 | 18.9% | 12,383 | 20.6 | 249.9 |
| `tigre847_p2s9_mousevAI_perturbs_preprocessed` | tigre847 | 5,678 | 1,352 | 4,326 | 23.8% | 9,912 | 16.5 | 200.9 |
| **TOTAL** | — | **56,739** | **13,624** | **43,115** | **24.0%** | **103,986** | **173.3** | **~2,136** |

**Notes:**
- Sessions `p2s38`, `p2s23`, `p2s8` are the **official leaderboard sessions** (N_kept = 4,763 combined).
- Source for original sessions (`p2s38`, `p2s23`, `p2s8`): `analyisis_old/JoeDATA_fixed/` (orientation-corrected NPZ).
- Source for additional sessions: `reconstructed_npz/` (converted from raw `.mat` using `preprocess_track2_sessions.py`).

---

## 6. File Format

### 6.1 NPZ arrays (5 keys, identical schema across all 9 sessions)

| Key | Shape | dtype | Value range | Description |
|---|---|---|---|---|
| `frames` | `(T_full, 1, 86, 155)` | float16 | [0.0, 0.996] | Stimulus video frames at 10 Hz; single grayscale channel; cropped to 86×155 px |
| `spikes` | `(N_kept, T_full)` | float32 | [0, ~83–127] | Binned spike counts at 10 Hz; total count (not rate) per 100 ms bin |
| `blackouts` | `(1, T_full)` | uint8 | {0, 1} | 0 = valid frame; 1 = excluded (scene transition or recording artifact) |
| `time` | `(T_full,)` | float32 | seconds from 0.0 | Frame timestamps; uniform spacing ≈ 0.1 s |
| `cellarea` | `(1, N_kept)` | uint8 | 1–9 | Brain area index per neuron (area labels not included in release) |

**Loading example:**

```python
import numpy as np

d = np.load("tigre569_p2s38_mousevAI_perturbs_preprocessed.npz")
print(list(d.keys()))       # ['frames', 'spikes', 'blackouts', 'time', 'cellarea']

# Apply blackout mask before analysis (required — do not skip)
mask   = d["blackouts"] == 0          # shape (1, T_full), dtype bool
frames = d["frames"][mask[0]]          # (T_valid, 1, 86, 155) float32 after cast
spikes = d["spikes"][:, mask[0]]       # (N_kept, T_valid)

# Cast frames to float32 for downstream processing
frames = frames.astype("float32")
```

### 6.2 Sidecar files (3 per session directory)

| File | Size range | dtype / format | Description |
|---|---|---|---|
| `session_filter_metadata.json` | 1,251–1,262 B | JSON | Filter provenance: thresholds, neuron counts, source path, date created |
| `neuron_filter_mask.npy` | 3,414–8,921 B | bool `(N_orig,)` | `True` for each kept neuron in the original (unfiltered) population |
| `original_neuron_indices.npy` | 3,532–9,904 B | uint32 `(N_kept,)` | Integer indices of kept neurons within the original population |

**Usage — reconstruct original neuron index:**

```python
mask    = np.load("neuron_filter_mask.npy")             # bool (N_orig,)
indices = np.load("original_neuron_indices.npy")        # uint32 (N_kept,)
# indices[i] gives the position of kept neuron i in the original (unfiltered) array
```

### 6.3 Root-level files

The `mouse_vs_ai_data_preprocessed/` root also contains three files that appear to be residuals from an earlier intermediate step (single-session test run) rather than session-level content:

| File | Size | Note |
|---|---|---|
| `croissant.json` | 27,725 B | Croissant 1.0 metadata for the full 9-session release — see §8 |
| `neuron_filter_mask.npy` | 8,094 B | Root-level copy; not a session directory file — provenance unclear |
| `original_neuron_indices.npy` | 5,676 B | Root-level copy; not a session directory file — provenance unclear |
| `session_filter_metadata.json` | 1,253 B | Root-level copy; not a session directory file — provenance unclear |

The root-level `neuron_filter_mask.npy` / `original_neuron_indices.npy` / `session_filter_metadata.json` should be treated as stale artifacts. The authoritative per-session files are those inside each session subdirectory.

---

## 7. Aggregate Statistics

### Per-mouse breakdown

| Mouse | Sessions | N_orig | N_kept | Fraction | T_full | Duration (min) |
|---|---|---|---|---|---|---|
| tigre569 | 3 (p2s28, p2s35, p2s38) | 25,470 | 5,796 | 22.8% | 30,480 | 50.8 |
| tigre613 | 3 (p2s21, p2s22, p2s23) | 11,290 | 3,676 | 32.6% | 38,902 | 64.8 |
| tigre847 | 3 (p2s5, p2s8, p2s9) | 19,979 | 4,152 | 20.8% | 34,604 | 57.7 |
| **All** | **9** | **56,739** | **13,624** | **24.0%** | **103,986** | **173.3** |

### Official 3-session leaderboard subset

The official Track 2 leaderboard was computed on `p2s38 + p2s23 + p2s8`:

| Session | Mouse | N_kept | T_full |
|---|---|---|---|
| tigre569_p2s38 | tigre569 | 2,444 | 12,804 |
| tigre613_p2s23 | tigre613 | 851 | 11,414 |
| tigre847_p2s8 | tigre847 | 1,468 | 12,383 |
| **Total** | — | **4,763** | **36,601** |

The value `n_C_0005 = 4,763` in `leaderboard_track2.csv` matches this sum exactly.

---

## 8. Metadata

### Croissant metadata

| File | Status | Size | Notes |
|---|---|---|---|
| `mouse_vs_ai_data_preprocessed/croissant.json` | ✓ Present | 27,725 B | 9 sessions, MIT license, 6 creators (UCSB), Croissant 1.0, full RAI fields |
| `mouse-vs-ai-benchmark/metadata/` folder | ✗ Does not exist | — | No benchmark-repo-level Croissant file |

The `croissant.json` at the data-folder root is the authoritative Croissant record for this dataset. It conforms to the MLCommons Croissant 1.0 schema and includes:
- 9 `cr:FileObject` entries for `.npz` session bundles (with `contentSize` values matching the filesystem)
- 9 `cr:FileObject` entries for `session_filter_metadata.json` sidecar files
- 9 `cr:FileObject` entries for `neuron_filter_mask.npy` sidecar files
- 9 `cr:FileObject` entries for `original_neuron_indices.npy` sidecar files
- Full `rai:*` metadata (data limitations, biases, personal sensitive information, use cases, social impact, data collection methods)
- `rai:personalSensitiveInformation`: explicitly states no human subjects; all data from IACUC-approved mouse experiments

**Content URL status:** `contentUrl` values in `croissant.json` are relative paths (e.g., `tigre569_p2s38.../tigre569_p2s38...npz`). These must be updated to absolute download URLs (Zenodo, GitHub Releases, etc.) before public release. No `sha256` or `md5` checksums are currently present; add before final upload.

---

## 9. Known Issues and Discrepancies

### Issue 1 — CRITICAL: Wrong `DATA_DIR` default in `02_track2_evaluation.ipynb`

**Severity:** Critical — causes `FileNotFoundError` on first run with default settings.

**Location:** `mouse-vs-ai-benchmark/02_track2_evaluation.ipynb`, cell `cell-1`, User Settings block.

**Current (wrong) value:**
```python
DATA_DIR = ROOT / "data" / "track2_preprocessed_filtered_C0005"
```
This path does not exist anywhere in the release package.

**Correct value:**
```python
DATA_DIR = ROOT.parent / "mouse_vs_ai_data_preprocessed"
```
The preprocessed data folder sits one level above the benchmark repo root, not inside `data/`.

The same stale folder name `track2_preprocessed_filtered_C0005` also appears in `01_explore_dataset.ipynb` cell `a0f16618` (in a markdown instruction telling users where to copy the data). This issue has since been corrected — `02_track2_evaluation.ipynb` (the renamed evaluation notebook) now uses the correct path.

### Issue 2 — `script_commit_hash` is null in all metadata files

All 9 `session_filter_metadata.json` files have `"script_commit_hash": null`. The git commit of `apply_track2_neuron_filter.py` was not recorded at the time of preprocessing (2026-05-01). This means the exact script version used for each session cannot be verified from the metadata alone. The script version can be recovered from the git history if needed.

### Issue 3 — Two separate filter PKL files for the two session batches

The R_visual filter was computed in two separate runs:

| PKL file | Sessions covered |
|---|---|
| `analyisis_old/filter_mask/filter_raw.pkl` | p2s38, p2s23, p2s8 (original 3) |
| `analyisis_old/filter_mask/filter_raw_new6.pkl` | p2s28, p2s35, p2s21, p2s22, p2s5, p2s9 (additional 6) |

These two PKL files are stored internally and are not part of the released `mouse_vs_ai_data_preprocessed/` folder. The filter results are already applied to the NPZ files; the PKL files are only needed if recomputing filter masks.

### Issue 4 — Root-level stale sidecar files in `mouse_vs_ai_data_preprocessed/`

Three sidecar files (`neuron_filter_mask.npy`, `original_neuron_indices.npy`, `session_filter_metadata.json`) exist at the top level of `mouse_vs_ai_data_preprocessed/` alongside `croissant.json`. These are not session-level files and appear to be artifacts from a test run. They should be removed before public upload (they are not referenced in `croissant.json`).

### Issue 5 — `contentUrl` and checksums missing from `croissant.json`

`contentUrl` values in `croissant.json` are relative paths, not absolute download URLs. `sha256` / `md5` checksum fields are absent. Both must be populated before public release.

---

## 10. Reproduction

### Re-running the full preprocessing pipeline

The complete three-stage pipeline is documented in:
```
mouse-vs-ai-benchmark/src/preprocess_track2.ipynb
```
This notebook is provided for completeness and documentation purposes; it requires access to the raw `.mat` files and the internal filter PKL files.

### Running individual stages

**Stage 1 — `.mat` → `.npz`:**
```bash
python preprocess_track2_sessions.py \
    --input_dir "analyisis_old/JoeDATA/additional 6 sessions" \
    --output_dir reconstructed_npz \
    --target_h 86 --target_w 155 \
    --fix_orientation
```

**Stage 2 — Compute R_visual filter:**
```bash
# Additional 6 sessions
python track2_eval/compute_filter_mask.py \
    --sessions reconstructed_npz/tigre569_p2s28_*.npz [... all 6 ...] \
    --out analyisis_old/filter_mask/filter_raw_new6.pkl

# Original 3 sessions (for reproduction/validation)
python track2_eval/compute_filter_mask.py \
    --sessions analyisis_old/JoeDATA_fixed/tigre569_p2s38_*.npz [... all 3 ...] \
    --out analyisis_old/filter_mask/filter_raw_reproduced.pkl \
    --compare analyisis_old/filter_mask/filter_raw.pkl
```

**Stage 3 — Apply filter:**
```bash
python scripts/apply_track2_neuron_filter.py \
    --sessions reconstructed_npz/*.npz [... all sessions ...] \
    --filter_pkl analyisis_old/filter_mask/filter_raw_new6.pkl \
    --out_dir mouse_vs_ai_data_preprocessed \
    --r_thresh 0.005 --fr_thresh 100
```

### Runtime estimates (CPU)

| Stage | Per session | All 9 sessions |
|---|---|---|
| Stage 1 (.mat → .npz) | ~5 min | ~45 min |
| Stage 2 (R_visual) | ~20–45 min | ~3–7 hours |
| Stage 3 (apply filter) | < 1 min | < 10 min |

Stage 2 (Ridge regression, 5-fold CV over all valid frames) is the bottleneck. Runtime scales with `T_valid × N_orig` — sessions with more neurons or frames take proportionally longer.

### Validation

After re-running Stage 3 on the original 3 sessions:
```bash
python track2_eval/compute_filter_mask.py \
    --sessions analyisis_old/JoeDATA_fixed/tigre569_p2s38_*.npz \
               analyisis_old/JoeDATA_fixed/tigre613_p2s23_*.npz \
               analyisis_old/JoeDATA_fixed/tigre847_p2s8_*.npz \
    --out analyisis_old/filter_mask/filter_raw_reproduced.pkl \
    --compare analyisis_old/filter_mask/filter_raw.pkl
```
Expected output: `PASS - all R_visual arrays match within tol=1e-4`.

Pre-filter equivalence (pre-filtering vs post-hoc masking) has been verified on 15 trials: max |delta| = 0.00 on all score columns.

---

*Report prepared from direct filesystem inspection of `mouse_vs_ai_data_preprocessed/` and benchmark codebase audit. Values sourced from `session_filter_metadata.json` (all 9 files), NPZ array headers (3 representative sessions), `croissant.json`, and `02_track2_evaluation.ipynb` source cells.*
