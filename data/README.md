# Data files

This folder contains two static reference files included in the repository.
Leaderboard CSVs and full evaluation results will be added when the paper is published.

---

## Files

| File | Rows | Description |
|---|---|---|
| `mouse_performance.csv` | 5 | Mouse behavioural performance across 5 recording sessions |
| `model_metadata.csv` | 236 | Model architecture metadata (backbone, depth, parameters) |

---

### `mouse_performance.csv`

| Column | Description |
|---|---|
| `subject_id` | Session identifier (one row per recording session / mouse) |
| `Normal`, `Fog`, `Clutter`, `Contrast`, `RDK` | Success rate per perturbation condition |
| `heldout_performance` | Mean of Clutter, Contrast, RDK |
| `performance_drop` | Normal minus heldout_performance |
| `robustness_ratio` | heldout_performance / Normal |

Five sessions from three mice (tigre569, tigre613, tigre847). Values are the fraction of
trials in which the mouse made a correct choice.

### `model_metadata.csv`

| Column | Description |
|---|---|
| `id` | Submission ID |
| `owner` | Team name |
| `num_layers` | Total number of layers in the ONNX model |
| `num_params` | Total number of parameters |
| `has_recurrence` | Whether the model contains recurrent connections |
| `backbone` | Visual encoder architecture family (cnn, resnet, moe, transformer, …) |
| `policy_algo` | Reinforcement learning algorithm (ppo, a2c, …) |
| `training_steps` | Number of RL training steps. NaN for models whose filenames do not follow the standard naming convention. |

---

## Perturbation conditions

Five visual conditions were used in Track 1:

| Condition | Type | Role |
|---|---|---|
| Normal | Unperturbed | Reference |
| Fog | Global luminance reduction | Seen during training |
| Clutter | Background clutter | **Held-out** |
| Contrast | Contrast reduction | **Held-out** |
| RDK | Random dot kinematogram | **Held-out** |

Metric definitions (identical for AI and mouse):
- `heldout_performance` = mean(Clutter, Contrast, RDK)
- `performance_drop` = Normal − heldout_performance
- `robustness_ratio` = heldout_performance / Normal

---

## What is NOT in this folder

- **Neural recording data** — the preprocessed Track 2 session bundles (~2 GB) are
  distributed separately. See the project README for the download link and setup instructions.
- **Leaderboard CSVs** — full competition leaderboards will be added when the paper is
  published. Until then, `src/04_figures.ipynb` cannot be run from scratch.
