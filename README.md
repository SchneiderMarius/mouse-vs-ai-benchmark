# Mouse vs. AI: Robust Foraging Benchmark

Evaluation package for the **NeurIPS 2025 Evaluations & Datasets** submission.
Two tracks assess AI agents on the same virtual-reality foraging task performed by mice:

- **Track 1 — Visual Robustness** — agent success rate across five visual perturbation conditions
- **Track 2 — Neural Alignment** — how well agent representations predict mouse visual cortex spike trains

| | |
|---|---|
| 🌐 Competition website | https://robustforaging.github.io |
| 📄 Paper | *coming soon* |
| 📦 Neural data (~2 GB) | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YB8J31 |
| 🎮 Unity builds (Track 1) | *[INSERT DOWNLOAD LINK]* |

---

## Quick Start

### Install

```bash
git clone https://github.com/SchneiderMarius/mouse-vs-ai-benchmark.git
cd mouse-vs-ai-benchmark
pip install -r requirements.txt
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate mouse-vs-ai
```

### Explore the neural dataset (< 1 min)

1. Download the preprocessed data (~2 GB) — link above
2. Place `mouse_vs_ai_data_preprocessed/` one level above this repo (see layout below)
3. Open `01_explore_dataset.ipynb` and run all cells

### Run the Track 2 evaluation on a baseline model (~30–90 min on CPU)

1. Place a baseline ONNX model in `models/baselines/Baseline_NatureCNN/`
2. Open `02_track2_evaluation.ipynb`
3. In the **User Settings** cell set `MODEL_NAME = "Baseline_NatureCNN"`
4. Run all cells — the final cell prints the `track2_score`

Baseline ONNX models are provided in `models/baselines/`. Reproducing their reported scores
confirms the pipeline is working correctly on your machine.

---

## Track 1 — Visual Robustness

Track 1 evaluation requires the competition Unity game builds (Windows / macOS / Linux).

**Download Unity builds:** *[INSERT DOWNLOAD LINK]*

Agents (ONNX format) are scored across five visual conditions:

| Condition | Type | Split |
|---|---|---|
| Normal | Unperturbed | Reference |
| Fog | Global luminance reduction | Seen |
| Clutter | Background clutter | **Held-out** |
| Contrast | Contrast reduction | **Held-out** |
| RDK | Random dot kinematogram | **Held-out** |

Primary metric: mean success rate across the three held-out conditions (Clutter, Contrast, RDK).

---

## Track 2 — Neural Alignment

Track 2 measures whether a model's internal representations linearly predict individual
neuron spike trains recorded from mouse visual cortex.

**Setup:**

```
mouseai-master/
├── mouse-vs-ai-benchmark/                    ← this repo
└── mouse_vs_ai_data_preprocessed/            ← download separately
    ├── tigre569_p2s38_mousevAI_perturbs_preprocessed.npz
    ├── tigre613_p2s23_mousevAI_perturbs_preprocessed.npz
    ├── tigre847_p2s8_mousevAI_perturbs_preprocessed.npz
    └── ...  (9 .npz files total)
```

**Scoring pipeline:**
ONNX activations → Sparse Random Projection (ε = 0.2) → Ridge regression (α = 1.0) →
5-fold contiguous cross-validation → Pearson r per neuron → mean across visually-tuned
neurons (C0.005: firing rate ≥ 100 spikes AND R_visual ≥ 0.005).

**Output:** `track2_score` — mean Pearson r; higher is better.

---

## Dataset

Paired visual frames and spike trains from mouse visual cortex during active VR navigation.

| Property | Value |
|---|---|
| Animals | 3 transgenic mice (GCaMP6s, head-fixed) |
| Recording sessions | 9 total (3 used for the official leaderboard) |
| Neurons retained | 13,624 (from ~56,700 recorded; C0.005 quality filter) |
| Brain areas | V1 and higher visual areas (LM, LI, AL, RL, AM, PM) |
| Sampling rate | 10 Hz (100 ms bins) |
| Stimulus frame size | 86 × 155 px, grayscale |
| Total recording time | ~173 min |
| File format | One `.npz` per session — arrays: `frames`, `spikes`, `blackouts`, `time`, `cellarea` |
| Download size | ~2.1 GB |

**Download:** https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YB8J31

Minimal loading snippet:

```python
import numpy as np

d      = np.load("tigre569_p2s38_mousevAI_perturbs_preprocessed.npz", allow_pickle=False)
mask   = d["blackouts"][0] == 0      # exclude scene-transition frames
frames = d["frames"][mask]           # (T_valid, 1, 86, 155)  float16
spikes = d["spikes"][:, mask]        # (N_neurons, T_valid)   float32
```

Full preprocessing and filter documentation:
[`docs/preprocessed_track2_dataset_report.md`](docs/preprocessed_track2_dataset_report.md)

---

## Repository Layout

```
mouse-vs-ai-benchmark/
│
├── 01_explore_dataset.ipynb     Browse frames, spike trains, neuron filter statistics
├── 02_track2_evaluation.ipynb   Evaluate an ONNX model on Track 2 (neural alignment)
│
├── src/
│   ├── data_loading.py          Load .npz session bundles and apply blackout mask
│   ├── activation_extraction.py ONNX layer activation extraction
│   ├── regression_readout.py    SRP + Ridge cross-validation scoring loop
│   ├── metrics.py               Score aggregation (Score A / B / C variants)
│   ├── utils.py                 Config, pickle I/O, debug helpers
│   └── 04_figures.ipynb         Reproduce paper figures (requires leaderboard CSVs*)
│
├── models/
│   └── baselines/               Official baseline ONNX models
│
├── data/
│   ├── mouse_performance.csv    Behavioural performance — 3 mice, 5 perturbations
│   └── model_metadata.csv       Architecture metadata for submitted models
│
├── figures/
│   ├── main/                    Main paper figure panels (PDF + PNG)
│   └── supplement/              Supplementary figure panels (PDF + PNG)
│
├── docs/
│   └── preprocessed_track2_dataset_report.md   Technical dataset reference
│
└── results/                     Evaluation outputs written here (gitignored)
```

> *\* Full leaderboard CSVs will be released with the paper.
> Until then, `src/04_figures.ipynb` will raise a `FileNotFoundError` on the data-loading cell.
> The generated figures are already committed in `figures/`.*

---

## Citation

```bibtex
@inproceedings{schneider2025mousevsai,
  title     = {Mouse vs.\ {AI}: A Benchmark for Visual Robustness and Neural Alignment
               in Reinforcement Learning Agents},
  author    = {Schneider, Marius and Canzano, Joe and Hou, Yuchen and Peng, Jing and
               Smith, Spencer LaVere and Beyeler, Michael},
  booktitle = {Advances in Neural Information Processing Systems ---
               Evaluations \& Datasets Track},
  year      = {2025},
  url       = {https://robustforaging.github.io}
}
```

---

## License

MIT — see [`LICENSE`](LICENSE).
