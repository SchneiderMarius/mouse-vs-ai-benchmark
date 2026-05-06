# Mouse vs. AI: Robust Foraging Benchmark

Evaluation package for the **Mouse vs. AI** benchmark ([competition website](https://robustforaging.github.io) · [paper](PAPER_LINK)).

## Quick Start — Track 2 (Neural Alignment)

```bash
git clone https://github.com/SchneiderMarius/mouse-vs-ai-benchmark.git
cd mouse-vs-ai-benchmark
pip install -r requirements.txt
```

Download the preprocessed neural dataset (~2 GB) from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YB8J31) and place the `.npz` files in `data/`:

```
mouse-vs-ai-benchmark/
├── data/
│   ├── tigre569_p2s38_mousevAI_perturbs_preprocessed.npz
│   ├── tigre613_p2s23_mousevAI_perturbs_preprocessed.npz
│   └── tigre847_p2s8_mousevAI_perturbs_preprocessed.npz
├── models/baselines/          ← baseline ONNX models included
├── 01_explore_dataset.ipynb   ← browse frames, spikes, and neuron stats
└── 02_track2_evaluation.ipynb ← evaluate any ONNX model
```

Run `01_explore_dataset.ipynb` to browse the data, then `02_track2_evaluation.ipynb` to score a model. Set `MODEL_NAME` in the notebook to any model in `models/` and run all cells.

## Track 1 (Visual Robustness)

Track 1 evaluation runs inside the Unity game engine. Download builds for your platform from the [competition website](https://robustforaging.github.io).

## Citation

```bibtex
@inproceedings{schneider2025mousevsai,
  title     = {Mouse vs.\ {AI}: A Benchmark for Visual Robustness and Neural Alignment},
  author    = {Schneider, Marius and Canzano, Joe and Hou, Yuchen and Peng, Jing
               and Smith, Spencer LaVere and Beyeler, Michael},
  booktitle = {NeurIPS --- Evaluations \& Datasets Track},
  year      = {2025},
  url       = {https://robustforaging.github.io}
}
```

License: [MIT](LICENSE)
