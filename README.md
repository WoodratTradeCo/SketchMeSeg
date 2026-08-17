# CADSketch-Seg

Point-wise segmentation of CAD-style vector sketches. The input is a
stroke-based vector drawing (x/y samples grouped into strokes); the output is
a segment label for every point (e.g. for a gear: teeth / body / shaft).

> **Status.** The paper describing this work has been submitted to **Pattern
> Recognition Letters**. This repository currently provides a small sample of
> the mechanical-sketch data and the core implementation. The final version of
> the data and code will be released once the paper is accepted.

## Method

The model has two parallel branches built on the same point sequence:

* **Geometric branch** -- 4 stacked DynGCN blocks (dynamic kNN graph
  convolution, k = 8) that read the raw geometry. Their outputs are
  concatenated into a 128-d feature `F_geo`.
* **Sequence branch** -- a 1-layer Bi-GRU (hidden 32, output 64) followed by 4
  Transformer encoder layers (d_model = 64, 4 heads, FFN width 192, dropout
  0.1). The self-attention is biased by the T-Grd encoding: centrality
  embeddings, a geometric bias `b_ij` learned from local curvature and
  shortest-path distance, and a stroke bias `c_ij` aggregated over path edges.

The branch features are concatenated (128 + 64), fused by an MLP into 256
dims, pooled per stroke, classified, and the stroke logits are broadcast back
to the points. The loss is a per-point (log) cross-entropy, so the network can
also serve in a stroke-only setting.

Implementation: `netV3.py`, shared blocks in `modules/`.

## Data

Mechanical sketch classes (gear, screw) with per-point labels. Each line of an
`.ndjson` file is one sketch:

```json
{"drawing": [[xs, ys, labels], [xs, ys, labels], ...]}
```

Each inner list is one stroke; `labels` holds the segment id of every point.
Coordinates are normalized to [0, 1] per sketch and the sketches are already
resampled to 512 points. The `Dataset/` folder only contains a partial sample;
the full dataset will be released with the final version.

## Setup

Python 3.8+, PyTorch, PyTorch Geometric, torch-scatter, torch-cluster.

```bash
pip install torch torch-scatter torch-sparse torch-cluster torch-geometric
```

`dataset.py` uses plain `torch_geometric.data.Data` and works with current pyg
releases.

## Train

```bash
python train.py --data-folder Dataset --dataset MC2 --class-name gear
```

Hyper-parameters live in `options.py` (defaults: 100 epochs, Adam lr 0.01,
weight decay 5e-4, cosine annealing down to 1e-5). An evaluation is run after
every epoch.

## Evaluation

Two metrics are reported on the test set:

* **P** -- point accuracy weighted by segment length;
* **C** -- stroke-level accuracy, where a stroke counts as correct if more
  than 75% of its length-weighted points agree with the ground truth.

## Licence

All rights reserved. The code is made available for academic use. The final
release will accompany the accepted paper.