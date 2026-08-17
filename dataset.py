import json
import os.path as osp

import numpy as np
import torch
from torch_geometric.data import Data


class SketchDataset(torch.utils.data.Dataset):
    """Reads NDJSON sketches and converts them to torch_geometric Data.

    Layout of one sketch in the NDJSON file::

        {"drawing": [[xs, ys, labels], [xs, ys, labels], ...]}

    Each inner list is one stroke; ``labels`` is the per-point segment label.
    Coordinates are normalized to [0, 1] per sketch. Edges connect consecutive
    points inside a stroke (both directions); ``stroke_idx`` marks which stroke
    every point belongs to.
    """

    def __init__(self, root, class_name, split='train'):
        self.class_name = class_name
        self.split = split
        self.json_dir = osp.join(root, '{}_{}.ndjson'.format(class_name, split))
        self.pt_dir = osp.join(root, '{}_{}.pt'.format(class_name, split))

        if osp.exists(self.pt_dir):
            self.processed_data = torch.load(self.pt_dir)
        else:
            self.processed_data = self._process()

    def _process(self):
        with open(self.json_dir, 'r') as f:
            raw = [json.loads(line)['drawing'] for line in f]

        data_list = []
        for sketch in raw:
            strokes = [np.array(s) for s in sketch]

            # stack all points, remembering the stroke id of each one
            stroke_idx = np.concatenate(
                [np.full(len(s[0]), i) for i, s in enumerate(strokes)])
            point = np.concatenate([s.transpose()[:, :2] for s in strokes]).astype(float)

            # per-sketch normalization into [0, 1]
            lo = point.min(axis=0)
            hi = point.max(axis=0)
            point = (point - lo) / (hi - lo)

            label = np.concatenate([s[2] for s in strokes], axis=0)

            # intra-stroke edges (undirected)
            edge_index = []
            offset = 0
            for s in strokes:
                n = len(s[0])
                for i in range(n - 1):
                    edge_index += [[offset + i, offset + i + 1],
                                   [offset + i + 1, offset + i]]
                offset += n
            edge_index = np.array(edge_index).transpose()

            data_list.append(Data(
                x=torch.FloatTensor(point),
                edge_index=torch.LongTensor(edge_index),
                y=torch.LongTensor(label),
                stroke_idx=torch.LongTensor(stroke_idx),
            ))

        torch.save(data_list, self.pt_dir)
        return data_list

    def __getitem__(self, index):
        return self.processed_data[index]

    def __len__(self):
        return len(self.processed_data)