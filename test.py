import numpy as np
import torch


def test(data, model, device, loss_func):
    """Forward pass of one batch; returns loss and per-point predictions."""
    x = data.x.to(device).requires_grad_(True)
    label = data.y.to(device)
    edge_index = data.edge_index.to(device)
    stroke_data = {
        'stroke_idx': data.stroke_idx.to(device),
        'batch': data.batch.to(device),
        'pos': x,
    }

    out = model(x, edge_index, stroke_data)
    predict = torch.argmax(out, dim=1).cpu().numpy()
    loss = loss_func(out, label)
    return loss, predict


def eval_with_len(opt, predict, dataset='test'):
    """Evaluation against the original NDJSON sketches.

    Every stroke of a sketch consumes a contiguous slice of its prediction
    vector (the training data was already resampled to a fixed point count, so
    the two align). Two metrics are reported:

    * P-metric : point accuracy weighted by segment length
    * C-metric : stroke-level accuracy, a stroke counts as correct when more
                 than 75% of its length-weighted points are correct

    Strokes labelled with -1 (unlabelled) are skipped.
    """
    import json

    with open('{}/{}/{}_{}.ndjson'.format(
            opt.data_folder, opt.dataset, opt.class_name, dataset), 'r') as f:
        test_data = [json.loads(line) for line in f]

    p_metric_list = []
    c_metric_list = []
    for i, sample in enumerate(test_data):
        predict_result = predict[i]
        sketch = sample['drawing']

        p_right, p_sum = 0, 0
        c_right, c_sum = 0, 0
        for stroke in sketch:
            if stroke[2][0] == -1:      # unlabelled stroke
                continue
            c_sum += 1

            # stroke length used as the per-point weight
            stroke_len = [1.0]
            for j in range(1, len(stroke[0])):
                dx = stroke[0][j] - stroke[0][j - 1]
                dy = stroke[1][j] - stroke[1][j - 1]
                stroke_len.append(((dx ** 2 + dy ** 2) ** 0.5))
            stroke_len = np.array(stroke_len)

            stroke_labels = stroke[2]
            pred_labels = np.array(predict_result[:len(stroke_labels)])
            predict_result = predict_result[len(stroke_labels):]

            right = np.array(pred_labels == stroke_labels, dtype=int)
            stroke_p_sum = stroke_len.sum()
            p_sum += stroke_p_sum
            p_right += (right * stroke_len).sum()

            if (right * stroke_len).sum() / stroke_p_sum > 0.75:
                c_right += 1

        p_metric_list.append(p_right / p_sum)
        c_metric_list.append(c_right / c_sum)

    return p_metric_list, c_metric_list