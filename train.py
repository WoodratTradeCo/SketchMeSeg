import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

try:
    from torch_geometric.loader import DataLoader
except ImportError:                 # older pyg exposes it under data
    from torch_geometric.data import DataLoader

from options import BaseOptions, set_seed
from dataset import SketchDataset
from writer import Writer
from netV3 import CADSketch_Seg
from test import test, eval_with_len

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def step(data, model, optimizer, loss_func):
    x = data.x.to(device)
    label = data.y.to(device)
    edge_index = data.edge_index.to(device)
    stroke_data = {
        'stroke_idx': data.stroke_idx.to(device),
        'batch': data.batch.to(device),
        'pos': x,
    }

    optimizer.zero_grad()
    out = model(x, edge_index, stroke_data)
    loss = loss_func(out, label)
    loss.backward()
    optimizer.step()
    return loss


def train():
    opt = BaseOptions().parse()

    train_dataset = SketchDataset(root=os.path.join(opt.data_folder, opt.dataset),
                                  class_name=opt.class_name, split='train')
    test_dataset = SketchDataset(root=os.path.join(opt.data_folder, opt.dataset),
                                 class_name=opt.class_name, split='test')
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size,
                              shuffle=True, num_workers=opt.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size,
                             shuffle=False, num_workers=opt.num_workers)
    print('{} samples for train, {} samples for test'.format(
        len(train_dataset), len(test_dataset)))

    model = CADSketch_Seg(opt).to(device)
    loss_func = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr,
                                 weight_decay=opt.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, opt.epoch,
                                               eta_min=opt.eta_min)

    writer = Writer(opt)

    best_P = 0.0
    for epoch in range(opt.epoch):
        # ---------------- train epoch ----------------
        model.train()
        train_loss = 0.0
        for i, data in enumerate(train_loader):
            loss = step(data, model, optimizer, loss_func)
            train_loss += loss.item()
            if i % opt.print_freq == 0:
                writer.print_train_loss(epoch, i, loss)
        writer.print_epoch_train_loss(train_loss / len(train_loader))
        scheduler.step()

        # ---------------- test epoch ----------------
        model.eval()
        predict_list, loss_list = [], []
        with torch.no_grad():
            for data in test_loader:
                test_loss, predict = test(data, model, device, loss_func)
                loss_list.append(test_loss.item())
                predict_list.extend(predict.reshape(-1, opt.points_num).tolist())

            p_metric, c_metric = eval_with_len(opt, predict_list, dataset='test')
            P_metric = np.average(p_metric)
            C_metric = np.average(c_metric)
            if best_P < P_metric:
                best_P = P_metric
            writer.print_test_result(
                epoch, np.average(loss_list), P_metric, C_metric, best_P)


if __name__ == '__main__':
    set_seed(42)
    train()