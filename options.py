import os
import random
import argparse

import numpy as np
import torch


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class BaseOptions:
    """All hyper-parameters, exposed through the command line."""

    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def initialize(self):
        # data
        self.parser.add_argument('--data-folder', type=str, default='Dataset',
                                 help='folder that contains the datasets')
        self.parser.add_argument('--dataset', type=str, default='MC2',
                                 help='name of the dataset')
        self.parser.add_argument('--class-name', type=str, default='gear',
                                 help='name of the class to train/test on')
        self.parser.add_argument('--points-num', type=int, default=512,
                                 help='number of points per sketch')
        self.parser.add_argument('--out-segment', type=int, default=3,
                                 help='number of segment labels')

        # training
        self.parser.add_argument('--batch-size', type=int, default=16)
        self.parser.add_argument('--num-workers', type=int, default=0)
        self.parser.add_argument('--epoch', type=int, default=100)
        self.parser.add_argument('--lr', type=float, default=0.01,
                                 help='initial learning rate (Adam)')
        self.parser.add_argument('--weight-decay', type=float, default=5e-4)
        self.parser.add_argument('--eta-min', type=float, default=1e-5)
        self.parser.add_argument('--print-freq', type=int, default=10)
        self.parser.add_argument('--log', type=bool, default=True,
                                 help='write training logs')

        # network
        self.parser.add_argument('--in-feature', type=int, default=2,
                                 help='number of coordinates per point')

    def parse(self):
        self.initialize()
        return self.parser.parse_args()