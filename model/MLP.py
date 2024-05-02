# -*- coding: utf-8 -*-
"""
@Auth ： ZQB
@Time ： 2023/5/21 17:46
@File ：MLP.py
"""
import torch
from torch import nn


class MLPBottomModel(nn.Module):
    def __init__(self, i_f, o_f):
        super().__init__()
        self.dense = nn.Linear(i_f, o_f)
        nn.init.xavier_normal_(self.dense.weight)

    def forward(self, x):
        x = self.dense(x)
        x = torch.relu(x)
        return x


class MLPTopModel(nn.Module):
    def __init__(self, i_f, o_f):
        super().__init__()
        self.dense_1 = nn.Linear(i_f, 20, bias=False)
        nn.init.xavier_normal_(self.dense_1.weight)
        self.dense_2 = nn.Linear(20, o_f, bias=False)
        nn.init.xavier_normal_(self.dense_2.weight)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = torch.relu(x)
        return x


class MLP(nn.Module):
    def __init__(self, n_f):
        super().__init__()
        self.dense_1 = nn.Linear(n_f, n_f)
        nn.init.xavier_normal_(self.dense_1.weight)
        self.dense_2 = nn.Linear(n_f, 2)
        nn.init.xavier_normal_(self.dense_2.weight)

    def forward(self, x):
        x = self.dense_1(x)
        x = torch.relu(x)
        x = self.dense_2(x)
        return x
