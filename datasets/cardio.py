# -*- coding: utf-8 -*-
"""
@Auth ： ZQB
@Time ： 2023/7/6 16:45
@File ：cardio.py
"""
import numpy as np
import pandas as pd
from utils import timer
from .basic_dataset import BasicDataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import time
import math


def pressure_range(x):
    """ Function to adjust pressure numbers as 3 or 2 digits only"""
    if math.log10(x) >= 4:
        z = x // 100
    elif math.log10(x) >= 3:
        z = x // 10
    elif math.log10(x) < 2 and x < 50:
        z = x * 10
    else:
        z = x
    return z


class Cardio(BasicDataset):
    def __init__(self, cfg, client_rank, train=True, is_label_owner=False):
        super(Cardio, self).__init__(cfg, client_rank, train,
                                     is_label_owner)
        key = ''
        if cfg.trainer == 'mlp':
            key = 'mlp_conf'
        elif cfg.trainer == 'lr':
            key = 'lr_conf'
        elif cfg.trainer == 'knn':
            key = 'knn_conf'
        self.csv_path = cfg[key].data_path

        self._get_train_and_test_tensor_data()
        print(self.train_data_tensor.shape)
        print(self.train_label_tensor.shape)
        print(self.test_data_tensor.shape)
        print(self.test_label_tensor.shape)

    @timer
    def _load_data_from_csv(self):
        data = pd.read_csv(self.csv_path, sep=';')
        data = data.drop('id', axis=1)

        data["ap_hi"] = data["ap_hi"].abs()
        data["ap_hi"] = data["ap_hi"].apply(pressure_range)
        data = data[data["ap_hi"].between(80, 370)]

        data["ap_lo"] = data["ap_lo"].abs()
        data = data[data["ap_lo"] != 0]
        data["ap_lo"] = data["ap_lo"].apply(pressure_range)
        data = data[data["ap_lo"].between(60, 360)]

        data["pulse_pressure"] = data["ap_hi"] - data["ap_lo"]
        data = data.loc[:, ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'pulse_pressure', 'cholesterol',
                            'gluc', 'smoke', 'alco', 'active', 'cardio']]
        data = data[data["pulse_pressure"].between(10, 120)]

        y = data['cardio']
        x = data.drop('cardio', axis=1)
        # print(x.info())

        if self.cfg.trainer == 'mlp':
            enc = OneHotEncoder()
            y = enc.fit_transform(y.values.reshape(-1, 1)).toarray()
            y = y.astype(np.int)

        return x, y
