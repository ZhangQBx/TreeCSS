# -*- coding: utf-8 -*-
"""
@Auth ： ZQB
@Time ： 2023/5/19 17:49
@File ：choose_gpu.py
"""
import torch


def get_cuda_device(gpu_id, trainer):
    """
    get torch device from gpu id
    :param trainer: trainer type
    :param model: model type
    :param gpu_id: which gpu
    :return:
    """
    if torch.cuda.is_available() and trainer == 'mlp':
        device = torch.device('cuda', gpu_id)
    else:
        print(">>>GPU is not available")
        device = torch.device('cpu')

    return device
