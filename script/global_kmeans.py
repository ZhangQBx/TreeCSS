# -*- coding: utf-8 -*-
"""
@Auth ： ZQB
@Time ： 2023/8/14 10:45
@File ：global_kmeans.py
"""
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets.bank import Bank
import hydra
from omegaconf import DictConfig
import os
import numpy as np
from tqdm import tqdm

SIZE = 3
K = 20
ITER = 10000

"""
Global kmeans test for Bank.
"""

def euclidean_dist(vector_a, vector_b):
    return np.sqrt(sum(np.power((vector_a - vector_b), 2)))


def init_process(cfg, rank):
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ["MASTER_PORT"] = "5555"

    dist.init_process_group('gloo', rank=rank, world_size=SIZE)
    dataset = Bank(cfg, rank, True, False)
    # print(len(dataset.train_data_numpy))
    if rank == 0:
        initial_centroids = torch.rand(K, 11)
        # print(initial_centroids.tolist())
    else:
        initial_centroids = torch.zeros(K, 11)
    dist.broadcast(initial_centroids, src=0)
    # print(f"{rank}, {initial_centroids}")
    # features = dataset.train_data_numpy.shape[1]
    # print(features)
    if rank == 0:
        centroids = initial_centroids[:, :3]
    elif rank == 1:
        centroids = initial_centroids[:, 3:6]
    else:
        centroids = initial_centroids[:, 6:10]

    cluster_id = []
    for iter in tqdm(range(ITER)):
        dist_list = torch.zeros(7000, K)
        for index, data in enumerate(dataset.train_data_numpy):
            for centroid_index in range(K):
                euc_dist = euclidean_dist(centroids[centroid_index],
                                          data)
                dist_list[index][centroid_index] = euc_dist
        all_gather_dist_list = [torch.zeros(7000, K) for _ in range(SIZE)]
        dist.all_gather(all_gather_dist_list, dist_list)
        summed_dist_list = all_gather_dist_list[0] + all_gather_dist_list[1] + all_gather_dist_list[2]
        summed_dist_list_np = summed_dist_list.numpy()
        cluster_id = np.argmin(summed_dist_list_np, axis=1)

        for centroid_index in range(K):
            data_belong_to_centroid_index = dataset.train_data_numpy[np.nonzero(cluster_id == centroid_index)]
            centroids[centroid_index] = torch.tensor(np.mean(data_belong_to_centroid_index, axis=0))
        if rank == 0:
            print(f"{iter}, {cluster_id}")
    print(cluster_id)


@hydra.main(version_base=None, config_path="../conf", config_name="conf")
def launch(cfg: DictConfig):
    process = []
    mp.set_start_method("spawn")
    for rank in range(SIZE):
        p = mp.Process(target=init_process, args=(cfg, rank))
        p.start()
        process.append(p)

    for p in process:
        p.join()


if __name__ == "__main__":
    launch()
