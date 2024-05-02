# -*- coding: utf-8 -*-
"""
@Auth ： ZQB
@Time ： 2023/5/8 16:30
@File ：basic_trainer.py
"""
import random

import torch
import torch.utils.data as tud
from datasets import dataset
import tenseal as ts
import numpy as np
from utils import timer
from sklearn.cluster import KMeans
from collections import Counter
from rpc.psi import generate_rsa_keys, encode_local_id_use_pk, decode_ids, \
    encode_and_hash_local_ids_use_sk, get_double_psi_result_genexps, \
    encode_empty_psi_result, get_final_psi_result
import rpc.grpc_file.vfl_server_service_pb2_grpc as vfl_server_service_pb2_grpc
import rpc.grpc_file.vfl_server_service_pb2 as vfl_server_service_pb2
import rpc.grpc_file.vfl_client_service_pb2_grpc as vfl_client_service_pb2_grpc
import rpc.grpc_file.vfl_client_service_pb2 as vfl_client_service_pb2
import rpc.grpc_file.vfl_label_owner_service_pb2_grpc as vfl_label_owner_service_pb2_grpc
import rpc.grpc_file.vfl_label_owner_service_pb2 as vfl_label_owner_service_pb2
import time
import grpc
from distutils.util import strtobool
from tqdm import tqdm
from copy import deepcopy


def euclidean_dist(vector_a, vector_b):
    """

    :param vector_a: a
    :param vector_b: b
    :return: euclidean distance between a and b
    """
    return np.sqrt(sum(np.power((vector_a - vector_b), 2)))


class BasicTrainer:
    def __init__(self, rank, logger, cfg, he_file, device, ip_address,
                 type_, is_label_owner=False, is_regression=False):
        self.type = type_
        self.rank = rank
        self.logger = logger
        self.cfg = cfg
        self.device = device
        self.is_label_owner = is_label_owner
        self.epochs = None
        self.batch_size = None
        self.batch_size_normal = None
        self.batch_size_cc = None
        self.lr = None
        self.lr_gamma = None
        self.lr_step = None
        self.optimizer = None
        self.early_stop = False
        self.is_regression = is_regression
        self.sleep = 0.001
        self.ip_address = ip_address

        max_msg_size = 1000000000
        self.options = [('grpc.max_send_message_length', max_msg_size),
                        ('grpc.max_receive_message_length', max_msg_size)]
        he_ctx_bytes = open(he_file, "rb").read()
        self.he_key = ts.context_from(he_ctx_bytes)
        torch.manual_seed(self.cfg.defs.seed)

        self.origin_train_dataset = dataset(cfg, rank, True, is_label_owner, self.type)
        self.cc_train_dataset = None
        self.train_dataset = None
        self.test_dataset = dataset(cfg, rank, False, is_label_owner, self.type)

        self.origin_train_loader = None
        self.cc_train_loader = None

        self.train_loader = None
        self.test_loader = None

        self.num_origin_train_batches = None
        self.num_cc_train_batches = None

        self.num_train_batches = None
        self.num_test_batches = None

        # For KMEANS
        self.k = None
        self.cluster_data = None
        self.cluster_centers = None
        self.cluster_labels = None
        self.remove_outliers_rate = self.cfg.defs.vertical_fl.remove_outliers_rate
        self.outlier_data_index = set()
        self.weight_list = None

        # For RSA psi
        self.use_psi = cfg.defs.use_psi
        self.psi_only = cfg.defs.psi_only
        self.vfl_client_status = None
        self.rsa_pk = None
        self.rsa_sk = None
        self.rsa_pk_comm_status = False
        self.client_enc_ids_pk = []
        self.client_ra_list = []
        self.client_enc_ids_comm_status = False
        self.client_dec_ids = []
        self.server_hash_enc_ids = []
        self.client_dec_ids_comm_status = False
        self.server_hash_enc_ids_comm_status = False
        self.psi_result = None
        if not self.use_psi:
            self.psi_result = [i for i in range(len(self.origin_train_dataset))]
        self.label_owner_psi_permission = False

        # For align index
        self.align_index = None
        self.use_align_label = self.cfg.defs.use_align_label

        # self._generate_origin_dataloader()

    @timer
    def train_test_vertical_model(self):
        if self.cfg.defs.vertical_fl.train_type == 'cc':
            self._preprocess_local_data()
            time.sleep(0.5)
        else:
            self._rsa_psi([i for i in range(len(self.origin_train_dataset))],
                          self.ip_address, self.rank, 1)
            # start_time = time.time()
            # self._rsa_psi([i for i in range((self.rank+1) * 10000)],
            #               self.ip_address, self.rank, 1)
            # self.logger.warning(f'PSI Time: {time.time()-start_time}')

        if self.psi_only:
            return
        if not self.is_label_owner:
            self._train()
            time.sleep(1)
            self._test()

    def select_align_index(self, label, distance, weight):
        assert len(label) == len(distance)
        dist_dict = dict()
        result_dict = dict()

        # print(len(label[0]))
        # print(len(distance))
        for i in range(len(self.psi_result)):
            temp = []
            dist = 0
            index = self.psi_result[i]
            label_class, label_types = self._get_align_item_label(index)
            for item in label:
                temp.append(item[i])
            temp = tuple(temp)
            for item in distance:
                dist += item[i]
            if temp not in dist_dict.keys():
                if not self.use_align_label:
                    dist_dict[temp] = dist
                    result_dict[temp] = index
                else:
                    dist_dict[temp] = [-1 for _ in range(label_types)]
                    dist_dict[temp][label_class] = dist
                    result_dict[temp] = [-1 for _ in range(label_types)]
                    result_dict[temp][label_class] = index
            else:
                if not self.use_align_label:
                    if dist < dist_dict[temp]:
                        dist_dict[temp] = dist
                        result_dict[temp] = index

                else:
                    if dist_dict[temp][label_class] == -1:
                        dist_dict[temp][label_class] = dist
                        result_dict[temp][label_class] = index
                    else:
                        if dist < dist_dict[temp][label_class]:
                            dist_dict[temp][label_class] = dist
                            result_dict[temp][label_class] = index

        if not self.use_align_label:
            align_index_list = list(result_dict.values())
            align_index_list.sort()
        else:
            result_value = np.array(list(result_dict.values())).flatten()
            align_index_set = set(result_value)
            align_index_list = list(align_index_set)
            if -1 in align_index_list:
                align_index_list.remove(-1)
            else:
                print("No remove.")
        # align_index_list = list(result_dict.values())
        random.shuffle(align_index_list)

        weight_list = []
        for index in align_index_list:
            weight_list.append(weight[index])

        weight_list = np.array(weight_list)
        self.weight_list = torch.from_numpy(weight_list.reshape(-1, 1)).float()
        self.align_index = align_index_list

        return align_index_list

    def _get_align_item_label(self, index):
        raise NotImplementedError

    def _train(self):
        assert self.epochs is not None
        if self.rank == 0:
            d = {'batch_size': self.batch_size, 'Clients': self.origin_train_dataset.num_clients}
            for epoch in tqdm(range(self.epochs), desc="Training Process", postfix=d):
                self._train_client_iteration(epoch)
                if self.early_stop:
                    tqdm.write(">>>Train finish.")
                    return
                # print(f">>>Epoch {epoch + 1}, complete. (Client {self.rank})")
            tqdm.write(">>>Train finish.")
        else:
            for epoch in range(self.epochs):
                self._train_client_iteration(epoch)
                if self.early_stop:
                    return
        # print(">>>Train finish.")

    def _test(self):
        self._test_client_iteration()
        if self.rank == 0:
            tqdm.write(">>>Test finish.")

    def _train_client_iteration(self, epoch):
        raise NotImplementedError

    def _test_client_iteration(self):
        raise NotImplementedError

    def _generate_origin_dataloader(self):
        if self.cfg.defs.vertical_fl.train_type == 'normal':
            self.__generate_origin_train_dataloader()
        self.__generate_test_dataloader()

    def _adjust_learning_rate(self, epoch):

        """
        adjust lr while training
        :return:
        """
        raise NotImplementedError

    @timer
    def _preprocess_local_data(self):
        if self.is_label_owner:
            while not self.label_owner_psi_permission:
                time.sleep(self.sleep)
            self._rsa_psi([i for i in range(len(self.origin_train_dataset))],
                          self.ip_address, self.rank, 1)
            # self._rsa_psi([i for i in range(100000)],
            #               self.ip_address, self.rank, 1)
            if self.psi_only:
                return
        else:
            # self.__kmeans_cluster()
            # remain_indexes_set = self.__remove_outliers()
            response = self.__invite_label_owner_psi()
            if not response.psi_permission:
                raise ValueError("Invitation failed.")
            self._rsa_psi([i for i in range(len(self.origin_train_dataset))],
                          self.ip_address, self.rank, 1)
            # self._rsa_psi([i for i in range(100000)],
            #               self.ip_address, self.rank, 1)
            if self.psi_only:
                return
            self.__kmeans_cluster()
            # remain_indexes_set = self.__remove_outliers()
            self.__calculate_weight()
            self.__get_align_index()
        while not self.align_index:
            time.sleep(self.sleep)
        self.cc_train_dataset = self.origin_train_dataset.update_dataset_via_indexes(self.align_index)
        self.__generate_align_train_dataloader()

    @timer
    def _rsa_psi(self, id_list, ip_address, cid, qid):
        if not self.use_psi:
            return
        psi_id_list = id_list
        psi_result_status = [False, None]
        self.__init_vfl_client_status(ip_address)
        while True:
            if psi_id_list is not None:
                num_of_ids = len(psi_id_list)
            else:
                num_of_ids = 0
            vfl_server_response = self.__get_vfl_server_psi_status(num_of_ids, cid, qid, psi_result_status)
            if vfl_server_response[1]:
                self.__decode_he_psi_result(vfl_server_response[2])

                return

            psi_id_list, psi_result_status = self.__rsa_double_psi(psi_id_list, cid, qid, vfl_server_response[0])

    def _get_vfl_server_rpc_stub(self):
        vfl_server_address = self.cfg.server_conf.vfl_server.host + ":" + \
                             self.cfg.server_conf.vfl_server.port
        vfl_server_channel = grpc.insecure_channel(vfl_server_address, options=self.options)
        vfl_server_stub = vfl_server_service_pb2_grpc.VFLServerServiceStub(vfl_server_channel)

        return vfl_server_stub

    def __reset_rsa_psi_status_per_round(self):
        self.rsa_pk = None
        self.rsa_sk = None
        self.rsa_pk_comm_status = False
        self.client_enc_ids_pk = []
        self.client_ra_list = []
        self.client_enc_ids_comm_status = False
        self.client_dec_ids = []
        self.server_hash_enc_ids = []
        self.client_dec_ids_comm_status = False
        self.server_hash_enc_ids_comm_status = False

    def __reset_all_rsa_psi_status(self):
        self.vfl_client_status = None
        self.rsa_pk = None
        self.rsa_sk = None
        self.rsa_pk_comm_status = False
        self.client_enc_ids_pk = []
        self.client_ra_list = []
        self.client_enc_ids_comm_status = False
        self.client_dec_ids = []
        self.server_hash_enc_ids = []
        self.client_dec_ids_comm_status = False
        self.server_hash_enc_ids_comm_status = False
        self.psi_result = None

    def __generate_origin_train_dataloader(self):
        self.logger.warning(f"Origin train Dataset length:{len(self.origin_train_dataset)}")
        self.train_dataset = deepcopy(self.origin_train_dataset)

        self.origin_train_loader = tud.DataLoader(self.origin_train_dataset, self.batch_size_normal)
        self.train_loader = self.origin_train_loader

        self.num_origin_train_batches = len(self.origin_train_loader)
        self.num_train_batches = self.num_origin_train_batches
        self.batch_size = self.batch_size_normal

    def __generate_align_train_dataloader(self):
        self.logger.warning(f"Align train Dataset length:{len(self.cc_train_dataset)}")
        self.train_dataset = deepcopy(self.cc_train_dataset)

        self.cc_train_loader = tud.DataLoader(self.cc_train_dataset, self.batch_size_cc)
        self.train_loader = self.cc_train_loader

        self.num_cc_train_batches = len(self.cc_train_loader)
        self.num_train_batches = self.num_cc_train_batches
        self.batch_size = self.batch_size_cc

    def __generate_test_dataloader(self):
        if self.test_loader is not None:
            return
        # batch_size = self.cfg.defs.vertical_fl.batch_size

        self.logger.warning(f"Test Dataset length:{len(self.test_dataset)}")
        self.test_loader = tud.DataLoader(self.test_dataset, self.batch_size_normal)
        self.num_test_batches = len(self.test_loader)

    @timer
    def __kmeans_cluster(self):
        """
        use KMEANS to clust local data
        :return:
        """
        if self.k is None:
            raise ValueError("K is None.")
        self.cluster_data = self.origin_train_dataset.train_data_numpy.tolist()

        cluster_pred = KMeans(n_clusters=self.k).fit(self.cluster_data)
        self.cluster_labels = cluster_pred.labels_
        self.cluster_centers = cluster_pred.cluster_centers_

    def __detect_outliers(self):
        """
        detect outer points after kmeans
        :return: data indexes need to be removed
        """
        euc_dist_list = [{} for _ in range(self.k)]
        counter_dict = Counter(self.cluster_labels)

        for index, data in enumerate(self.cluster_data):
            cluster_center_index = self.cluster_labels[index]
            euc_dist_list[cluster_center_index][index] = euclidean_dist(data,
                                                                        self.cluster_centers[
                                                                            cluster_center_index])

        # outlier_data_index = set()
        for cluster_index in range(self.k):
            num_addition_data = int(counter_dict[cluster_index] * self.remove_outliers_rate)
            sorted_euc_dist_list = sorted(euc_dist_list[cluster_index].items(),
                                          key=lambda x: x[1], reverse=True)
            addition_data_index = [item[0] for item in sorted_euc_dist_list][:num_addition_data]

            try:
                self.outlier_data_index = self.outlier_data_index | set(addition_data_index)
            except TypeError:
                self.outlier_data_index.add(addition_data_index)

    def __remove_outliers(self):
        """
        remove outer points
        :return: remain data indexes
        """
        remain_indexes_set = set([i for i in range(len(self.origin_train_dataset))])

        self.__detect_outliers()

        print(f"Amount of Outliers: {len(self.outlier_data_index)}")
        print(f"Former dataset length: {len(remain_indexes_set)}")
        remain_indexes_set -= self.outlier_data_index
        print(f"Latter dataset length: {len(remain_indexes_set)}")

        return remain_indexes_set

    def __calculate_weight(self):
        """
        calculate weight for data instance use Euclidean distance
        :return:
        """
        self.weight_list = [0 for _ in range(len(self.psi_result))]
        euc_dist_list = [{} for _ in range(self.k)]

        for index, data in enumerate(self.cluster_data):
            cluster_center_index = self.cluster_labels[index]
            euc_dist_list[cluster_center_index][index] = euclidean_dist(data,
                                                                        self.cluster_centers[cluster_center_index])

        for cluster_index in range(self.k):
            sorted_euc_dist_list = sorted(euc_dist_list[cluster_index].items(),
                                          key=lambda x: x[1], reverse=True)
            data_index = [item[0] for item in sorted_euc_dist_list]

            # unnormalize
            weight = [(index + 1) / len(data_index) for index in range(len(data_index))]
            # normalize
            # normalize_base = sum([i + 1 for i in range(len(data_index))])
            # weight = [(index + 1) / normalize_base for index in range(len(data_index))]

            for index, i in enumerate(data_index):
                if self.weight_list[i] != 0:
                    raise ValueError
                self.weight_list[i] = weight[index]

        # print(self.rank, ":", self.weight_list[:10])

    def __init_vfl_client_status(self, ip_address):
        self.vfl_client_status = [ip_address, True, 0]

    def __update_vfl_client_status(self, store_psi_result, current_round):
        self.vfl_client_status[1] = store_psi_result
        self.vfl_client_status[2] = current_round

    def __generate_final_psi_result(self):
        if len(self.psi_result) == 0:
            self.psi_result = encode_empty_psi_result()
        psi_final_result = ts.ckks_vector(self.he_key, self.psi_result)

        return psi_final_result

    def __decode_he_psi_result(self, encode_bytes):
        psi_enc_result = ts.ckks_vector_from(self.he_key, encode_bytes)
        psi_dec_result = psi_enc_result.decrypt()
        self.__reset_all_rsa_psi_status()

        self.psi_result = get_final_psi_result(psi_dec_result)
        self.psi_result.sort()
        print("RSA-PSI Finished.")

    @timer
    def __get_vfl_server_psi_status(self, num_of_ids, cid, qid, psi_result_status):
        # print(f"[Rank {self.rank}, Round {self.vfl_client_status[2]}]: Request for server status.")
        vfl_client_psi_status_str = []
        carry_psi_final_result = psi_result_status[0]
        psi_final_result = psi_result_status[1].serialize() if carry_psi_final_result \
            else None

        for item in self.vfl_client_status:
            vfl_client_psi_status_str.append(str(item))

        vfl_server_stub = self._get_vfl_server_rpc_stub()

        request = vfl_server_service_pb2.client_psi_status_request(
            cid=cid,
            qid=qid,
            vfl_client_status=vfl_client_psi_status_str,
            data_length=num_of_ids,
            carry_psi_final_result=psi_result_status[0],
            psi_final_result=psi_final_result
        )
        print(f"[Rank {self.rank}, Round {self.vfl_client_status[2]}]: Request for server status.")
        # print("Request for VFLServer psi status...")

        response = vfl_server_stub.get_vfl_server_psi_status(request)
        vfl_server_status = response.vfl_server_status

        if response.carry_psi_final_result:
            return [vfl_server_status, True, response.psi_final_result]
        else:
            return [vfl_server_status, False, None]

    def __get_vfl_client_rpc_stub(self, ip_address):
        vfl_client_channel = grpc.insecure_channel(ip_address, options=self.options)
        vfl_client_stub = vfl_client_service_pb2_grpc.VFLClientServiceStub(vfl_client_channel)

        return vfl_client_stub

    def __get_vfl_label_owner_rpc_stub(self, ip_address):
        vfl_label_owner_channel = grpc.insecure_channel(ip_address, options=self.options)
        vfl_label_owner_stub = vfl_label_owner_service_pb2_grpc.VFLLabelOwnerServiceStub(vfl_label_owner_channel)

        return vfl_label_owner_stub

    def __send_rsa_pk(self, cid, qid, ip_address):
        self.rsa_pk, self.rsa_sk = generate_rsa_keys()
        vfl_client_stub = self.__get_vfl_client_rpc_stub(ip_address)
        request = vfl_client_service_pb2.rsa_public_key_request(
            cid=cid,
            qid=qid,
            pk_N=bytes(str(self.rsa_pk[0]).encode('utf-8')),
            pk_e=self.rsa_pk[1]
        )
        # print("Sending PSI key...")

        response = vfl_client_stub.send_rsa_public_key(request)
        if not response.recv_status:
            print("Failed.")
        else:
            self.rsa_pk_comm_status = True

    def __send_rsa_pk_label_owner(self, cid, qid, ip_address):
        self.rsa_pk, self.rsa_sk = generate_rsa_keys()
        vfl_label_owner_stub = self.__get_vfl_label_owner_rpc_stub(ip_address)
        request = vfl_label_owner_service_pb2.rsa_public_key_request_label_owner(
            cid=cid,
            qid=qid,
            pk_N=bytes(str(self.rsa_pk[0]).encode('utf-8')),
            pk_e=self.rsa_pk[1]
        )
        # print("Sending PSI key to label 0wner...")

        response = vfl_label_owner_stub.send_rsa_public_key_label_owner(request)
        if not response.recv_status:
            print("Failed.")
        else:
            self.rsa_pk_comm_status = True

    def __get_enc_ids_pk_str(self, local_id):
        self.client_enc_ids_pk, self.client_ra_list = encode_local_id_use_pk(local_id, self.rsa_pk)
        client_enc_ids_pk_str = []
        for enc_id in self.client_enc_ids_pk:
            client_enc_ids_pk_str.append(str(enc_id))

        return client_enc_ids_pk_str

    def __send_client_enc_ids_use_pk(self, local_id, cid, qid, ip_address):
        vfl_client_stub = self.__get_vfl_client_rpc_stub(ip_address)

        client_enc_ids_pk_str = self.__get_enc_ids_pk_str(local_id)
        request = vfl_client_service_pb2.send_client_enc_ids_request(
            cid=cid,
            qid=qid,
            client_enc_ids_pk_str=client_enc_ids_pk_str
        )
        # print("Sending encrypted client ids...")

        response = vfl_client_stub.send_client_enc_ids(request)
        if not response.recv_status:
            print("Failed.")
        else:
            self.client_enc_ids_comm_status = True

    def __send_client_enc_ids_use_pk_label_owner(self, local_id, cid, qid, ip_address):
        vfl_label_owner_stub = self.__get_vfl_label_owner_rpc_stub(ip_address)

        client_enc_ids_pk_str = self.__get_enc_ids_pk_str(local_id)
        request = vfl_label_owner_service_pb2.send_client_enc_ids_request_label_owner(
            cid=cid,
            qid=qid,
            client_enc_ids_pk_str=client_enc_ids_pk_str
        )
        # print("Sending encrypted client ids to Label Owner...")

        response = vfl_label_owner_stub.send_client_enc_ids_label_owner(request)
        if not response.recv_status:
            print("Failed.")
        else:
            self.client_enc_ids_comm_status = True

    @timer
    def __send_server_enc_id_use_sk_and_client_dec_id(self, local_ids, cid, qid, ip_address):
        vfl_client_stub = self.__get_vfl_client_rpc_stub(ip_address)

        client_dec_ids = decode_ids(self.client_enc_ids_pk, self.rsa_sk)
        server_hash_enc_ids = encode_and_hash_local_ids_use_sk(local_ids, self.rsa_sk)

        client_dec_ids_str = []
        for dec_id in client_dec_ids:
            client_dec_ids_str.append(str(dec_id))

        # server_hash_enc_ids_bytes = []
        # for hash_enc_id in database_server.server_hash_enc_ids:
        #     server_hash_enc_ids_bytes.append(bytes(str(hash_enc_id).encode('utf-8')))

        request = vfl_client_service_pb2.send_server_enc_ids_and_client_dec_ids_request(
            cid=cid,
            qid=qid,
            client_dec_ids=client_dec_ids_str,
            server_hash_enc_ids=server_hash_enc_ids
        )
        # print("Sending encrypted server ids and decrypted client ids...")

        response = vfl_client_stub.send_server_enc_ids_and_client_dec_ids(request)
        if not (response.client_dec_ids_recv_status and response.server_hash_enc_ids_recv_status):
            print("Failed.")
        else:
            self.client_dec_ids_comm_status = True
            self.server_hash_enc_ids_comm_status = True

    @timer
    def __send_server_enc_id_use_sk_and_client_dec_id_label_owner(self, local_ids, cid, qid, ip_address):
        vfl_label_owner_stub = self.__get_vfl_label_owner_rpc_stub(ip_address)

        client_dec_ids = decode_ids(self.client_enc_ids_pk, self.rsa_sk)
        server_hash_enc_ids = encode_and_hash_local_ids_use_sk(local_ids, self.rsa_sk)

        client_dec_ids_str = []
        for dec_id in client_dec_ids:
            client_dec_ids_str.append(str(dec_id))

        request = vfl_label_owner_service_pb2.send_server_enc_ids_and_client_dec_ids_request_label_owner(
            cid=cid,
            qid=qid,
            client_dec_ids=client_dec_ids_str,
            server_hash_enc_ids=server_hash_enc_ids
        )
        # print("Sending encrypted server ids and decrypted client ids...")

        response = vfl_label_owner_stub.send_server_enc_ids_and_client_dec_ids_label_owner(request)
        if not (response.client_dec_ids_recv_status and response.server_hash_enc_ids_recv_status):
            print("Failed.")
        else:
            self.client_dec_ids_comm_status = True
            self.server_hash_enc_ids_comm_status = True

    @timer
    def __rsa_double_psi(self, id_list, cid, qid, vfl_server_status):
        # psi_participator_num = int(vfl_server_status[0])
        total_rounds = int(vfl_server_status[1])
        current_round = int(vfl_server_status[2])
        # participator_index = int(vfl_server_status[3])
        # group_index = int(vfl_server_status[4])
        store_psi_result = True if strtobool(vfl_server_status[5]) else False
        ip_address = vfl_server_status[6]
        carry_final_psi_result = False
        psi_final_result = None
        connect_to_label_owner = False

        host = self.cfg.server_conf.vfl_label_owner.host
        port = self.cfg.server_conf.vfl_label_owner.port
        if ip_address == host + ":" + port:
            connect_to_label_owner = True

        if ip_address != '0':
            # Stage I
            if not store_psi_result:
                if connect_to_label_owner:
                    self.__send_rsa_pk_label_owner(cid, qid, ip_address)
                else:
                    self.__send_rsa_pk(cid, qid, ip_address)

            # Waiting for status...
            while not (self.rsa_pk_comm_status and self.rsa_pk):
                time.sleep(0.1)

            print(f"[Rank {self.rank}, Round {current_round}]: RSA public key exchange success.")

            # Stage II
            if store_psi_result:
                if connect_to_label_owner:
                    self.__send_client_enc_ids_use_pk_label_owner(id_list, cid, qid, ip_address)
                else:
                    self.__send_client_enc_ids_use_pk(id_list, cid, qid, ip_address)

            while not self.client_enc_ids_comm_status:
                time.sleep(0.1)

            print(f"[Rank {self.rank}, Round {current_round}]: Exchange encode client ids success.")

            # Stage III
            if not store_psi_result:
                if connect_to_label_owner:
                    self.__send_server_enc_id_use_sk_and_client_dec_id_label_owner(id_list, cid, qid, ip_address)
                else:
                    self.__send_server_enc_id_use_sk_and_client_dec_id(id_list, cid, qid, ip_address)

            while not (self.client_dec_ids_comm_status and self.server_hash_enc_ids_comm_status):
                time.sleep(0.1)

            print(
                f"[Rank {self.rank}, Round {current_round}]: Exchange encode server ids and decode client ids success.")

            # Stage IV
            if store_psi_result:
                self.psi_result = get_double_psi_result_genexps(id_list,
                                                                self.client_dec_ids,
                                                                self.client_ra_list,
                                                                self.rsa_pk,
                                                                self.server_hash_enc_ids)
                # print(database_server.psi_result)
                if (current_round == total_rounds) or \
                        ((len(self.psi_result) == 0) and (not carry_final_psi_result)):
                    psi_final_result = self.__generate_final_psi_result()
                    carry_final_psi_result = True

                # if (len(database_server.psi_result) == 0) and (carry_final_psi_result == False):
                #     generate_final_psi_result(database_server, he_context_path)
                #     carry_final_psi_result = True

            else:
                self.psi_result = None
        else:
            if store_psi_result:
                self.psi_result = id_list
                if current_round == total_rounds:
                    psi_final_result = self.__generate_final_psi_result()
                    carry_final_psi_result = True
            else:
                pass
        # print("Double PSI process done.")
        # print("Public key: ", database_server.rsa_pk)
        # print("================")
        # print("Private key: ",database_server.rsa_sk)
        # print("================")
        # print("Random number list: ", database_server.client_ra_list)
        # print("================")
        # print("Client_enc_ids_pk: ", database_server.client_enc_ids_pk)
        # print("================")
        # print("Client_dec_ids: ", database_server.client_dec_ids)
        # print("================")
        # print("Server_hash_enc_ids: ", database_server.server_hash_enc_ids)
        # print("================")
        # print("PSI_result: ", database_server.psi_result)

        # Update local status
        self.__update_vfl_client_status(store_psi_result, current_round)
        self.__reset_rsa_psi_status_per_round()
        return self.psi_result, [carry_final_psi_result, psi_final_result]

    def __invite_label_owner_psi(self):
        vfl_server_stub = self._get_vfl_server_rpc_stub()

        request = vfl_server_service_pb2.kmeans_finish_request(
            cid=self.rank,
            kmeans_finish=True
        )

        response = vfl_server_stub.invite_label_owner_psi_server(request)
        return response

    def __get_label_owner_psi_result(self):
        vfl_server_stub = self._get_vfl_server_rpc_stub()

        request = vfl_server_service_pb2.label_owner_psi_result_request(
            cid=self.rank,
            is_label_owner=True
        )

        response = vfl_server_stub.get_label_owner_psi_result(request)
        self.__decode_he_psi_result(response.psi_final_result)

    def __get_align_index(self, enc=False):
        # index, label, dist = self.__generate_cluster_info_bytes()
        label, dist, weight = self.__generate_cluster_info(enc=enc)
        vfl_server_stub = self._get_vfl_server_rpc_stub()

        request = vfl_server_service_pb2.client_cluster_info_request(
            cid=self.rank,
            # cluster_label=label.serialize(),
            # distance=dist.serialize(),
            # data_weight=weight.serialize()
            cluster_label=label,
            cluster_distance=dist,
            data_weight=weight
        )

        response = vfl_server_stub.get_client_align_index(request)
        # self.align_index = ts.ckks_vector_from(self.he_key, response.align_index).decrypt()
        self.align_index = response.align_index

        # print(len(self.align_index))

    def __generate_cluster_info(self, enc=False):
        # index_bytes = self.__generate_index_bytes()
        label, dist, weight = self.__generate_label_dist_weight(enc)

        return label, dist, weight

    def __generate_index(self, enc=False):
        if enc:
            enc_index = ts.ckks_vector(self.he_key, self.psi_result)
            return enc_index
        else:
            return self.psi_result

    def __generate_label_dist_weight(self, enc=False):
        cluster_labels = []
        euc_dist = []
        weight = []

        for index in self.psi_result:
            label = self.cluster_labels[index]
            cluster_labels.append(label)
            euc_dist.append(euclidean_dist(self.cluster_data[index], self.cluster_centers[label]))
            weight.append(self.weight_list[index])

        if enc:
            cluster_labels = ts.ckks_vector(self.he_key, cluster_labels)
            euc_dist = ts.ckks_vector(self.he_key, euc_dist)
            weight = ts.ckks_vector(self.he_key, weight)

        return cluster_labels, euc_dist, weight

