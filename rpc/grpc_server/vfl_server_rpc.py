# -*- coding: utf-8 -*-
"""
@Auth ： ZQB
@Time ： 2023/5/15 16:25
@File ：lr_server_rpc.py
"""
import numpy as np

import rpc.grpc_file.vfl_server_service_pb2_grpc as vfl_server_service_pb2_grpc
import rpc.grpc_file.vfl_server_service_pb2 as vfl_server_service_pb2
import rpc.grpc_file.vfl_label_owner_service_pb2_grpc as vfl_label_owner_service_pb2_grpc
import rpc.grpc_file.vfl_label_owner_service_pb2 as vfl_label_owner_service_pb2
import rpc.grpc_file.vfl_client_service_pb2_grpc as vfl_client_service_pb2_grpc
import rpc.grpc_file.vfl_client_service_pb2 as vfl_client_service_pb2
import tenseal as ts
import math
import time
import grpc
import torch
from array import array
from collections import deque
from model import MLPTopModel
import torch.optim as optim


class VFLServer(vfl_server_service_pb2_grpc.VFLServerServiceServicer):
    def __init__(self, num_clients, pk_file, options, device, cfg, trainer='lr'):
        self.cfg = cfg
        self.num_clients = num_clients
        pk_ctx_bytes = open(pk_file, "rb").read()
        self.he_pk = ts.context_from(pk_ctx_bytes)
        self.sleep = 0.001
        self.options = options
        self.device = device
        self.client_id = []
        torch.manual_seed(self.cfg.defs.seed)

        # knn
        if trainer == 'knn':
            self.knn_client_id = []
            self.num_knn_request = 0
            self.num_knn_response = 0
            self.current_data_index = []
            self.check_data_index = False
            self.knn_dist = []

        # mlp
        if trainer == 'mlp':
            self.mlp_client_id = []
            self.num_mlp_request = 0
            self.num_mlp_response = 0
            self.epochs = self.cfg.mlp_conf.epochs
            self.current_epoch = 0
            self.lr = self.cfg.mlp_conf.lr
            self.lr_gamma = self.cfg.mlp_conf.lr_gamma
            self.lr_step = self.cfg.mlp_conf.lr_step
            self.n_bottom_out = self.cfg.mlp_conf.n_bottom_out
            self.n_top_out = self.cfg.mlp_conf.n_top_out
            self.top_model = MLPTopModel(self.n_bottom_out * self.num_clients,
                                         self.n_top_out)
            self.top_model.to(self.device)
            self.top_optimizer = optim.Adam(self.top_model.parameters(), lr=self.lr)
            # self.mlp_add_info_queue = deque()
            self.concat_list = [0 for _ in range(self.num_clients)]
            self.bottom_grad_list = None

        # train lr
        self.lr_client_id = []
        self.lr_forward_res = []
        self.receive_flag = False
        self.num_lr_request = 0
        self.num_lr_response = 0
        self.current_batch_index = None
        self.sum_lr_forward = None
        self.is_summed = False
        self.batch_grad = None
        self.early_stop = False

        # test lr
        self.is_calculate_accuracy = False
        self.continue_test_iter_flag = False

        # for rsa psi
        # self.regular_mode = True
        self.optim_mode = False
        self.num_psi_request = 0
        self.num_psi_response = 0
        self.psi_add_info_queue = deque()
        self.psi_cid_list = array('I')
        self.psi_IP_list = []
        self.psi_store_psi_result_list = []
        self.num_psi_participators = 0
        self.total_psi_rounds = 1000000
        self.current_psi_round = 0
        self.initial_participators = []
        self.waiting_for_initial_participators_status = False
        self.waiting_for_initialize = False
        self.group_index_list = array('i')
        self.psi_cid_length_dict = {}
        self.psi_comm_IP_index = array('i')
        self.psi_update_status = False
        self.psi_check_result_count = 0
        self.psi_final_result_status = False
        self.psi_final_result = None
        self.reset_psi_status_per_round_finished = False

        # For invite label owner to join psi
        self.num_invite_request = 0
        self.num_invite_response = 0
        self.kmeans_finish_status_list = []
        self.psi_permission = False

        # For align index
        self.num_align_request = 0
        self.num_align_response = 0
        self.align_add_info_queue = deque()
        self.align_cluster_label = []
        self.align_distance = []
        self.align_info = []
        self.align_index = None

    def gather_lr_train_forward(self, request, context):
        """

        :param request: grpc request
        :param context: grpc response
        :return: grpc message
        """
        cid = request.cid
        first_request_cid = None
        if not len(self.lr_client_id):
            first_request_cid = cid
            self.current_batch_index = request.batch_index
        while request.batch_index != self.current_batch_index:
            time.sleep(self.sleep)

        # enc_forward_bytes = request.forward_result
        forward_result = request.forward_result
        # enc_forward = ts.ckks_vector_from(self.he_pk, enc_forward_bytes)

        self.lr_client_id.append(cid)
        # self.lr_forward_res.append(enc_forward)
        self.lr_forward_res.append(torch.tensor(forward_result))
        self.num_lr_request += 1

        while self.num_lr_request % self.num_clients != 0:
            time.sleep(self.sleep)
        if cid == first_request_cid:
            self.__sum_lr_forward()
            self.__get_lr_train_batch_gradient_from_label_owner()

        while not self.batch_grad:
            time.sleep(self.sleep)

        response = vfl_server_service_pb2.lr_train_forward_response(
            cid=cid,
            batch_gradient=self.batch_grad,
            early_stop=self.early_stop
        )
        self.num_lr_response += 1

        while self.num_lr_response % self.num_clients != 0:
            time.sleep(self.sleep)

        if cid == first_request_cid:
            self.__reset_lr_train_status()
        time.sleep(self.sleep)

        return response

    def gather_lr_test_forward(self, request, context):
        """

        :param request: grpc request
        :param context: grpc response
        :return: grpc message
        """
        # print(self.is_train)
        cid = request.cid
        first_request_cid = None
        if not len(self.lr_client_id):
            first_request_cid = cid
            self.current_batch_index = request.batch_index
        while request.batch_index != self.current_batch_index:
            time.sleep(self.sleep)

        # enc_forward_bytes = request.test_forward
        # enc_forward = ts.ckks_vector_from(self.he_pk, enc_forward_bytes)
        test_forward = request.test_forward

        self.lr_client_id.append(cid)
        self.lr_forward_res.append(torch.tensor(test_forward))
        self.num_lr_request += 1

        while self.num_lr_request % self.num_clients != 0:
            time.sleep(self.sleep)

        if cid == first_request_cid:
            self.__sum_lr_forward()
            response = self.__send_lr_test_forward_result_to_label_owner()
            self.continue_test_iter_flag = response.receive_flag
            self.is_calculate_accuracy = True

        while not self.is_calculate_accuracy:
            time.sleep(self.sleep)

        response = vfl_server_service_pb2.lr_test_forward_response(
            cid=cid,
            continue_iter=self.continue_test_iter_flag
        )
        self.num_lr_response += 1

        while self.num_lr_response % self.num_clients != 0:
            time.sleep(self.sleep)
        if cid == first_request_cid:
            self.__reset_lr_test_status()
        time.sleep(self.sleep)

        return response

    def gather_mlp_train_bottom_forward(self, request, context):
        """

        :param request:
        :param context:
        :return:
        """
        cid = request.cid
        batch_index = request.batch_index
        epoch = request.epoch
        # bottom_forward = request.bottom_forward
        bottom_forward = self.__get_mlp_bottom_forward_rpc_msg(request)

        first_request_cid = None
        if not len(self.mlp_client_id):
            first_request_cid = cid

        self.mlp_client_id.append(cid)
        self.concat_list[cid] = torch.tensor(bottom_forward)
        self.num_mlp_request += 1

        while self.num_mlp_request % self.num_clients != 0:
            time.sleep(self.sleep)

        if cid == first_request_cid:
            concat_bottom_forward = torch.cat(self.concat_list, dim=1)
            concat_bottom_forward.requires_grad = True
            concat_bottom_forward.retain_grad()
            top_forward, grad = self.__get_mlp_train_batch_gradient_from_label_owner(concat_bottom_forward,
                                                                                     batch_index)
            # grad = []
            # for item in batch_grad_msg:
            #     grad.append(item.grad)

            # self.__adjust_mlp_learning_rate(epoch)
            self.top_optimizer.zero_grad()
            # grad = torch.tensor(grad)
            # grad = grad.to(self.device)
            top_forward.backward(grad)
            self.top_optimizer.step()
            # print(concat_bottom_forward.grad)
            self.__split_bottom_grad(concat_bottom_forward.grad)

        while not self.bottom_grad_list:
            time.sleep(self.sleep)

        # print(f"{cid}, {self.bottom_grad_list[cid]}")
        batch_gradient = []
        for item in self.bottom_grad_list[cid]:
            single_grad = vfl_server_service_pb2.internal_split_grad(
                grad=item
            )
            batch_gradient.append(single_grad)

        response = vfl_server_service_pb2.mlp_train_batch_gradient_response(
            cid=cid,
            batch_gradient=batch_gradient,
            early_stop=self.early_stop
        )
        self.num_mlp_response += 1
        while self.num_mlp_response % self.num_clients != 0:
            time.sleep(self.sleep)

        if cid == first_request_cid:
            self.__reset_mlp_train_status()
            # if (self.current_epoch + 1) == self.epochs:
            #     print(">>>Train finish.")
        time.sleep(self.sleep)

        return response

    def gather_mlp_test_bottom_forward(self, request, context):
        cid = request.cid
        batch_index = request.batch_index
        bottom_forward = self.__get_mlp_bottom_forward_rpc_msg(request)

        first_request_cid = None
        if not len(self.mlp_client_id):
            first_request_cid = cid

        self.mlp_client_id.append(cid)
        self.concat_list[cid] = torch.tensor(bottom_forward)
        self.num_mlp_request += 1

        while self.num_mlp_request % self.num_clients != 0:
            time.sleep(self.sleep)

        if cid == first_request_cid:
            concat_bottom_forward = torch.cat(self.concat_list, dim=1)
            response = self.__send_mlp_test_forward_result_to_label_owner(concat_bottom_forward,
                                                                          batch_index)
            self.continue_test_iter_flag = response.receive_flag
            self.is_calculate_accuracy = True

        while not self.is_calculate_accuracy:
            time.sleep(self.sleep)

        response = vfl_server_service_pb2.mlp_test_bottom_forward_response(
            cid=cid,
            continue_iter=self.continue_test_iter_flag
        )
        self.num_mlp_response += 1

        while self.num_mlp_response % self.num_clients != 0:
            time.sleep(self.sleep)

        if cid == first_request_cid:
            self.__reset_mlp_test_status()
        time.sleep(self.sleep)

        return response

    def gather_knn_distance(self, request, context):
        cid = request.cid
        first_request_cid = None
        if not len(self.knn_client_id):
            first_request_cid = cid

        test_data_index = request.index
        self.knn_client_id.append(cid)
        self.current_data_index.append(test_data_index)
        self.knn_dist.append(request.dist)
        self.num_knn_request += 1

        while self.num_knn_request % self.num_clients != 0:
            time.sleep(self.sleep)

        if cid == first_request_cid:
            check_arr = np.array(self.current_data_index)
            if not np.std(check_arr):
                self.check_data_index = True
            else:
                raise ValueError

        while not self.check_data_index:
            time.sleep(self.sleep)

        if cid == first_request_cid:
            sum_knn_dist = self.__sum_knn_distance()
            response = self.__send_knn_distance_to_label_owner(test_data_index, sum_knn_dist)
            self.continue_test_iter_flag = response.receive_flag
            self.is_calculate_accuracy = True

        while not self.is_calculate_accuracy:
            time.sleep(self.sleep)

        response = vfl_server_service_pb2.knn_distance_response(
            cid=cid,
            continue_iter=self.continue_test_iter_flag
        )
        self.num_knn_response += 1

        while self.num_knn_response % self.num_clients != 0:
            time.sleep(self.sleep)

        if cid == first_request_cid:
            self.__reset_knn_status()
        time.sleep(self.sleep)

        return response

    def get_vfl_server_psi_status(self, request, context):
        cid = request.cid
        qid = request.qid
        data_server_status = request.vfl_client_status
        data_server_psi_round = int(data_server_status[2])
        data_length = request.data_length
        # print(cid, data_length)
        carry_psi_final_result = request.carry_psi_final_result
        psi_final_result = request.psi_final_result

        assert self.current_psi_round == data_server_psi_round, \
            "PSI current round don't match."
        if (data_server_psi_round == 0) and self.waiting_for_initial_participators_status:
            raise RuntimeError(f"PSI service already in use. (Client-ID {cid}).")

        if self.current_psi_round == 0:
            first_request_cid = None
            if (data_server_psi_round == 0) and (cid not in self.psi_cid_list):
                if len(self.psi_cid_list) == 0:
                    first_request_cid = cid

                self.psi_add_info_queue.append(cid)
                while self.psi_add_info_queue[0] != cid:
                    time.sleep(self.sleep)
                self.psi_cid_list.append(cid)
                self.psi_IP_list.append(data_server_status[0])
                self.psi_store_psi_result_list.append(bool(data_server_status[1]))
                self.psi_add_info_queue.popleft()
            else:
                raise RuntimeError(f"DataServer {cid} already requested.")

            # Waiting 8s for other participators to join
            if cid == first_request_cid:
                time.sleep(6)
                self.waiting_for_initial_participators_status = True
            while not self.waiting_for_initial_participators_status:
                time.sleep(self.sleep)
            while len(self.psi_add_info_queue) != 0 and self.waiting_for_initial_participators_status:
                time.sleep(self.sleep)

            assert len(self.psi_cid_list) == len(self.psi_IP_list) == len(self.psi_store_psi_result_list), \
                "PSI initialization failed."

            if cid == first_request_cid:
                self.num_psi_participators = len(self.psi_cid_list)
                self.__total_psi_rounds()
                self.reset_psi_status_per_round_finished = True
                print("All participator collected.")
                # print(self.psi_cid_list)
                # print(self.psi_IP_list)
                print(f"Psi participators : {self.num_psi_participators}")
                print(f"Total psi rounds: {self.total_psi_rounds}")
                self.waiting_for_initialize = True
            while not self.waiting_for_initialize:
                time.sleep(self.sleep)

        if self.psi_store_psi_result_list[self.psi_cid_list.index(cid)]:
            self.psi_cid_length_dict[cid] = data_length

        while not self.reset_psi_status_per_round_finished:
            time.sleep(self.sleep)
        self.num_psi_request += 1
        # waiting_time_count = 0
        while self.num_psi_request % self.num_psi_participators != 0:
            time.sleep(self.sleep)

        self.reset_psi_status_per_round_finished = False
        if carry_psi_final_result:
            self.psi_final_result = psi_final_result
            self.psi_final_result_status = True
        self.psi_check_result_count += 1

        # print(carry_psi_final_result)
        while self.psi_check_result_count % self.num_psi_participators != 0:
            time.sleep(self.sleep)
        # print(self.psi_final_result_status)

        # if (self.current_psi_round == self.total_psi_rounds == int(data_server_status[2])) and \
        #         (self.psi_final_result_status == True):
        if self.psi_final_result_status:
            response = vfl_server_service_pb2.vfl_server_psi_status_response(
                cid=cid,
                qid=qid,
                vfl_server_status=[],
                carry_psi_final_result=self.psi_final_result_status,
                psi_final_result=self.psi_final_result
            )
            self.num_psi_response += 1
            while self.num_psi_response % self.num_psi_participators != 0:
                time.sleep(self.sleep)
            time.sleep(self.sleep)
            self.__reset_all_psi_status()
            if cid == 0:
                print(">>>PSI Finish.")

            return response

        if cid == self.psi_cid_list[0]:
            self.current_psi_round += 1
            if not self.optim_mode:
                # self.psi_cid_list = sorted(self.psi_cid_list)
                self.__update_group_index_list_regular()
                # print(self.group_index_list)
                self.__get_psi_comm_IP()
                # print(self.psi_store_psi_result_list)
                # print(self.psi_comm_IP_index)
                # print(f"group_index: {self.group_index_list}")
                self.__update_store_psi_result_status_regular()
                # print(self.psi_store_psi_result_list)
                self.psi_update_status = True
            else:
                self.__update_group_index_list_and_store_psi_result_status_and_psi_comm_IP_optim()
                self.psi_update_status = True

        while not self.psi_update_status:
            time.sleep(self.sleep)

        vfl_server_status = self.__generate_vfl_server_status(cid)
        vfl_server_status_str = []
        for item in vfl_server_status:
            vfl_server_status_str.append(str(item))

        response = vfl_server_service_pb2.vfl_server_psi_status_response(
            cid=cid,
            qid=qid,
            vfl_server_status=vfl_server_status_str,
            carry_psi_final_result=carry_psi_final_result,
            psi_final_result=psi_final_result
        )

        self.num_psi_response += 1
        while self.num_psi_response % self.num_psi_participators != 0:
            time.sleep(self.sleep)
        if cid == self.psi_cid_list[0]:
            self.__reset_psi_status_per_round()
            self.reset_psi_status_per_round_finished = True

        return response

    def invite_label_owner_psi_server(self, request, context):
        """

        :param request: grpc request
        :param context: grpc response
        :return: grpc message
        """
        cid = request.cid

        first_request_cid = None
        if not len(self.client_id):
            first_request_cid = cid

        self.client_id.append(cid)
        self.kmeans_finish_status_list.append(request.kmeans_finish)
        self.num_invite_request += 1

        while self.num_invite_request % self.num_clients != 0:
            time.sleep(self.sleep)

        if cid == first_request_cid:
            if sum(self.kmeans_finish_status_list) != self.num_clients:
                raise ValueError("Not every client finish kmeans.")
            response = self.__invite_label_owner_join_psi()
            if response.recv_status:
                self.psi_permission = True
            else:
                raise ValueError("Label Owner error.")
        while not self.psi_permission:
            time.sleep(self.sleep)

        response = vfl_server_service_pb2.kmeans_finish_response(
            cid=cid,
            psi_permission=self.psi_permission
        )
        self.num_invite_response += 1

        while self.num_invite_response % self.num_clients != 0:
            time.sleep(self.sleep)

        self.__reset_invite_psi_status()
        time.sleep(self.sleep)

        return response

    def get_label_owner_psi_result(self, request, context):
        """

        :param request: grpc request
        :param context: grpc response
        :return: grpc message
        """
        label_owner_cid = request.cid
        assert request.is_label_owner is True

        while not self.psi_final_result_status:
            time.sleep(self.sleep)

        response = vfl_server_service_pb2.vfl_server_psi_result_response(
            cid=label_owner_cid,
            psi_final_result=self.psi_final_result
        )

        return response

    def get_client_align_index(self, request, context):
        """

        :param request: grpc request
        :param context: grpc response
        :return: grpc message
        """
        cid = request.cid
        label = request.cluster_label
        distance = request.cluster_distance
        weight = request.data_weight

        # label = ts.ckks_vector_from(self.he_pk, cluster_label)
        # dist = ts.ckks_vector_from(self.he_pk, distance)
        cluster_info = vfl_label_owner_service_pb2.cluster_info(
            cluster_label=label,
            cluster_distance=distance,
            data_weight=weight
        )

        first_request_cid = None
        if not len(self.client_id):
            first_request_cid = cid

        self.align_add_info_queue.append(cid)
        self.client_id.append(cid)
        while self.align_add_info_queue[0] != cid:
            time.sleep(self.sleep)
        # self.align_cluster_label.append(label)
        # self.align_distance.append(distance)
        self.align_info.append(cluster_info)
        self.align_add_info_queue.popleft()
        self.num_align_request += 1

        while self.num_align_request % self.num_clients != 0:
            time.sleep(self.sleep)

        if cid == first_request_cid:
            response = self.__send_cluster_info_to_label_owner()
            self.align_index = response.align_index

        while not self.align_index:
            time.sleep(self.sleep)

        response = vfl_server_service_pb2.vfl_server_align_index_response(
            cid=cid,
            align_index=self.align_index
        )
        self.num_align_response += 1
        while self.num_align_response % self.num_clients != 0:
            time.sleep(self.sleep)

        self.__reset_align_index_status()
        time.sleep(self.sleep)

        return response

    def __reset_lr_train_status(self):
        self.lr_client_id = []
        self.lr_forward_res = []
        self.receive_flag = False
        self.num_lr_request = 0
        self.num_lr_response = 0
        self.current_batch_index = None
        self.sum_lr_forward = None
        self.is_summed = False
        self.batch_grad = None
        # print(">>>Reset Status.")

    def __reset_lr_test_status(self):
        self.lr_client_id = []
        self.lr_forward_res = []
        self.num_lr_request = 0
        self.num_lr_response = 0
        self.current_batch_index = None
        self.sum_lr_forward = None
        self.is_summed = False
        self.is_calculate_accuracy = False
        self.continue_test_iter_flag = False

    def __reset_mlp_train_status(self):
        self.mlp_client_id = []
        self.num_mlp_request = 0
        self.num_mlp_response = 0
        # self.current_epoch = 0
        self.concat_list = [0 for _ in range(self.num_clients)]
        self.bottom_grad_list = None

    def __reset_mlp_test_status(self):
        self.mlp_client_id = []
        self.num_mlp_request = 0
        self.num_mlp_response = 0
        self.concat_list = [0 for _ in range(self.num_clients)]
        self.is_calculate_accuracy = False
        self.continue_test_iter_flag = False

    def __reset_knn_status(self):
        self.knn_client_id = []
        self.num_knn_request = 0
        self.num_knn_response = 0
        self.current_data_index = []
        self.check_data_index = False
        self.knn_dist = []
        self.is_calculate_accuracy = False
        self.continue_test_iter_flag = False

    def __reset_psi_status_per_round(self):
        self.num_psi_request = 0
        self.num_psi_response = 0
        self.psi_check_result_count = 0
        self.psi_update_status = False
        self.psi_cid_length_dict = {}

    def __reset_all_psi_status(self):
        # self.regular_mode = True
        # self.optim_mode = False
        self.num_psi_request = 0
        self.num_psi_response = 0
        self.psi_add_info_queue = deque()
        self.psi_cid_list = array('I')
        self.psi_IP_list = []
        self.psi_store_psi_result_list = []
        self.num_psi_participators = 0
        self.total_psi_rounds = 1000000
        self.current_psi_round = 0
        self.initial_participators = []
        self.waiting_for_initial_participators_status = False
        self.waiting_for_initialize = False
        self.group_index_list = array('i')
        self.psi_cid_length_dict = {}
        self.psi_comm_IP_index = array('i')
        self.psi_update_status = False
        self.psi_check_result_count = 0
        self.psi_final_result_status = False
        self.psi_final_result = None
        self.reset_psi_status_per_round_finished = False

    def __reset_align_index_status(self):
        self.client_id = []
        self.num_align_request = 0
        self.num_align_response = 0
        self.align_add_info_queue = deque()
        self.align_cluster_label = []
        self.align_distance = []
        self.align_info = []
        self.align_index = None

    def __reset_invite_psi_status(self):
        self.client_id = []
        self.num_invite_request = 0
        self.num_invite_response = 0
        self.kmeans_finish_status_list = []
        self.psi_permission = False

    def __total_psi_rounds(self):
        if self.num_psi_participators <= 1:
            self.total_psi_rounds = 1
        else:
            self.total_psi_rounds = max(int(math.log2(self.num_psi_participators - 1)), 0) + 1

    def __update_group_index_list_regular(self):
        self.group_index_list = array('i')
        for i in range(len(self.psi_cid_list)):
            self.group_index_list.append(int((i // math.pow(2, self.current_psi_round))))

    def __update_group_index_list_and_store_psi_result_status_and_psi_comm_IP_optim(self):
        # self.group_index_list.clear()
        self.group_index_list = array('i', (-1 for _ in range(self.num_psi_participators)))
        self.psi_store_psi_result_list = [False for _ in range(self.num_psi_participators)]
        comm_IP_index_list = array('i', (-1 for _ in range(self.num_psi_participators)))

        sorted_list_base_on_length = sorted(self.psi_cid_length_dict.items(),
                                            key=lambda x: x[1], reverse=True)
        sorted_dict = dict(sorted_list_base_on_length)
        sorted_cid_list = list(sorted_dict.keys())
        # print(sorted_dict)
        # print(sorted_cid_list)
        size_cid_list = len(sorted_cid_list)
        bound = size_cid_list // 2
        for i in range(bound):
            high_length_cid = sorted_cid_list[i]
            if size_cid_list % 2 != 0:
                low_length_cid = sorted_cid_list[i + bound + 1]
            else:
                low_length_cid = sorted_cid_list[i + bound]
            high_length_cid_index = self.psi_cid_list.index(high_length_cid)
            low_length_cid_index = self.psi_cid_list.index(low_length_cid)

            self.group_index_list[high_length_cid_index] = i
            self.group_index_list[low_length_cid_index] = i
            self.psi_store_psi_result_list[low_length_cid_index] = True

            comm_IP_index_list[high_length_cid_index] = self.psi_cid_list.index(low_length_cid)
            comm_IP_index_list[low_length_cid_index] = self.psi_cid_list.index(high_length_cid)

        if size_cid_list % 2 != 0:
            self.group_index_list[self.psi_cid_list.index(sorted_cid_list[bound])] = bound
            self.psi_store_psi_result_list[self.psi_cid_list.index(sorted_cid_list[bound])] = True

        # print(comm_IP_index_list)
        # print(self.psi_cid_list)
        # print(self.psi_store_psi_result_list)
        # print(self.group_index_list)
        # print(self.psi_IP_list)
        # time.sleep(10000)
        self.psi_comm_IP_index = comm_IP_index_list

    def __get_psi_comm_IP(self):
        comm_IP_index_list = array('i', (-1 for _ in range(self.num_psi_participators)))
        i = 0
        while i < self.num_psi_participators:
            j = i + 1
            if self.psi_store_psi_result_list[i]:
                if i in comm_IP_index_list:
                    comm_IP_index_list[i] = comm_IP_index_list.index(i)
                    i = j
                    continue
                while j < self.num_psi_participators:
                    if self.psi_store_psi_result_list[j]:
                        comm_IP_index_list[i] = j
                        break
                    j += 1
            i = j

        self.psi_comm_IP_index = comm_IP_index_list

    def __update_store_psi_result_status_regular(self):
        self.psi_store_psi_result_list.clear()
        id_set = set()
        for group_index in self.group_index_list:
            if group_index not in id_set:
                self.psi_store_psi_result_list.append(True)
                id_set.add(group_index)
            else:
                self.psi_store_psi_result_list.append(False)

    def __generate_vfl_server_status(self, cid):
        data_server_index = self.psi_cid_list.index(cid)
        if self.psi_comm_IP_index[data_server_index] == -1:
            ip_address = 0
        else:
            ip_address = self.psi_IP_list[self.psi_comm_IP_index[data_server_index]]
        vfl_server_status = [self.num_psi_participators, self.total_psi_rounds, self.current_psi_round,
                             data_server_index, self.group_index_list[data_server_index],
                             self.psi_store_psi_result_list[data_server_index], ip_address]

        return vfl_server_status

    def __get_vfl_label_owner_rpc_stub(self):
        vfl_label_owner_address = self.cfg.server_conf.vfl_label_owner.host + ":" + \
                                  self.cfg.server_conf.vfl_label_owner.port
        vfl_label_owner_channel = grpc.insecure_channel(vfl_label_owner_address,
                                                        options=self.options)
        vfl_label_owner_stub = vfl_label_owner_service_pb2_grpc.VFLLabelOwnerServiceStub(vfl_label_owner_channel)

        return vfl_label_owner_stub

    def __invite_label_owner_join_psi(self):
        vfl_label_owner_stub = self.__get_vfl_label_owner_rpc_stub()

        request = vfl_label_owner_service_pb2.invite_psi_request(
            server_id=1,
            invite_psi=True
        )

        response = vfl_label_owner_stub.invite_label_owner_psi(request)
        return response

    def __sum_lr_forward(self):
        """
        add the forward results from clients together
        :return:
        """
        self.sum_lr_forward = sum(self.lr_forward_res).numpy()
        self.is_summed = True
        # print(self.sum_lr_forward)
        # print(self.sum_lr_forward.decrypt())

    def __send_lr_train_forward_result_to_label_owner(self):
        vfl_label_owner_stub = self.__get_vfl_label_owner_rpc_stub()

        request = vfl_label_owner_service_pb2.lr_train_forward_request_label_owner(
            server_id=1,
            batch_index=self.current_batch_index,
            # summed_forward_result=self.sum_lr_forward.serialize()
            summed_forward_result=self.sum_lr_forward
        )

        response = vfl_label_owner_stub.get_lr_train_batch_gradient(request)

        return response

    def __send_lr_train_batch_gradient_to_client(self, key, batch_grad):
        lr_client_address = self.cfg.server_conf[key].host + ":" + self.cfg.server_conf[key].port

        lr_client_channel = grpc.insecure_channel(lr_client_address,
                                                  options=self.options)
        lr_client_stub = vfl_client_service_pb2_grpc.VFLClientServiceStub(lr_client_channel)

        request = vfl_client_service_pb2.lr_train_batch_gradient_request(
            server_id=1,
            batch_index=self.current_batch_index,
            # batch_gradient=batch_grad.serialize()
            batch_gradient=batch_grad
        )

        response = lr_client_stub.send_lr_train_batch_gradient(request)

        return response

    def __get_lr_train_batch_gradient_from_label_owner(self):
        assert self.is_summed is True

        response = self.__send_lr_train_forward_result_to_label_owner()
        self.batch_grad = response.batch_gradient
        self.early_stop = response.early_stop
        # enc_batch_grad = ts.ckks_vector_from(self.he_pk, batch_grad)

        # print(enc_batch_grad)
        # print(enc_batch_grad.decrypt())

        # return batch_grad

    def __send_lr_test_forward_result_to_label_owner(self):
        vfl_label_owner_stub = self.__get_vfl_label_owner_rpc_stub()
        request = vfl_label_owner_service_pb2.lr_test_forward_request_label_owner(
            server_id=1,
            batch_index=self.current_batch_index,
            # summed_test_result=self.sum_lr_forward.serialize()
            summed_test_result=self.sum_lr_forward
        )
        response = vfl_label_owner_stub.calculate_lr_test_accuracy(request)

        return response

    def __get_mlp_train_batch_gradient_from_label_owner(self, concat_bottom_f, batch_index):
        vfl_label_owner_stub = self.__get_vfl_label_owner_rpc_stub()
        top_forward = self.__mlp_top_forward_iteration(concat_bottom_f)
        top_f_numpy = top_forward.detach().cpu().numpy()
        msg = self.__generate_mlp_top_forward_rpc_msg(top_f_numpy)

        request = vfl_label_owner_service_pb2.mlp_train_top_forward_request(
            server_id=1,
            batch_index=batch_index,
            top_forward=msg
        )

        response = vfl_label_owner_stub.get_mlp_train_batch_gradient(request)
        batch_grad = response.batch_gradient
        self.early_stop = response.early_stop

        grad = []
        for item in batch_grad:
            grad.append(item.grad)
        grad = torch.tensor(grad)
        grad = grad.to(self.device)

        return top_forward, grad

    def __send_mlp_test_forward_result_to_label_owner(self, concat_bottom_f, batch_index):
        vfl_label_owner_stub = self.__get_vfl_label_owner_rpc_stub()
        top_forward = self.__mlp_top_forward_iteration(concat_bottom_f)
        top_f_numpy = top_forward.detach().cpu().numpy()
        msg = self.__generate_mlp_top_forward_rpc_msg(top_f_numpy)

        request = vfl_label_owner_service_pb2.mlp_test_top_forward_request(
            server_id=1,
            batch_index=batch_index,
            top_forward=msg
        )

        response = vfl_label_owner_stub.calculate_mlp_test_accuracy(request)

        return response

    def __send_knn_distance_to_label_owner(self, test_data_index, dist):
        vfl_label_owner_stub = self.__get_vfl_label_owner_rpc_stub()

        request = vfl_label_owner_service_pb2.knn_distance_request_label_owner(
            server_id=1,
            index=test_data_index,
            dist=dist
        )

        response = vfl_label_owner_stub.calculate_knn_accuracy(request)

        return response

    def __send_cluster_info_to_label_owner(self):
        vfl_label_owner_stub = self.__get_vfl_label_owner_rpc_stub()
        # print(len(self.align_cluster_label))
        # print(len(self.align_cluster_label[0]))
        request = vfl_label_owner_service_pb2.server_cluster_info_request(
            server_id=1,
            info=self.align_info
        )
        response = vfl_label_owner_stub.calculate_client_align_index(request)

        return response

    def __mlp_top_forward_iteration(self, concat_bottom_f):
        concat_bottom_f = concat_bottom_f.to(self.device)
        top_forward = self.top_model(concat_bottom_f)
        # top_f_numpy = top_forward.detach().cpu().numpy()
        # print(top_f_numpy)

        return top_forward

    def __adjust_mlp_learning_rate(self, epoch):
        if self.current_epoch != epoch:
            self.current_epoch = epoch
            if self.current_epoch in self.lr_step:
                self.lr *= self.lr_gamma
                for param_group in self.top_optimizer.param_groups:
                    param_group['lr'] = self.lr

                print(f">>>Epoch{epoch}: Learning rate decay.")
                print(">>>Learning rate: ", self.lr)
        else:
            return

    def __split_bottom_grad(self, bottom_grad):
        shape_list = [self.n_bottom_out] * self.num_clients
        split_list = list(torch.split(bottom_grad, shape_list, dim=-1))
        bottom_grad_list = []
        for item in split_list:
            bottom_grad_list.append(item.numpy())

        self.bottom_grad_list = bottom_grad_list

    def __get_mlp_bottom_forward_rpc_msg(self, request):
        bottom_forward = []
        for item in request.bottom_forward:
            bottom_forward.append(item.forward)

        return bottom_forward

    def __generate_mlp_top_forward_rpc_msg(self, top_f_numpy):
        msg = []
        for item in top_f_numpy:
            single_msg = vfl_label_owner_service_pb2.internal_top_forward(
                forward=item
            )
            msg.append(single_msg)

        return msg

    def __sum_knn_distance(self):
        sum_knn_dist = np.sum(self.knn_dist, axis=0)

        return sum_knn_dist
