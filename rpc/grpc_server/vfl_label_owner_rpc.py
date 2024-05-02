# -*- coding: utf-8 -*-
"""
@Auth ： ZQB
@Time ： 2023/5/16 12:17
@File ：lr_label_owner_rpc.py
"""
import time

import rpc.grpc_file.vfl_label_owner_service_pb2_grpc as vfl_label_owner_service_pb2_grpc
import rpc.grpc_file.vfl_label_owner_service_pb2 as vfl_label_owner_service_pb2
from trainer import LRTrainer, MLPTrainer, KNNTrainer
from utils import prepare_logger
import tenseal as ts
import torch
import numpy as np
import copy


class VFLLabelOwner(vfl_label_owner_service_pb2_grpc.VFLLabelOwnerServiceServicer):
    def __init__(self, rank, log_path, cfg, he_file, device, ip_address,
                 is_label_owner=True, is_regression=False, trainer='lr'):
        logger = prepare_logger(rank, log_path, cfg.defs.mode)
        if trainer == 'lr':
            self.trainer = LRTrainer(rank, logger, cfg, he_file, device, ip_address,
                                     trainer, is_label_owner, is_regression)

            self.criterion = torch.nn.BCELoss()
        elif trainer == 'mlp':
            self.trainer = MLPTrainer(rank, logger, cfg, he_file, device, ip_address,
                                      trainer, is_label_owner, is_regression)
            self.criterion = torch.nn.CrossEntropyLoss()
        elif trainer == 'knn':
            self.trainer = KNNTrainer(rank, logger, cfg, he_file, device, ip_address,
                                      trainer, is_label_owner, is_regression)

        if self.trainer.is_regression:
            self.criterion = torch.nn.MSELoss()

        # For train and test model
        self.epoch = 0
        self.epoch_loss = 0
        self.epoch_loss_list = []
        self.right_pred = 0
        self.current_batch_index = -1
        self.test_success_flag = False
        self.use_weight = cfg.defs.use_weight

    def send_rsa_public_key_label_owner(self, request, context):
        """
        :param request:
        :param context:
        :return: Process status
        """
        cid = request.cid
        qid = request.qid
        pk_N = request.pk_N
        pk_e = request.pk_e
        # recv_status = False

        if pk_N and pk_e:
            self.trainer.rsa_pk = (int(pk_N), pk_e)
            self.trainer.rsa_pk_comm_status = True
            # print("Public Key received.")
            # print(self.rsa_pk)

        response = vfl_label_owner_service_pb2.rsa_public_key_response_label_owner(
            cid=cid,
            qid=qid,
            recv_status=self.trainer.rsa_pk_comm_status
        )

        return response

    def send_client_enc_ids_label_owner(self, request, context):
        """

        :param request:
        :param context:
        :return:
        """
        cid = request.cid
        qid = request.qid
        client_enc_ids_pk_str = request.client_enc_ids_pk_str

        for enc_id_str in client_enc_ids_pk_str:
            self.trainer.client_enc_ids_pk.append(int(enc_id_str))

        self.trainer.client_enc_ids_comm_status = True
        response = vfl_label_owner_service_pb2.send_client_enc_ids_response_label_owner(
            cid=cid,
            qid=qid,
            recv_status=self.trainer.client_enc_ids_comm_status
        )

        return response

    def send_server_enc_ids_and_client_dec_ids_label_owner(self, request, context):
        """

        :param request:
        :param context:
        :return:
        """
        cid = request.cid
        qid = request.qid
        client_dec_ids = request.client_dec_ids
        server_hash_enc_ids = request.server_hash_enc_ids

        for dec_id in client_dec_ids:
            self.trainer.client_dec_ids.append(int(dec_id))
        self.trainer.client_dec_ids_comm_status = True

        # for hash_enc_id in server_hash_enc_ids:
        self.trainer.server_hash_enc_ids = server_hash_enc_ids
        self.trainer.server_hash_enc_ids_comm_status = True

        response = vfl_label_owner_service_pb2.send_server_enc_ids_and_client_dec_ids_response_label_owner(
            cid=cid,
            qid=qid,
            client_dec_ids_recv_status=self.trainer.client_dec_ids_comm_status,
            server_hash_enc_ids_recv_status=self.trainer.server_hash_enc_ids_comm_status
        )

        return response

    def invite_label_owner_psi(self, request, context):
        """

        :param request:
        :param context:
        :return:
        """
        server_id = request.server_id
        invite_psi = request.invite_psi
        if not invite_psi:
            raise ValueError
        self.trainer.label_owner_psi_permission = invite_psi

        response = vfl_label_owner_service_pb2.invite_psi_response(
            server_id=server_id,
            recv_status=True
        )

        return response

    def get_lr_train_batch_gradient(self, request, context):
        """

        :param request: grpc request
        :param context: grpc response
        :return: grpc message
        """
        server_id = request.server_id
        batch_index = request.batch_index

        assert ((self.current_batch_index + 1) == batch_index) or \
               ((self.current_batch_index - batch_index) == self.trainer.num_train_batches - 1)
        self.current_batch_index = batch_index

        # enc_summed_forward_result_bytes = request.summed_forward_result
        #
        # enc_summed_forward_result = ts.ckks_vector_from(self.trainer.he_key,
        #                                                 enc_summed_forward_result_bytes)
        # summed_forward_result = enc_summed_forward_result.decrypt()
        # print(f"summed: {summed_forward_result}")
        summed_forward_result = request.summed_forward_result
        summed_forward_result_tensor = torch.tensor(summed_forward_result, requires_grad=True)
        # print(f"summed: {summed_forward_result_tensor}")

        grad, early_stop = self.__calculate_lr_train_batch_gradient(summed_forward_result_tensor)
        grad = grad.numpy()
        # print(grad)
        # enc_grad = ts.ckks_vector(self.trainer.he_key, grad)

        response = vfl_label_owner_service_pb2.lr_train_forward_response_label_owner(
            server_id=server_id,
            # batch_gradient=enc_grad.serialize()
            batch_gradient=grad,
            early_stop=early_stop
        )

        return response

    def calculate_lr_test_accuracy(self, request, context):
        """

        :param request: grpc request
        :param context: grpc response
        :return: grpc message
        """
        server_id = request.server_id
        batch_index = request.batch_index
        assert ((self.current_batch_index + 1) == batch_index) or \
               ((self.current_batch_index - batch_index) == self.trainer.num_train_batches - 1)
        self.current_batch_index = batch_index

        # test_result_bytes = request.summed_test_result
        # enc_test_result = ts.ckks_vector_from(self.trainer.he_key,
        #                                       test_result_bytes)
        # test_result = enc_test_result.decrypt()
        test_result = request.summed_test_result
        test_result_tensor = torch.tensor(test_result)

        self.__calculate_lr_test_batch_accuracy(test_result_tensor)
        response = vfl_label_owner_service_pb2.lr_test_forward_response_label_owner(
            server_id=server_id,
            receive_flag=self.test_success_flag
        )

        self.test_success_flag = False

        return response

    def get_mlp_train_batch_gradient(self, request, context):
        """

        :param request: grpc request
        :param context: grpc response
        :return: grpc message
        """
        server_id = request.server_id
        batch_index = request.batch_index
        assert ((self.current_batch_index + 1) == batch_index) or \
               ((self.current_batch_index - batch_index) == self.trainer.num_train_batches - 1)
        self.current_batch_index = batch_index

        top_forward = self.__get_mlp_top_forward_rpc_msg(request)
        # print(f">>>{top_forward}")
        top_forward_tensor = torch.tensor(top_forward, dtype=torch.float, requires_grad=True)
        # print(top_forward_tensor)
        grad, early_stop = self.__calculate_mlp_train_batch_gradient(top_forward_tensor)
        grad = grad.numpy()

        batch_grad = []
        for item in grad:
            single_grad = vfl_label_owner_service_pb2.internal_batch_gradient(
                grad=item
            )
            batch_grad.append(single_grad)

        response = vfl_label_owner_service_pb2.mlp_train_top_forward_response(
            server_id=server_id,
            batch_gradient=batch_grad,
            early_stop=early_stop
        )

        return response

    def calculate_mlp_test_accuracy(self, request, context):
        server_id = request.server_id
        batch_index = request.batch_index

        assert ((self.current_batch_index + 1) == batch_index) or \
               ((self.current_batch_index - batch_index) == self.trainer.num_train_batches - 1)
        self.current_batch_index = batch_index

        top_forward = self.__get_mlp_top_forward_rpc_msg(request)
        top_forward_tensor = torch.tensor(top_forward)

        self.__calculate_mlp_test_batch_accuracy(top_forward_tensor)
        response = vfl_label_owner_service_pb2.mlp_test_top_forward_response(
            server_id=server_id,
            receive_flag=self.test_success_flag
        )

        self.test_success_flag = False

        return response

    def calculate_knn_accuracy(self, request, context):
        server_id = request.server_id
        dist = request.dist
        test_data_index = request.index
        index_top_k = np.argsort(dist)[:self.trainer.top_k]

        self.__calculate_knn_prediction(test_data_index, index_top_k)
        response = vfl_label_owner_service_pb2.knn_distance_response_label_owner(
            server_id=server_id,
            receive_flag=self.test_success_flag
        )

        self.test_success_flag = False

        return response

    def calculate_client_align_index(self, request, context):
        """

        :param request: grpc request
        :param context: grpc response
        :return: grpc message
        """
        server_id = request.server_id
        info = request.info

        label = []
        distance = []
        weight = []

        for i in info:
            # info_label = ts.ckks_vector_from(self.trainer.he_key, i.cluster_label)
            # info_label.decrypt()
            # label.append(info_label)
            # info_dist = ts.ckks_vector_from(self.trainer.he_key, i.cluster_distance)
            # info_dist.decrypt()
            # distance.append(info_dist)
            # info_weight = ts.ckks_vector_from(self.trainer.he_key, i.data_weight)
            # info_weight.decrypt()
            # weight.append(info_weight)
            label.append(i.cluster_label)
            distance.append(i.cluster_distance)
            weight.append(i.data_weight)

        weight = np.sum(np.array(weight), axis=0)

        # print(len(weight))
        align_index = self.trainer.select_align_index(label, distance, weight)
        # enc_align_index = ts.ckks_vector(self.trainer.he_key, align_index)

        response = vfl_label_owner_service_pb2.label_owner_align_index_response(
            server_id=server_id,
            # align_index=enc_align_index,
            align_index=align_index
        )

        return response

    def __reset_train_status(self):
        self.current_batch_index = -1
        self.epoch = 0
        self.epoch_loss = 0

    def __reset_test_status(self):
        self.current_batch_index = -1
        self.right_pred = 0
        self.test_success_flag = False

    def __calculate_lr_train_batch_gradient(self, lr_f):
        """
        use lr forward result to calculate gradient
        :return:
        """
        # loss = None
        grad = None
        early_stop = False
        for batch_index, label in enumerate(self.trainer.train_loader):
            if batch_index != self.current_batch_index:
                continue
            num_train_batches = self.trainer.num_train_batches
            batch_size = self.trainer.batch_size
            batch_weight = None
            if self.trainer.weight_list is not None:
                start = batch_index * batch_size
                if batch_index < num_train_batches:
                    end = start + batch_size
                    batch_weight = self.trainer.weight_list[start:end, :].reshape(-1)
                else:
                    batch_weight = self.trainer.weight_list[start:, :].reshape(-1)

            lr_f.retain_grad()
            # print(f">>>{forward_result_cuda}")
            # label = label.unsqueeze(dim=1)
            # print(label.shape)
            # print(forward_result.shape)
            if self.trainer.is_regression:
                loss = self.criterion(lr_f, label)
            else:
                h = torch.sigmoid(lr_f)
                if batch_weight is None:
                    loss = self.criterion(h, label)
                else:
                    if self.use_weight:
                        criterion = torch.nn.BCELoss(batch_weight)
                    else:
                        criterion = self.criterion
                    loss = criterion(h, label)
            loss.backward()
            # print(f">>>{forward_result_cuda.grad}")
            grad = lr_f.grad
            self.epoch_loss += loss

            if self.current_batch_index == num_train_batches - 1:
                print(f">>>Epoch:{self.epoch + 1}, complete. (Label Owner).")
                print(f">>>Loss {float(self.epoch_loss)}. (Label Owner).")
                self.trainer.logger.info(f">>>Epoch: {self.epoch + 1}, train loss: {self.epoch_loss}")
                self.epoch += 1
                self.epoch_loss_list.append(self.epoch_loss.item())
                # print(self.epoch_loss_list)
                if len(self.epoch_loss_list) >= 5:
                    print(self.epoch_loss_list[-5:])
                    max_loss = max(self.epoch_loss_list[-5:])
                    min_loss = min(self.epoch_loss_list[-5:])
                    if max_loss - min_loss <= 1e-4:
                        early_stop = True
                        print(">>>Early Stop")
                        print(">>>LR Train finish.")
                self.epoch_loss = 0

                if self.trainer.epochs == self.epoch:
                    self.__reset_train_status()
                    print(">>>LR Train finish.")
            break

        # print(loss)
        return grad, early_stop

    def __calculate_lr_test_batch_accuracy(self, lr_f):
        """
        use lr forward result to calculate accuracy
        :return:
        """
        for batch_index, label in enumerate(self.trainer.test_loader):
            if batch_index != self.current_batch_index:
                continue

            y_pred_sig = torch.sigmoid(lr_f).detach().numpy()
            prediction = (y_pred_sig > 0.5).astype(int)
            self.right_pred += np.sum(prediction == label.numpy())

            if self.current_batch_index == (self.trainer.num_test_batches - 1):
                print(f">>>Test acc:{self.right_pred / len(self.trainer.test_dataset)}")
                self.trainer.logger.warning(f">>>Test acc:{self.right_pred / len(self.trainer.test_dataset)}")
                self.__reset_test_status()
            self.test_success_flag = True

    def __calculate_mlp_train_batch_gradient(self, mlp_f):
        """
        use lr forward result to calculate gradient
        :param mlp_f:
        :return:
        """
        grad = None
        early_stop = False
        for batch_index, label in enumerate(self.trainer.train_loader):
            if batch_index != self.current_batch_index:
                continue
            num_train_batches = self.trainer.num_train_batches
            batch_size = self.trainer.batch_size
            batch_weight = None
            # print(self.trainer.weight_list)
            if self.trainer.weight_list is not None:
                start = batch_index * batch_size
                if batch_index < num_train_batches:
                    end = start + batch_size
                    batch_weight = self.trainer.weight_list[start:end, :].reshape(-1)
                else:
                    batch_weight = self.trainer.weight_list[start:, :].reshape(-1)
            # print(batch_index)
            mlp_f.retain_grad()
            # loss = self.criterion(mlp_f, label.long())
            if batch_weight is None:
                loss = self.criterion(mlp_f, label)
            else:
                if self.use_weight:
                    criterion = torch.nn.CrossEntropyLoss(reduction='none')
                    loss = criterion(mlp_f, label)
                    loss *= batch_weight
                    loss = torch.mean(loss)
                else:
                    loss = self.criterion(mlp_f, label)

            loss.backward()
            grad = mlp_f.grad

            self.epoch_loss += loss

            if self.current_batch_index == (num_train_batches - 1):
                print(f">>>Epoch:{self.epoch + 1}, complete. (Label Owner).")
                self.trainer.logger.info(f">>>Epoch: {self.epoch + 1}, train loss: {self.epoch_loss}")
                self.epoch += 1
                self.epoch_loss_list.append(self.epoch_loss.item())
                # print(self.epoch_loss_list)
                if len(self.epoch_loss_list) >= 5:
                    print(self.epoch_loss_list[-5:])
                    max_loss = max(self.epoch_loss_list[-5:])
                    min_loss = min(self.epoch_loss_list[-5:])
                    if max_loss - min_loss <= 1e-4:
                        early_stop = True
                        print(">>>Early Stop")
                        print(">>>LR Train finish.")
                self.epoch_loss = 0

                if self.trainer.epochs == self.epoch:
                    self.__reset_train_status()
                    print(">>>MLP Train finish.")
            break

        return grad, early_stop

    def __calculate_mlp_test_batch_accuracy(self, mlp_f):
        for batch_index, label in enumerate(self.trainer.test_loader):
            if batch_index != self.current_batch_index:
                continue

            _, predicted = mlp_f.max(1)
            # num_data += len(data_x)
            label = torch.from_numpy(label.nonzero().numpy()[:, 1])
            self.right_pred += predicted.eq(label).sum().item()

            if self.current_batch_index == self.trainer.num_test_batches - 1:
                print(f">>>Test acc:{self.right_pred / len(self.trainer.test_dataset)}")
                self.trainer.logger.warning(f">>>Test acc:{self.right_pred / len(self.trainer.test_dataset)}")
                self.__reset_test_status()
            self.test_success_flag = True

    def __get_mlp_top_forward_rpc_msg(self, request):
        top_forward = []
        for item in request.top_forward:
            top_forward.append(item.forward)

        return top_forward

    def __calculate_knn_prediction(self, test_data_index, index_top_k):

        label_count = dict()
        for index in index_top_k:
            label = int(self.trainer.train_dataset.train_label_numpy[index])
            if label not in label_count.keys():
                label_count[label] = 0
            label_count[label] += 1

        pred = max(label_count, key=label_count.get)
        target = int(self.trainer.test_dataset.test_label_numpy[test_data_index])
        self.right_pred += (pred == target)

        if test_data_index == (len(self.trainer.test_dataset) - 1):
            print(f">>>KNN complete. (Label Owner).")
            print(f">>>KNN Test acc:{self.right_pred / len(self.trainer.test_dataset)}")
            self.trainer.logger.warning(f">>>Test acc:{self.right_pred / len(self.trainer.test_dataset)}")
            self.__reset_test_status()
        self.test_success_flag = True
