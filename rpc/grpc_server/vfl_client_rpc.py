# -*- coding: utf-8 -*-
"""
@Auth ： ZQB
@Time ： 2023/5/16 20:43
@File ：lr_client_rpc.py
"""
import rpc.grpc_file.vfl_client_service_pb2_grpc as vfl_client_service_pb2_grpc
import rpc.grpc_file.vfl_client_service_pb2 as vfl_client_service_pb2
from trainer import LRTrainer, MLPTrainer, KNNTrainer
from utils import prepare_logger
import tenseal as ts


class VFLClient(vfl_client_service_pb2_grpc.VFLClientServiceServicer):
    def __init__(self, rank, log_path, cfg, he_file, device, ip_address,
                 is_label_owner=False, is_regression=False, trainer='lr'):
        logger = prepare_logger(rank, log_path, cfg.defs.mode)
        if trainer == 'lr':
            self.trainer = LRTrainer(rank, logger, cfg, he_file, device, ip_address,
                                     trainer, is_label_owner, is_regression)
        elif trainer == 'mlp':
            self.trainer = MLPTrainer(rank, logger, cfg, he_file, device, ip_address,
                                      trainer, is_label_owner, is_regression)
        elif trainer == 'knn':
            self.trainer = KNNTrainer(rank, logger, cfg, he_file, device, ip_address,
                                      trainer, is_label_owner, is_regression)

    def send_rsa_public_key(self, request, context):
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

        response = vfl_client_service_pb2.rsa_public_key_response(
            cid=cid,
            qid=qid,
            recv_status=self.trainer.rsa_pk_comm_status
        )

        return response

    def send_client_enc_ids(self, request, context):
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
        response = vfl_client_service_pb2.send_client_enc_ids_response(
            cid=cid,
            qid=qid,
            recv_status=self.trainer.client_enc_ids_comm_status
        )

        return response

    def send_server_enc_ids_and_client_dec_ids(self, request, context):
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

        response = vfl_client_service_pb2.send_server_enc_ids_and_client_dec_ids_response(
            cid=cid,
            qid=qid,
            client_dec_ids_recv_status=self.trainer.client_dec_ids_comm_status,
            server_hash_enc_ids_recv_status=self.trainer.server_hash_enc_ids_comm_status
        )

        return response
