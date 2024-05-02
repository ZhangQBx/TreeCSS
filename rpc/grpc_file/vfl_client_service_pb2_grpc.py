# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import rpc.grpc_file.vfl_client_service_pb2 as vfl__client__service__pb2


class VFLClientServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.send_lr_train_batch_gradient = channel.unary_unary(
                '/VFLClientService/send_lr_train_batch_gradient',
                request_serializer=vfl__client__service__pb2.lr_train_batch_gradient_request.SerializeToString,
                response_deserializer=vfl__client__service__pb2.lr_train_batch_gradient_response.FromString,
                )
        self.send_rsa_public_key = channel.unary_unary(
                '/VFLClientService/send_rsa_public_key',
                request_serializer=vfl__client__service__pb2.rsa_public_key_request.SerializeToString,
                response_deserializer=vfl__client__service__pb2.rsa_public_key_response.FromString,
                )
        self.send_client_enc_ids = channel.unary_unary(
                '/VFLClientService/send_client_enc_ids',
                request_serializer=vfl__client__service__pb2.send_client_enc_ids_request.SerializeToString,
                response_deserializer=vfl__client__service__pb2.send_client_enc_ids_response.FromString,
                )
        self.send_server_enc_ids_and_client_dec_ids = channel.unary_unary(
                '/VFLClientService/send_server_enc_ids_and_client_dec_ids',
                request_serializer=vfl__client__service__pb2.send_server_enc_ids_and_client_dec_ids_request.SerializeToString,
                response_deserializer=vfl__client__service__pb2.send_server_enc_ids_and_client_dec_ids_response.FromString,
                )


class VFLClientServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def send_lr_train_batch_gradient(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def send_rsa_public_key(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def send_client_enc_ids(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def send_server_enc_ids_and_client_dec_ids(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_VFLClientServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'send_lr_train_batch_gradient': grpc.unary_unary_rpc_method_handler(
                    servicer.send_lr_train_batch_gradient,
                    request_deserializer=vfl__client__service__pb2.lr_train_batch_gradient_request.FromString,
                    response_serializer=vfl__client__service__pb2.lr_train_batch_gradient_response.SerializeToString,
            ),
            'send_rsa_public_key': grpc.unary_unary_rpc_method_handler(
                    servicer.send_rsa_public_key,
                    request_deserializer=vfl__client__service__pb2.rsa_public_key_request.FromString,
                    response_serializer=vfl__client__service__pb2.rsa_public_key_response.SerializeToString,
            ),
            'send_client_enc_ids': grpc.unary_unary_rpc_method_handler(
                    servicer.send_client_enc_ids,
                    request_deserializer=vfl__client__service__pb2.send_client_enc_ids_request.FromString,
                    response_serializer=vfl__client__service__pb2.send_client_enc_ids_response.SerializeToString,
            ),
            'send_server_enc_ids_and_client_dec_ids': grpc.unary_unary_rpc_method_handler(
                    servicer.send_server_enc_ids_and_client_dec_ids,
                    request_deserializer=vfl__client__service__pb2.send_server_enc_ids_and_client_dec_ids_request.FromString,
                    response_serializer=vfl__client__service__pb2.send_server_enc_ids_and_client_dec_ids_response.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'VFLClientService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class VFLClientService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def send_lr_train_batch_gradient(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/VFLClientService/send_lr_train_batch_gradient',
            vfl__client__service__pb2.lr_train_batch_gradient_request.SerializeToString,
            vfl__client__service__pb2.lr_train_batch_gradient_response.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def send_rsa_public_key(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/VFLClientService/send_rsa_public_key',
            vfl__client__service__pb2.rsa_public_key_request.SerializeToString,
            vfl__client__service__pb2.rsa_public_key_response.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def send_client_enc_ids(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/VFLClientService/send_client_enc_ids',
            vfl__client__service__pb2.send_client_enc_ids_request.SerializeToString,
            vfl__client__service__pb2.send_client_enc_ids_response.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def send_server_enc_ids_and_client_dec_ids(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/VFLClientService/send_server_enc_ids_and_client_dec_ids',
            vfl__client__service__pb2.send_server_enc_ids_and_client_dec_ids_request.SerializeToString,
            vfl__client__service__pb2.send_server_enc_ids_and_client_dec_ids_response.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)