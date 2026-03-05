"""
Inference engine gRPC protocol definitions.

This module contains the protocol buffer definitions for the inference engine
gRPC API. The protobuf files are compiled into Python code using grpcio-tools.
"""

__all__ = [
    "inference_engine_pb2",
    "inference_engine_pb2_grpc",
]

try:
    from . import inference_engine_pb2
    from . import inference_engine_pb2_grpc
except ImportError:
    inference_engine_pb2 = None  # type: ignore[misc, assignment]
    inference_engine_pb2_grpc = None  # type: ignore[misc, assignment]
