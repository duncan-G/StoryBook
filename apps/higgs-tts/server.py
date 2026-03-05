#!/usr/bin/env python3
"""Inference engine gRPC server.

Exposes the AudioEngine over gRPC so clients can send generation requests
and stream responses. Supports graceful shutdown on SIGINT/SIGTERM.

Run:
  python server.py
  python server.py --host 0.0.0.0 --port 50051
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys

import grpc

from inference_grpc import inference_engine_pb2 as pb2
from inference_grpc import inference_engine_pb2_grpc as pb2_grpc

from src.generation.engine import AudioEngine
from src.generation.engine_servicer import InferenceEngineServicer
from src.telemetry import setup_telemetry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI & server setup
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser; host/port can also be set via HOST and PORT env vars."""
    p = argparse.ArgumentParser(description="Inference engine gRPC server")
    p.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.getenv("PORT", "50051")))
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
    )
    return p


def enable_reflection(server: grpc.aio.Server) -> None:
    """Enable gRPC server reflection so tools (e.g. grpcurl) can discover services."""
    try:
        from grpc_reflection.v1alpha import reflection

        service_names = (
            pb2.DESCRIPTOR.services_by_name["InferenceEngine"].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(service_names, server)
        logger.info("gRPC reflection enabled")
    except ImportError:
        logger.info("gRPC reflection disabled (pip install grpcio-reflection)")


# ---------------------------------------------------------------------------
# gRPC server lifecycle
# ---------------------------------------------------------------------------


async def serve(engine: AudioEngine, host: str, port: int) -> None:
    """Run the gRPC server until SIGINT or SIGTERM; then stop gracefully."""
    servicer = InferenceEngineServicer(engine)
    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ]
    )
    pb2_grpc.add_InferenceEngineServicer_to_server(servicer, server)
    enable_reflection(server)

    address = f"{host}:{port}"
    server.add_insecure_port(address)
    await server.start()
    logger.info("Inference gRPC server started on %s", address)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    try:
        await stop.wait()
    finally:
        logger.info("Shutting down gRPC server …")
        await server.stop(grace=5.0)
        logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse args, build the AudioEngine, then run the gRPC server."""
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    setup_telemetry(service_name="higgs-tts")
    engine = AudioEngine()

    try:
        asyncio.run(serve(engine, host=args.host, port=args.port))
    except Exception:
        logger.exception("Server failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
