#!/bin/bash
# Run CogniVault gRPC server with container-style config (no Docker)

set -e

PORT=${PORT:-50051}

exec python server.py --host 0.0.0.0 --port "$PORT"
