# screenplay-parser: FastAPI service for screenplay PDF parsing
# Python 3.12, uv for install, uvicorn to run
FROM python:3.12-slim-bookworm

WORKDIR /app

# Install uv and system deps (e.g. for pdfminer if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# Copy dependency manifests first for better layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy inference_grpc and compile protos (generates code compatible with grpcio 1.78.0)
COPY inference_grpc/ inference_grpc/
RUN uv run python inference_grpc/compile_protos.py

# Copy application code
COPY main.py screenplay_parser.py gemini_qa.py chunking.py store.py database.py models.py config.py telemetry.py generate.py ./

# Default port for FastAPI/uvicorn
ENV PORT=8000
EXPOSE 8000

# Run with uvicorn (no --reload in production)
CMD ["sh", "-c", "uv run uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
