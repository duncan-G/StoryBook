# Use NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:25.02-py3

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install torchaudio as it's not included in the base image
# We point directly to the CUDA 12.8 wheelhouse to match the NVIDIA 25.02 environment
RUN pip install torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128


# Default to an interactive shell so you can exec and run scripts manually.
# Override CMD in `docker run` if you want a specific script.
CMD ["/bin/bash"]
