# Multi-stage Dockerfile for batch testing
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0,1,2,3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt test-requirements.txt ./

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir -r test-requirements.txt

# Copy source code
COPY . .

# Create directories for batch testing
RUN mkdir -p /app/test-images /app/batch_test_results /app/trained-models

# Set permissions
RUN chmod +x /app/batch_tester.py

# Default command for batch testing
CMD ["python3", "batch_tester.py", "--config", "configs/batch_testing_config.yaml"]