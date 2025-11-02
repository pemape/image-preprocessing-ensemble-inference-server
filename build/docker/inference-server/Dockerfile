# Multi-stage Dockerfile for PyTorch LBCNN Inference System
# Based on: "Ensemble of pre-processing techniques with CNN for diabetic retinopathy detection"

# Stage 1: Base image with system dependencies
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Stage 2: Dependencies installation
FROM base as dependencies

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install PyTorch (CPU version for smaller image, GPU version commented below)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# For GPU support, uncomment the line below and comment the CPU version above:
# RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Stage 3: Application
FROM dependencies as application

# Set working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/models \
             /app/configs \
             /app/logs \
             /app/input \
             /app/output \
             /app/temp \
             /app/debug && \
    chown -R appuser:appuser /app

# Copy application files
COPY --chown=appuser:appuser fundus_preprocessor.py /app/
COPY --chown=appuser:appuser diabetic_retinopathy_classifier.py /app/
COPY --chown=appuser:appuser fundus_inference_server.py /app/
COPY --chown=appuser:appuser client.py /app/

# Copy configuration files
COPY --chown=appuser:appuser preprocessing_config.yaml /app/configs/
COPY --chown=appuser:appuser classifier_config.yaml /app/configs/
COPY --chown=appuser:appuser classifier_config_hard_voting.yaml /app/configs/
COPY --chown=appuser:appuser classifier_config_cpu.yaml /app/configs/

# Copy model files from the trained models directory
# This will copy all .pt files from your lbcnn_pytorch directory structure
COPY --chown=appuser:appuser ../lbcnn_pytorch/hns/DR/Original500x500/EfficientNetB4/Nadam/8/301_08_2023_07_50_40/Aptos5_8_EfficientNetB4_opt=Nadam_lr=0.001_best_ckp.pt /app/models/efficientnetb4_aptos5_original_fold8.pt
COPY --chown=appuser:appuser ../lbcnn_pytorch/hns/DR/RGBClahe500x500/EfficientNetB4/Nadam/8/201_08_2023_14_54_36/Aptos5_8_EfficientNetB4_opt=Nadam_lr=0.001_best_ckp.pt /app/models/efficientnetb4_aptos5_rgbclahe_fold8.pt
COPY --chown=appuser:appuser ../lbcnn_pytorch/hns/DR/Original500x500/Xception/Nadam/8/301_09_2023_17_18_45/Aptos5_8_Xception_opt=Nadam_lr=0.001_best_ckp.pt /app/models/xception_aptos5_original_fold8.pt
COPY --chown=appuser:appuser ../lbcnn_pytorch/hns/DR/RGBClahe500x500/Xception/Nadam/8/101_05_2023_18_34_40/Aptos5_8_Xception_opt=Nadam_lr=0.001_best_ckp.pt /app/models/xception_aptos5_rgbclahe_fold8.pt

# Copy additional model files (best performing models from each architecture/preprocessing combination)
COPY --chown=appuser:appuser ../lbcnn_pytorch/hns/DR/Original500x500/EfficientNetB4/Nadam/7/301_08_2023_07_41_03/Aptos5_7_EfficientNetB4_opt=Nadam_lr=0.001_best_ckp.pt /app/models/efficientnetb4_aptos5_original_fold7.pt
COPY --chown=appuser:appuser ../lbcnn_pytorch/hns/DR/RGBClahe500x500/EfficientNetB4/Nadam/7/201_08_2023_14_45_00/Aptos5_7_EfficientNetB4_opt=Nadam_lr=0.001_best_ckp.pt /app/models/efficientnetb4_aptos5_rgbclahe_fold7.pt
COPY --chown=appuser:appuser ../lbcnn_pytorch/hns/DR/Original500x500/Xception/Nadam/7/301_09_2023_17_09_08/Aptos5_7_Xception_opt=Nadam_lr=0.001_best_ckp.pt /app/models/xception_aptos5_original_fold7.pt
COPY --chown=appuser:appuser ../lbcnn_pytorch/hns/DR/RGBClahe500x500/Xception/Nadam/7/101_05_2023_18_25_04/Aptos5_7_Xception_opt=Nadam_lr=0.001_best_ckp.pt /app/models/xception_aptos5_rgbclahe_fold7.pt

# Create a startup script
COPY --chown=appuser:appuser <<EOF /app/start_server.sh
#!/bin/bash
echo "Starting Fundus Inference Server..."
echo "Available models:"
ls -la /app/models/
echo ""
echo "Available configs:"
ls -la /app/configs/
echo ""
echo "Starting server with full inference pipeline..."
python fundus_inference_server.py \
    --preprocessing-config /app/configs/preprocessing_config.yaml \
    --classifier-config /app/configs/classifier_config.yaml \
    --host 0.0.0.0 \
    --port 5000
EOF

RUN chmod +x /app/start_server.sh

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Default command
CMD ["/app/start_server.sh"]

# Stage 4: Development (optional)
FROM application as development

USER root

# Install development tools
RUN pip install --no-cache-dir \
    jupyter \
    ipython \
    tensorboard \
    matplotlib \
    seaborn \
    plotly \
    memory-profiler \
    line-profiler \
    pytest

# Install additional debugging tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    htop \
    tree \
    git \
    && rm -rf /var/lib/apt/lists/*

USER appuser

# Expose additional ports for development
EXPOSE 8888 6006

# Development command
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Stage 5: Production optimized
FROM application as production

# Remove development files and optimize
USER root
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    find /app -name "*.pyc" -delete && \
    find /app -name "__pycache__" -type d -exec rm -rf {} + || true

USER appuser

# Production command with optimizations
CMD ["python", "-O", "fundus_inference_server.py", \
     "--preprocessing-config", "/app/configs/preprocessing_config.yaml", \
     "--classifier-config", "/app/configs/classifier_config.yaml", \
     "--host", "0.0.0.0", \
     "--port", "5000"]