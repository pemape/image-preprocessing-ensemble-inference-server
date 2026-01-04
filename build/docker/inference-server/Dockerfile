# Fundus Inference Server - Flask API Docker Image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies for image processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    libgl1

# Create working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu130 && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY fundus_preprocessor.py .
COPY diabetic_retinopathy_classifier.py .
COPY fundus_inference_server.py .
COPY redis_cache_manager.py .

# Copy ensemble inference module
COPY ensemble_inference/ ./ensemble_inference/

# Copy configuration files
COPY configs/ ./configs/

# Copy DVC configuration for pulling models
COPY .dvc/ ./.dvc/
COPY .dvcignore .
COPY models/ ./models/

# Create necessary directories for models and logs
RUN mkdir -p /app/models /app/logs

# Expose Flask API port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start Flask API server
CMD ["python", "fundus_inference_server.py", \
    "--preprocessing-config", "configs/preprocessing_config.yaml", \
    "--classifier-config", "configs/classifier_config.yaml", \
    "--host", "0.0.0.0", \
    "--port", "5000"]
