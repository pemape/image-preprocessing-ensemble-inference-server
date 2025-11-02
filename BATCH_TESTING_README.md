# Batch Testing System for Fundus Image Classifier

A comprehensive batch testing framework for evaluating fundus image classification models with multi-GPU support, continuous result writing, and detailed performance analytics.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation & Setup](#installation--setup)
4. [Quick Start](#quick-start)
5. [Configuration](#configuration)
6. [Usage Examples](#usage-examples)
7. [Output Format](#output-format)
8. [Multi-GPU Support](#multi-gpu-support)
9. [Reconciler Pattern & Continuous Monitoring](#reconciler-pattern--continuous-monitoring)
10. [Docker Deployment](#docker-deployment)
11. [VS Code Integration](#vs-code-integration)
12. [Performance Optimization](#performance-optimization)
13. [Troubleshooting](#troubleshooting)
14. [API Reference](#api-reference)

## Overview

The Batch Testing System provides a robust framework for evaluating diabetic retinopathy classification models on large datasets. It processes images in batches, applies ensemble preprocessing and classification, and generates comprehensive JSON reports with detailed metrics.

### Key Capabilities

- **Large-Scale Processing**: Handle thousands of images efficiently
- **Multi-GPU Support**: Automatic distribution across available GPUs
- **Memory Safe**: Continuous JSON writing prevents memory overflow
- **Crash Resilient**: Results preserved even if process interrupts
- **Real-time Monitoring**: Progress tracking and intermediate summaries
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, confusion matrix
- **Reconciler Pattern**: Continuous monitoring for new images and automatic processing

## Features

### 🚀 Performance Features
- **Multi-GPU Data Parallelism**: Automatic GPU detection and utilization
- **Continuous JSON Writing**: No memory accumulation, safe for large datasets
- **Batch Processing**: Configurable batch sizes for optimal throughput
- **Progress Monitoring**: Real-time logging and intermediate checkpoints

### 📊 Evaluation Features
- **Ground Truth Integration**: CSV-based label comparison
- **Comprehensive Metrics**: Per-class and overall performance statistics
- **Confusion Matrix**: Detailed classification analysis
- **Processing Time Analytics**: Performance profiling and optimization insights

### 🔧 Operational Features
- **Flexible Configuration**: YAML-based settings for all parameters
- **Docker Support**: Containerized deployment with GPU access
- **VS Code Integration**: Built-in tasks and debug configurations
- **Error Handling**: Graceful failure recovery and detailed error logging
- **Reconciler Pattern**: Kubernetes-style continuous monitoring and processing

## Installation & Setup

### Prerequisites

```bash
# Python 3.8+ required
python --version

# Install dependencies
pip install -r requirements.txt
pip install -r test-requirements.txt

# For GPU support (optional but recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Verify Installation

```bash
# Test batch configuration
python -c "import yaml; config = yaml.safe_load(open('configs/batch_testing_config.yaml', 'r')); print('✓ Batch testing config is valid')"

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPUs detected: {torch.cuda.device_count()}')"
```

## Quick Start

### 1. Prepare Your Data

Create your image directory and optional ground truth file:

```
test-images/
├── image_001.jpg
├── image_002.jpg
└── ...

# Optional: ground_truth.csv
image_id,true_label
image_001.jpg,0
image_002.jpg,2
```

### 2. Configure Testing

Edit `configs/batch_testing_config.yaml`:

```yaml
input:
  images_folder: "test-images"
  ground_truth_file: "ground_truth.csv"  # Optional

output:
  results_folder: "batch_test_results"

batch_processing:
  batch_size: 16
  num_workers: 4

gpu:
  enabled: true
  use_data_parallel: true
```

### 3. Run Batch Testing

```bash
# Basic usage
python batch_tester.py --config configs/batch_testing_config.yaml

# With custom parameters
python batch_tester.py \
  --config configs/batch_testing_config.yaml \
  --images-folder ./my-test-images \
  --batch-size 32

# Force CPU usage
python batch_tester.py \
  --config configs/batch_testing_config.yaml \
  --force-cpu
```

## Configuration

### Complete Configuration Reference

```yaml
# Input/Output Configuration
input:
  images_folder: "test-images"              # Image directory path
  ground_truth_file: "ground_truth.csv"    # Optional CSV with labels
  supported_formats: [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

output:
  results_folder: "batch_test_results"     # Output directory
  save_processed_images: false            # Save preprocessed variants
  save_individual_results: true           # Save progress summaries

# Model Configuration
model:
  preprocessing_config: "configs/preprocessing_config.yaml"
  classifier_config: "configs/classifier_config.yaml"
  voting_strategy: "soft"                 # "soft" or "hard"

# Batch Processing Configuration
batch_processing:
  batch_size: 16                          # Images per batch
  num_workers: 4                          # Data loader workers
  prefetch_factor: 2                      # Batches to prefetch

# GPU Configuration
gpu:
  enabled: true                           # Enable GPU usage
  use_data_parallel: true                 # Multi-GPU support
  device_ids: []                          # Specific GPUs (empty = all)
  force_cpu: false                        # Force CPU usage

# Performance Monitoring
performance:
  log_interval: 10                        # Log every N batches
  save_interval: 50                       # Save summary every N batches
  memory_efficient: true                  # Memory optimization

# Testing Parameters
testing:
  confidence_threshold: 0.5               # Minimum confidence
  calculate_metrics: true                 # Enable metric calculation
  class_names:
    - "No DR"
    - "Mild DR"
    - "Moderate DR"
    - "Severe DR"
    - "Proliferative DR"

# Output Format
report:
  include_metadata: true                  # System information
  include_confusion_matrix: true          # Classification matrix
  include_per_class_metrics: true         # Per-class statistics
  include_processing_times: true          # Performance metrics
  save_failed_images_list: true           # Error tracking
```

## Usage Examples

### Basic Testing

```bash
# Test small dataset
python batch_tester.py --config configs/batch_testing_config.yaml

# View results
cat batch_test_results/test_results_*.json | jq '.batch_metadata.dataset_info'
```

### Advanced Testing with Custom Parameters

```bash
# Large dataset with optimized settings
python batch_tester.py \
  --config configs/batch_testing_config.yaml \
  --images-folder /path/to/large/dataset \
  --output-folder /path/to/results \
  --batch-size 64 \
  --ground-truth /path/to/labels.csv
```

### Development and Debugging

```bash
# Small batch for debugging
python batch_tester.py \
  --config configs/batch_testing_config.yaml \
  --batch-size 2 \
  --force-cpu

# Monitor progress in real-time
tail -f batch_test_results/test_results_*.json
```

## Output Format

### Primary Output: `test_results_YYYYMMDD_HHMMSS.json`

The main results file contains:

```json
{
  "batch_metadata": {
    "testing_session": {
      "session_id": "batch_test_20251102_143056",
      "start_time": "2025-11-02T14:30:56Z",
      "end_time": "2025-11-02T14:45:23Z",
      "duration_seconds": 867.34
    },
    "configuration": { /* Full config details */ },
    "system_info": {
      "cuda_available": true,
      "num_gpus": 2,
      "device_names": ["NVIDIA RTX 4090", "NVIDIA RTX 4090"],
      "pytorch_version": "2.1.0"
    },
    "model_info": {
      "ensemble_size": 5,
      "architectures": ["efficientnetb4"],
      "voting_strategy": "soft"
    }
  },
  "processing_statistics": {
    "total_batches": 125,
    "average_batch_time": 2.34,
    "average_time_per_image": 0.146,
    "total_processing_time": 292.5
  },
  "evaluation_metrics": {
    "overall": {
      "accuracy": 0.892,
      "macro_f1": 0.856,
      "weighted_f1": 0.889
    },
    "per_class": {
      "No DR": {
        "precision": 0.94,
        "recall": 0.91,
        "f1_score": 0.925
      }
    },
    "confusion_matrix": [[450, 12, 3, 0, 1], [...]],
    "average_confidence": 0.847
  },
  "batch_results": [
    {
      "image_id": "image_001.jpg",
      "logits": [0.85, 0.10, 0.03, 0.01, 0.01],
      "predicted_label": 0,
      "predicted_class": "No DR",
      "confidence": 0.85,
      "true_label": 0,
      "processing_times": {
        "preprocessing_ms": 45.2,
        "classification_ms": 123.8,
        "total_ms": 169.0
      },
      "status": "success"
    }
  ]
}
```

### Secondary Outputs

- **`metrics_summary_YYYYMMDD_HHMMSS.json`** - Standalone metrics file
- **`progress_summary_batch_X.json`** - Intermediate progress checkpoints
- **`processed_images/`** - Preprocessed image variants (if enabled)

## Multi-GPU Support

### Automatic GPU Detection

The system automatically detects and utilizes all available GPUs:

```python
# GPU configuration is handled automatically
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs")
```

### Configuration Options

```yaml
gpu:
  enabled: true
  use_data_parallel: true
  device_ids: [0, 1, 2, 3]  # Specific GPUs or empty for all
  force_cpu: false
```

### Performance Scaling

Expected performance improvements with multiple GPUs:

| GPUs | Batch Size | Speedup | Memory Usage |
|------|------------|---------|--------------|
| 1    | 16         | 1.0x    | ~6GB        |
| 2    | 32         | 1.8x    | ~12GB       |
| 4    | 64         | 3.2x    | ~24GB       |

## Reconciler Pattern & Continuous Monitoring

The batch tester can be configured to run in a **reconciler pattern**, continuously monitoring the images folder for changes and automatically processing new images as they arrive. This is similar to Kubernetes controllers that watch for resource changes and take action.

### 🎯 Use Cases

- **Real-time Processing**: Process images as they arrive from medical imaging systems
- **Automated Workflows**: Integrate with PACS systems or image acquisition pipelines
- **Continuous Evaluation**: Monitor model performance on incoming data streams
- **Production Monitoring**: Detect performance drift or data quality issues over time

### Configuration for Reconciler Mode

Add the following section to your `batch_testing_config.yaml`:

```yaml
# Reconciler Pattern Configuration
reconciler:
  enabled: true                           # Enable reconciler mode
  watch_interval: 30                      # Check for changes every N seconds
  process_new_only: true                  # Only process new/unprocessed images
  state_tracking:
    enable_state_file: true               # Track processed images
    state_file: "processed_images_state.json"
    cleanup_old_entries: true            # Remove old entries from state
    max_state_entries: 10000             # Maximum state entries to keep
  
  # File system monitoring
  folder_monitoring:
    watch_subdirectories: true           # Monitor subdirectories
    file_patterns: ["*.jpg", "*.jpeg", "*.png"]
    ignore_hidden_files: true           # Skip hidden files
    min_file_age: 5                     # Wait N seconds before processing (file stability)
    
  # Processing triggers
  triggers:
    batch_when_count: 16                # Process when N new images found
    batch_when_timeout: 300             # Process when timeout reached (seconds)
    force_process_on_shutdown: true     # Process remaining images on shutdown
    
  # Output management
  output:
    incremental_reports: true           # Generate reports for each reconciliation cycle
    consolidate_reports: true           # Merge all cycle reports into main report
    archive_old_reports: true           # Move old reports to archive folder
```

### Running in Reconciler Mode

#### Command Line Usage

```bash
# Start reconciler mode
python batch_tester.py \
  --config configs/batch_testing_config.yaml \
  --reconciler-mode

# With custom watch interval
python batch_tester.py \
  --config configs/batch_testing_config.yaml \
  --reconciler-mode \
  --watch-interval 60

# Reconciler with specific folder monitoring
python batch_tester.py \
  --config configs/batch_testing_config.yaml \
  --reconciler-mode \
  --images-folder /path/to/watch/folder \
  --min-file-age 10
```

#### Docker Reconciler Deployment

```yaml
# docker-compose.reconciler.yml
version: '3.8'

services:
  batch-reconciler:
    build:
      context: .
      dockerfile: Dockerfile.batch
    image: fundus-batch-tester:latest
    container_name: fundus-batch-reconciler
    restart: unless-stopped
    volumes:
      - ./watch-images:/app/test-images
      - ./batch_test_results:/app/batch_test_results
      - ./trained-models:/app/trained-models
      - ./configs:/app/configs
      - ./state:/app/state  # Persistent state storage
    environment:
      - CUDA_VISIBLE_DEVICES=0,1
      - PYTHONUNBUFFERED=1
    command: >
      python3 batch_tester.py 
      --config configs/batch_testing_config.yaml
      --reconciler-mode
      --watch-interval 30
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

```bash
# Deploy reconciler
docker-compose -f docker-compose.reconciler.yml up -d

# Monitor logs
docker-compose -f docker-compose.reconciler.yml logs -f batch-reconciler
```

### Reconciler Implementation Details

#### State Tracking

The reconciler maintains a state file to track processed images:

```json
{
  "last_reconciliation": "2025-11-02T15:30:45Z",
  "total_reconciliation_cycles": 157,
  "processed_images": {
    "image_001.jpg": {
      "processed_at": "2025-11-02T14:30:15Z",
      "file_hash": "sha256:abc123...",
      "file_size": 2048576,
      "processing_status": "success",
      "result_file": "test_results_20251102_143015.json"
    },
    "image_002.jpg": {
      "processed_at": "2025-11-02T14:32:20Z",
      "file_hash": "sha256:def456...",
      "file_size": 1876543,
      "processing_status": "success",
      "result_file": "test_results_20251102_143220.json"
    }
  },
  "statistics": {
    "total_processed": 1247,
    "successful": 1198,
    "failed": 49,
    "average_processing_time": 2.34
  }
}
```

#### Change Detection Algorithm

```python
# Pseudo-code for reconciler logic
def reconcile_folder():
    current_images = scan_folder()
    state = load_state_file()
    
    new_images = []
    for image_path in current_images:
        file_hash = calculate_file_hash(image_path)
        file_age = time.now() - file_stat(image_path).modified_time
        
        if (image_path not in state.processed_images or 
            state.processed_images[image_path].hash != file_hash):
            
            if file_age >= min_file_age:  # File is stable
                new_images.append(image_path)
    
    if len(new_images) >= batch_when_count or timeout_reached():
        process_batch(new_images)
        update_state_file(new_images)
```

#### Processing Modes

1. **Immediate Processing**: Process images as soon as they're detected
2. **Batch Accumulation**: Wait for N images or timeout before processing
3. **Scheduled Processing**: Process at specific intervals regardless of file count

### Reconciler Configuration Examples

#### High-Frequency Medical Imaging

```yaml
reconciler:
  enabled: true
  watch_interval: 10                    # Check every 10 seconds
  process_new_only: true
  
  triggers:
    batch_when_count: 4                 # Process quickly with small batches
    batch_when_timeout: 60              # Maximum 1-minute delay
    
  output:
    incremental_reports: true
    consolidate_reports: true
```

#### Research Dataset Processing

```yaml
reconciler:
  enabled: true
  watch_interval: 300                   # Check every 5 minutes
  process_new_only: true
  
  triggers:
    batch_when_count: 32                # Larger batches for efficiency
    batch_when_timeout: 1800            # Process every 30 minutes
    
  folder_monitoring:
    watch_subdirectories: true          # Monitor nested folders
    min_file_age: 30                    # Wait for file stability
```

#### Production Monitoring

```yaml
reconciler:
  enabled: true
  watch_interval: 60                    # Check every minute
  process_new_only: true
  
  state_tracking:
    cleanup_old_entries: true           # Prevent state file bloat
    max_state_entries: 50000            # Keep recent history
    
  triggers:
    batch_when_count: 8                 # Balance latency vs efficiency
    batch_when_timeout: 300             # 5-minute maximum delay
    
  output:
    incremental_reports: true
    archive_old_reports: true           # Manage disk space
```

### Monitoring & Observability

#### Log Output Example

```
2025-11-02 15:30:45 - BatchReconciler - INFO - Starting reconciler mode, watching: /app/test-images
2025-11-02 15:30:45 - BatchReconciler - INFO - Reconciliation interval: 30 seconds
2025-11-02 15:30:45 - BatchReconciler - INFO - Batch triggers: 16 images or 300 seconds timeout
2025-11-02 15:31:15 - BatchReconciler - INFO - Reconciliation cycle #1: Found 3 new images
2025-11-02 15:31:45 - BatchReconciler - INFO - Reconciliation cycle #2: Found 8 new images (total: 11)
2025-11-02 15:32:15 - BatchReconciler - INFO - Reconciliation cycle #3: Found 6 new images (total: 17)
2025-11-02 15:32:15 - BatchReconciler - INFO - Batch threshold reached (17 >= 16), processing batch
2025-11-02 15:32:15 - BatchTester - INFO - Processing 17 images in 2 batches
2025-11-02 15:32:47 - BatchTester - INFO - Batch processing completed: 17 successful, 0 failed
2025-11-02 15:32:47 - BatchReconciler - INFO - Updated state file with 17 new entries
```

#### Health Checks

```bash
# Check reconciler status
curl http://localhost:8080/reconciler/status

# Response
{
  "reconciler_active": true,
  "last_reconciliation": "2025-11-02T15:32:47Z",
  "total_cycles": 3,
  "pending_images": 0,
  "next_check_in": 23,
  "statistics": {
    "total_processed": 17,
    "successful": 17,
    "failed": 0,
    "average_cycle_time": 2.3
  }
}
```

### Integration with External Systems

#### PACS Integration

```python
# Example PACS webhook handler
@app.route('/pacs/new-study', methods=['POST'])
def handle_new_study():
    study_data = request.json
    image_paths = study_data['image_paths']
    
    # Copy images to watched folder
    for image_path in image_paths:
        shutil.copy(image_path, '/app/test-images/')
    
    # Reconciler will automatically detect and process
    return {'status': 'queued', 'images': len(image_paths)}
```

#### Message Queue Integration

```python
# Example RabbitMQ consumer
def process_image_queue():
    def callback(ch, method, properties, body):
        message = json.loads(body)
        image_url = message['image_url']
        
        # Download and save image
        image_path = download_image(image_url, '/app/test-images/')
        
        # Reconciler will process automatically
        ch.basic_ack(delivery_tag=method.delivery_tag)
    
    channel.basic_consume(queue='image_processing', on_message_callback=callback)
    channel.start_consuming()
```

#### Kubernetes CronJob

```yaml
# k8s-reconciler-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: fundus-batch-reconciler
spec:
  schedule: "*/5 * * * *"  # Every 5 minutes
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: batch-reconciler
            image: fundus-batch-tester:latest
            args:
            - python3
            - batch_tester.py
            - --config
            - configs/batch_testing_config.yaml
            - --reconciler-mode
            - --watch-interval
            - "300"
            volumeMounts:
            - name: images-volume
              mountPath: /app/test-images
            - name: results-volume
              mountPath: /app/batch_test_results
            resources:
              limits:
                nvidia.com/gpu: 1
              requests:
                memory: "4Gi"
                cpu: "2"
          volumes:
          - name: images-volume
            persistentVolumeClaim:
              claimName: images-pvc
          - name: results-volume
            persistentVolumeClaim:
              claimName: results-pvc
          restartPolicy: OnFailure
```

### Performance Considerations for Reconciler Mode

#### Resource Usage

- **CPU**: ~5-10% baseline for file monitoring
- **Memory**: Minimal overhead for state tracking
- **Disk I/O**: State file updates and folder scanning
- **Network**: None (unless integrating with external systems)

#### Scaling Guidelines

| Watch Folder Size | Watch Interval | Batch Size | Expected Latency |
|-------------------|----------------|------------|------------------|
| < 1,000 images    | 10 seconds     | 4-8        | 10-60 seconds   |
| 1,000-10,000     | 30 seconds     | 16-32      | 30-300 seconds  |
| 10,000-100,000   | 60 seconds     | 32-64      | 60-600 seconds  |
| > 100,000        | 300 seconds    | 64-128     | 300+ seconds    |

#### Optimization Tips

1. **State File Management**: Enable cleanup for large datasets
2. **Batch Sizing**: Balance latency vs GPU utilization
3. **File Stability**: Set appropriate `min_file_age` for network storage
4. **Resource Limits**: Use Docker/Kubernetes resource constraints

### Error Handling in Reconciler Mode

#### Graceful Degradation

```yaml
reconciler:
  error_handling:
    max_consecutive_failures: 5        # Stop after N failed cycles
    failure_backoff: 60               # Wait N seconds after failure
    skip_problematic_files: true      # Continue despite individual file errors
    alert_on_persistent_failures: true # Send notifications
```

#### Recovery Mechanisms

- **State File Corruption**: Automatic backup and recovery
- **Processing Failures**: Retry with exponential backoff
- **Resource Exhaustion**: Graceful degradation and alerting
- **Network Issues**: Queue images for retry when connectivity restored

### Command Line Options for Reconciler Mode

```bash
python batch_tester.py [standard options] [reconciler options]

Reconciler Options:
  --reconciler-mode         Enable reconciler pattern
  --watch-interval SECONDS  Override watch interval
  --min-file-age SECONDS    Minimum file age before processing
  --batch-when-count N      Process when N images accumulated
  --batch-when-timeout SEC  Process when timeout reached
  --state-file PATH         Custom state file location
  --no-state-tracking       Disable state persistence
  --process-existing        Process existing images on startup
```

The reconciler pattern transforms the batch tester from a one-time processing tool into a continuous monitoring and processing system, perfect for production environments where images arrive continuously and need immediate processing.

## Docker Deployment

### Build Batch Testing Container

```bash
# Build the container
docker build -f Dockerfile.batch -t fundus-batch-tester:latest .

# Run with GPU support
docker run --rm --gpus all \
  -v ./test-images:/app/test-images \
  -v ./batch_test_results:/app/batch_test_results \
  -v ./trained-models:/app/trained-models \
  -v ./configs:/app/configs \
  fundus-batch-tester:latest
```

### Docker Compose

```bash
# GPU-enabled testing
docker-compose -f docker-compose.batch.yml up batch-tester

# CPU-only testing
docker-compose -f docker-compose.batch.yml up batch-tester-cpu
```

### Container Environment Variables

```bash
# Control GPU visibility
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run with custom configuration
docker run --rm --gpus all \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -v ./configs:/app/configs \
  fundus-batch-tester:latest \
  python batch_tester.py --config configs/custom_config.yaml
```

## VS Code Integration

### Available Tasks

Use `Ctrl+Shift+P` → "Tasks: Run Task" to access:

- **Run Batch Testing** - Standard batch testing
- **Run Batch Testing (CPU Only)** - Force CPU usage
- **Run Batch Testing (Custom Images)** - Interactive folder selection
- **Build Batch Testing Docker Image** - Container building
- **Test Batch Configuration** - Validate configuration files

### Debug Configurations

Use `F5` to start debugging with:

- **Debug: Batch Testing** - Standard debugging with breakpoints
- **Debug: Batch Testing (CPU Only)** - CPU-only debugging
- **Debug: Batch Testing (Small Batch for Debug)** - Use batch size 2 for detailed debugging
- **Debug: Batch Testing (Custom Images)** - Interactive debugging with custom paths

### Keyboard Shortcuts

- `F5` - Start debugging
- `Ctrl+F5` - Run without debugging
- `Shift+F5` - Stop debugging
- `F9` - Toggle breakpoint

## Performance Optimization

### Batch Size Tuning

Choose optimal batch size based on your hardware:

```python
# Memory-limited systems
batch_size = 4  # ~2GB VRAM

# Balanced performance
batch_size = 16  # ~6GB VRAM

# High-performance systems
batch_size = 32  # ~12GB VRAM
```

### Worker Process Configuration

```yaml
batch_processing:
  num_workers: 4      # CPU cores / 2
  prefetch_factor: 2  # 2-4 for SSD, 1-2 for HDD
```

### Memory Optimization

```yaml
performance:
  memory_efficient: true    # Sequential processing
  
output:
  save_processed_images: false  # Reduce I/O overhead
```

### Expected Performance

| Hardware Configuration | Images/sec | Batch Size | Memory Usage |
|------------------------|------------|------------|--------------|
| RTX 4090 (24GB)       | 6.8        | 32         | ~18GB       |
| RTX 3080 (10GB)       | 4.2        | 16         | ~8GB        |
| CPU Only (32 cores)   | 0.8        | 8          | ~4GB        |

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM) Errors

```bash
# Symptoms
RuntimeError: CUDA out of memory

# Solutions
# Reduce batch size
python batch_tester.py --batch-size 8

# Force CPU usage
python batch_tester.py --force-cpu

# Enable memory efficiency
# Edit config: memory_efficient: true
```

#### 2. File Permission Issues

```bash
# Symptoms
PermissionError: [Errno 13] Permission denied

# Solutions
# Check directory permissions
chmod 755 batch_test_results/

# Run with elevated permissions (Windows)
# Run as Administrator
```

#### 3. Model Loading Failures

```bash
# Symptoms
FileNotFoundError: Model file not found

# Solutions
# Verify model paths in classifier_config.yaml
# Check if models exist in trained-models/
ls -la trained-models/

# Update model paths
# Edit configs/classifier_config.yaml
```

#### 4. Ground Truth Format Issues

```bash
# Symptoms
KeyError: 'true_label' not found

# Solutions
# Verify CSV format
head ground_truth.csv
# Should show: image_id,true_label

# Check column names match exactly
# Fix header: image_id,true_label (no spaces)
```

### Performance Issues

#### Slow Processing

```bash
# Check GPU utilization
nvidia-smi

# Monitor CPU usage
htop

# Profile memory usage
python -m memory_profiler batch_tester.py
```

#### Debugging Steps

1. **Enable Debug Mode**:
   ```bash
   python batch_tester.py --batch-size 2 --force-cpu
   ```

2. **Check Configuration**:
   ```bash
   python -c "import yaml; print(yaml.safe_load(open('configs/batch_testing_config.yaml')))"
   ```

3. **Validate Models**:
   ```bash
   python -c "from diabetic_retinopathy_classifier import DiabeticRetinopathyClassifier; DiabeticRetinopathyClassifier('configs/classifier_config.yaml')"
   ```

### Error Recovery

The system is designed to be resilient:

- **Partial Results**: Even if processing fails, results up to that point are saved
- **Resume Capability**: Can restart from where it left off (manual implementation needed)
- **Error Logging**: Detailed error information in failed_images section

## API Reference

### Command Line Interface

```bash
python batch_tester.py [OPTIONS]

Options:
  --config PATH              Configuration file path
  --images-folder PATH       Override images folder
  --output-folder PATH       Override output folder
  --ground-truth PATH        Override ground truth file
  --batch-size INTEGER       Override batch size
  --force-cpu               Force CPU usage
  --help                    Show help message
```

### Configuration Schema

```yaml
# Main configuration structure
input:
  images_folder: str          # Required
  ground_truth_file: str      # Optional
  supported_formats: list     # Default: [".jpg", ".jpeg", ".png"]

output:
  results_folder: str         # Default: "batch_test_results"
  save_processed_images: bool # Default: false
  save_individual_results: bool # Default: true

model:
  preprocessing_config: str   # Required
  classifier_config: str      # Required
  voting_strategy: str        # "soft" or "hard"

batch_processing:
  batch_size: int            # Default: 16
  num_workers: int           # Default: 4
  prefetch_factor: int       # Default: 2

gpu:
  enabled: bool              # Default: true
  use_data_parallel: bool    # Default: true
  device_ids: list           # Default: [] (all GPUs)
  force_cpu: bool            # Default: false

performance:
  log_interval: int          # Default: 10
  save_interval: int         # Default: 50
  memory_efficient: bool     # Default: true

testing:
  confidence_threshold: float # Default: 0.5
  calculate_metrics: bool     # Default: true
  class_names: list          # Required

report:
  include_metadata: bool      # Default: true
  include_confusion_matrix: bool # Default: true
  include_per_class_metrics: bool # Default: true
  include_processing_times: bool # Default: true
  save_failed_images_list: bool # Default: true
```

### Python API

```python
from batch_tester import BatchTester

# Initialize tester
tester = BatchTester('configs/batch_testing_config.yaml')

# Override configuration programmatically
tester.config['batch_processing']['batch_size'] = 32
tester.config['gpu']['force_cpu'] = True

# Run testing
tester.run_batch_testing()

# Access results
print(f"Total processed: {tester.json_writer.total_results}")
print(f"Successful: {tester.json_writer.successful_results}")
```

### Output JSON Schema

```python
# Main result structure
{
  "batch_metadata": {
    "testing_session": {
      "session_id": str,
      "start_time": str,      # ISO format
      "end_time": str,        # ISO format
      "duration_seconds": float
    },
    "configuration": dict,    # Full config
    "system_info": dict,      # Hardware/software info
    "dataset_info": dict,     # Dataset statistics
    "model_info": dict        # Model configuration
  },
  "processing_statistics": {
    "total_batches": int,
    "average_batch_time": float,
    "average_time_per_image": float,
    "total_processing_time": float,
    "min_time_per_image": float,
    "max_time_per_image": float,
    "std_time_per_image": float
  },
  "evaluation_metrics": {
    "overall": {
      "accuracy": float,
      "macro_precision": float,
      "macro_recall": float,
      "macro_f1": float,
      "weighted_precision": float,
      "weighted_recall": float,
      "weighted_f1": float,
      "total_samples": int
    },
    "per_class": {
      "<class_name>": {
        "precision": float,
        "recall": float,
        "f1_score": float,
        "support": int
      }
    },
    "confusion_matrix": list[list[int]],
    "average_confidence": float,
    "confidence_std": float
  },
  "batch_results": [
    {
      "image_id": str,
      "logits": list[float],      # Raw model outputs
      "predicted_label": int,     # Class index
      "predicted_class": str,     # Class name
      "confidence": float,        # Max probability
      "true_label": int,          # Ground truth (if available)
      "processing_times": {
        "preprocessing_ms": float,
        "classification_ms": float,
        "total_ms": float
      },
      "status": str              # "success" or "failed"
    }
  ],
  "failed_images": [             # Optional
    {
      "image_id": str,
      "error": str,
      "timestamp": str
    }
  ]
}
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make your changes and test thoroughly
4. Commit your changes: `git commit -am 'Add new feature'`
5. Push to the branch: `git push origin feature/new-feature`
6. Submit a pull request

## License

This project is part of the fundus image processing ensemble system. Please refer to the main project license for usage terms.

## Support

For issues and questions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review existing issues in the repository
3. Create a new issue with detailed error information and system configuration

---

**Last Updated**: November 2, 2025
**Version**: 1.0.0