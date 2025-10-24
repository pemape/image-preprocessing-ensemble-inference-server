# Fundus Image Preprocessing System Makefile (Windows Compatible)
# Based on: "Ensemble of pre-processing techniques with CNN for diabetic retinopathy detection"
# Paper: https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.12987

# Configuration
PYTHON = python
PIP = pip
CONFIG_FILE = preprocessing_config.yaml
SERVER_HOST = 0.0.0.0
SERVER_PORT = 8080
TEST_IMAGE = test_image.jpg
OUTPUT_DIR = ./output
LOG_LEVEL = INFO

# Preprocessing specific variables
PREPROCESS_CONFIG = configs\preprocessing_config.yaml
PREPROCESS_INPUT = "D:\FEI STU\ing\2roc\DATABASE\Aptos\train_images\00a8624548a9.png"
PREPROCESS_OUTPUT = ./processed_images  # Changed to match config default

.PHONY: help install install-dev setup check test server client demo clean format lint docs preprocess preprocess-debug

# Default target
help: ## Show this help message
	@echo "Fundus Image Preprocessing System"
	@echo "Available commands:"
	@echo "  install      - Install required dependencies"
	@echo "  install-dev  - Install development dependencies"  
	@echo "  setup        - Setup project directories"
	@echo "  preprocess   - Run preprocessing on test image"
	@echo "  preprocess-batch   - Run preprocessing on test image batch"
	@echo "  server       - Start inference server"
	@echo "  server-debug - Start server in debug mode"
	@echo "  client-info  - Get server information"
	@echo "  client-health- Check server health"
	@echo "  test-config  - Test configuration validity"
	@echo "  format       - Format code with black"
	@echo "  lint         - Run code linting"
	@echo "  clean        - Clean temporary files"
	@echo "  version      - Show version information"

# Installation and Setup
install: ## Install required dependencies
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt
	@echo "Dependencies installed successfully!"

install-dev: ## Install development dependencies
	@echo "Installing development dependencies..."
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov black flake8 mypy jupyter
	@echo "Development dependencies installed successfully!"

setup: install ## Setup the project
	@echo "Setting up project..."
	@if not exist "$(OUTPUT_DIR)" mkdir "$(OUTPUT_DIR)"
	@if not exist "debug" mkdir "debug"
	@if not exist "logs" mkdir "logs"
	@if not exist "client_output" mkdir "client_output"
	@echo "Project setup completed!"

# Testing and Validation
test-config: ## Test configuration file validity
	@echo "Testing configuration file..."
	$(PYTHON) -c "import yaml; yaml.safe_load(open('$(CONFIG_FILE)', 'r')); print('Config file is valid')"
	@echo "Configuration file is valid!"

check: format lint test-config ## Run all checks
	@echo "All checks completed!"

# Server Operations
server: ## Start the inference server
	@echo "Starting inference server on $(SERVER_HOST):$(SERVER_PORT)..."
	$(PYTHON) fundus_inference_server.py --config $(CONFIG_FILE) --host $(SERVER_HOST) --port $(SERVER_PORT) --log-level $(LOG_LEVEL)

server-debug: ## Start server in debug mode
	@echo "Starting inference server in debug mode..."
	$(PYTHON) fundus_inference_server.py --config $(CONFIG_FILE) --host $(SERVER_HOST) --port $(SERVER_PORT) --debug --log-level DEBUG

# Client Operations  
client-info: ## Get server information
	@echo "Getting server information..."
	$(PYTHON) client.py --server http://$(SERVER_HOST):$(SERVER_PORT) --info

client-config: ## Get server configuration
	@echo "Getting server configuration..."
	$(PYTHON) client.py --server http://$(SERVER_HOST):$(SERVER_PORT) --config

client-stats: ## Get server statistics
	@echo "Getting server statistics..."
	$(PYTHON) client.py --server http://$(SERVER_HOST):$(SERVER_PORT) --stats

client-health: ## Check server health
	@echo "Checking server health..."
	$(PYTHON) client.py --server http://$(SERVER_HOST):$(SERVER_PORT) --health

client-process: ## Process test image
	@echo "Processing image $(TEST_IMAGE)..."
	$(PYTHON) client.py --server http://$(SERVER_HOST):$(SERVER_PORT) --image $(TEST_IMAGE) --output $(OUTPUT_DIR)

# Code Quality
format: ## Format code using black
	@echo "Formatting code..."
	black *.py
	@echo "Code formatted!"

format-check: ## Check code formatting
	@echo "Checking code formatting..."
	black --check *.py

lint: ## Run linting with flake8
	@echo "Running linter..."
	flake8 *.py --max-line-length=100 --ignore=E203,W503

# Documentation
docs: ## Generate documentation
	@echo "Generating documentation..."
	@if not exist "docs" mkdir "docs"
	$(PYTHON) -c "import fundus_preprocessor; help(fundus_preprocessor.FundusPreprocessor)" > docs/preprocessor_help.txt
	$(PYTHON) -c "import fundus_inference_server; help(fundus_inference_server.FundusInferenceServer)" > docs/server_help.txt
	@echo "Documentation generated in docs/"

# Benchmarking
benchmark: ## Run performance benchmarks
	@echo "Running performance benchmarks..."
	$(PYTHON) -c "import time; import numpy as np; from fundus_preprocessor import FundusPreprocessor; test_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8); preprocessor = FundusPreprocessor('$(CONFIG_FILE)'); start_time = time.time(); result = preprocessor.process_image(test_image, 'benchmark'); end_time = time.time(); print(f'Processing time: {end_time - start_time:.3f} seconds'); print(f'Variants: {len(result)}')"

# Cleanup
clean: ## Clean temporary files
	@echo "Cleaning temporary files..."
	@if exist "__pycache__" rmdir /s /q "__pycache__"
	@if exist "*.pyc" del /q "*.pyc"
	@if exist ".pytest_cache" rmdir /s /q ".pytest_cache"
	@if exist "htmlcov" rmdir /s /q "htmlcov"
	@if exist ".coverage" del /q ".coverage"
	@if exist "$(OUTPUT_DIR)" rmdir /s /q "$(OUTPUT_DIR)"
	@if exist $(PREPROCESS_OUTPUT) rmdir /s /q $(PREPROCESS_OUTPUT)
	@if exist "debug" rmdir /s /q "debug"
	@if exist "client_output" rmdir /s /q "client_output"
	@echo "Cleaned temporary files!"

clean-outputs: ## Clean all output directories
	@echo "Cleaning output directories..."
	@if exist "processed_images" rmdir /s /q "processed_images"
	@if exist "direct_output" rmdir /s /q "direct_output"
	@if exist "$(PREPROCESS_OUTPUT)" rmdir /s /q "$(PREPROCESS_OUTPUT)"
	@echo "Cleaned output directories!"

clean-logs: ## Clean log files
	@echo "Cleaning log files..."
	@if exist "logs" rmdir /s /q "logs"
	@echo "Cleaned log files!"

clean-all: clean clean-outputs clean-logs ## Clean everything
	@echo "Everything cleaned!"

# Development helpers
dev-setup: install-dev setup ## Complete development setup
	@echo "Development environment ready!"

# Requirements management
requirements: ## Generate requirements.txt
	@echo "Generating requirements.txt..."
	$(PIP) freeze > requirements.txt
	@echo "Requirements file updated!"

requirements-check: ## Check requirements
	@echo "Checking requirements..."
	$(PIP) check

# Version and info
version: ## Show version information
	@echo "Fundus Image Preprocessing System"
	@echo "Version: 1.0.0" 
	@$(PYTHON) --version
	@echo "Paper: https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.12987"

info: version ## Show system information
	@echo ""
	@echo "Configuration:"
	@echo "  Config file: $(CONFIG_FILE)"
	@echo "  Server: $(SERVER_HOST):$(SERVER_PORT)"
	@echo "  Output dir: $(OUTPUT_DIR)"
	@echo "  Log level: $(LOG_LEVEL)"

# Simple demo
demo: ## Run simple demo
	@echo "Starting simple demo..."
	@echo "1. Testing configuration..."
	$(MAKE) test-config
	@echo "2. Checking if server files exist..."
	@if exist "fundus_inference_server.py" echo "Server file found" else echo "Server file missing"
	@if exist "client.py" echo "Client file found" else echo "Client file missing"
	@echo "Demo completed! Use 'make server' to start the server."

# Preprocessing Operations
preprocess: ## Run fundus preprocessing on test image
	@echo "Running fundus preprocessing..."
	@echo "Input: $(PREPROCESS_INPUT)"
	@echo "Config: $(PREPROCESS_CONFIG)"
	$(PYTHON) fundus_preprocessor.py --config $(PREPROCESS_CONFIG) --input $(PREPROCESS_INPUT)

preprocess-batch: ## Run preprocessing on multiple images (use BATCH_DIR=path)
	@echo "Running batch preprocessing..."
	@echo "Batch directory: $(BATCH_DIR)"
	$(PYTHON) fundus_preprocessor.py --config $(PREPROCESS_CONFIG) --input "$(BATCH_DIR)" --batch
	@echo "Batch preprocessing completed!"
