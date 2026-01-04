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

# Docker Configuration
DOCKER_IMAGE_NAME = fundus-inference-server
DOCKER_IMAGE_TAG = latest
DOCKER_CONTAINER_NAME = fundus-inference-api
DOCKER_PORT = 5000
DOCKER_NETWORK = fundus-network
DOCKER_REDIS_CONTAINER = fundus-redis
DOCKER_COMPOSE_FILE = build/docker/docker-compose.yml

# Preprocessing specific variables
PREPROCESS_CONFIG = configs\preprocessing_config.yaml
PREPROCESS_INPUT = "./test-images/007-2809-100.jpg"
PREPROCESS_OUTPUT = final_resized_processed_images  # Changed to match config default

# OpenAPI Generator variables
OPENAPI_SPEC = api/ensemble-inference.openapi.yaml
OPENAPI_GENERATOR_IMAGE = openapitools/openapi-generator-cli
GENERATOR_CONFIG = scripts/generator-cfg.yaml
PROJECT_ROOT = $(shell cd)
API_OUTPUT_DIR = api

.PHONY: help install install-dev setup check test server client demo clean format lint docs preprocess preprocess-debug openapi-generate openapi-validate docker-build docker-run docker-stop docker-remove docker-logs

# Default target
help: ## Show this help message
	@echo "Fundus Image Preprocessing System"
	@echo "Available commands:"
	@echo "  install      - Install required dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  setup        - Setup project directories"
	@echo "  preprocess   - Run preprocessing on test image"
	@echo "  server       - Start inference server"
	@echo "  server-debug - Start server in debug mode"
	@echo "  server-redis - Start server with Redis caching enabled"
	@echo "  redis-start  - Start Redis server in Docker"
	@echo "  redis-stop   - Stop Redis server"
	@echo "  redis-remove - Remove Redis container"
	@echo "  redis-restart- Restart Redis server"
	@echo "  redis-logs   - Show Redis server logs"
	@echo "  redis-cli    - Connect to Redis CLI"
	@echo "  redis-info   - Get Redis server info"
	@echo "  redis-flush  - Flush all Redis cache data"
	@echo "  cache-stats  - Get cache statistics from server"
	@echo "  cache-health - Check cache health status"
	@echo "  cache-clear  - Clear server cache via API"
	@echo "  client-info  - Get server information"
	@echo "  client-health- Check server health"
	@echo "  test-config  - Test configuration validity"
	@echo "  openapi-validate - Validate OpenAPI specification"
	@echo "  openapi-generate - Generate client SDK from OpenAPI spec"
	@echo "  openapi-generate-python - Generate Python client SDK"
	@echo "  openapi-generate-docs - Generate HTML documentation"
	@echo "  openapi-generate-all - Generate all SDKs and documentation"
	@echo "  compose-up       - Start services using Docker Compose (RECOMMENDED)"
	@echo "  compose-down     - Stop services using Docker Compose"
	@echo "  compose-logs     - View Docker Compose logs"
	@echo "  compose-build    - Build Docker image with Compose"
	@echo "  docker-build     - Build Docker image (legacy)"
	@echo "  docker-run       - Run Docker container (legacy)"
	@echo "  docker-stop      - Stop Docker container (legacy)"
	@echo "  docker-remove    - Remove Docker container (legacy)"
	@echo "  docker-logs      - Show Docker container logs (legacy)"
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

server-redis: ## Start server with Redis caching enabled
	@echo "Starting inference server with Redis caching..."
	$(PYTHON) fundus_inference_server.py --preprocessing-config configs\preprocessing_config.yaml --classifier-config configs\classifier_config.yaml --host $(SERVER_HOST) --port $(SERVER_PORT) --redis-enabled true

# Redis Cache Operations
redis-start: ## Start Redis server in Docker
	@echo "Starting Redis server..."
	docker run --name redis-server -d -p 6379:6379 redis
	@echo "Redis server started on port 6379"

redis-stop: ## Stop Redis server
	@echo "Stopping Redis server..."
	docker stop redis-server
	@echo "Redis server stopped"

redis-remove: ## Remove Redis container
	@echo "Removing Redis container..."
	docker rm redis-server
	@echo "Redis container removed"

redis-restart: redis-stop redis-remove redis-start ## Restart Redis server
	@echo "Redis server restarted"

redis-logs: ## Show Redis server logs
	@echo "Showing Redis logs..."
	docker logs redis-server

redis-cli: ## Connect to Redis CLI
	@echo "Connecting to Redis CLI..."
	docker exec -it redis-server redis-cli

redis-info: ## Get Redis server info
	@echo "Getting Redis server info..."
	docker exec -it redis-server redis-cli INFO

redis-flush: ## Flush all Redis cache data
	@echo "Flushing all Redis data..."
	docker exec -it redis-server redis-cli FLUSHALL
	@echo "Redis cache cleared"

cache-stats: ## Get cache statistics from server
	@echo "Getting cache statistics..."
	curl -s http://$(SERVER_HOST):$(SERVER_PORT)/cache/stats | python -m json.tool

cache-health: ## Check cache health status
	@echo "Checking cache health..."
	curl -s http://$(SERVER_HOST):$(SERVER_PORT)/cache/health | python -m json.tool

cache-clear: ## Clear server cache via API
	@echo "Clearing server cache..."
	curl -X POST -s http://$(SERVER_HOST):$(SERVER_PORT)/cache/clear | python -m json.tool

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

# OpenAPI Generator Operations
openapi-validate: ## Validate OpenAPI specification
	@echo "Validating OpenAPI specification..."
	docker run --rm -v "$(PROJECT_ROOT):/local" $(OPENAPI_GENERATOR_IMAGE) validate -i /local/$(OPENAPI_SPEC)
	@echo "OpenAPI specification is valid!"

# Docker Compose Operations (Recommended)
compose-build: ## Build Docker image with Docker Compose
	@echo "Building Docker Compose services..."
	docker-compose -f $(DOCKER_COMPOSE_FILE) build
	@echo "Build completed!"

compose-up: ## Start services using Docker Compose (Redis + Inference Server)
	@echo "Cleaning up old containers..."
	@docker rm -f fundus-redis fundus-inference-api >nul 2>&1 || true
	@echo "Starting services with Docker Compose..."
	docker-compose -f $(DOCKER_COMPOSE_FILE) up -d
	@echo "Services started!"
	@echo "  API: http://localhost:5000"
	@echo "  Redis: localhost:6379"
	@echo ""
	@echo "View logs: make compose-logs"

compose-down: ## Stop services using Docker Compose
	@echo "Stopping services..."
	docker-compose -f $(DOCKER_COMPOSE_FILE) down
	@echo "Services stopped"

compose-logs: ## View Docker Compose logs
	@echo "Showing Docker Compose logs..."
	docker-compose -f $(DOCKER_COMPOSE_FILE) logs -f

compose-restart: compose-down compose-up ## Restart services with Docker Compose

compose-clean: compose-down ## Clean Docker Compose resources
	@echo "Cleaning Docker Compose volumes..."
	docker-compose -f $(DOCKER_COMPOSE_FILE) down -v
	@echo "Cleanup completed"

# Legacy Docker Operations (Use compose-* instead)
docker-build: ## Build Docker image for inference server
	@echo "Building Docker image: $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)"
	docker build -t $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG) -f build/docker/inference-server/Dockerfile .
	@echo "Docker image built successfully!"
	@echo "Note: Use 'make compose-up' for full stack with Redis"

docker-run: ## Run Docker container
	@echo "Removing old inference server container if exists..."
	@docker rm -f $(DOCKER_CONTAINER_NAME) >nul 2>&1 || echo "No old container to remove"
	@echo "Starting Docker container: $(DOCKER_CONTAINER_NAME)..."
	docker run -d --name $(DOCKER_CONTAINER_NAME) -p $(DOCKER_PORT):5000 \
		-e REDIS_HOST=$(DOCKER_REDIS_CONTAINER) \
		$(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)
	@echo "Container running on http://localhost:$(DOCKER_PORT)"

docker-stop: ## Stop Docker container
	@echo "Stopping Docker container..."
	docker stop $(DOCKER_CONTAINER_NAME) 2>/dev/null || true
	@echo "Container stopped"

docker-redis-stop: ## Stop Redis container
	@echo "Stopping Redis container..."
	docker stop $(DOCKER_REDIS_CONTAINER) 2>/dev/null || true
	@echo "Redis stopped"

docker-remove: docker-stop ## Remove Docker container
	@echo "Removing Docker container..."
	docker rm $(DOCKER_CONTAINER_NAME) 2>/dev/null || true
	@echo "Container removed"

docker-redis-remove: docker-redis-stop ## Remove Redis container
	@echo "Removing Redis container..."
	docker rm $(DOCKER_REDIS_CONTAINER) 2>/dev/null || true
	@echo "Redis removed"

docker-logs: ## Show Docker container logs
	@echo "Showing Docker container logs..."
	docker logs -f $(DOCKER_CONTAINER_NAME)

docker-redis-logs: ## Show Redis logs
	@echo "Showing Redis logs..."
	docker logs -f $(DOCKER_REDIS_CONTAINER)

docker-clean: docker-remove docker-redis-remove ## Clean Docker images and containers
	@echo "Removing Docker image..."
	docker rmi $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG) 2>/dev/null || true
	@echo "Docker cleanup completed"

openapi-generate: ## Generate client SDK from OpenAPI spec (using generator config)
	@echo "Generating client SDK from OpenAPI specification..."
	@if not exist "$(API_OUTPUT_DIR)" mkdir "$(API_OUTPUT_DIR)"
	docker run --rm -v "$(PROJECT_ROOT):/local" $(OPENAPI_GENERATOR_IMAGE) generate -c /local/$(GENERATOR_CONFIG)
	@echo "Client SDK generated successfully!"

openapi-generate-docs: ## Generate HTML documentation from OpenAPI spec
	@echo "Generating HTML documentation..."
	@if not exist "$(API_OUTPUT_DIR)\docs" mkdir "$(API_OUTPUT_DIR)\docs"
	docker run --rm -v "$(PROJECT_ROOT):/local" $(OPENAPI_GENERATOR_IMAGE) generate \
		-i /local/$(OPENAPI_SPEC) \
		-g html2 \
		-o /local/$(API_OUTPUT_DIR)/docs
	@echo "HTML documentation generated in $(API_OUTPUT_DIR)/docs"

openapi-generate-markdown: ## Generate Markdown documentation from OpenAPI spec
	@echo "Generating Markdown documentation..."
	@if not exist "$(API_OUTPUT_DIR)\markdown" mkdir "$(API_OUTPUT_DIR)\markdown"
	docker run --rm -v "$(PROJECT_ROOT):/local" $(OPENAPI_GENERATOR_IMAGE) generate \
		-i /local/$(OPENAPI_SPEC) \
		-g markdown \
		-o /local/$(API_OUTPUT_DIR)/markdown
	@echo "Markdown documentation generated in $(API_OUTPUT_DIR)/markdown"

openapi-generate-all: openapi-validate openapi-generate-python openapi-generate-typescript openapi-generate-docs ## Generate all SDKs and documentation
	@echo "All client SDKs and documentation generated successfully!"

openapi-list-generators: ## List all available OpenAPI generators
	@echo "Available OpenAPI generators:"
	docker run --rm $(OPENAPI_GENERATOR_IMAGE) list

openapi-clean: ## Clean generated API artifacts
	@echo "Cleaning generated API artifacts..."
	@if exist "$(API_OUTPUT_DIR)" rmdir /s /q "$(API_OUTPUT_DIR)"
	@echo "Generated API artifacts cleaned!"

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

clean-all: clean clean-outputs clean-logs openapi-clean ## Clean everything
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
