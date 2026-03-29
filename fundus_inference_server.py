"""
Fundus Image Processing and Classification Server
Modular system supporting:
1. Ensemble preprocessing (5 variants)
2. Diabetic retinopathy classification (Xception/EfficientNetB4 with ensemble voting)
"""

import os
import time
import json
import base64
import logging
import threading
from collections import defaultdict
from io import BytesIO
from typing import Dict, List, Optional, Union
import cv2
import numpy as np
from flask import Flask, request, jsonify
from pathlib import Path
import argparse
import uuid
from datetime import datetime, timezone
from concurrent.futures import Future, TimeoutError as FuturesTimeoutError

# Import generated OpenAPI models
from ensemble_inference.models.process_result import ProcessResult
from ensemble_inference.models.operation_status import OperationStatus
from ensemble_inference.models.image_properties import ImageProperties
from ensemble_inference.models.image_processing_times import ImageProcessingTimes
from ensemble_inference.models.classification_result import ClassificationResult
from ensemble_inference.models.class_probability import ClassProbability
from ensemble_inference.models.voting_strategy_enum import VotingStrategyEnum
from ensemble_inference.models.preprocess_response import PreprocessResponse
from ensemble_inference.models.classify_response import ClassifyResponse
from ensemble_inference.models.preprocessed_images import PreprocessedImages
from ensemble_inference.models.batch_processing_metrics import BatchProcessingMetrics
from ensemble_inference.models.model_configuration import ModelConfiguration
from ensemble_inference.models.preprocess_response_metadata import (
    PreprocessResponseMetadata,
)
from ensemble_inference.models.process_metadata import ProcessMetadata
from ensemble_inference.models.process_response import ProcessResponse
from ensemble_inference.models.version_info import VersionInfo

# Import our modules
from fundus_preprocessor import FundusPreprocessor
from diabetic_retinopathy_classifier import DiabeticRetinopathyClassifier
from redis_cache_manager import RedisCacheManager


class DynamicBatchInferenceManager:
    """Queue-based dynamic batching manager for classifier inference."""

    def __init__(
        self,
        classifier: DiabeticRetinopathyClassifier,
        logger: logging.Logger,
        enabled: bool = False,
        max_batch_size: int = 8,
        max_wait_ms: int = 15,
        max_queue_size: int = 256,
    ):
        self.classifier = classifier
        self.logger = logger
        self.enabled = enabled
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_wait_ms = max(1, int(max_wait_ms))
        self.max_queue_size = max(1, int(max_queue_size))

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._pending: List[Dict] = []
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None

        if self.enabled:
            self._worker = threading.Thread(
                target=self._run_worker,
                name="dynamic-batch-worker",
                daemon=True,
            )
            self._worker.start()
            self.logger.info(
                "Dynamic batching enabled (max_batch_size=%s, max_wait_ms=%s, max_queue_size=%s)",
                self.max_batch_size,
                self.max_wait_ms,
                self.max_queue_size,
            )

    def stop(self, timeout: float = 2.0) -> None:
        """Stop batch worker gracefully."""
        if not self.enabled:
            return

        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()

        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=timeout)

    def submit(
        self,
        preprocessed_images: Dict[str, np.ndarray],
        voting_strategy: str,
        timeout_seconds: float = 15.0,
    ) -> Dict:
        """Submit one inference request to the dynamic batch queue."""
        if not self.enabled:
            return self.classifier.classify(
                preprocessed_images, voting_strategy=voting_strategy
            )

        future: Future = Future()
        item = {
            "images": preprocessed_images,
            "voting_strategy": voting_strategy,
            "future": future,
            "enqueued_at": time.monotonic(),
        }

        with self._condition:
            if len(self._pending) >= self.max_queue_size:
                raise RuntimeError("Dynamic batch queue is full")

            self._pending.append(item)
            self._condition.notify()

        return future.result(timeout=timeout_seconds)

    def _build_batch_metrics(
        self,
        enqueued_at: float,
        execution_started_at: float,
        execution_finished_at: float,
        batch_size: int,
        batch_id: str,
    ) -> Dict:
        """Build per-request batch timing metrics in milliseconds."""
        return {
            "batch_id": batch_id,
            "batch_wait_ms": round(
                max(0.0, (execution_started_at - enqueued_at) * 1000.0), 2
            ),
            "batch_execution_ms": round(
                max(0.0, (execution_finished_at - execution_started_at) * 1000.0),
                2,
            ),
            "batch_total_ms": round(
                max(0.0, (execution_finished_at - enqueued_at) * 1000.0), 2
            ),
            "batch_size": batch_size,
        }

    def _run_worker(self) -> None:
        while not self._stop_event.is_set():
            batch = self._collect_batch()
            if not batch:
                continue

            self._execute_batch(batch)

    def _collect_batch(self) -> List[Dict]:
        with self._condition:
            while not self._pending and not self._stop_event.is_set():
                self._condition.wait(timeout=0.25)

            if self._stop_event.is_set() and not self._pending:
                return []

            if not self._pending:
                return []

            batch = [self._pending.pop(0)]
            deadline = batch[0]["enqueued_at"] + (self.max_wait_ms / 1000.0)

            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break

                if not self._pending:
                    self._condition.wait(timeout=remaining)
                    continue

                batch.append(self._pending.pop(0))

            return batch

    def _execute_batch(self, batch: List[Dict]) -> None:
        grouped: Dict[str, List[Dict]] = defaultdict(list)
        for item in batch:
            grouped[item["voting_strategy"]].append(item)

        for voting_strategy, items in grouped.items():
            try:
                execution_started_at = time.monotonic()
                batch_id = f"BATCH-{uuid.uuid4().hex[:12]}"
                images_batch = [item["images"] for item in items]
                results = self.classifier.classify_batch(
                    images_batch,
                    voting_strategy=voting_strategy,
                )
                execution_finished_at = time.monotonic()

                for item, result in zip(items, results):
                    if not item["future"].done():
                        item["future"].set_result(
                            {
                                "classification_result": result,
                                "batch_metrics": self._build_batch_metrics(
                                    enqueued_at=item["enqueued_at"],
                                    execution_started_at=execution_started_at,
                                    execution_finished_at=execution_finished_at,
                                    batch_size=len(items),
                                    batch_id=batch_id,
                                ),
                            }
                        )
            except Exception as e:
                self.logger.error(
                    "Dynamic batch inference failed (strategy=%s): %s",
                    voting_strategy,
                    e,
                )
                for item in items:
                    if not item["future"].done():
                        item["future"].set_exception(e)


class FundusInferenceServer:
    """Main server class integrating preprocessing and classification."""

    def __init__(
        self,
        preprocessing_config: str,
        classifier_config: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = 5000,
        debug: bool = False,
        redis_config: Optional[Dict] = None,
    ):
        """
        Initialize the inference server.

        Args:
            preprocessing_config: Path to preprocessing configuration
            classifier_config: Path to classifier configuration (optional)
            host: Server host
            port: Server port
            debug: Debug mode
            redis_config: Redis configuration dictionary (optional)
        """
        self.host = host
        self.port = port
        self.debug = debug

        # Setup logging
        self.logger = self._setup_logging()

        # Initialize modules
        self.preprocessor = FundusPreprocessor(preprocessing_config)
        self.classifier = None

        if classifier_config and os.path.exists(classifier_config):
            try:
                self.classifier = DiabeticRetinopathyClassifier(classifier_config)
                self.logger.info("Classifier module loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load classifier: {e}")
                self.classifier = None
        else:
            self.logger.info("Running in preprocessing-only mode")

        # Initialize Redis cache (optional)
        self.cache = self._initialize_cache(redis_config or {})

        # Initialize dynamic batch manager (optional)
        self.dynamic_batcher = self._initialize_dynamic_batcher()

        # Setup Flask app
        self.app = self._create_flask_app()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("FundusInferenceServer")

        level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(
            level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        return logger

    def _initialize_cache(self, redis_config: Dict) -> RedisCacheManager:
        """
        Initialize Redis cache manager.

        Args:
            redis_config: Redis configuration dictionary

        Returns:
            RedisCacheManager instance
        """
        return RedisCacheManager(
            enabled=redis_config.get("enabled", False),
            host=redis_config.get("host", "localhost"),
            port=redis_config.get("port", 6379),
            db=redis_config.get("db", 0),
            password=redis_config.get("password"),
            ttl=redis_config.get("ttl", 86400),
            key_prefix=redis_config.get("key_prefix", "fundus_inference"),
        )

    def _initialize_dynamic_batcher(self) -> DynamicBatchInferenceManager:
        """Initialize dynamic batch manager from classifier config."""
        if not self.classifier:
            return DynamicBatchInferenceManager(
                classifier=None,
                logger=self.logger,
                enabled=False,
            )

        batching_cfg = self.classifier.config.get("dynamic_batching", {})
        enabled = bool(batching_cfg.get("enabled", False))

        return DynamicBatchInferenceManager(
            classifier=self.classifier,
            logger=self.logger,
            enabled=enabled,
            max_batch_size=batching_cfg.get("max_batch_size", 8),
            max_wait_ms=batching_cfg.get("max_wait_ms", 15),
            max_queue_size=batching_cfg.get("max_queue_size", 256),
        )

    # ========== Response Schema Helpers ==========

    def _get_api_version(self) -> str:
        """Get API version."""
        return "1.0.0"

    def _get_model_version(self) -> str:
        """Get model version identifier."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        return f"{timestamp}-v2.1"

    def _generate_request_id(self, prefix: str = "REQ") -> str:
        """Generate unique request ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"{prefix}-{timestamp}-{unique_id}"

    def _get_utc_timestamp(self) -> str:
        """Get current UTC timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _get_diagnosis_certainty(self, confidence: float) -> str:
        """
        Determine diagnosis certainty level based on confidence score.

        Args:
            confidence: Confidence score (0-1)

        Returns:
            Certainty level: HIGH, MEDIUM, or LOW
        """
        if confidence >= 0.75:
            return "HIGH"
        elif confidence >= 0.50:
            return "MEDIUM"
        else:
            return "LOW"

    def _get_class_mapping(self) -> Dict[int, Dict[str, str]]:
        """
        Get class ID to label mapping.

        Returns:
            Dictionary mapping class index to ID and full label
        """
        return {
            0: {"id": "DR_0", "label": "No Diabetic Retinopathy"},
            1: {"id": "DR_1", "label": "Mild Non-Proliferative DR"},
            2: {"id": "DR_2", "label": "Moderate Non-Proliferative DR"},
            3: {"id": "DR_3", "label": "Severe Non-Proliferative DR"},
            4: {"id": "DR_4", "label": "Proliferative DR"},
        }

    def _get_label_to_classifier_mapping(self) -> Dict[str, str]:
        """
        Get mapping from full labels to short classifier labels.

        Returns:
            Dictionary mapping full label to short label
        """
        return {
            "No Diabetic Retinopathy": "No DR",
            "Mild Non-Proliferative DR": "Mild DR",
            "Moderate Non-Proliferative DR": "Moderate DR",
            "Severe Non-Proliferative DR": "Severe DR",
            "Proliferative DR": "Proliferative DR",
        }

    def _convert_cached_response_to_classifier_format(
        self, cached_response: Dict
    ) -> Dict:
        """
        Convert cached structured response back to classifier format.

        Args:
            cached_response: Cached response in structured format

        Returns:
            Dictionary in classifier format
        """
        if "classification" not in cached_response:
            return {}

        cached_result = cached_response["classification"]
        class_probs = cached_result["class_probabilities"]
        label_map = self._get_label_to_classifier_mapping()

        # Get the predicted class from cached response
        predicted_class_label = cached_result["predicted_class_label"]
        predicted_class_short = label_map.get(
            predicted_class_label, predicted_class_label
        )

        prediction_dict = {
            "predicted_class": predicted_class_short,
            "confidence": cached_result["confidence_score"],
            "model_info": cached_response.get("metadata", {}).get(
                "model_configuration", {}
            ),
        }

        # Add all class probabilities with their original labels
        for cp in class_probs:
            classifier_label = label_map.get(cp["label"], cp["label"])
            prediction_dict[classifier_label] = cp["probability"]

        return prediction_dict

    def _build_classification_result(
        self,
        classification_results: Dict,
    ) -> ClassificationResult:
        """
        Build structured classification result following the API schema.

        Args:
            classification_results: Raw classification results from classifier
            image_shape: Original image shape (height, width)
            processed_shape: Processed image shape (height, width)

        Returns:
            Structured result dictionary
        """
        class_mapping = self._get_class_mapping()

        # Extract class names and probabilities
        class_names = [
            "No DR",
            "Mild DR",
            "Moderate DR",
            "Severe DR",
            "Proliferative DR",
        ]

        # Build class probabilities array
        class_probabilities = []
        for idx, name in enumerate(class_names):
            prob = classification_results.get(name, 0.0)
            class_info = class_mapping[idx]
            class_probabilities.append(
                {
                    "id": class_info["id"],
                    "label": class_info["label"],
                    "probability": round(prob, 6),
                }
            )

        # Find predicted class
        predicted_class_name = classification_results.get("predicted_class", "No DR")
        predicted_idx = (
            class_names.index(predicted_class_name)
            if predicted_class_name in class_names
            else 0
        )
        predicted_class_info = class_mapping[predicted_idx]

        confidence = classification_results.get("confidence", 0.0)

        result = ClassificationResult(
            predicted_class_id=predicted_class_info["id"],
            predicted_class_label=predicted_class_info["label"],
            confidence_score=round(confidence, 6),
            diagnosis_certainty=self._get_diagnosis_certainty(confidence),
            class_probabilities=class_probabilities,
        )
        return result

    def _build_single_process_response(
        self,
        image_shape: tuple,
        processed_shape: tuple,
        preprocessing_time: float,
        classification_time: float,
        classification_results: Optional[Dict] = None,
        include_images: bool = False,
        variants: Optional[Dict] = None,
        image_filename: Optional[str] = None,
    ) -> Dict:
        """
        Build structured response for single image processing.

        Args:
            image_shape: Original image shape (height, width)
            processed_shape: Processed image shape (height, width)
            preprocessing_time: Time spent on preprocessing (seconds)
            classification_time: Time spent on classification (seconds)
            classification_results: Classification results (optional)
            include_images: Whether to include base64 encoded images
            variants: Preprocessed image variants (optional)
            image_filename: Original image filename (optional)

        Returns:
            Structured response dictionary
        """
        total_time = preprocessing_time + classification_time

        response = {
            "status": "SUCCESS",
            "request_id": self._generate_request_id(),
            "timestamp_utc": self._get_utc_timestamp(),
            "image_name": image_filename,
            "processing_times": {
                "preprocessing_ms": round(preprocessing_time * 1000, 2),
                "inference_ms": round(classification_time * 1000, 2),
                "total_ms": round(total_time * 1000, 2),
                "per_image_ms": round(
                    total_time * 1000, 2
                ),  # Same as total for single image
            },
            "metadata": {
                "image_properties": {
                    "original_width_px": image_shape[1],
                    "original_height_px": image_shape[0],
                    "processed_width_px": processed_shape[1],
                    "processed_height_px": processed_shape[0],
                },
                "version_info": {
                    "api_version": self._get_api_version(),
                    "model_version": self._get_model_version(),
                },
            },
        }

        # Add model configuration if classifier is available
        if self.classifier and classification_results:
            model_info = classification_results.get("model_info", {})
            architectures = model_info.get("architectures", [])
            datasets = model_info.get("datasets", [])
            voting_strategy = model_info.get("voting_strategy", "soft")

            response["metadata"]["model_configuration"] = {
                "model_architecture": (
                    architectures[0] if architectures else "EfficientNetB4"
                ),
                "ensemble_size": model_info.get("ensemble_size", 1),
                "trained_datasets": (
                    datasets if datasets else ["APTOS-2019-DR-Classification"]
                ),
                "voting_strategy": f"{voting_strategy.capitalize()}-Voting",
            }

            # Add classification result
            response["classification"] = self._build_classification_result(
                classification_results,
            ).to_dict()

        # Optionally include preprocessed images
        if include_images and variants:
            response["preprocessed_images"] = {}
            for variant_name, variant_image in variants.items():
                encoded_image = self._encode_image(variant_image)
                response["preprocessed_images"][variant_name] = encoded_image

        return response

    def _get_voting_strategy_values(self) -> List[str]:
        values = [v for k, v in vars(VotingStrategyEnum).items() if k.isupper()]
        return values

    def _create_flask_app(self) -> Flask:
        """Create and configure Flask application."""
        app = Flask(__name__)

        # Set max content length from environment variable or use default 8MB
        max_content_mb = int(os.getenv("MAX_CONTENT_LENGTH_MB", "8"))
        app.config["MAX_CONTENT_LENGTH"] = (
            max_content_mb * 1024 * 1024
        )  # Convert MB to bytes

        # Register routes
        app.route("/health", methods=["GET"])(self.health_check)
        app.route("/info", methods=["GET"])(self.get_info)
        app.route("/config", methods=["GET"])(self.get_config)
        app.route("/models", methods=["GET"])(self.get_models)

        # Cache management endpoints
        app.route("/cache/stats", methods=["GET"])(self.get_cache_stats)
        app.route("/cache/health", methods=["GET"])(self.get_cache_health)
        app.route("/cache/clear", methods=["POST"])(self.clear_cache)

        # Processing endpoints
        app.route("/preprocess", methods=["POST"])(self.preprocess_image)
        app.route("/classify", methods=["POST"])(self.classify_image)
        app.route("/process", methods=["POST"])(
            self.full_process
        )  # Preprocess + Classify

        return app

    def health_check(self):
        """Health check endpoint."""
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "modules": {
                "preprocessor": True,
                "classifier": self.classifier is not None,
            },
        }
        return jsonify(health_status)

    def get_info(self):
        """Get server information."""
        info = {
            "name": "Fundus Image Processing and Classification Server",
            "version": "1.0.0",
            "modules": {
                "preprocessing": {
                    "enabled": True,
                    "variants": 5,
                    "clipping_methods": 4,
                },
                "classification": {
                    "enabled": self.classifier is not None,
                    "models": (
                        len(self.classifier.ensemble.models) if self.classifier else 0
                    ),
                    "voting_strategy": (
                        self.classifier.ensemble.voting_strategy
                        if self.classifier
                        else None
                    ),
                    "dynamic_batching": {
                        "enabled": self.dynamic_batcher.enabled,
                        "max_batch_size": self.dynamic_batcher.max_batch_size,
                        "max_wait_ms": self.dynamic_batcher.max_wait_ms,
                        "max_queue_size": self.dynamic_batcher.max_queue_size,
                    },
                },
            },
            "endpoints": {
                "GET /health": "Health check",
                "GET /info": "Server information",
                "GET /config": "Configuration details",
                "GET /models": "Model information",
                "POST /preprocess": "Image preprocessing only",
                "POST /classify": "Classification from preprocessed images",
                "POST /process": "Full pipeline (preprocess + classify)",
            },
            "paper_reference": "https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.12987",
        }

        return jsonify(info)

    def get_config(self):
        """Get configuration details."""
        config_info = {
            "preprocessing": {
                "target_resolution": self.preprocessor.config.get("general", {}).get(
                    "target_resolution"
                ),
                "normalize_pixels": self.preprocessor.config.get("general", {}).get(
                    "normalize_pixels"
                ),
                "black_border_clipping": self.preprocessor.config.get(
                    "black_border_clipping", {}
                ).get("enabled"),
                "parallel_processing": self.preprocessor.config.get(
                    "performance", {}
                ).get("parallel_variants"),
            }
        }

        if self.classifier:
            config_info["classification"] = {
                "voting_strategy": self.classifier.config.get("voting_strategy"),
                "force_cpu": self.classifier.config.get("force_cpu"),
                "confidence_threshold": self.classifier.config.get(
                    "confidence_threshold"
                ),
                "dynamic_batching": self.classifier.config.get(
                    "dynamic_batching", {}
                ),
            }

        return jsonify(config_info)

    def get_models(self):
        """Get model information."""
        if not self.classifier:
            return jsonify({"error": "Classification module not available"}), 400

        model_info = self.classifier.get_model_info()
        return jsonify(model_info)

    def get_cache_stats(self):
        """Get cache statistics endpoint."""
        cache_stats = self.cache.get_stats()
        return jsonify(cache_stats)

    def get_cache_health(self):
        """Get cache health status endpoint."""
        health_status = self.cache.health_check()
        return jsonify(health_status)

    def clear_cache(self):
        """Clear cache endpoint."""
        try:
            # Get optional pattern from request
            data = request.get_json() if request.is_json else {}
            pattern = data.get("pattern") if data else None

            deleted_count = self.cache.clear_all(pattern)

            return jsonify(
                {
                    "status": "success",
                    "message": f"Cleared {deleted_count} cached entries",
                    "deleted_count": deleted_count,
                    "pattern": pattern or "all",
                }
            )
        except Exception as e:
            self.logger.error(f"Cache clear error: {e}")
            return jsonify({"error": str(e)}), 500

    def preprocess_image(self):
        """Preprocess a single image."""
        try:
            # Get image from request
            image_data = self._get_image_from_request(request)
            if image_data is None:
                return jsonify({"error": "No valid image provided"}), 400

            image, _ = image_data

            # Process image
            start_time = time.time()
            variants = self.preprocessor.process_image(image, "uploaded_image")
            processing_time = time.time() - start_time

            # Check if user wants to include encoded images
            include_encoded_images = (
                request.args.get("include_encoded_images", "false").lower() == "true"
            )

            response = PreprocessResponse()
            response.status = OperationStatus.SUCCESS
            response.metadata = PreprocessResponseMetadata()
            response.metadata.image_properties = ImageProperties(
                original_width_px=image.shape[1],
                original_height_px=image.shape[0],
                processed_width_px=variants["original"].shape[1],
                processed_height_px=variants["original"].shape[0],
            )
            response.metadata.variant_count = len(variants)
            response.metadata.processing_time_seconds = round(processing_time, 4)

            # Conditionally include base64-encoded variants
            if include_encoded_images:
                response_variants = {}
                for variant_name, variant_image in variants.items():
                    encoded_image = self._encode_image(variant_image)
                    response_variants[variant_name] = encoded_image
                response.preprocessed_images = PreprocessedImages.from_dict(
                    response_variants
                )
            else:
                response.preprocessed_images = None

            return jsonify(response.to_dict())

        except Exception as e:
            self.logger.error(f"Preprocessing error: {e}")
            return jsonify({"error": str(e)}), 500

    def classify_image(self):
        """Classify from preprocessed images."""
        try:
            if not self.classifier:
                return jsonify({"error": "Classification module not available"}), 400

            # Get voting strategy from query parameter (default to config value)
            voting_strategy = request.args.get(
                "voting_strategy", self.classifier.ensemble.voting_strategy
            )

            # Validate voting strategy
            if voting_strategy not in ["soft", "hard"]:
                return (
                    jsonify(
                        {
                            "error": f"Invalid voting_strategy '{voting_strategy}'. Must be 'soft' or 'hard'"
                        }
                    ),
                    400,
                )

            # Get preprocessed images from request
            preprocessed_images = self._get_preprocessed_images_from_request(request)
            if not preprocessed_images:
                return jsonify({"error": "No valid preprocessed images provided"}), 400

            # Classify (optionally through dynamic batching)
            start_time = time.time()
            queue_timeout = self.classifier.config.get("dynamic_batching", {}).get(
                "request_timeout_s", 15
            )
            batch_metrics = None

            if self.dynamic_batcher.enabled:
                try:
                    batch_response = self.dynamic_batcher.submit(
                        preprocessed_images,
                        voting_strategy=voting_strategy,
                        timeout_seconds=queue_timeout,
                    )
                    results = batch_response["classification_result"]
                    batch_metrics = batch_response.get("batch_metrics")
                except RuntimeError as e:
                    return jsonify({"error": str(e)}), 429
                except FuturesTimeoutError:
                    return (
                        jsonify(
                            {
                                "error": "Inference queue timeout",
                                "timeout_seconds": queue_timeout,
                            }
                        ),
                        503,
                    )
            else:
                results = self.classifier.classify(
                    preprocessed_images,
                    voting_strategy=voting_strategy,
                )

            classification_time = time.time() - start_time

            response = ClassifyResponse()
            response.status = OperationStatus.SUCCESS
            response.cached = False
            response.classification_time_seconds = round(classification_time, 4)
            response.batch_processing_metrics = (
                BatchProcessingMetrics.from_dict(batch_metrics)
                if batch_metrics
                else None
            )
            response.classification = self._build_classification_result(results)

            return jsonify(response.to_dict())

        except Exception as e:
            self.logger.error(f"Classification error: {e}")
            return jsonify({"error": str(e)}), 500

    def full_process(self):
        """Full pipeline: preprocess + classify."""
        try:
            # VALIDATION: Ensure only ONE image is provided (single endpoint)
            files = request.files.getlist("image")
            if len(files) > 1:
                return (
                    jsonify(
                        {
                            "error": "Single image endpoint accepts only one image.",
                            "images_provided": len(files),
                            "max_allowed": 1,
                        }
                    ),
                    400,
                )

            # Get image from request and extract filename
            image_data = self._get_image_from_request(request)
            if not image_data:
                return jsonify({"error": "No valid image provided"}), 400

            image, image_filename = image_data

            # Get model configuration for cache key
            if self.classifier:
                voting_strategy = request.args.get(
                    "voting_strategy", self.classifier.ensemble.voting_strategy
                )
                model_info = self.classifier.get_model_info()
                model_architecture = model_info.get(
                    "architectures", ["EfficientNetB4"]
                )[0]
                ensemble_size = len(self.classifier.ensemble.models)
            else:
                voting_strategy = "soft"
                model_architecture = "unknown"
                ensemble_size = 0

            # CHECK CACHE: Try to get cached result with model configuration
            cached_result = self.cache.get(
                image_filename,
                image,
                voting_strategy=voting_strategy,
                model_architecture=model_architecture,
                ensemble_size=ensemble_size,
            )
            if cached_result is not None:
                self.logger.debug(f"Cache HIT for image: {image_filename}")

                # Update processing times to reflect cache retrieval (near-zero)
                if (
                    "process_result" in cached_result
                    and "image_processing_times" in cached_result["process_result"]
                ):
                    image_processing_times = ImageProcessingTimes(
                        preprocessing_ms=0.0,
                        inference_ms=0.0,
                        total_ms=0.0,
                    )

                    cached_result["process_result"][
                        "image_processing_times"
                    ] = image_processing_times.to_dict()

                    # Keep batch metrics in a dedicated sibling object.
                    cached_result["process_result"]["batch_processing_metrics"] = {
                        "batch_id": None,
                        "batch_wait_ms": 0.0,
                        "batch_execution_ms": 0.0,
                        "batch_total_ms": 0.0,
                        "batch_size": 0,
                    }

                cached_metadata = cached_result.get("process_metadata", {})
                cached_version_info = cached_metadata.get("version_info", {})
                version_info = VersionInfo(
                    api_version=cached_version_info.get(
                        "api_version", self._get_api_version()
                    ),
                    model_version=cached_version_info.get(
                        "model_version", self._get_model_version()
                    ),
                )

                model_configuration = None
                cached_model_cfg = cached_metadata.get("model_configuration")
                if isinstance(cached_model_cfg, dict):
                    model_configuration = ModelConfiguration(
                        model_architecture=cached_model_cfg.get(
                            "model_architecture", "EfficientNetB4"
                        ),
                        ensemble_size=cached_model_cfg.get("ensemble_size", 1),
                        trained_datasets=cached_model_cfg.get(
                            "trained_datasets", ["APTOS-2019-DR-Classification"]
                        ),
                        voting_strategy=cached_model_cfg.get(
                            "voting_strategy", "Soft-Voting"
                        ),
                    )

                process_metadata = ProcessMetadata(
                    request_id=self._generate_request_id(),
                    timestamp_utc=self._get_utc_timestamp(),
                    version_info=version_info,
                    model_configuration=model_configuration,
                )
                cached_result["process_metadata"] = process_metadata.to_dict()

                # Mark as cached in result
                if (
                    "process_result" in cached_result
                    and "cached" in cached_result["process_result"]
                ):
                    cached_result["process_result"]["cached"] = True

                return jsonify(cached_result)

            # Cache miss - proceed with processing
            self.logger.debug(f"Cache MISS for image: {image_filename}")

            # Preprocess
            preprocessing_start = time.time()
            preprocessed_image_variants = self.preprocessor.process_image(
                image, image_filename
            )
            preprocessing_time = time.time() - preprocessing_start

            # Classify if classifier available
            classification_results_from_classifier = None
            classification_time = 0

            if self.classifier:
                # Get voting strategy from query parameter (default to config value)
                voting_strategy = request.args.get(
                    "voting_strategy", self.classifier.ensemble.voting_strategy
                )

                # Validate voting strategy
                if voting_strategy not in self._get_voting_strategy_values():
                    return (
                        jsonify(
                            {
                                "error": f"Invalid voting_strategy '{voting_strategy}'. Must be in {self._get_voting_strategy_values()}"
                            }
                        ),
                        400,
                    )

                classification_start = time.time()
                queue_timeout = self.classifier.config.get(
                    "dynamic_batching", {}
                ).get("request_timeout_s", 15)
                batch_metrics = None

                if self.dynamic_batcher.enabled:
                    try:
                        batch_response = self.dynamic_batcher.submit(
                            preprocessed_image_variants,
                            voting_strategy=voting_strategy,
                            timeout_seconds=queue_timeout,
                        )
                        classification_results_from_classifier = batch_response[
                            "classification_result"
                        ]
                        batch_metrics = batch_response.get("batch_metrics")
                    except RuntimeError as e:
                        return jsonify({"error": str(e)}), 429
                    except FuturesTimeoutError:
                        return (
                            jsonify(
                                {
                                    "error": "Inference queue timeout",
                                    "timeout_seconds": queue_timeout,
                                }
                            ),
                            503,
                        )
                else:
                    classification_results_from_classifier = self.classifier.classify(
                        preprocessed_image_variants,
                        voting_strategy=voting_strategy,
                    )

                classification_time = time.time() - classification_start

            # Check if user wants to include preprocessed images
            include_encoded_images = (
                request.args.get("include_encoded_images", "false").lower() == "true"
            )

            # Build structured response using typed models
            if classification_results_from_classifier:
                # Create ImageProperties instance
                image_properties = ImageProperties(
                    original_width_px=image.shape[1],
                    original_height_px=image.shape[0],
                    processed_width_px=preprocessed_image_variants["original"].shape[1],
                    processed_height_px=preprocessed_image_variants["original"].shape[
                        0
                    ],
                )

                # Create ImageProcessingTimes instance
                image_processing_times = ImageProcessingTimes(
                    preprocessing_ms=round(preprocessing_time * 1000, 2),
                    inference_ms=round(classification_time * 1000, 2),
                    total_ms=round(
                        (preprocessing_time + classification_time) * 1000, 2
                    ),
                )

                # Build classification result using typed models
                classification_dict = self._build_classification_result(
                    classification_results_from_classifier,
                ).to_dict()

                # Create ClassProbability instances
                class_probabilities = []
                for cp_dict in classification_dict["class_probabilities"]:
                    class_prob = ClassProbability(
                        id=cp_dict["id"],
                        label=cp_dict["label"],
                        probability=cp_dict["probability"],
                    )
                    class_probabilities.append(class_prob)

                # Create ClassificationResult instance
                classification_result = ClassificationResult(
                    predicted_class_id=classification_dict["predicted_class_id"],
                    predicted_class_label=classification_dict["predicted_class_label"],
                    confidence_score=classification_dict["confidence_score"],
                    diagnosis_certainty=classification_dict["diagnosis_certainty"],
                    class_probabilities=class_probabilities,
                )

                # Create ProcessResult instance
                process_result = ProcessResult(
                    status=OperationStatus.SUCCESS,
                    cached=False,
                    image_name=image_filename,
                    image_index=0,
                    image_properties=image_properties,
                    image_processing_times=image_processing_times,
                    classification=classification_result,
                )
                process_result.batch_processing_metrics = (
                    BatchProcessingMetrics.from_dict(batch_metrics)
                    if batch_metrics
                    else None
                )

                # Optionally include preprocessed images
                if include_encoded_images:
                    preprocessed_images = {}
                    for (
                        variant_name,
                        variant_image,
                    ) in preprocessed_image_variants.items():
                        encoded_image = self._encode_image(variant_image)
                        preprocessed_images[variant_name] = encoded_image
                    process_result.preprocessed_images = preprocessed_images

                model_info = self.classifier.get_model_info()
                architectures = model_info.get("architectures", [])
                datasets = model_info.get("datasets", [])

                model_configuration = ModelConfiguration(
                    model_architecture=(
                        architectures[0] if architectures else "EfficientNetB4"
                    ),
                    ensemble_size=len(self.classifier.ensemble.models),
                    trained_datasets=(
                        datasets if datasets else ["APTOS-2019-DR-Classification"]
                    ),
                    voting_strategy=f"{voting_strategy.capitalize()}-Voting",
                )

                process_metadata = ProcessMetadata(
                    request_id=self._generate_request_id(),
                    timestamp_utc=self._get_utc_timestamp(),
                    version_info=VersionInfo(
                        api_version=self._get_api_version(),
                        model_version=self._get_model_version(),
                    ),
                    model_configuration=model_configuration,
                )

                process_response = ProcessResponse(
                    process_metadata=process_metadata,
                    process_result=process_result,
                )
                response = process_response.to_dict()

                # STORE IN CACHE: Save result for future requests with model configuration
                # Don't cache if include_images=true (too large)
                if not include_encoded_images:
                    cache_stored = self.cache.set(
                        image_filename,
                        image,
                        response,
                        voting_strategy=voting_strategy,
                        model_architecture=model_architecture,
                        ensemble_size=ensemble_size,
                    )
                    if cache_stored:
                        self.logger.debug(f"Result cached for image: {image_filename}")
            else:
                # Preprocessing-only mode (classifier not available)
                # Create ImageProperties instance
                image_properties = ImageProperties(
                    original_width_px=image.shape[1],
                    original_height_px=image.shape[0],
                    processed_width_px=preprocessed_image_variants["original"].shape[1],
                    processed_height_px=preprocessed_image_variants["original"].shape[
                        0
                    ],
                )

                # Create ImageProcessingTimes instance
                image_processing_times = ImageProcessingTimes(
                    preprocessing_ms=round(preprocessing_time * 1000, 2),
                    inference_ms=0.0,
                    total_ms=round(preprocessing_time * 1000, 2),
                )

                # Create ProcessResult instance
                process_result = ProcessResult(
                    status=OperationStatus.SUCCESS,
                    cached=False,
                    image_name=image_filename,
                    image_properties=image_properties,
                    image_processing_times=image_processing_times,
                )

                # Optionally include preprocessed images
                if include_encoded_images:
                    preprocessed_images = {}
                    for (
                        variant_name,
                        variant_image,
                    ) in preprocessed_image_variants.items():
                        encoded_image = self._encode_image(variant_image)
                        preprocessed_images[variant_name] = encoded_image
                    process_result.preprocessed_images = preprocessed_images

                process_metadata = ProcessMetadata(
                    request_id=self._generate_request_id(),
                    timestamp_utc=self._get_utc_timestamp(),
                    version_info=VersionInfo(
                        api_version=self._get_api_version(),
                        model_version=self._get_model_version(),
                    ),
                    model_configuration=None,
                )

                process_response = ProcessResponse(
                    process_metadata=process_metadata,
                    process_result=process_result,
                )
                response = process_response.to_dict()

            return jsonify(response)

        except Exception as e:
            self.logger.error(f"Full processing error: {e}")
            return jsonify({"error": str(e)}), 500

    def _get_image_from_request(self, req) -> tuple:
        """Extract image from Flask request."""
        try:
            if "image" not in req.files:
                return None

            file = req.files["image"]
            if file.filename == "":
                return None

            # Read image
            image_bytes = file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                return None

            return (image, file.filename)

        except Exception as e:
            self.logger.error(f"Error extracting image from request: {e}")
            return None

    def _get_images_from_request(self, req) -> List[tuple]:
        """Extract multiple images from Flask request.

        Returns:
            List of tuples (image, filename)
        """
        images = []

        try:
            files = req.files.getlist("images")

            for file in files:
                if file.filename != "":
                    image_bytes = file.read()
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if image is not None:
                        images.append((image, file.filename))

            return images

        except Exception as e:
            self.logger.error(f"Error extracting images from request: {e}")
            return []

    def _get_preprocessed_images_from_request(self, req) -> Dict[str, np.ndarray]:
        """Extract preprocessed images from request (base64 encoded)."""
        try:
            data = req.get_json()
            if not data or "preprocessed_images" not in data:
                return {}

            variants = {}
            for variant_name, encoded_image in data["preprocessed_images"].items():
                decoded_image = (
                    self._decode_image(encoded_image).astype(np.float32) / 255.0
                )
                if decoded_image is not None:
                    variants[variant_name] = decoded_image

            return variants

        except Exception as e:
            self.logger.error(f"Error extracting preprocessed images: {e}")
            return {}

    def _encode_image(self, image: np.ndarray) -> str:
        """Encode image to base64 string."""
        # Ensure image is in correct format
        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image

        # Encode to JPEG
        _, buffer = cv2.imencode(".jpg", image_bgr)
        return base64.b64encode(buffer).decode("utf-8")

    def _decode_image(self, encoded_image: str) -> Optional[np.ndarray]:
        """Decode base64 string to image."""
        try:
            image_bytes = base64.b64decode(encoded_image)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is not None:
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            return image

        except Exception as e:
            self.logger.error(f"Error decoding image: {e}")
            return None

    def run(self):
        """Start the server."""
        self.logger.info(f"Starting Fundus Inference Server on {self.host}:{self.port}")
        try:
            self.app.run(host=self.host, port=self.port, debug=self.debug)
        finally:
            self.dynamic_batcher.stop()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fundus Image Processing and Classification Server"
    )
    parser.add_argument(
        "--preprocessing-config",
        type=str,
        default="preprocessing_config.yaml",
        help="Path to preprocessing configuration file",
    )
    parser.add_argument(
        "--classifier-config",
        type=str,
        default=None,
        help="Path to classifier configuration file",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # Redis configuration arguments (override config file)
    parser.add_argument(
        "--redis-enabled",
        type=lambda x: x.lower() == "true",
        default=None,
        help="Enable Redis caching (true/false)",
    )
    parser.add_argument("--redis-host", type=str, help="Redis host")
    parser.add_argument("--redis-port", type=int, help="Redis port")
    parser.add_argument("--redis-db", type=int, help="Redis database number")
    parser.add_argument("--redis-password", type=str, help="Redis password")
    parser.add_argument("--redis-ttl", type=int, help="Redis TTL in seconds")

    args = parser.parse_args()

    # Load Redis configuration from classifier config file
    redis_config = {}
    if args.classifier_config and os.path.exists(args.classifier_config):
        try:
            import yaml

            with open(args.classifier_config, "r") as f:
                config = yaml.safe_load(f)
                redis_config = config.get("redis", {})
        except Exception as e:
            print(f"Warning: Could not load Redis config from file: {e}")

    # Override with command-line arguments if provided
    if args.redis_enabled is not None:
        redis_config["enabled"] = args.redis_enabled
    if args.redis_host:
        redis_config["host"] = args.redis_host
    if args.redis_port:
        redis_config["port"] = args.redis_port
    if args.redis_db is not None:
        redis_config["db"] = args.redis_db
    if args.redis_password:
        redis_config["password"] = args.redis_password
    if args.redis_ttl:
        redis_config["ttl"] = args.redis_ttl

    # Create server and run
    server = FundusInferenceServer(
        preprocessing_config=args.preprocessing_config,
        classifier_config=args.classifier_config,
        host=args.host,
        port=args.port,
        debug=args.debug,
        redis_config=redis_config,
    )

    server.run()


if __name__ == "__main__":
    main()
