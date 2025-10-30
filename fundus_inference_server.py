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
from io import BytesIO
from typing import Dict, List, Optional, Union
import cv2
import numpy as np
from flask import Flask, request, jsonify
from pathlib import Path
import argparse
import uuid
from datetime import datetime, timezone

# Import generated OpenAPI models
from ensemble_inference.models.processing_times import ProcessingTimes

# Import our modules
from fundus_preprocessor import FundusPreprocessor
from diabetic_retinopathy_classifier import DiabeticRetinopathyClassifier
from redis_cache_manager import RedisCacheManager


class FundusInferenceServer:
    """Main server class integrating preprocessing and classification."""

    def __init__(
        self,
        preprocessing_config: str,
        classifier_config: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = 5000,
        debug: bool = False,
        max_batch_size: int = 100,
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
            max_batch_size: Maximum number of images allowed in a batch request
            redis_config: Redis configuration dictionary (optional)
        """
        self.host = host
        self.port = port
        self.debug = debug
        self.max_batch_size = max_batch_size

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

        # Setup Flask app
        self.app = self._create_flask_app()

        # Server statistics
        self.stats = {
            "total_requests": 0,
            "preprocessing_requests": 0,
            "classification_requests": 0,
            "errors": 0,
            "start_time": time.time(),
        }

        self.logger.info(
            f"FundusInferenceServer initialized successfully (max_batch_size={max_batch_size})"
        )

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
        if "result" not in cached_response:
            return {}

        cached_result = cached_response["result"]
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
        image_shape: tuple,
        processed_shape: tuple,
    ) -> Dict:
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

        # Build result structure
        result = {
            "predicted_class_id": predicted_class_info["id"],
            "predicted_class_label": predicted_class_info["label"],
            "confidence_score": round(confidence, 6),
            "diagnosis_certainty": self._get_diagnosis_certainty(confidence),
            "class_probabilities": class_probabilities,
        }

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

        # Create ProcessingTimes model instance
        processing_times = ProcessingTimes(
            _00_preprocessing_ms=round(preprocessing_time * 1000, 2),
            _01_inference_ms=round(classification_time * 1000, 2),
            total_ms=round(total_time * 1000, 2),
            per_image_ms=round(total_time * 1000, 2),  # Same as total for single image
        )

        response = {
            "status": "SUCCESS",
            "request_id": self._generate_request_id(),
            "timestamp_utc": self._get_utc_timestamp(),
            "image_name": image_filename,
            "processing_times": processing_times.to_dict(),
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
            response["result"] = self._build_classification_result(
                classification_results,
                image_shape,
                processed_shape,
            )

        # Optionally include preprocessed images
        if include_images and variants:
            response["preprocessed_images"] = {}
            for variant_name, variant_image in variants.items():
                encoded_image = self._encode_image(variant_image)
                response["preprocessed_images"][variant_name] = encoded_image

        return response

    def _build_batch_process_response(
        self,
        images: List[tuple],
        preprocessing_results: List,
        classification_results_list: List,
        preprocessing_time: float,
        classification_time: float,
        voting_strategy: str,
    ) -> Dict:
        """
        Build structured response for batch processing.

        Args:
            images: List of (image, filename) tuples
            preprocessing_results: List of preprocessing results
            classification_results_list: List of classification results
            preprocessing_time: Total preprocessing time (seconds)
            classification_time: Total classification time (seconds)
            voting_strategy: Voting strategy used

        Returns:
            Structured response dictionary
        """
        batch_size = len(images)
        total_time = preprocessing_time + classification_time
        per_image_time = total_time / batch_size if batch_size > 0 else 0

        # Count successes and failures
        successful_count = sum(
            1 for r in classification_results_list if r.get("status") == "success"
        )
        failed_count = batch_size - successful_count

        response = {
            "status": "SUCCESS",
            "request_id": self._generate_request_id("BATCH"),
            "timestamp_utc": self._get_utc_timestamp(),
            "batch_details": {
                "batch_size": batch_size,
                "processed_successfully": successful_count,
                "processed_failure": failed_count,
            },
            "total_processing_times": {
                "00_preprocessing_ms": round(preprocessing_time * 1000, 2),
                "01_inference_ms": round(classification_time * 1000, 2),
                "total_ms": round(total_time * 1000, 2),
                "per_image_ms": round(per_image_time * 1000, 2),
            },
            "metadata": {
                "version_info": {
                    "api_version": self._get_api_version(),
                    "model_version": self._get_model_version(),
                },
            },
        }

        # Add model configuration if classifier is available
        if self.classifier:
            model_info = self.classifier.get_model_info()
            architectures = model_info.get("architectures", [])

            response["metadata"]["model_configuration"] = {
                "model_architecture": (
                    architectures[0] if architectures else "EfficientNetB4"
                ),
                "ensemble_size": len(self.classifier.ensemble.models),
                "trained_datasets": ["APTOS-2019-DR-Classification"],
                "voting_strategy": f"{voting_strategy.capitalize()}-Voting",
            }

        # Build results array
        results = []
        for i, (classification_result, preprocessing_result) in enumerate(
            zip(classification_results_list, preprocessing_results)
        ):
            image, filename = images[i]

            if classification_result.get("status") == "success":
                # Calculate individual image processing time (estimate)
                individual_time = total_time / batch_size

                result_entry = {
                    "image_index": i,
                    "image_name": filename,
                    "status": "SUCCESS",
                    "image_processing_times": {
                        "00_preprocessing_ms": round(
                            (preprocessing_time / batch_size) * 1000, 2
                        ),
                        "01_inference_ms": round(
                            (classification_time / batch_size) * 1000, 2
                        ),
                        "total_ms": round(individual_time * 1000, 2),
                    },
                    "result": self._build_classification_result(
                        classification_result.get("prediction", {}),
                        image.shape[:2],
                        (500, 500),  # Assume processed size
                    ),
                }
            else:
                # Failed processing
                result_entry = {
                    "image_index": i,
                    "image_name": filename,
                    "status": "FAILED",
                    "error": classification_result.get("error", "Unknown error"),
                }

            results.append(result_entry)

        response["results"] = results

        return response

    # ========== End Response Schema Helpers ==========

    def _create_flask_app(self) -> Flask:
        """Create and configure Flask application."""
        app = Flask(__name__)
        app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

        # Register routes
        app.route("/health", methods=["GET"])(self.health_check)
        app.route("/info", methods=["GET"])(self.get_info)
        app.route("/stats", methods=["GET"])(self.get_stats)
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

        # Batch processing
        app.route("/batch/preprocess", methods=["POST"])(self.batch_preprocess)
        app.route("/batch/process", methods=["POST"])(self.batch_process)

        return app

    def health_check(self):
        """Health check endpoint."""
        self.stats["total_requests"] += 1

        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime": time.time() - self.stats["start_time"],
            "modules": {
                "preprocessor": True,
                "classifier": self.classifier is not None,
            },
        }

        return jsonify(health_status)

    def get_info(self):
        """Get server information."""
        self.stats["total_requests"] += 1

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
                },
            },
            "endpoints": {
                "GET /health": "Health check",
                "GET /info": "Server information",
                "GET /stats": "Server statistics",
                "GET /config": "Configuration details",
                "GET /models": "Model information",
                "POST /preprocess": "Image preprocessing only",
                "POST /classify": "Classification from preprocessed images",
                "POST /process": "Full pipeline (preprocess + classify)",
                "POST /batch/preprocess": "Batch preprocessing",
                "POST /batch/process": "Batch full processing",
            },
            "paper_reference": "https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.12987",
        }

        return jsonify(info)

    def get_stats(self):
        """Get server statistics."""
        self.stats["total_requests"] += 1

        uptime = time.time() - self.stats["start_time"]

        statistics = {
            "uptime_seconds": uptime,
            "uptime_formatted": f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s",
            "total_requests": self.stats["total_requests"],
            "preprocessing_requests": self.stats["preprocessing_requests"],
            "classification_requests": self.stats["classification_requests"],
            "errors": self.stats["errors"],
            "requests_per_minute": (
                self.stats["total_requests"] / (uptime / 60) if uptime > 0 else 0
            ),
        }

        return jsonify(statistics)

    def get_config(self):
        """Get configuration details."""
        self.stats["total_requests"] += 1

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
            }

        return jsonify(config_info)

    def get_models(self):
        """Get model information."""
        self.stats["total_requests"] += 1

        if not self.classifier:
            return jsonify({"error": "Classification module not available"}), 400

        model_info = self.classifier.get_model_info()
        return jsonify(model_info)

    def get_cache_stats(self):
        """Get cache statistics endpoint."""
        self.stats["total_requests"] += 1

        cache_stats = self.cache.get_stats()
        return jsonify(cache_stats)

    def get_cache_health(self):
        """Get cache health status endpoint."""
        self.stats["total_requests"] += 1

        health_status = self.cache.health_check()
        return jsonify(health_status)

    def clear_cache(self):
        """Clear cache endpoint."""
        self.stats["total_requests"] += 1

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
            self.stats["total_requests"] += 1
            self.stats["preprocessing_requests"] += 1

            # Get image from request
            image_data = self._get_image_from_request(request)
            if image_data is None:
                return jsonify({"error": "No valid image provided"}), 400

            image, _ = image_data

            # Process image
            start_time = time.time()
            variants = self.preprocessor.process_image(image, "uploaded_image")
            processing_time = time.time() - start_time

            # Convert to base64 for JSON response
            response_variants = {}
            for variant_name, variant_image in variants.items():
                encoded_image = self._encode_image(variant_image)
                response_variants[variant_name] = encoded_image

            response = {
                "status": "success",
                "processing_time_seconds": processing_time,
                "variants": response_variants,
                "metadata": {
                    "original_size": list(image.shape[:2]),
                    "processed_size": list(variants["original"].shape[:2]),
                    "variant_count": len(variants),
                },
            }

            return jsonify(response)

        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error(f"Preprocessing error: {e}")
            return jsonify({"error": str(e)}), 500

    def classify_image(self):
        """Classify from preprocessed images."""
        try:
            self.stats["total_requests"] += 1
            self.stats["classification_requests"] += 1

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

            # Temporarily set voting strategy if different from default
            original_strategy = self.classifier.ensemble.voting_strategy
            self.classifier.ensemble.voting_strategy = voting_strategy

            try:
                # Classify
                start_time = time.time()
                results = self.classifier.classify(preprocessed_images)
                classification_time = time.time() - start_time

                response = {
                    "status": "success",
                    "classification_time_seconds": classification_time,
                    "prediction": results,
                }

                return jsonify(response)
            finally:
                # Restore original voting strategy
                self.classifier.ensemble.voting_strategy = original_strategy

        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error(f"Classification error: {e}")
            return jsonify({"error": str(e)}), 500

    def full_process(self):
        """Full pipeline: preprocess + classify."""
        try:
            self.stats["total_requests"] += 1
            self.stats["preprocessing_requests"] += 1
            self.stats["classification_requests"] += 1

            # VALIDATION: Ensure only ONE image is provided (single endpoint)
            files = request.files.getlist("image")
            if len(files) > 1:
                return (
                    jsonify(
                        {
                            "error": "Single image endpoint accepts only one image. Use /batch/process for multiple images.",
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
                self.logger.info(f"Cache HIT for image: {image_filename}")

                # Update processing times to reflect cache retrieval (near-zero)
                cached_result["processing_times"] = {
                    "00_preprocessing_ms": 0.0,
                    "01_inference_ms": 0.0,
                    "total_ms": 0.0,
                    "per_image_ms": 0.0,
                }

                # Update timestamp to current time
                cached_result["timestamp_utc"] = self._get_utc_timestamp()

                # Generate new request ID for this cached request
                cached_result["request_id"] = self._generate_request_id()

                # Add cache indicators
                cached_result["cached"] = True
                cached_result["cache_hit"] = True

                return jsonify(cached_result)

            # Cache miss - proceed with processing
            self.logger.debug(f"Cache MISS for image: {image_filename}")

            # Preprocess
            preprocessing_start = time.time()
            variants = self.preprocessor.process_image(image, image_filename)
            preprocessing_time = time.time() - preprocessing_start

            # Classify if classifier available
            classification_results = None
            classification_time = 0

            if self.classifier:
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

                # Temporarily set voting strategy if different from default
                original_strategy = self.classifier.ensemble.voting_strategy
                self.classifier.ensemble.voting_strategy = voting_strategy

                try:
                    classification_start = time.time()
                    classification_results = self.classifier.classify(variants)
                    classification_time = time.time() - classification_start
                finally:
                    # Restore original voting strategy
                    self.classifier.ensemble.voting_strategy = original_strategy

            # Check if user wants to include preprocessed images
            include_images = (
                request.form.get("include_images", "false").lower() == "true"
            )

            # Build structured response using the new schema
            if classification_results:
                response = self._build_single_process_response(
                    image_shape=image.shape[:2],
                    processed_shape=variants["original"].shape[:2],
                    preprocessing_time=preprocessing_time,
                    classification_time=classification_time,
                    classification_results=classification_results,
                    include_images=include_images,
                    variants=variants if include_images else None,
                    image_filename=image_filename,
                )

                # Add cache indicator
                response["cached"] = False
                response["cache_hit"] = False

                # STORE IN CACHE: Save result for future requests with model configuration
                # Don't cache if include_images=true (too large)
                if not include_images:
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
                response = {
                    "status": "SUCCESS",
                    "request_id": self._generate_request_id(),
                    "timestamp_utc": self._get_utc_timestamp(),
                    "image_name": image_filename,
                    "cached": False,
                    "cache_hit": False,
                    "processing_times": {
                        "preprocessing_ms": round(preprocessing_time * 1000, 2),
                        "inference_ms": 0.0,
                        "total_ms": round(preprocessing_time * 1000, 2),
                        "per_image_ms": round(preprocessing_time * 1000, 2),
                    },
                    "metadata": {
                        "image_properties": {
                            "original_width_px": image.shape[1],
                            "original_height_px": image.shape[0],
                            "processed_width_px": variants["original"].shape[1],
                            "processed_height_px": variants["original"].shape[0],
                        },
                        "version_info": {
                            "api_version": self._get_api_version(),
                            "model_version": self._get_model_version(),
                        },
                    },
                    "warning": "Classification module not available - preprocessing only",
                }

                if include_images:
                    response["preprocessed_images"] = {}
                    for variant_name, variant_image in variants.items():
                        encoded_image = self._encode_image(variant_image)
                        response["preprocessed_images"][variant_name] = encoded_image

            return jsonify(response)

        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error(f"Full processing error: {e}")
            return jsonify({"error": str(e)}), 500

    def batch_preprocess(self):
        """Batch preprocessing endpoint."""
        try:
            self.stats["total_requests"] += 1

            # Get images from request (returns list of tuples: (image, filename))
            image_data = self._get_images_from_request(request)
            if not image_data:
                return jsonify({"error": "No valid images provided"}), 400

            # Separate images and filenames
            images = [img for img, _ in image_data]
            filenames = [name for _, name in image_data]

            # Process batch
            start_time = time.time()
            results = self.preprocessor.process_batch(images, filenames)
            processing_time = time.time() - start_time

            # Convert results
            response_results = []
            for i, variants in enumerate(results):
                if variants:  # Success
                    response_variants = {}
                    for variant_name, variant_image in variants.items():
                        encoded_image = self._encode_image(variant_image)
                        response_variants[variant_name] = encoded_image

                    response_results.append(
                        {
                            "status": "success",
                            "image_index": i,
                            "image_name": filenames[i],
                            "variants": response_variants,
                        }
                    )
                else:  # Failed
                    response_results.append(
                        {
                            "status": "failed",
                            "image_index": i,
                            "image_name": filenames[i],
                            "error": "Processing failed",
                        }
                    )

            self.stats["preprocessing_requests"] += len(images)

            response = {
                "status": "success",
                "batch_size": len(images),
                "successful": sum(
                    1 for r in response_results if r["status"] == "success"
                ),
                "processing_time_seconds": processing_time,
                "results": response_results,
            }

            return jsonify(response)

        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error(f"Batch preprocessing error: {e}")
            return jsonify({"error": str(e)}), 500

    def batch_process(self):
        """Batch full processing endpoint with Redis caching support."""
        try:
            self.stats["total_requests"] += 1

            # VALIDATION: steps
            if not self.classifier:
                return (
                    jsonify(
                        {
                            "error": "Classification module not available for batch processing"
                        }
                    ),
                    400,
                )

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

            # Get images from request (returns list of tuples: (image, filename))
            image_data = self._get_images_from_request(request)
            if not image_data:
                return jsonify({"error": "No valid images provided"}), 400

            # VALIDATION: Check batch size limit
            total_images = len(image_data)
            if total_images > self.max_batch_size:
                return (
                    jsonify(
                        {
                            "error": f"Batch size exceeds maximum allowed limit",
                            "images_provided": total_images,
                            "max_batch_size": self.max_batch_size,
                            "suggestion": f"Please split your request into multiple batches of max {self.max_batch_size} images each",
                        }
                    ),
                    400,
                )

            # CHECK CACHE: Try to get cached results for each image
            cached_results = []
            images_to_process = []
            images_to_process_indices = []
            cache_hit_count = 0

            # Get model info once for cache operations
            model_info = self.classifier.get_model_info()
            model_architecture = model_info.get("architectures", ["EfficientNetB4"])[0]
            ensemble_size = len(self.classifier.ensemble.models)

            for i, (image, filename) in enumerate(image_data):
                cached_result = self.cache.get(
                    filename,
                    image,
                    voting_strategy=voting_strategy,
                    model_architecture=model_architecture,
                    ensemble_size=ensemble_size,
                )
                if cached_result is not None:
                    # Cache hit - use cached result
                    cache_hit_count += 1
                    self.logger.info(f"Cache HIT for batch image [{i}]: {filename}")
                    cached_results.append(
                        {"index": i, "result": cached_result, "from_cache": True}
                    )
                else:
                    # Cache miss - need to process
                    self.logger.debug(f"Cache MISS for batch image [{i}]: {filename}")
                    images_to_process.append((image, filename))
                    images_to_process_indices.append(i)

            self.logger.info(
                f"Batch cache stats: {cache_hit_count}/{total_images} hits, "
                f"{len(images_to_process)} to process"
            )

            # Process only non-cached images
            preprocessing_time = 0
            classification_time = 0
            processing_results = []

            if images_to_process:
                # Separate images and filenames for processing
                images = [img for img, _ in images_to_process]
                filenames = [name for _, name in images_to_process]

                # Process batch
                start_time = time.time()
                preprocessing_results = self.preprocessor.process_batch(
                    images, filenames
                )
                preprocessing_time = time.time() - start_time

                # Temporarily set voting strategy if different from default
                original_strategy = self.classifier.ensemble.voting_strategy
                self.classifier.ensemble.voting_strategy = voting_strategy

                try:
                    # Classify each result
                    classification_start = time.time()

                    for i, variants in enumerate(preprocessing_results):
                        if variants:  # Preprocessing succeeded
                            try:
                                classification_results = self.classifier.classify(
                                    variants
                                )

                                # Build the complete response for this image
                                image, filename = images_to_process[i]
                                individual_response = self._build_single_process_response(
                                    image_shape=image.shape[:2],
                                    processed_shape=(500, 500),
                                    preprocessing_time=preprocessing_time
                                    / len(images_to_process),
                                    classification_time=0,  # Will be calculated after
                                    classification_results=classification_results,
                                    include_images=False,
                                    variants=None,
                                    image_filename=filename,
                                )

                                processing_results.append(
                                    {
                                        "index": images_to_process_indices[i],
                                        "result": {
                                            "status": "success",
                                            "prediction": classification_results,
                                        },
                                        "response": individual_response,
                                        "image": image,
                                        "filename": filename,
                                        "from_cache": False,
                                    }
                                )
                            except Exception as e:
                                processing_results.append(
                                    {
                                        "index": images_to_process_indices[i],
                                        "result": {
                                            "status": "classification_failed",
                                            "error": str(e),
                                        },
                                        "from_cache": False,
                                    }
                                )
                        else:  # Preprocessing failed
                            processing_results.append(
                                {
                                    "index": images_to_process_indices[i],
                                    "result": {
                                        "status": "preprocessing_failed",
                                        "error": "Preprocessing failed",
                                    },
                                    "from_cache": False,
                                }
                            )

                    classification_time = time.time() - classification_start
                finally:
                    # Restore original voting strategy
                    self.classifier.ensemble.voting_strategy = original_strategy

                # STORE IN CACHE: Cache successful results
                for proc_result in processing_results:
                    if (
                        proc_result["result"].get("status") == "success"
                        and "response" in proc_result
                    ):
                        cache_stored = self.cache.set(
                            proc_result["filename"],
                            proc_result["image"],
                            proc_result["response"],
                        )
                        if cache_stored:
                            self.logger.debug(
                                f"Result cached for batch image: {proc_result['filename']}"
                            )

            self.stats["preprocessing_requests"] += len(images_to_process)
            self.stats["classification_requests"] += len(images_to_process)

            # Combine cached and processed results in original order
            all_results = cached_results + processing_results
            all_results.sort(key=lambda x: x["index"])

            # Build final classification results list
            classification_results_list = []
            # Map from structured response back to classifier format
            for res in all_results:
                if res.get("from_cache"):
                    # Extract the classification info from cached response
                    cached_response = res["result"]
                    classification_results_list.append(
                        {
                            "status": "success",
                            "prediction": self._convert_cached_response_to_classifier_format(
                                cached_response
                            ),
                            "cached": True,
                        }
                    )
                else:
                    classification_results_list.append(res["result"])

            # Build structured response using the new schema
            batch_size = len(image_data)
            total_time = preprocessing_time + classification_time
            per_image_time = total_time / batch_size if batch_size > 0 else 0

            # Count successes and failures
            successful_count = sum(
                1 for r in classification_results_list if r.get("status") == "success"
            )
            failed_count = batch_size - successful_count

            response = {
                "status": "SUCCESS",
                "request_id": self._generate_request_id("BATCH"),
                "timestamp_utc": self._get_utc_timestamp(),
                "batch_details": {
                    "batch_size": batch_size,
                    "processed_successfully": successful_count,
                    "processed_failure": failed_count,
                },
                "cache_stats": {
                    "cache_hits": cache_hit_count,
                    "cache_misses": len(images_to_process),
                    "cached_percentage": (
                        round((cache_hit_count / batch_size) * 100, 2)
                        if batch_size > 0
                        else 0
                    ),
                },
                "total_processing_times": {
                    "preprocessing_ms": round(preprocessing_time * 1000, 2),
                    "inference_ms": round(classification_time * 1000, 2),
                    "total_ms": round(total_time * 1000, 2),
                    "per_image_ms": round(per_image_time * 1000, 2),
                },
                "metadata": {
                    "version_info": {
                        "api_version": self._get_api_version(),
                        "model_version": self._get_model_version(),
                    },
                },
            }

            # Add model configuration if classifier is available
            if self.classifier:
                model_info = self.classifier.get_model_info()
                architectures = model_info.get("architectures", [])

                response["metadata"]["model_configuration"] = {
                    "model_architecture": (
                        architectures[0] if architectures else "EfficientNetB4"
                    ),
                    "ensemble_size": len(self.classifier.ensemble.models),
                    "trained_datasets": ["APTOS-2019-DR-Classification"],
                    "voting_strategy": f"{voting_strategy.capitalize()}-Voting",
                }

            # Build results array
            results = []
            for i, (image, filename) in enumerate(image_data):
                classification_result = classification_results_list[i]
                is_cached = classification_result.get("cached", False)

                if classification_result.get("status") == "success":
                    # Calculate individual image processing time (estimate)
                    if is_cached:
                        # Cached results have minimal processing time
                        individual_time = 0
                        individual_preprocessing = 0
                        individual_inference = 0
                    else:
                        individual_time = (
                            total_time / len(images_to_process)
                            if images_to_process
                            else 0
                        )
                        individual_preprocessing = (
                            preprocessing_time / len(images_to_process)
                            if images_to_process
                            else 0
                        )
                        individual_inference = (
                            classification_time / len(images_to_process)
                            if images_to_process
                            else 0
                        )

                    result_entry = {
                        "image_index": i,
                        "image_name": filename,
                        "status": "SUCCESS",
                        "cached": is_cached,
                        "image_processing_times": {
                            "preprocessing_ms": round(
                                individual_preprocessing * 1000, 2
                            ),
                            "inference_ms": round(individual_inference * 1000, 2),
                            "total_ms": round(individual_time * 1000, 2),
                        },
                        "result": self._build_classification_result(
                            classification_result.get("prediction", {}),
                            image.shape[:2],
                            (500, 500),  # Assume processed size
                        ),
                    }
                else:
                    # Failed processing
                    result_entry = {
                        "image_index": i,
                        "image_name": filename,
                        "status": "FAILED",
                        "cached": False,
                        "error": classification_result.get("error", "Unknown error"),
                    }

                results.append(result_entry)

            response["results"] = results

            return jsonify(response)

        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error(f"Batch processing error: {e}")
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
            if not data or "variants" not in data:
                return {}

            variants = {}
            for variant_name, encoded_image in data["variants"].items():
                decoded_image = self._decode_image(encoded_image)
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
        self.app.run(host=self.host, port=self.port, debug=self.debug)


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
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=100,
        help="Maximum number of images allowed in a batch request",
    )

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
        max_batch_size=args.max_batch_size,
        redis_config=redis_config,
    )

    server.run()


if __name__ == "__main__":
    main()
