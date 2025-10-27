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

# Import our modules
from fundus_preprocessor import FundusPreprocessor
from diabetic_retinopathy_classifier import DiabeticRetinopathyClassifier


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

        response = {
            "status": "SUCCESS",
            "request_id": self._generate_request_id(),
            "timestamp_utc": self._get_utc_timestamp(),
            "image_name": image_filename,
            "processing_times": {
                "00_preprocessing_ms": round(preprocessing_time * 1000, 2),
                "01_inference_ms": round(classification_time * 1000, 2),
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

    def preprocess_image(self):
        """Preprocess a single image."""
        try:
            self.stats["total_requests"] += 1
            self.stats["preprocessing_requests"] += 1

            # Get image from request
            image = self._get_image_from_request(request)
            if image is None:
                return jsonify({"error": "No valid image provided"}), 400

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
            file = request.files.get("image")
            if not file or file.filename == "":
                return jsonify({"error": "No valid image provided"}), 400

            image_filename = file.filename

            # Read image
            image_bytes = file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                return jsonify({"error": "Failed to decode image"}), 400

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
            else:
                # Preprocessing-only mode (classifier not available)
                response = {
                    "status": "SUCCESS",
                    "request_id": self._generate_request_id(),
                    "timestamp_utc": self._get_utc_timestamp(),
                    "image_name": image_filename,
                    "processing_times": {
                        "00_preprocessing_ms": round(preprocessing_time * 1000, 2),
                        "01_inference_ms": round(classification_time * 1000, 2),
                        "total_ms": round(
                            (preprocessing_time + classification_time) * 1000, 2
                        ),
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
        """Batch full processing endpoint."""
        try:
            self.stats["total_requests"] += 1

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

            # Separate images and filenames
            images = [img for img, _ in image_data]
            filenames = [name for _, name in image_data]

            # Process batch
            start_time = time.time()
            preprocessing_results = self.preprocessor.process_batch(images, filenames)
            preprocessing_time = time.time() - start_time

            # Temporarily set voting strategy if different from default
            original_strategy = self.classifier.ensemble.voting_strategy
            self.classifier.ensemble.voting_strategy = voting_strategy

            try:
                # Classify each result
                classification_start = time.time()
                classification_results_list = []

                for i, variants in enumerate(preprocessing_results):
                    if variants:  # Preprocessing succeeded
                        try:
                            classification_results = self.classifier.classify(variants)
                            classification_results_list.append(
                                {
                                    "status": "success",
                                    "prediction": classification_results,
                                }
                            )
                        except Exception as e:
                            classification_results_list.append(
                                {
                                    "status": "classification_failed",
                                    "error": str(e),
                                }
                            )
                    else:  # Preprocessing failed
                        classification_results_list.append(
                            {
                                "status": "preprocessing_failed",
                                "error": "Preprocessing failed",
                            }
                        )

                classification_time = time.time() - classification_start
            finally:
                # Restore original voting strategy
                self.classifier.ensemble.voting_strategy = original_strategy

            self.stats["preprocessing_requests"] += len(images)
            self.stats["classification_requests"] += len(images)

            # Build structured response using the new schema
            response = self._build_batch_process_response(
                images=image_data,
                preprocessing_results=preprocessing_results,
                classification_results_list=classification_results_list,
                preprocessing_time=preprocessing_time,
                classification_time=classification_time,
                voting_strategy=voting_strategy,
            )

            return jsonify(response)

        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error(f"Batch processing error: {e}")
            return jsonify({"error": str(e)}), 500

    def _get_image_from_request(self, req) -> Optional[np.ndarray]:
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

            return image

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
        encoded_image = base64.b64encode(buffer).decode("utf-8")

        return encoded_image

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

    args = parser.parse_args()

    # Create server and run
    server = FundusInferenceServer(
        preprocessing_config=args.preprocessing_config,
        classifier_config=args.classifier_config,
        host=args.host,
        port=args.port,
        debug=args.debug,
    )

    server.run()


if __name__ == "__main__":
    main()
