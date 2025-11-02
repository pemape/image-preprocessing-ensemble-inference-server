"""
Batch Testing Script for Fundus Image Classifier
Supports multi-GPU processing and generates detailed JSON reports with continuous writing
"""

import os
import json
import time
import argparse
import logging
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import yaml
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from collections import defaultdict
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# Import our modules
from fundus_preprocessor import FundusPreprocessor
from diabetic_retinopathy_classifier import DiabeticRetinopathyClassifier


class ImageDataset(Dataset):
    """Dataset class for batch loading images."""

    def __init__(self, image_paths: List[Path], ground_truth: Optional[Dict] = None):
        """
        Initialize dataset.

        Args:
            image_paths: List of image file paths
            ground_truth: Optional dictionary mapping image_id to true_label
        """
        self.image_paths = image_paths
        self.ground_truth = ground_truth or {}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_id = image_path.name

        # Load image
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            # Return None for failed images
            return None

        true_label = self.ground_truth.get(image_id, -1)

        return {
            "image": image,
            "image_id": image_id,
            "image_path": str(image_path),
            "true_label": true_label,
        }


def collate_fn(batch):
    """Custom collate function to handle None values."""
    # Filter out None values
    batch = [item for item in batch if item is not None]
    return batch


class ContinuousJSONWriter:
    """Handles continuous writing of JSON results to avoid memory issues."""

    def __init__(self, output_file: Path):
        """Initialize the continuous JSON writer."""
        self.output_file = output_file
        self.batch_count = 0
        self.total_results = 0
        self.successful_results = 0
        self.failed_results = 0

        # Initialize the JSON file with metadata and start batch_results array
        self._initialize_file()

    def _initialize_file(self):
        """Initialize the JSON file structure."""
        initial_structure = {
            "batch_metadata": {},  # Will be updated at the end
            "processing_statistics": {},  # Will be updated at the end
            "evaluation_metrics": {},  # Will be calculated at the end
            "batch_results": [],  # Will be continuously appended
        }

        with open(self.output_file, "w") as f:
            json.dump(initial_structure, f, indent=2)

    def append_batch_results(self, batch_results: List[Dict]):
        """Append batch results to the JSON file."""
        # Read current file
        with open(self.output_file, "r") as f:
            data = json.load(f)

        # Append new results
        data["batch_results"].extend(batch_results)

        # Update counters
        self.batch_count += 1
        self.total_results += len(batch_results)
        self.successful_results += len(
            [r for r in batch_results if r["status"] == "success"]
        )
        self.failed_results += len(
            [r for r in batch_results if r["status"] == "failed"]
        )

        # Write back to file
        with open(self.output_file, "w") as f:
            json.dump(data, f, indent=2)

    def update_metadata(self, metadata: Dict):
        """Update the metadata section."""
        with open(self.output_file, "r") as f:
            data = json.load(f)

        data["batch_metadata"] = metadata

        with open(self.output_file, "w") as f:
            json.dump(data, f, indent=2)

    def update_processing_stats(self, stats: Dict):
        """Update processing statistics."""
        with open(self.output_file, "r") as f:
            data = json.load(f)

        data["processing_statistics"] = stats

        with open(self.output_file, "w") as f:
            json.dump(data, f, indent=2)

    def update_evaluation_metrics(self, metrics: Dict):
        """Update evaluation metrics."""
        with open(self.output_file, "r") as f:
            data = json.load(f)

        data["evaluation_metrics"] = metrics

        with open(self.output_file, "w") as f:
            json.dump(data, f, indent=2)

    def add_failed_images(self, failed_images: List[Dict]):
        """Add failed images list to the JSON."""
        with open(self.output_file, "r") as f:
            data = json.load(f)

        data["failed_images"] = failed_images

        with open(self.output_file, "w") as f:
            json.dump(data, f, indent=2)

    def get_current_results(self) -> List[Dict]:
        """Get current results from the file for metric calculation."""
        with open(self.output_file, "r") as f:
            data = json.load(f)
        return data.get("batch_results", [])


class BatchTester:
    """Main batch testing class with multi-GPU support and continuous JSON writing."""

    def __init__(self, config_path: str):
        """
        Initialize batch tester.

        Args:
            config_path: Path to batch testing configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.device = self._setup_device()

        # Initialize components
        self.preprocessor = self._setup_preprocessor()
        self.classifier = self._setup_classifier()

        # Setup output directory
        self.output_dir = Path(self.config["output"]["results_folder"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize continuous writer - create filename immediately
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.output_file = self.output_dir / f"test_results_{timestamp}.json"
        self.json_writer = ContinuousJSONWriter(self.output_file)

        # Lightweight tracking (no storing results in memory)
        self.failed_images = []
        self.processing_times = []

        self.logger.info(f"BatchTester initialized successfully")
        self.logger.info(f"Results will be continuously written to: {self.output_file}")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("BatchTester")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        return logger

    def _setup_device(self) -> torch.device:
        """Setup compute device with multi-GPU support."""
        gpu_config = self.config["gpu"]

        if gpu_config["force_cpu"] or not gpu_config["enabled"]:
            device = torch.device("cpu")
            self.logger.info("Using CPU for processing")
        elif torch.cuda.is_available():
            if gpu_config["device_ids"]:
                # Use specific GPU IDs
                device_ids = gpu_config["device_ids"]
                device = torch.device(f"cuda:{device_ids[0]}")
                self.logger.info(f"Using GPUs: {device_ids}")
            else:
                # Use all available GPUs
                device = torch.device("cuda:0")
                num_gpus = torch.cuda.device_count()
                self.logger.info(f"Using {num_gpus} GPUs")
        else:
            device = torch.device("cpu")
            self.logger.info("CUDA not available, using CPU")

        return device

    def _setup_preprocessor(self) -> FundusPreprocessor:
        """Setup image preprocessor."""
        preprocessing_config = self.config["model"]["preprocessing_config"]
        return FundusPreprocessor(preprocessing_config)

    def _setup_classifier(self) -> DiabeticRetinopathyClassifier:
        """Setup classifier with multi-GPU support."""
        classifier_config = self.config["model"]["classifier_config"]
        classifier = DiabeticRetinopathyClassifier(classifier_config)

        # Set voting strategy from config
        voting_strategy = self.config["model"].get("voting_strategy", "soft")
        classifier.ensemble.voting_strategy = voting_strategy

        # Setup data parallel if multiple GPUs available
        gpu_config = self.config["gpu"]
        if (
            gpu_config["enabled"]
            and not gpu_config["force_cpu"]
            and gpu_config["use_data_parallel"]
            and torch.cuda.device_count() > 1
        ):

            device_ids = gpu_config.get(
                "device_ids", list(range(torch.cuda.device_count()))
            )
            for model_info in classifier.ensemble.models:
                model = model_info["model"]
                if len(device_ids) > 1:
                    model = nn.DataParallel(model, device_ids=device_ids)
                    model_info["model"] = model
                    self.logger.info(
                        f"Enabled DataParallel for model on devices: {device_ids}"
                    )

        return classifier

    def _load_ground_truth(self) -> Dict[str, int]:
        """Load ground truth labels if available."""
        ground_truth_file = self.config["input"].get("ground_truth_file")
        if not ground_truth_file or not Path(ground_truth_file).exists():
            self.logger.info("No ground truth file provided or file not found")
            return {}

        # Get configurable column names
        csv_columns = self.config["input"].get("csv_columns", {})
        image_id_column = csv_columns.get("image_id_column", "image_id")
        label_column = csv_columns.get("label_column", "true_label")

        # Get optional label mapping
        label_mapping = self.config["testing"].get("label_mapping", {})

        ground_truth = {}
        try:
            with open(ground_truth_file, "r") as f:
                reader = csv.DictReader(f)

                # Validate required columns exist
                if image_id_column not in reader.fieldnames:
                    raise ValueError(
                        f"Column '{image_id_column}' not found in CSV. Available columns: {reader.fieldnames}"
                    )
                if label_column not in reader.fieldnames:
                    raise ValueError(
                        f"Column '{label_column}' not found in CSV. Available columns: {reader.fieldnames}"
                    )

                for row in reader:
                    image_id = row[image_id_column]
                    label_value = row[label_column]

                    # Apply label mapping if configured
                    if label_mapping and label_value in label_mapping:
                        true_label = label_mapping[label_value]
                    else:
                        # Try to convert to int, handle string labels
                        try:
                            true_label = int(label_value)
                        except ValueError:
                            # If it's a string label, try to map it to class index
                            class_names = self.config["testing"]["class_names"]
                            if label_value in class_names:
                                true_label = class_names.index(label_value)
                            else:
                                self.logger.warning(
                                    f"Unknown label '{label_value}' for image {image_id}, skipping"
                                )
                                continue

                    ground_truth[image_id] = true_label

            self.logger.info(f"Loaded ground truth for {len(ground_truth)} images")
            self.logger.info(
                f"Using columns: image_id='{image_id_column}', label='{label_column}'"
            )
            if label_mapping:
                self.logger.info(f"Applied label mapping: {label_mapping}")

        except Exception as e:
            self.logger.error(f"Failed to load ground truth file: {e}")

        return ground_truth

    def _get_image_paths(self) -> List[Path]:
        """Get list of image paths to process."""
        images_folder = Path(self.config["input"]["images_folder"])
        if not images_folder.exists():
            raise ValueError(f"Images folder not found: {images_folder}")

        supported_formats = self.config["input"]["supported_formats"]
        image_paths = []

        for format_ext in supported_formats:
            image_paths.extend(images_folder.glob(f"*{format_ext}"))
            # image_paths.extend(images_folder.glob(f"*{format_ext.upper()}"))

        image_paths.sort()
        self.logger.info(f"Found {len(image_paths)} images to process")

        return image_paths

    def _create_data_loader(
        self, image_paths: List[Path], ground_truth: Dict
    ) -> DataLoader:
        """Create data loader for batch processing."""
        dataset = ImageDataset(image_paths, ground_truth)

        batch_config = self.config["batch_processing"]

        dataloader = DataLoader(
            dataset,
            batch_size=batch_config["batch_size"],
            shuffle=False,
            num_workers=batch_config["num_workers"],
            prefetch_factor=batch_config["prefetch_factor"],
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available() and self.config["gpu"]["enabled"],
        )

        return dataloader

    def _process_batch(self, batch: List[Dict]) -> List[Dict]:
        """Process a batch of images."""
        batch_results = []
        batch_start_time = time.time()

        for item in batch:
            result = self._process_single_image(item)
            batch_results.append(result)

        batch_time = time.time() - batch_start_time
        self.processing_times.append(
            {
                "batch_size": len(batch),
                "batch_time": batch_time,
                "time_per_image": batch_time / len(batch) if batch else 0,
            }
        )

        return batch_results

    def _process_single_image(self, item: Dict) -> Dict:
        """Process a single image."""
        image_id = item["image_id"]
        image = item["image"]
        true_label = item["true_label"]

        start_time = time.time()

        try:
            # Preprocess image
            preprocessing_start = time.time()
            preprocessed_variants = self.preprocessor.process_image(image, image_id)
            preprocessing_time = time.time() - preprocessing_start

            # Classify image
            classification_start = time.time()
            classification_results = self.classifier.classify(preprocessed_variants)
            classification_time = time.time() - classification_start

            total_time = time.time() - start_time

            # Extract logits and predicted class
            class_names = self.config["testing"]["class_names"]
            logits = [classification_results.get(name, 0.0) for name in class_names]
            predicted_class = classification_results.get("predicted_class", "No DR")
            predicted_label = (
                class_names.index(predicted_class)
                if predicted_class in class_names
                else 0
            )
            confidence = classification_results.get("confidence", 0.0)

            result = {
                "image_id": image_id,
                "logits": logits,
                "predicted_label": predicted_label,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "true_label": true_label,
                "processing_times": {
                    "preprocessing_ms": preprocessing_time * 1000,
                    "classification_ms": classification_time * 1000,
                    "total_ms": total_time * 1000,
                },
                "status": "success",
            }

            # Save processed images if configured
            if self.config["output"]["save_processed_images"]:
                self._save_processed_images(image_id, preprocessed_variants)

        except Exception as e:
            self.logger.error(f"Failed to process image {image_id}: {e}")
            self.failed_images.append(
                {
                    "image_id": image_id,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            result = {
                "image_id": image_id,
                "logits": [0.0] * len(self.config["testing"]["class_names"]),
                "predicted_label": -1,
                "predicted_class": "unknown",
                "confidence": 0.0,
                "true_label": true_label,
                "processing_times": {
                    "preprocessing_ms": 0.0,
                    "classification_ms": 0.0,
                    "total_ms": 0.0,
                },
                "status": "failed",
                "error": str(e),
            }

        return result

    def _save_processed_images(self, image_id: str, variants: Dict):
        """Save preprocessed image variants."""
        processed_dir = self.output_dir / "processed_images" / image_id.split(".")[0]
        processed_dir.mkdir(parents=True, exist_ok=True)

        for variant_name, variant_image in variants.items():
            output_path = processed_dir / f"{variant_name}.jpg"

            # Convert to uint8 if needed
            if variant_image.dtype == np.float32 or variant_image.dtype == np.float64:
                if variant_image.max() <= 1.0:
                    variant_image = (variant_image * 255).astype(np.uint8)
                else:
                    variant_image = variant_image.astype(np.uint8)

            # Convert RGB to BGR for OpenCV
            variant_bgr = cv2.cvtColor(variant_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), variant_bgr)

    def _calculate_metrics_from_file(self) -> Dict:
        """Calculate evaluation metrics from results stored in file."""
        if not self.config["testing"]["calculate_metrics"]:
            return {}

        # Get results from file instead of memory
        all_results = self.json_writer.get_current_results()

        # Filter successful results with ground truth
        valid_results = [
            r for r in all_results if r["status"] == "success" and r["true_label"] != -1
        ]

        if not valid_results:
            self.logger.warning(
                "No valid results with ground truth for metric calculation"
            )
            return {}

        true_labels = [r["true_label"] for r in valid_results]
        predicted_labels = [r["predicted_label"] for r in valid_results]
        confidences = [r["confidence"] for r in valid_results]

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predicted_labels, average=None, zero_division=0
        )

        # Overall metrics
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)

        # Weighted metrics
        weighted_precision, weighted_recall, weighted_f1, _ = (
            precision_recall_fscore_support(
                true_labels, predicted_labels, average="weighted", zero_division=0
            )
        )

        # Confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)

        # Per-class metrics
        class_names = self.config["testing"]["class_names"]
        per_class_metrics = {}
        for i, class_name in enumerate(class_names):
            per_class_metrics[class_name] = {
                "precision": float(precision[i]) if i < len(precision) else 0.0,
                "recall": float(recall[i]) if i < len(recall) else 0.0,
                "f1_score": float(f1[i]) if i < len(f1) else 0.0,
                "support": int(support[i]) if i < len(support) else 0,
            }

        metrics = {
            "overall": {
                "accuracy": float(accuracy),
                "macro_precision": float(macro_precision),
                "macro_recall": float(macro_recall),
                "macro_f1": float(macro_f1),
                "weighted_precision": float(weighted_precision),
                "weighted_recall": float(weighted_recall),
                "weighted_f1": float(weighted_f1),
                "total_samples": len(valid_results),
            },
            "per_class": per_class_metrics,
            "confusion_matrix": cm.tolist(),
            "average_confidence": float(np.mean(confidences)),
            "confidence_std": float(np.std(confidences)),
        }

        return metrics

    def _save_intermediate_summary(self, batch_idx: int):
        """Save intermediate summary without duplicating results."""
        if not self.config["output"]["save_individual_results"]:
            return

        summary_file = self.output_dir / f"progress_summary_batch_{batch_idx}.json"

        # Create lightweight summary
        summary_data = {
            "session_info": {
                "processed_batches": batch_idx + 1,
                "total_images_processed": self.json_writer.total_results,
                "successful_images": self.json_writer.successful_results,
                "failed_images": self.json_writer.failed_results,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "processing_stats": {
                "batches_completed": len(self.processing_times),
                "average_batch_time": (
                    float(np.mean([pt["batch_time"] for pt in self.processing_times]))
                    if self.processing_times
                    else 0
                ),
                "average_time_per_image": (
                    float(
                        np.mean([pt["time_per_image"] for pt in self.processing_times])
                    )
                    if self.processing_times
                    else 0
                ),
            },
            "main_results_file": str(self.output_file.name),
        }

        with open(summary_file, "w") as f:
            json.dump(summary_data, f, indent=2)

    def _generate_metadata(
        self, start_time: datetime, end_time: datetime, total_images: int
    ) -> Dict:
        """Generate testing metadata."""
        total_time = (end_time - start_time).total_seconds()

        metadata = {
            "testing_session": {
                "session_id": f"batch_test_{start_time.strftime('%Y%m%d_%H%M%S')}",
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": total_time,
                "timestamp": end_time.isoformat(),
            },
            "configuration": {
                "model_config": {
                    "preprocessing_config": self.config["model"][
                        "preprocessing_config"
                    ],
                    "classifier_config": self.config["model"]["classifier_config"],
                    "voting_strategy": self.config["model"]["voting_strategy"],
                },
                "batch_processing": self.config["batch_processing"],
                "gpu_config": self.config["gpu"],
                "testing_parameters": self.config["testing"],
            },
            "system_info": {
                "cuda_available": torch.cuda.is_available(),
                "num_gpus": (
                    torch.cuda.device_count() if torch.cuda.is_available() else 0
                ),
                "device_names": (
                    [
                        torch.cuda.get_device_name(i)
                        for i in range(torch.cuda.device_count())
                    ]
                    if torch.cuda.is_available()
                    else []
                ),
                "python_version": f"{mp.sys.version_info.major}.{mp.sys.version_info.minor}.{mp.sys.version_info.micro}",
                "pytorch_version": torch.__version__,
            },
            "dataset_info": {
                "input_folder": self.config["input"]["images_folder"],
                "total_images_found": total_images,
                "total_images_processed": self.json_writer.total_results,
                "successful_processing": self.json_writer.successful_results,
                "failed_processing": self.json_writer.failed_results,
                "ground_truth_available": bool(
                    self.config["input"].get("ground_truth_file")
                ),
            },
        }

        # Add model information if available
        if hasattr(self.classifier, "get_model_info"):
            metadata["model_info"] = self.classifier.get_model_info()

        return metadata

    def run_batch_testing(self):
        """Run the complete batch testing process with continuous JSON writing."""
        start_time = datetime.now(timezone.utc)
        self.logger.info("Starting batch testing...")

        # Load ground truth
        ground_truth = self._load_ground_truth()

        # Get image paths
        image_paths = self._get_image_paths()
        if not image_paths:
            raise ValueError("No images found to process")

        # Create data loader
        dataloader = self._create_data_loader(image_paths, ground_truth)

        # Process batches
        total_batches = len(dataloader)
        log_interval = self.config["performance"]["log_interval"]
        save_interval = self.config["performance"]["save_interval"]

        self.logger.info(
            f"Processing {len(image_paths)} images in {total_batches} batches"
        )
        self.logger.info(
            f"Results are being continuously written to: {self.output_file}"
        )

        try:
            for batch_idx, batch in enumerate(dataloader):
                if not batch:  # Empty batch (all failed to load)
                    continue
                self.logger.info(f"Processing {batch_idx + 1}/{total_batches} batches")

                # Process batch
                batch_results = self._process_batch(batch)

                # IMMEDIATELY write results to file (no memory accumulation)
                self.json_writer.append_batch_results(batch_results)

                # Log progress
                if (batch_idx + 1) % log_interval == 0:
                    self.logger.info(
                        f"Processed {batch_idx + 1}/{total_batches} batches, "
                        f"{self.json_writer.total_results} images "
                        f"({self.json_writer.successful_results} successful, "
                        f"{self.json_writer.failed_results} failed)"
                    )

                # Save intermediate summary (lightweight)
                if (batch_idx + 1) % save_interval == 0:
                    self._save_intermediate_summary(batch_idx)
                    self.logger.info(
                        f"Intermediate summary saved at batch {batch_idx + 1}"
                    )

            end_time = datetime.now(timezone.utc)

            # Update metadata
            metadata = self._generate_metadata(start_time, end_time, len(image_paths))
            self.json_writer.update_metadata(metadata)

            # Calculate and update processing statistics
            processing_stats = {}
            if self.processing_times:
                batch_times = [pt["batch_time"] for pt in self.processing_times]
                per_image_times = [pt["time_per_image"] for pt in self.processing_times]

                processing_stats = {
                    "total_batches": len(self.processing_times),
                    "average_batch_time": float(np.mean(batch_times)),
                    "average_time_per_image": float(np.mean(per_image_times)),
                    "total_processing_time": float(np.sum(batch_times)),
                    "min_time_per_image": float(np.min(per_image_times)),
                    "max_time_per_image": float(np.max(per_image_times)),
                    "std_time_per_image": float(np.std(per_image_times)),
                }

            self.json_writer.update_processing_stats(processing_stats)

            # Calculate and update evaluation metrics
            metrics = self._calculate_metrics_from_file()
            self.json_writer.update_evaluation_metrics(metrics)

            # Add failed images list if configured
            if self.config["report"]["save_failed_images_list"] and self.failed_images:
                self.json_writer.add_failed_images(self.failed_images)

            # Save metrics summary separately
            if metrics:
                timestamp = end_time.strftime("%Y%m%d_%H%M%S")
                metrics_file = self.output_dir / f"metrics_summary_{timestamp}.json"
                with open(metrics_file, "w") as f:
                    json.dump(metrics, f, indent=2)
                self.logger.info(f"Metrics summary saved to: {metrics_file}")

            # Log summary
            self.logger.info(f"Batch testing completed!")
            self.logger.info(
                f"Total images processed: {self.json_writer.total_results}"
            )
            self.logger.info(f"Successful: {self.json_writer.successful_results}")
            self.logger.info(f"Failed: {self.json_writer.failed_results}")
            self.logger.info(
                f"Processing time: {(end_time - start_time).total_seconds():.2f} seconds"
            )
            self.logger.info(f"Final results saved to: {self.output_file}")

        except Exception as e:
            self.logger.error(f"Error during batch processing: {e}")
            # Even if there's an error, the results up to that point are safely saved
            self.logger.info(f"Partial results are saved in: {self.output_file}")
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch Testing for Fundus Image Classifier"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/batch_testing_config.yaml",
        help="Path to batch testing configuration file",
    )
    parser.add_argument(
        "--images-folder", type=str, help="Override images folder from config"
    )
    parser.add_argument(
        "--output-folder", type=str, help="Override output folder from config"
    )
    parser.add_argument(
        "--ground-truth", type=str, help="Override ground truth file from config"
    )
    parser.add_argument(
        "--batch-size", type=int, help="Override batch size from config"
    )
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU usage")

    args = parser.parse_args()

    # Load and modify config based on arguments
    config_path = args.config

    # Validate config file exists
    if not Path(config_path).exists():
        print(f"Error: Configuration file not found: {config_path}")
        return

    # Create batch tester
    try:
        batch_tester = BatchTester(config_path)

        # Override config with command line arguments
        if args.images_folder:
            batch_tester.config["input"]["images_folder"] = args.images_folder
        if args.output_folder:
            batch_tester.config["output"]["results_folder"] = args.output_folder
            batch_tester.output_dir = Path(args.output_folder)
            batch_tester.output_dir.mkdir(parents=True, exist_ok=True)
        if args.ground_truth:
            batch_tester.config["input"]["ground_truth_file"] = args.ground_truth
        if args.batch_size:
            batch_tester.config["batch_processing"]["batch_size"] = args.batch_size
        if args.force_cpu:
            batch_tester.config["gpu"]["force_cpu"] = True

        # Run batch testing
        batch_tester.run_batch_testing()

    except Exception as e:
        print(f"Error during batch testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
