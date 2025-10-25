"""
Fundus Image Preprocessor
Based on: "Ensemble of pre-processing techniques with CNN for diabetic retinopathy detection"
Paper: https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.12987

This module implements the complete preprocessing pipeline that generates 5 image variants:
1. Original clipped image
2. RGB-CLAHE enhanced image
3. Min-pooling enhanced image
4. Lab-CLAHE enhanced image
5. MaxGreenGsc enhanced image
"""

import cv2
import numpy as np
import yaml
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import os

# Optional imports for advanced features
try:
    from skimage import filters
    from scipy import ndimage

    ADVANCED_FILTERING = True
except ImportError:
    ADVANCED_FILTERING = False
    logging.warning(
        "Advanced filtering features disabled. Install scikit-image and scipy for full functionality."
    )


class FundusPreprocessor:
    """
    Main preprocessor class that implements the 5-variant preprocessing pipeline
    from the ensemble preprocessing paper.
    """

    def __init__(self, config_path: str):
        """
        Initialize the preprocessor with configuration.

        Args:
            config_path: Path to the YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self._validate_config()

        # Create output directories
        self._create_directories()

        self.logger.info("FundusPreprocessor initialized successfully")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {config_path}: {e}")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging based on configuration."""
        logger = logging.getLogger("FundusPreprocessor")
        debug_config = self.config.get("debug", {})
        log_severity = debug_config.get("log_level", "DEBUG")

        if log_severity.upper() == "DEBUG":
            level = logging.DEBUG
        else:
            level = logging.INFO

        logging.basicConfig(
            level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        return logger

    def _validate_config(self):
        """Validate configuration parameters."""
        required_sections = ["general", "black_border_clipping", "image_variants"]
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")

    def _create_directories(self):
        """Create necessary output directories."""
        # Only create debug directories if debug is BOTH enabled AND save_intermediate_images is true
        debug_config = self.config.get("debug", {})
        debug_fully_enabled = debug_config.get("enabled", False)

        if debug_fully_enabled:
            debug_paths = debug_config.get("intermediate_paths", {})
            for path_key, path_value in debug_paths.items():
                # Create the directory - path_value should be a directory path
                os.makedirs(path_value, exist_ok=True)
            self.logger.debug("Created debug directories")
        else:
            self.logger.debug(
                f"Debug directories skipped - enabled: {debug_config.get('enabled', False)}"
            )

        # Only create main output directory if NOT in debug mode or if explicitly needed
        output_dir = self.config.get("general", {}).get("output_directory")
        if output_dir and not debug_fully_enabled:
            os.makedirs(output_dir, exist_ok=True)
            self.logger.debug(f"Created output directory: {output_dir}")
        elif debug_fully_enabled:
            self.logger.debug(
                "Debug mode active - skipping main output directory creation"
            )
        else:
            self.logger.warning("No output directory specified in config")

    def process_image(
        self, image: np.ndarray, image_id: str = None
    ) -> Dict[str, np.ndarray]:
        """
        Process a single fundus image and generate all 5 variants.

        Args:
            image: Input fundus image as numpy array (BGR format from cv2)
            image_id: Optional identifier for the image (for debugging/logging)

        Returns:
            Dictionary containing all processed image variants
        """
        start_time = time.time()
        self.logger.info(f"Starting processing for image {image_id or 'unknown'}")

        # Step 1: Input validation
        if not self._validate_input_image(image):
            raise ValueError("Input image validation failed")

        # Convert BGR to RGB for processing
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Step 2: Black border clipping
        clipped_image = self._clip_black_borders(rgb_image)

        if self.config.get("debug", {}).get("enabled", False):
            self._save_debug_image(clipped_image, "black_border_clipped", image_id)

        # Step 3: Generate 5 image variants
        variants = {}

        if self.config.get("performance", {}).get("parallel_variants", True):
            variants = self._process_variants_parallel(clipped_image, image_id)
        else:
            variants = self._process_variants_sequential(clipped_image, image_id)

        # Step 4: Final resolution processing and normalization
        final_variants = {}
        for variant_name, variant_image in variants.items():
            processed = self._final_processing(variant_image, variant_name)
            final_variants[variant_name] = processed

            # Save final processed variants - always save if output directory is configured
            output_dir = self.config.get("general", {}).get("output_directory")
            debug_enabled = self.config.get("debug", {}).get("enabled", False)

            if debug_enabled or output_dir:
                self._save_final_variant(processed, variant_name, image_id)

        processing_time = time.time() - start_time
        self.logger.info(f"Processing completed in {processing_time:.2f} seconds")

        if self.config.get("debug", {}).get("log_processing_time", True):
            self.logger.info(f"Processing time breakdown: {processing_time:.2f}s total")

        return final_variants

    def _validate_input_image(self, image: np.ndarray) -> bool:
        """Validate input image according to configuration."""
        if not self.config.get("quality_control", {}).get("enabled", True):
            return True

        validation_config = self.config.get("quality_control", {}).get(
            "input_validation", {}
        )

        # Check image dimensions
        height, width = image.shape[:2]
        min_res = validation_config.get("min_resolution", [224, 224])
        max_res = validation_config.get("max_resolution", [4096, 4096])

        if width < min_res[0] or height < min_res[1]:
            self.logger.error(
                f"Image resolution {width}x{height} below minimum {min_res[0]}x{min_res[1]}"
            )
            return False

        if width > max_res[0] or height > max_res[1]:
            self.logger.error(
                f"Image resolution {width}x{height} above maximum {max_res[0]}x{max_res[1]}"
            )
            return False

        # Check for corruption
        if validation_config.get("check_corruption", True):
            if np.any(np.isnan(image)) or np.any(np.isinf(image)):
                self.logger.error("Image contains NaN or infinite values")
                return False

        return True

    def _clip_black_borders(self, image: np.ndarray) -> np.ndarray:
        """
        Remove black borders from fundus image using all 4 methods and return the best result.
        In debug mode, saves outputs from all methods for comparison.
        Methods implemented: morphological, contour_detection, adaptive_threshold, fixed_threshold

        Args:
            image: RGB image as numpy array

        Returns:
            Best clipped image based on crop area
        """
        if not self.config.get("black_border_clipping", {}).get("enabled", True):
            return image

        original_area = image.shape[0] * image.shape[1]
        results = {}
        debug_config = self.config.get("debug", {})
        debug_enabled = debug_config.get("enabled", False)

        debug_intermediate_paths = debug_config.get("save_intermediate_images", False)
        # Get the main method from config
        main_method = self.config.get("black_border_clipping", {}).get(
            "method", "fixed_threshold"
        )

        # Test all 4 methods for comparison (always run all in debug mode)
        methods_to_test = [
            "morphological",
            "contour_detection",
            "adaptive_threshold",
            "fixed_threshold",
        ]

        for method in methods_to_test:
            try:
                if method == "morphological":
                    result_image = self._clip_morphological_direct(image)
                elif method == "contour_detection":
                    result_image = self._clip_contour_detection_direct(image)
                elif method == "adaptive_threshold":
                    result_image = self._clip_adaptive_threshold_direct(image)
                elif method == "fixed_threshold":
                    result_image = self._clip_fixed_threshold_direct(image)

                crop_area = result_image.shape[0] * result_image.shape[1]
                crop_ratio = crop_area / original_area

                if crop_ratio > 0.3:  # Minimum acceptable crop ratio
                    results[method] = (result_image, crop_ratio)
                    self.logger.debug(f"{method} method: crop ratio = {crop_ratio:.3f}")

                    # Save debug image for this method (always save all methods in debug mode)
                    if debug_enabled and debug_intermediate_paths:
                        self._save_border_clipping_debug_image(
                            result_image, method, crop_ratio
                        )
                else:
                    self.logger.warning(
                        f"{method} method: crop ratio {crop_ratio:.3f} below minimum threshold"
                    )
                    if debug_enabled:
                        # Save even failed attempts for analysis
                        self._save_border_clipping_debug_image(
                            result_image, f"{method}_failed", crop_ratio
                        )

            except Exception as e:
                self.logger.warning(f"{method} method failed: {e}")
                if debug_enabled and debug_intermediate_paths:
                    # Save original image as fallback for failed method
                    self._save_border_clipping_debug_image(
                        image, f"{method}_error", 0.0
                    )

        # Select the best result or use the main method if specified
        if not results:
            self.logger.warning("All clipping methods failed, returning original image")
            if debug_enabled:
                self._save_border_clipping_debug_image(
                    image, "no_clipping_applied", 1.0
                )
            return image

        # Use main method if it succeeded, otherwise use the best result
        if main_method in results:
            selected_method = main_method
            selected_image = results[main_method][0]
            selected_ratio = results[main_method][1]
            self.logger.info(
                f"Using main method '{main_method}' with crop ratio {selected_ratio:.3f}"
            )
        else:
            # Fallback to best method if main method failed
            selected_method = max(results.keys(), key=lambda k: results[k][1])
            selected_image = results[selected_method][0]
            selected_ratio = results[selected_method][1]
            self.logger.warning(
                f"Main method '{main_method}' failed, using best alternative '{selected_method}' with crop ratio {selected_ratio:.3f}"
            )

        # Save final clipping report
        if debug_enabled:
            # Create a comparison summary
            self._create_border_clipping_comparison_report(
                results, selected_method, main_method
            )

        return selected_image

    def _save_border_clipping_debug_image(
        self,
        image: np.ndarray,
        method_name: str,
        crop_ratio: float,
        image_id: str = None,
    ):
        """Save border clipping debug images with detailed naming."""
        # Check BOTH debug.enabled AND save_intermediate_images
        debug_config = self.config.get("debug", {})
        if not (
            debug_config.get("enabled", False)
            and debug_config.get("save_intermediate_images", False)
        ):
            return

        debug_paths = debug_config.get("intermediate_paths", {})
        black_border_path = debug_paths.get(
            "black_border_clipped", "./debug/01_black_border_clipped/"
        )

        try:
            # Create the debug directory if it doesn't exist
            os.makedirs(black_border_path, exist_ok=True)

            # Ensure image is in correct format for saving
            if image.dtype == np.float32:
                save_image = (image * 255).astype(np.uint8)
            else:
                save_image = image

            # Convert RGB to BGR for OpenCV saving
            if len(save_image.shape) == 3:
                save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)

            # Create detailed filename with crop ratio
            base_name = image_id or "image"
            filename = f"{base_name}_{method_name}_crop_{crop_ratio:.3f}.jpg"
            filepath = os.path.join(black_border_path, filename)

            # Save image
            cv2.imwrite(filepath, save_image)
            self.logger.debug(f"Saved border clipping debug image: {filepath}")

        except Exception as e:
            self.logger.error(
                f"Failed to save border clipping debug image {method_name}: {e}"
            )

    def _create_border_clipping_comparison_report(
        self,
        results: dict,
        selected_method: str,
        main_method: str,
        image_id: str = None,
    ):
        """Create a detailed comparison report of all border clipping methods."""
        # Check BOTH debug.enabled
        debug_config = self.config.get("debug", {})
        if not (debug_config.get("enabled", False)):
            return

        debug_paths = debug_config.get("intermediate_paths", {})
        black_border_path = debug_paths.get(
            "black_border_clipped", "./debug/01_black_border_clipped/"
        )

        try:
            # Create report data
            report = {
                "timestamp": time.time(),
                "image_id": image_id or "unknown",
                "main_method_configured": main_method,
                "selected_method": selected_method,
                "total_methods_tested": len(results)
                + (4 - len(results)),  # Include failed methods
                "successful_methods": len(results),
                "method_results": {},
            }

            # Add results for each method
            for method, (_, crop_ratio) in results.items():
                report["method_results"][method] = {
                    "success": True,
                    "crop_ratio": crop_ratio,
                    "selected": method == selected_method,
                    "is_main_method": method == main_method,
                }

            # Add info about failed methods
            all_methods = [
                "morphological",
                "contour_detection",
                "adaptive_threshold",
                "fixed_threshold",
            ]
            failed_methods = [m for m in all_methods if m not in results]
            for method in failed_methods:
                report["method_results"][method] = {
                    "success": False,
                    "crop_ratio": 0.0,
                    "selected": False,
                    "is_main_method": method == main_method,
                    "failure_reason": "Method failed or crop ratio below threshold",
                }

            # Save report as JSON
            base_name = image_id or "image"
            report_filename = f"report_{base_name}_border_clipping_comparison.json"
            report_filepath = os.path.join(black_border_path, report_filename)

            with open(report_filepath, "w") as f:
                json.dump(report, f, indent=2)

            self.logger.debug(
                f"Saved border clipping comparison report: {report_filepath}"
            )

            # Log summary to console
            self.logger.info("Border Clipping Methods Comparison:")
            for method in all_methods:
                if method in results:
                    crop_ratio = results[method][1]
                    status = "✓ SELECTED" if method == selected_method else "✓"
                    main_indicator = " (MAIN)" if method == main_method else ""
                    self.logger.info(
                        f"  {method}{main_indicator}: {status} crop_ratio={crop_ratio:.3f}"
                    )
                else:
                    main_indicator = " (MAIN)" if method == main_method else ""
                    self.logger.info(f"  {method}{main_indicator}: ✗ FAILED")

        except Exception as e:
            self.logger.error(
                f"Failed to create border clipping comparison report: {e}"
            )

    def _clip_morphological_direct(self, image: np.ndarray) -> np.ndarray:
        """Apply morphological operations to detect and remove black borders."""
        config = self.config.get("black_border_clipping", {}).get("morphological", {})

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply threshold
        threshold = config.get("threshold", 10)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # Morphological operations
        kernel_size = config.get("kernel_size", (5, 5))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size[0], kernel_size[1])
        )

        # Opening to remove noise
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Closing to fill gaps
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return image

        # Find largest contour (main fundus region)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Apply margin
        margin = config.get("margin", 5)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)

        return image[y : y + h, x : x + w]

    def _clip_contour_detection_direct(self, image: np.ndarray) -> np.ndarray:
        """Apply contour detection to find and crop fundus region."""
        config = self.config.get("black_border_clipping", {}).get(
            "contour_detection", {}
        )

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur
        blur_kernel = config.get("blur_kernel", 5)
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

        # Apply threshold
        threshold = config.get("threshold_value", 15)
        _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return image

        # Filter contours by area
        min_area = config.get("min_contour_area", 1000)
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

        if not valid_contours:
            return image

        # Find largest valid contour
        largest_contour = max(valid_contours, key=cv2.contourArea)

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Apply margin
        margin = config.get("margin", 10)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)

        return image[y : y + h, x : x + w]

    def _clip_adaptive_threshold_direct(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding to detect fundus boundaries."""
        config = self.config.get("black_border_clipping", {}).get(
            "adaptive_threshold", {}
        )

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply adaptive threshold
        block_size = config.get("block_size", 11)
        c_constant = config.get("c_constant", 2)

        adaptive_thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c_constant,
        )

        # Morphological operations to clean up
        kernel_size = config.get("kernel_size", 3)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return image

        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Apply margin
        margin = config.get("margin", 8)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)

        return image[y : y + h, x : x + w]

    def _clip_fixed_threshold_direct(self, image: np.ndarray) -> np.ndarray:
        """
        Apply fixed thresholding to detect fundus region using tolerance-based border detection.
        This method implements sophisticated border detection with different tolerances for
        dark borders and bright borders (common in some fundus datasets).
        """
        config = self.config.get("black_border_clipping", {}).get("fixed_threshold", {})

        # Get tolerance parameters
        default_tolerance = config.get("default_tolerance", 25)
        bright_border_tolerance = config.get("bright_border_tolerance", 249)
        bright_border_threshold = config.get("bright_border_threshold", 200)
        use_grayscale = config.get("use_grayscale_conversion", True)

        self.logger.debug(
            f"Fixed threshold params: default_tolerance={default_tolerance}, "
            f"bright_border_tolerance={bright_border_tolerance}, "
            f"bright_border_threshold={bright_border_threshold}, "
            f"use_grayscale={use_grayscale}"
        )

        # Step 1: Prepare image for processing
        if use_grayscale:
            # Convert to grayscale for border detection
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
        else:
            # Use original image (assuming it's already grayscale or we want to work with RGB)
            gray = image.copy()

        # Step 2: Create binary mask based on tolerance thresholds
        mask = np.zeros(gray.shape, dtype=np.uint8)

        if use_grayscale and len(image.shape) == 3:
            # Working with grayscale version of RGB image

            # Detect bright borders (pixels > bright_border_threshold get high tolerance)
            bright_border_mask = gray > bright_border_threshold

            # Apply different tolerances
            # For bright borders: use bright_border_tolerance
            mask[bright_border_mask] = (
                gray[bright_border_mask] < bright_border_tolerance
            ).astype(np.uint8) * 255

            # For normal/dark areas: use default_tolerance
            normal_mask = ~bright_border_mask
            mask[normal_mask] = (gray[normal_mask] > default_tolerance).astype(
                np.uint8
            ) * 255

        else:
            # Simple thresholding fallback
            _, mask = cv2.threshold(gray, default_tolerance, 255, cv2.THRESH_BINARY)

        # Step 3: Find the region of interest from the binary mask
        # Find non-zero pixels (the fundus region)
        coords = cv2.findNonZero(mask)

        if coords is None:
            self.logger.warning(
                "Fixed threshold: No valid region found, returning original image"
            )
            return image

        # Get bounding rectangle of the fundus region
        x, y, w, h = cv2.boundingRect(coords)

        # Step 4: Apply padding/margin
        padding_percent = self.config.get("black_border_clipping", {}).get(
            "padding_percent", 0.02
        )

        # Calculate margin based on image size or use fixed margin
        if padding_percent > 0:
            margin_x = int(image.shape[1] * padding_percent)
            margin_y = int(image.shape[0] * padding_percent)
        else:
            margin_x = margin_y = config.get("margin", 5)

        # Apply margins with bounds checking
        x = max(0, x - margin_x)
        y = max(0, y - margin_y)
        w = min(image.shape[1] - x, w + 2 * margin_x)
        h = min(image.shape[0] - y, h + 2 * margin_y)

        # Step 5: Validate crop ratio
        original_area = image.shape[0] * image.shape[1]
        crop_area = w * h
        crop_ratio = crop_area / original_area
        min_crop_ratio = self.config.get("black_border_clipping", {}).get(
            "min_crop_ratio", 0.3
        )

        if crop_ratio < min_crop_ratio:
            self.logger.warning(
                f"Fixed threshold: Crop ratio {crop_ratio:.3f} below minimum {min_crop_ratio}, returning original"
            )
            return image

        # Step 6: Return cropped image
        cropped = image[y : y + h, x : x + w]

        self.logger.debug(
            f"Fixed threshold: Successfully cropped from {image.shape[:2]} to {cropped.shape[:2]} "
            f"(ratio: {crop_ratio:.3f})"
        )

        return cropped

    def _process_variants_parallel(
        self, clipped_image: np.ndarray, image_id: str
    ) -> Dict[str, np.ndarray]:
        """Process all image variants in parallel."""
        variants = {}

        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all variant processing tasks
            futures = {
                executor.submit(
                    self._create_original_variant, clipped_image
                ): "original",
                executor.submit(
                    self._create_rgb_clahe_variant, clipped_image
                ): "rgb_clahe",
                executor.submit(
                    self._create_min_pooling_variant, clipped_image
                ): "min_pooling",
                executor.submit(
                    self._create_lab_clahe_variant, clipped_image
                ): "lab_clahe",
                executor.submit(
                    self._create_max_green_gsc_variant, clipped_image
                ): "max_green_gsc",
            }

            # Collect results
            for future in as_completed(futures):
                variant_name = futures[future]
                try:
                    result = future.result()
                    variants[variant_name] = result
                    self.logger.debug(f"Completed {variant_name} variant")
                    if self.config.get("debug", {}).get("enabled", False):
                        self._save_debug_image(result, variant_name, image_id)
                except Exception as e:
                    self.logger.error(f"Failed to process {variant_name} variant: {e}")
                    # Create fallback variant
                    variants[variant_name] = clipped_image.copy()

        return variants

    def _process_variants_sequential(
        self, clipped_image: np.ndarray, image_id: str
    ) -> Dict[str, np.ndarray]:
        """Process all image variants sequentially."""
        variants = {}

        try:
            variants["original"] = self._create_original_variant(clipped_image)
            if self.config.get("debug", {}).get("enabled", False):
                self._save_debug_image(variants["original"], "original", image_id)
        except Exception as e:
            self.logger.error(f"Failed to create original variant: {e}")
            variants["original"] = clipped_image.copy()

        try:
            variants["rgb_clahe"] = self._create_rgb_clahe_variant(clipped_image)
            if self.config.get("debug", {}).get("enabled", False):
                self._save_debug_image(variants["rgb_clahe"], "rgb_clahe", image_id)
        except Exception as e:
            self.logger.error(f"Failed to create RGB-CLAHE variant: {e}")
            variants["rgb_clahe"] = clipped_image.copy()

        try:
            variants["min_pooling"] = self._create_min_pooling_variant(clipped_image)
            if self.config.get("debug", {}).get("enabled", False):
                self._save_debug_image(variants["min_pooling"], "min_pooling", image_id)
        except Exception as e:
            self.logger.error(f"Failed to create min-pooling variant: {e}")
            variants["min_pooling"] = clipped_image.copy()

        try:
            variants["lab_clahe"] = self._create_lab_clahe_variant(clipped_image)
            if self.config.get("debug", {}).get("enabled", False):
                self._save_debug_image(variants["lab_clahe"], "lab_clahe", image_id)
        except Exception as e:
            self.logger.error(f"Failed to create Lab-CLAHE variant: {e}")
            variants["lab_clahe"] = clipped_image.copy()

        try:
            variants["max_green_gsc"] = self._create_max_green_gsc_variant(
                clipped_image
            )
            if self.config.get("debug", {}).get("enabled", False):
                self._save_debug_image(
                    variants["max_green_gsc"], "max_green_gsc", image_id
                )
        except Exception as e:
            self.logger.error(f"Failed to create MaxGreenGsc variant: {e}")
            variants["max_green_gsc"] = clipped_image.copy()

        return variants

    def _create_original_variant(self, image: np.ndarray) -> np.ndarray:
        """Create the original clipped image variant."""
        return image.copy()

    def _create_rgb_clahe_variant(self, image: np.ndarray) -> np.ndarray:
        """Create RGB-CLAHE enhanced variant."""
        config = self.config.get("image_variants", {}).get("rgb_clahe", {})

        # Apply CLAHE to each RGB channel
        clip_limit = config.get("clip_limit", 2.0)
        tile_grid_size = tuple(config.get("tile_grid_size", [8, 8]))

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        # Process each channel
        result = np.zeros_like(image)
        for i in range(3):  # RGB channels
            result[:, :, i] = clahe.apply(image[:, :, i])

        return result

    def _create_min_pooling_variant(self, image: np.ndarray) -> np.ndarray:
        """Create Ben Graham enhanced variant (originally labeled as min-pooling)."""
        config = self.config.get("image_variants", {}).get("min_pooling", {})

        # Get Ben Graham parameters from config
        enhancement_factor = config.get("enhancement_factor", 4)
        blur_factor = config.get("blur_factor", -4)
        brightness_offset = config.get("brightness_offset", 128)

        # Apply Gaussian blur
        gaussian_config = config.get("gaussian_blur", {})
        sigma_x = gaussian_config.get("sigma_x", 10)
        kernel_size = tuple(gaussian_config.get("kernel_size", [0, 0]))
        sigma_y = gaussian_config.get("sigma_y", 0)

        # Create Gaussian blurred version
        blurred_image = cv2.GaussianBlur(image, kernel_size, sigma_x, sigmaY=sigma_y)

        # Apply Ben Graham enhancement: final = enhancement_factor * original + blur_factor * blurred + brightness_offset
        # This is equivalent to: cv2.addWeighted(img, 4, blurred_image, -4, 128)
        # Feature enhancement : https://programmersought.com/article/46784682579/
        result = cv2.addWeighted(
            image, enhancement_factor, blurred_image, blur_factor, brightness_offset
        )

        # Post-processing options
        post_config = config.get("post_processing", {})

        # Clip values to valid range if enabled
        if post_config.get("clip_values", True):
            result = np.clip(result, 0, 255)

        # Normalize output if enabled
        if post_config.get("normalize", False):
            result = result.astype(np.float32) / 255.0

        self.logger.debug(
            f"Ben Graham enhancement applied: sigma_x={sigma_x}, enhancement_factor={enhancement_factor}, blur_factor={blur_factor}, brightness_offset={brightness_offset}"
        )

        return result

    def _create_lab_clahe_variant(self, image: np.ndarray) -> np.ndarray:
        """Create Lab-CLAHE enhanced variant."""
        config = self.config.get("image_variants", {}).get("lab_clahe", {})

        # Convert RGB to Lab
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        # Apply CLAHE to L channel
        clip_limit = config.get("clip_limit", 3.0)
        tile_grid_size = tuple(config.get("tile_grid_size", [8, 8]))

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        # Convert back to RGB
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return result

    def _create_max_green_gsc_variant(self, image: np.ndarray) -> np.ndarray:
        """Create MaxGreenGsc (Maximum Green + Grayscale Conversion) variant."""
        config = self.config.get("image_variants", {}).get("max_green_gsc", {})

        # Get max RGB filter configuration
        max_rgb_config = config.get("max_rgb_filter", {})
        collect_stats = max_rgb_config.get("collect_statistics", True)

        # Get channel combination configuration
        channel_config = config.get("channel_combination", {})
        use_clahe = channel_config.get("use_clahe", True)
        components = channel_config.get("components", {})

        # Get enhancement configuration
        enhancement_config = config.get("enhancement", {})
        clahe_config = enhancement_config.get("clahe", {})
        clahe_clip_limit = clahe_config.get("clip_limit", 4.0)
        clahe_tile_size = tuple(clahe_config.get("tile_grid_size", [8, 8]))

        # Step 1: Create max RGB filter (based on your original max_rgb_filter function)
        B, G, R = cv2.split(image)
        M = np.maximum(np.maximum(R, G), B)

        # Optional: Collect statistics about RGB channel dominance
        if collect_stats:
            diff_max_R = M == R
            diff_max_G = M == G
            diff_max_B = M == B

            red_dominance = np.sum(diff_max_R) / diff_max_R.size * 100
            green_dominance = np.sum(diff_max_G) / diff_max_G.size * 100
            blue_dominance = np.sum(diff_max_B) / diff_max_B.size * 100

            self.logger.debug(
                f"RGB dominance - R: {red_dominance:.1f}%, G: {green_dominance:.1f}%, B: {blue_dominance:.1f}%"
            )

        # Step 2: Get green channel
        green_channel = G

        # Step 3: Create grayscale version
        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Step 4: Apply CLAHE if enabled (based on your original build_max_green_gsc_3d_clahe_img)
        if use_clahe:
            clahe = cv2.createCLAHE(
                clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_size
            )

            # Apply CLAHE to each component
            max_equalized = clahe.apply(M)
            green_equalized = clahe.apply(green_channel)
            gsc_equalized = clahe.apply(grayscale)

            # Merge the CLAHE-enhanced channels (max_rgb, green, grayscale)
            result = cv2.merge((max_equalized, green_equalized, gsc_equalized))
        else:
            # Merge without CLAHE enhancement
            result = cv2.merge((M, green_channel, grayscale))

        # Step 5: RGB conversion method
        rgb_conversion_config = config.get("rgb_conversion", {})
        conversion_method = rgb_conversion_config.get("method", "merge_channels")

        if conversion_method == "replicate":
            # Replicate one channel to all three
            if len(result.shape) == 2:
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        elif conversion_method == "colorize":
            # Apply false coloring (if grayscale)
            if len(result.shape) == 2:
                colormap = getattr(
                    cv2,
                    f'COLORMAP_{rgb_conversion_config.get("colorize_map", "JET").upper()}',
                    cv2.COLORMAP_JET,
                )
                result = cv2.applyColorMap(result, colormap)
                result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        # For 'merge_channels', result is already a 3-channel image

        # Step 6: Optional vessel enhancement (if enabled)
        vessel_config = enhancement_config.get("vessel_enhancement", {})
        if vessel_config.get("enabled", False) and ADVANCED_FILTERING:
            try:
                vessel_method = vessel_config.get("method", "frangi")
                sigma_range = vessel_config.get("sigma_range", [1, 8])
                sigma_step = vessel_config.get("sigma_step", 2)

                # Convert to grayscale for vessel detection
                gray_for_vessels = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)

                if vessel_method == "frangi":
                    from skimage.filters import frangi

                    alpha = vessel_config.get("frangi_alpha", 0.5)
                    beta = vessel_config.get("frangi_beta", 0.5)
                    gamma = vessel_config.get("frangi_gamma", 15)

                    vessels = frangi(
                        gray_for_vessels,
                        sigmas=range(sigma_range[0], sigma_range[1], sigma_step),
                        alpha=alpha,
                        beta=beta,
                        gamma=gamma,
                    )
                    vessels = (vessels * 255).astype(np.uint8)
                    result = cv2.cvtColor(vessels, cv2.COLOR_GRAY2RGB)

                self.logger.debug(f"Applied {vessel_method} vessel enhancement")
            except Exception as e:
                self.logger.warning(f"Vessel enhancement failed: {e}")

        self.logger.debug(
            f"MaxGreenGsc variant created: use_clahe={use_clahe}, conversion_method={conversion_method}"
        )

        return result

    def _final_processing(self, image: np.ndarray, variant_name: str) -> np.ndarray:
        """Apply final processing steps to each variant."""
        config = self.config.get("general", {})

        # Resize to target resolution
        target_size = tuple(config.get("target_resolution", [224, 224]))
        if image.shape[:2] != target_size:
            resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        else:
            resized = image

        # Normalize pixel values
        if config.get("normalize_pixels", True):
            normalized = resized.astype(np.float32) / 255.0
        else:
            normalized = resized

        # Apply final quality checks
        if self.config.get("quality_control", {}).get("final_validation", True):
            if not self._validate_final_image(normalized, variant_name):
                self.logger.warning(f"Final validation failed for {variant_name}")

        return normalized

    def _validate_final_image(self, image: np.ndarray, variant_name: str) -> bool:
        """Validate final processed image."""
        # Check for NaN or infinite values
        if np.any(np.isnan(image)) or np.any(np.isinf(image)):
            self.logger.error(
                f"Final image {variant_name} contains NaN or infinite values"
            )
            return False

        # Check value range
        if image.min() < 0 or image.max() > 1:
            self.logger.warning(
                f"Final image {variant_name} has values outside [0,1] range"
            )

        # Check for completely black or white images
        if image.max() - image.min() < 0.01:
            self.logger.warning(f"Final image {variant_name} has very low contrast")

        return True

    def _save_debug_image(self, image: np.ndarray, stage_name: str, image_id: str):
        """Save intermediate images for debugging."""
        # Check BOTH debug.enabled
        debug_config = self.config.get("debug", {})
        if not (debug_config.get("enabled", False)):
            return

        debug_paths = debug_config.get("intermediate_paths", {})
        if stage_name not in debug_paths:
            return

        try:
            # Ensure image is in correct format for saving
            if image.dtype == np.float32:
                save_image = (image * 255).astype(np.uint8)
            else:
                save_image = image

            # Convert RGB to BGR for OpenCV saving
            if len(save_image.shape) == 3:
                save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)

            # Create filename
            filename = f"{image_id or 'unknown'}_{stage_name}.jpg"
            filepath = os.path.join(debug_paths[stage_name], filename)

            # Save image
            cv2.imwrite(filepath, save_image)
            self.logger.debug(f"Saved debug image: {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save debug image {stage_name}: {e}")

    def _save_final_variant(self, image: np.ndarray, variant_name: str, image_id: str):
        """Save final processed variants efficiently - avoid duplication."""
        debug_config = self.config.get("debug", {})
        debug_enabled = debug_config.get("enabled", False)

        # Get configured output directory - no fallback to avoid unwanted folder creation
        output_dir = self.config.get("general", {}).get("output_directory")

        if not output_dir:
            self.logger.warning(
                "No output directory specified in config - skipping final variant save"
            )
            return

        # In debug mode, save to debug intermediate_paths instead of duplicating
        if debug_enabled:
            debug_paths = debug_config.get("intermediate_paths", {})
            final_resized_path = debug_paths.get(
                "final_resized", "./debug/07_final_resized/"
            )
            save_location = final_resized_path
            self.logger.debug(
                f"Debug mode: saving final variants to {save_location} instead of {output_dir}"
            )
        else:
            # Normal mode: use configured output directory
            save_location = output_dir

        try:
            # Create the directory if it doesn't exist
            os.makedirs(save_location, exist_ok=True)

            # Ensure image is in correct format for saving
            if image.dtype == np.float32:
                save_image = (image * 255).astype(np.uint8)
            else:
                save_image = image

            # Convert RGB to BGR for OpenCV saving
            if len(save_image.shape) == 3:
                save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)

            # Create filename with resolution info for debug mode
            if debug_enabled:
                resolution = f"{image.shape[1]}x{image.shape[0]}"
                filename = (
                    f"{image_id or 'unknown'}_{variant_name}_final_{resolution}.jpg"
                )
            else:
                filename = f"{image_id or 'unknown'}_{variant_name}.jpg"

            filepath = os.path.join(save_location, filename)

            # Save image
            cv2.imwrite(filepath, save_image)
            self.logger.debug(f"Saved final variant image: {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save final variant image {variant_name}: {e}")

    def process_batch(
        self, images: List[np.ndarray], image_ids: List[str] = None
    ) -> List[Dict[str, np.ndarray]]:
        """
        Process a batch of images.

        Args:
            images: List of input images
            image_ids: Optional list of image identifiers

        Returns:
            List of dictionaries containing processed variants for each image
        """
        if image_ids is None:
            image_ids = [f"image_{i}" for i in range(len(images))]

        results = []

        if self.config.get("performance", {}).get("parallel_images", True):
            # Process images in parallel
            with ThreadPoolExecutor(
                max_workers=self.config.get("performance", {}).get("max_workers", 4)
            ) as executor:
                futures = {
                    executor.submit(self.process_image, img, img_id): (img, img_id)
                    for img, img_id in zip(images, image_ids)
                }

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        img, img_id = futures[future]
                        self.logger.error(f"Failed to process image {img_id}: {e}")
                        # Add empty result to maintain order
                        results.append({})
        else:
            # Process images sequentially
            for img, img_id in zip(images, image_ids):
                try:
                    result = self.process_image(img, img_id)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to process image {img_id}: {e}")
                    results.append({})

        return results


def create_default_config(output_path: str):
    """Create a default configuration file."""
    default_config = {
        "general": {
            "target_resolution": [224, 224],
            "normalize_pixels": True,
            "output_directory": "./processed_images",
        },
        "black_border_clipping": {
            "enabled": True,
            "morphological": {"threshold": 10, "kernel_size": 5, "margin": 5},
            "contour_detection": {
                "threshold": 15,
                "blur_kernel": 5,
                "min_contour_area": 1000,
                "margin": 10,
            },
            "adaptive_threshold": {
                "block_size": 11,
                "c_constant": 2,
                "kernel_size": 3,
                "margin": 8,
            },
            "fixed_threshold": {
                "default_tolerance": 25,
                "bright_border_tolerance": 249,
                "bright_border_threshold": 200,
                "use_grayscale_conversion": True,
                "margin": 5,
            },
        },
        "image_variants": {
            "rgb_clahe": {"clip_limit": 2.0, "tile_grid_size": [8, 8]},
            "min_pooling": {
                "sigma_x": 10,
                "enhancement_factor": 4,
                "blur_factor": -4,
                "brightness_offset": 128,
                "gaussian_blur": {"kernel_size": [0, 0], "sigma_y": 0},
                "post_processing": {"clip_values": True, "normalize": False},
            },
            "lab_clahe": {"clip_limit": 3.0, "tile_grid_size": [8, 8]},
            "max_green_gsc": {
                "apply_histogram_equalization": True,
                "apply_contrast_enhancement": True,
                "contrast_alpha": 1.2,
                "contrast_beta": 10,
            },
        },
        "quality_control": {
            "enabled": True,
            "input_validation": {
                "min_resolution": [224, 224],
                "max_resolution": [4096, 4096],
                "check_corruption": True,
            },
            "final_validation": True,
        },
        "performance": {
            "parallel_variants": True,
            "parallel_images": True,
            "max_workers": 4,
        },
        "debug": {
            "log_level": "INFO",
            "save_intermediate_images": False,
            "log_processing_time": True,
            "intermediate_paths": {
                "black_border_clipped": "./debug/clipped/",
                "rgb_clahe": "./debug/rgb_clahe/",
                "min_pooling": "./debug/min_pooling/",
                "lab_clahe": "./debug/lab_clahe/",
                "max_green_gsc": "./debug/max_green_gsc/",
                "final_resized": "./debug/07_final_resized/",
            },
        },
    }

    with open(output_path, "w") as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)

    print(f"Default configuration saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Fundus Image Preprocessor")
    parser.add_argument(
        "--create-config", type=str, help="Create default configuration file"
    )
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--input", type=str, help="Input image path")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument(
        "--batch", action="store_true", help="Process directory of images in batch mode"
    )

    args = parser.parse_args()

    if args.create_config:
        create_default_config(args.create_config)
    elif args.config and args.input:
        # Initialize preprocessor
        preprocessor = FundusPreprocessor(args.config)

        # Check debug settings from YAML config
        debug_config = preprocessor.config.get("debug", {})
        debug_enabled = debug_config.get("enabled", False) and debug_config.get(
            "save_intermediate_images", False
        )

        if debug_enabled:
            print("🔍 Debug mode enabled via YAML config")
        else:
            print("ℹ️ Debug mode disabled")

        if args.batch:
            # Process directory of images
            input_path = Path(args.input)
            if not input_path.is_dir():
                print(f"Error: {args.input} is not a directory for batch processing")
                exit(1)

            # Find all image files
            image_extensions = [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]
            image_files = []
            for ext in image_extensions:
                image_files.extend(input_path.glob(f"*{ext}"))
                image_files.extend(input_path.glob(f"*{ext.upper()}"))

            if not image_files:
                print(f"No image files found in {args.input}")
                exit(1)

            print(f"Found {len(image_files)} images to process in batch mode")

            # Determine output directory
            if debug_enabled:
                # Debug mode: use debug folder for everything including reports
                output_dir = "./debug"
                batch_output_dir = Path(output_dir)
                report_dir = batch_output_dir
                print(f"🔍 Debug mode: Using debug folder for batch output and reports")
            else:
                # Normal mode: use configured output directory
                output_dir = args.output or preprocessor.config.get("general", {}).get(
                    "output_directory"
                )
                if not output_dir:
                    print("Error: No output directory specified in args or config")
                    exit(1)

                batch_output_dir = Path(output_dir)
                report_dir = batch_output_dir
                os.makedirs(batch_output_dir, exist_ok=True)

            # # Create subdirectories for each variant if not in debug mode
            # if not debug_enabled:
            #     variant_dirs = {}
            #     for variant in ['original', 'rgb_clahe', 'min_pooling', 'lab_clahe', 'max_green_gsc']:
            #         variant_dir = batch_output_dir / variant
            #         # os.makedirs(variant_dir, exist_ok=True)
            #         variant_dirs[variant] = variant_dir

            # Process images with progress tracking
            start_time = time.time()
            successful_count = 0
            failed_count = 0
            batch_results = []

            for i, image_file in enumerate(image_files, 1):
                print(f"Processing {i}/{len(image_files)}: {image_file.name}")

                try:
                    # Load image
                    image = cv2.imread(str(image_file))
                    if image is None:
                        print(f"  ❌ Failed to load image: {image_file}")
                        failed_count += 1
                        continue

                    # Process image
                    results = preprocessor.process_image(image, image_file.stem)

                    batch_results.append(
                        {
                            "filename": image_file.name,
                            "status": "success",
                            "variants": len(results),
                            "processing_time": time.time() - start_time,
                        }
                    )

                    successful_count += 1
                    print(f"  ✓ Processed successfully ({len(results)} variants)")

                except Exception as e:
                    print(f"  ❌ Error processing {image_file.name}: {e}")
                    batch_results.append(
                        {
                            "filename": image_file.name,
                            "status": "failed",
                            "error": str(e),
                        }
                    )
                    failed_count += 1

            # Create batch processing report
            total_time = time.time() - start_time
            batch_report = {
                "batch_info": {
                    "input_directory": str(input_path),
                    "output_directory": str(batch_output_dir),
                    "total_images": len(image_files),
                    "successful": successful_count,
                    "failed": failed_count,
                    "processing_time_seconds": round(total_time, 2),
                    "average_time_per_image": round(total_time / len(image_files), 2),
                    "debug_mode": debug_enabled,
                },
                "results": batch_results,
                "config": {
                    "variants_enabled": [
                        name
                        for name, cfg in preprocessor.config.get(
                            "image_variants", {}
                        ).items()
                        if cfg.get("enabled", False)
                    ],
                    "target_resolution": preprocessor.config.get("general", {}).get(
                        "target_resolution"
                    ),
                    "parallel_processing": preprocessor.config.get(
                        "performance", {}
                    ).get("parallel_variants", True),
                },
            }

            # Save batch report
            report_path = (
                report_dir
                / f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(report_path, "w") as f:
                json.dump(batch_report, f, indent=2, ensure_ascii=False)

            print(f"\n📊 BATCH PROCESSING SUMMARY:")
            print(f"  • Total images: {len(image_files)}")
            print(f"  • Successful: {successful_count} ✓")
            print(f"  • Failed: {failed_count} ❌")
            print(f"  • Processing time: {total_time:.2f} seconds")
            print(f"  • Average per image: {total_time / len(image_files):.2f} seconds")

            if debug_enabled:
                print(f"  🔍 Debug mode: All files saved to debug folders")
                print(f"  📁 Check ./debug/ for intermediate and final images")
            else:
                print(f"  📁 Output directory: {batch_output_dir}")
                print(f"  📁 Variants saved to: {batch_output_dir}/[variant_name]/")

            print(f"  📄 Batch report: {report_path}")

        else:
            # Process single image
            image = cv2.imread(args.input)
            if image is None:
                print(f"Failed to load image: {args.input}")
                exit(1)

            print(f"Processing image: {Path(args.input).name}")

            # Process image
            results = preprocessor.process_image(image, Path(args.input).stem)

            # In debug mode, files are already saved by _save_final_variant to debug folder
            # Only save command-line copies if NOT in debug mode
            if not debug_enabled:
                # Normal mode: save results to specified output directory
                output_dir = args.output or preprocessor.config.get("general", {}).get(
                    "output_directory"
                )

                print(f"\n✅ Processing completed!")
                print(f"📁 Output directory: {output_dir}")
            else:
                # Debug mode: files already saved by preprocessor to debug folders
                debug_paths = preprocessor.config.get("debug", {}).get(
                    "intermediate_paths", {}
                )
                final_resized_path = debug_paths.get(
                    "final_resized", "./debug/07_final_resized/"
                )

                print(f"\n✅ Processing completed!")
                print(f"🔍 Debug mode: All files saved to debug folders")
                print(f"📁 Final variants: {final_resized_path}")
                print(f"📁 Intermediate images: ./debug/[01-06]_*/")

            if debug_enabled:
                print(f"🔍 Debug files saved to: ./debug/")
    else:
        parser.print_help()
