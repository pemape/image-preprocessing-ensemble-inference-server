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
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Optional imports for advanced features
try:
    from skimage import filters
    from scipy import ndimage
    ADVANCED_FILTERING = True
except ImportError:
    ADVANCED_FILTERING = False
    logging.warning("Advanced filtering features disabled. Install scikit-image and scipy for full functionality.")


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
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {config_path}: {e}")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging based on configuration."""
        logger = logging.getLogger('FundusPreprocessor')

        if self.config.get('debug', {}).get('verbose_logging', False):
            level = logging.DEBUG
        else:
            level = logging.INFO

        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        return logger

    def _validate_config(self):
        """Validate configuration parameters."""
        required_sections = ['general', 'black_border_clipping', 'image_variants']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")

    def _create_directories(self):
        """Create necessary output directories."""
        if self.config.get('debug', {}).get('save_intermediate_images', False):
            debug_paths = self.config.get('debug', {}).get('intermediate_paths', {})
            for path_key, path_value in debug_paths.items():
                # Create the directory - path_value should be a directory path
                os.makedirs(path_value, exist_ok=True)

        # Create main output directory
        output_dir = self.config.get('general', {}).get('output_directory', './processed_images')
        os.makedirs(output_dir, exist_ok=True)

    def process_image(self, image: np.ndarray, image_id: str = None) -> Dict[str, np.ndarray]:
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

        if self.config.get('debug', {}).get('save_intermediate_images', False):
            self._save_debug_image(clipped_image, 'black_border_clipped', image_id)

        # Step 3: Generate 5 image variants
        variants = {}

        if self.config.get('performance', {}).get('parallel_variants', True):
            variants = self._process_variants_parallel(clipped_image, image_id)
        else:
            variants = self._process_variants_sequential(clipped_image, image_id)

        # Step 4: Final resolution processing and normalization
        final_variants = {}
        for variant_name, variant_image in variants.items():
            processed = self._final_processing(variant_image, variant_name)
            final_variants[variant_name] = processed

            # Save final processed variants if debug is enabled
            if self.config.get('debug', {}).get('save_intermediate_images', False):
                self._save_final_variant(processed, variant_name, image_id)

        processing_time = time.time() - start_time
        self.logger.info(f"Processing completed in {processing_time:.2f} seconds")

        if self.config.get('debug', {}).get('log_processing_time', True):
            self.logger.info(f"Processing time breakdown: {processing_time:.2f}s total")

        return final_variants

    def _validate_input_image(self, image: np.ndarray) -> bool:
        """Validate input image according to configuration."""
        if not self.config.get('quality_control', {}).get('enabled', True):
            return True

        validation_config = self.config.get('quality_control', {}).get('input_validation', {})

        # Check image dimensions
        height, width = image.shape[:2]
        min_res = validation_config.get('min_resolution', [224, 224])
        max_res = validation_config.get('max_resolution', [4096, 4096])

        if width < min_res[0] or height < min_res[1]:
            self.logger.error(f"Image resolution {width}x{height} below minimum {min_res[0]}x{min_res[1]}")
            return False

        if width > max_res[0] or height > max_res[1]:
            self.logger.error(f"Image resolution {width}x{height} above maximum {max_res[0]}x{max_res[1]}")
            return False

        # Check for corruption
        if validation_config.get('check_corruption', True):
            if np.any(np.isnan(image)) or np.any(np.isinf(image)):
                self.logger.error("Image contains NaN or infinite values")
                return False

        return True

    def _clip_black_borders(self, image: np.ndarray) -> np.ndarray:
        """
        Remove black borders from fundus image using all 4 methods and return the best result.
        Methods implemented: morphological, contour_detection, adaptive_threshold, fixed_threshold

        Args:
            image: RGB image as numpy array

        Returns:
            Best clipped image based on crop area
        """
        if not self.config.get('black_border_clipping', {}).get('enabled', True):
            return image

        original_area = image.shape[0] * image.shape[1]
        results = {}

        # Method 1: Morphological Operations
        try:
            result_morph = self._clip_morphological_direct(image)
            crop_area = result_morph.shape[0] * result_morph.shape[1]
            crop_ratio = crop_area / original_area
            if crop_ratio > 0.3:  # Minimum acceptable crop ratio
                results['morphological'] = (result_morph, crop_ratio)
                self.logger.debug(f"Morphological method: crop ratio = {crop_ratio:.3f}")
        except Exception as e:
            self.logger.warning(f"Morphological method failed: {e}")

        # Method 2: Contour Detection
        try:
            result_contour = self._clip_contour_detection_direct(image)
            crop_area = result_contour.shape[0] * result_contour.shape[1]
            crop_ratio = crop_area / original_area
            if crop_ratio > 0.3:
                results['contour_detection'] = (result_contour, crop_ratio)
                self.logger.debug(f"Contour detection method: crop ratio = {crop_ratio:.3f}")
        except Exception as e:
            self.logger.warning(f"Contour detection method failed: {e}")

        # Method 3: Adaptive Threshold
        try:
            result_adaptive = self._clip_adaptive_threshold_direct(image)
            crop_area = result_adaptive.shape[0] * result_adaptive.shape[1]
            crop_ratio = crop_area / original_area
            if crop_ratio > 0.3:
                results['adaptive_threshold'] = (result_adaptive, crop_ratio)
                self.logger.debug(f"Adaptive threshold method: crop ratio = {crop_ratio:.3f}")
        except Exception as e:
            self.logger.warning(f"Adaptive threshold method failed: {e}")

        # Method 4: Fixed Threshold
        try:
            result_fixed = self._clip_fixed_threshold_direct(image)
            crop_area = result_fixed.shape[0] * result_fixed.shape[1]
            crop_ratio = crop_area / original_area
            if crop_ratio > 0.3:
                results['fixed_threshold'] = (result_fixed, crop_ratio)
                self.logger.debug(f"Fixed threshold method: crop ratio = {crop_ratio:.3f}")
        except Exception as e:
            self.logger.warning(f"Fixed threshold method failed: {e}")

        # Select best result based on crop area
        if not results:
            self.logger.warning("All clipping methods failed, returning original image")
            return image

        # Choose method with largest crop area (best preservation of image content)
        best_method = max(results.keys(), key=lambda k: results[k][1])
        best_image = results[best_method][0]

        self.logger.info(f"Selected {best_method} clipping method with crop ratio {results[best_method][1]:.3f}")
        return best_image

    def _clip_morphological_direct(self, image: np.ndarray) -> np.ndarray:
        """Apply morphological operations to detect and remove black borders."""
        config = self.config.get('black_border_clipping', {}).get('morphological', {})

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply threshold
        threshold = config.get('threshold', 10)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # Morphological operations
        kernel_size = config.get('kernel_size', 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Opening to remove noise
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Closing to fill gaps
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image

        # Find largest contour (main fundus region)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Apply margin
        margin = config.get('margin', 5)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)

        return image[y:y+h, x:x+w]

    def _clip_contour_detection_direct(self, image: np.ndarray) -> np.ndarray:
        """Apply contour detection to find and crop fundus region."""
        config = self.config.get('black_border_clipping', {}).get('contour_detection', {})

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur
        blur_kernel = config.get('blur_kernel', 5)
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

        # Apply threshold
        threshold = config.get('threshold', 15)
        _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image

        # Filter contours by area
        min_area = config.get('min_contour_area', 1000)
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

        if not valid_contours:
            return image

        # Find largest valid contour
        largest_contour = max(valid_contours, key=cv2.contourArea)

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Apply margin
        margin = config.get('margin', 10)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)

        return image[y:y+h, x:x+w]

    def _clip_adaptive_threshold_direct(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding to detect fundus boundaries."""
        config = self.config.get('black_border_clipping', {}).get('adaptive_threshold', {})

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply adaptive threshold
        block_size = config.get('block_size', 11)
        c_constant = config.get('c_constant', 2)

        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c_constant
        )

        # Morphological operations to clean up
        kernel_size = config.get('kernel_size', 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image

        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Apply margin
        margin = config.get('margin', 8)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)

        return image[y:y+h, x:x+w]

    def _clip_fixed_threshold_direct(self, image: np.ndarray) -> np.ndarray:
        """Apply fixed thresholding to detect fundus region."""
        config = self.config.get('black_border_clipping', {}).get('fixed_threshold', {})

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply fixed threshold
        threshold = config.get('threshold', 20)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # Find non-zero pixels
        coords = cv2.findNonZero(binary)

        if coords is None:
            return image

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(coords)

        # Apply margin
        margin = config.get('margin', 5)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)

        return image[y:y+h, x:x+w]

    def _process_variants_parallel(self, clipped_image: np.ndarray, image_id: str) -> Dict[str, np.ndarray]:
        """Process all image variants in parallel."""
        variants = {}

        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all variant processing tasks
            futures = {
                executor.submit(self._create_original_variant, clipped_image): 'original',
                executor.submit(self._create_rgb_clahe_variant, clipped_image): 'rgb_clahe',
                executor.submit(self._create_min_pooling_variant, clipped_image): 'min_pooling',
                executor.submit(self._create_lab_clahe_variant, clipped_image): 'lab_clahe',
                executor.submit(self._create_max_green_gsc_variant, clipped_image): 'max_green_gsc'
            }

            # Collect results
            for future in as_completed(futures):
                variant_name = futures[future]
                try:
                    result = future.result()
                    variants[variant_name] = result
                    self.logger.debug(f"Completed {variant_name} variant")
                    if self.config.get('debug', {}).get('save_intermediate_images', False):
                        self._save_debug_image(result, variant_name, image_id)
                except Exception as e:
                    self.logger.error(f"Failed to process {variant_name} variant: {e}")
                    # Create fallback variant
                    variants[variant_name] = clipped_image.copy()

        return variants

    def _process_variants_sequential(self, clipped_image: np.ndarray, image_id: str) -> Dict[str, np.ndarray]:
        """Process all image variants sequentially."""
        variants = {}

        try:
            variants['original'] = self._create_original_variant(clipped_image)
            if self.config.get('debug', {}).get('save_intermediate_images', False):
                self._save_debug_image(variants['original'], 'original', image_id)
        except Exception as e:
            self.logger.error(f"Failed to create original variant: {e}")
            variants['original'] = clipped_image.copy()

        try:
            variants['rgb_clahe'] = self._create_rgb_clahe_variant(clipped_image)
            if self.config.get('debug', {}).get('save_intermediate_images', False):
                self._save_debug_image(variants['rgb_clahe'], 'rgb_clahe', image_id)
        except Exception as e:
            self.logger.error(f"Failed to create RGB-CLAHE variant: {e}")
            variants['rgb_clahe'] = clipped_image.copy()

        try:
            variants['min_pooling'] = self._create_min_pooling_variant(clipped_image)
            if self.config.get('debug', {}).get('save_intermediate_images', False):
                self._save_debug_image(variants['min_pooling'], 'min_pooling', image_id)
        except Exception as e:
            self.logger.error(f"Failed to create min-pooling variant: {e}")
            variants['min_pooling'] = clipped_image.copy()

        try:
            variants['lab_clahe'] = self._create_lab_clahe_variant(clipped_image)
            if self.config.get('debug', {}).get('save_intermediate_images', False):
                self._save_debug_image(variants['lab_clahe'], 'lab_clahe', image_id)
        except Exception as e:
            self.logger.error(f"Failed to create Lab-CLAHE variant: {e}")
            variants['lab_clahe'] = clipped_image.copy()

        try:
            variants['max_green_gsc'] = self._create_max_green_gsc_variant(clipped_image)
            if self.config.get('debug', {}).get('save_intermediate_images', False):
                self._save_debug_image(variants['max_green_gsc'], 'max_green_gsc', image_id)
        except Exception as e:
            self.logger.error(f"Failed to create MaxGreenGsc variant: {e}")
            variants['max_green_gsc'] = clipped_image.copy()

        return variants

    def _create_original_variant(self, image: np.ndarray) -> np.ndarray:
        """Create the original clipped image variant."""
        return image.copy()

    def _create_rgb_clahe_variant(self, image: np.ndarray) -> np.ndarray:
        """Create RGB-CLAHE enhanced variant."""
        config = self.config.get('image_variants', {}).get('rgb_clahe', {})

        # Apply CLAHE to each RGB channel
        clip_limit = config.get('clip_limit', 2.0)
        tile_grid_size = tuple(config.get('tile_grid_size', [8, 8]))

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        # Process each channel
        result = np.zeros_like(image)
        for i in range(3):  # RGB channels
            result[:, :, i] = clahe.apply(image[:, :, i])

        return result

    def _create_min_pooling_variant(self, image: np.ndarray) -> np.ndarray:
        """Create min-pooling enhanced variant."""
        config = self.config.get('image_variants', {}).get('min_pooling', {})

        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply min pooling
        kernel_size = config.get('kernel_size', 3)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Min pooling using erosion
        min_pooled = cv2.erode(gray, kernel, iterations=1)

        # Enhance contrast
        enhanced = cv2.equalizeHist(min_pooled)

        # Convert back to RGB
        result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

        return result

    def _create_lab_clahe_variant(self, image: np.ndarray) -> np.ndarray:
        """Create Lab-CLAHE enhanced variant."""
        config = self.config.get('image_variants', {}).get('lab_clahe', {})

        # Convert RGB to Lab
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        # Apply CLAHE to L channel
        clip_limit = config.get('clip_limit', 3.0)
        tile_grid_size = tuple(config.get('tile_grid_size', [8, 8]))

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        # Convert back to RGB
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return result

    def _create_max_green_gsc_variant(self, image: np.ndarray) -> np.ndarray:
        """Create MaxGreenGsc (Maximum Green + Grayscale Conversion) variant."""
        config = self.config.get('image_variants', {}).get('max_green_gsc', {})

        # Extract green channel (index 1 in RGB)
        green_channel = image[:, :, 1]

        # Apply histogram equalization to green channel
        if config.get('apply_histogram_equalization', True):
            green_enhanced = cv2.equalizeHist(green_channel)
        else:
            green_enhanced = green_channel

        # Convert to 3-channel grayscale
        result = cv2.cvtColor(green_enhanced, cv2.COLOR_GRAY2RGB)

        # Optional: Apply additional contrast enhancement
        if config.get('apply_contrast_enhancement', True):
            alpha = config.get('contrast_alpha', 1.2)
            beta = config.get('contrast_beta', 10)
            result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)

        return result

    def _final_processing(self, image: np.ndarray, variant_name: str) -> np.ndarray:
        """Apply final processing steps to each variant."""
        config = self.config.get('general', {})

        # Resize to target resolution
        target_size = tuple(config.get('target_resolution', [224, 224]))
        if image.shape[:2] != target_size:
            resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        else:
            resized = image

        # Normalize pixel values
        if config.get('normalize_pixels', True):
            normalized = resized.astype(np.float32) / 255.0
        else:
            normalized = resized

        # Apply final quality checks
        if self.config.get('quality_control', {}).get('final_validation', True):
            if not self._validate_final_image(normalized, variant_name):
                self.logger.warning(f"Final validation failed for {variant_name}")

        return normalized

    def _validate_final_image(self, image: np.ndarray, variant_name: str) -> bool:
        """Validate final processed image."""
        # Check for NaN or infinite values
        if np.any(np.isnan(image)) or np.any(np.isinf(image)):
            self.logger.error(f"Final image {variant_name} contains NaN or infinite values")
            return False

        # Check value range
        if image.min() < 0 or image.max() > 1:
            self.logger.warning(f"Final image {variant_name} has values outside [0,1] range")

        # Check for completely black or white images
        if image.max() - image.min() < 0.01:
            self.logger.warning(f"Final image {variant_name} has very low contrast")

        return True

    def _save_debug_image(self, image: np.ndarray, stage_name: str, image_id: str):
        """Save intermediate images for debugging."""
        if not self.config.get('debug', {}).get('save_intermediate_images', False):
            return

        debug_paths = self.config.get('debug', {}).get('intermediate_paths', {})
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
        """Save final processed variants."""
        output_dir = self.config.get('general', {}).get('output_directory', './processed_images')
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
            filename = f"{image_id or 'unknown'}_{variant_name}.jpg"
            filepath = os.path.join(output_dir, filename)

            # Save image
            cv2.imwrite(filepath, save_image)
            self.logger.debug(f"Saved final variant image: {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save final variant image {variant_name}: {e}")

    def process_batch(self, images: List[np.ndarray], image_ids: List[str] = None) -> List[Dict[str, np.ndarray]]:
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

        if self.config.get('performance', {}).get('parallel_images', True):
            # Process images in parallel
            with ThreadPoolExecutor(max_workers=self.config.get('performance', {}).get('max_workers', 4)) as executor:
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
        'general': {
            'target_resolution': [224, 224],
            'normalize_pixels': True,
            'output_directory': './processed_images'
        },
        'black_border_clipping': {
            'enabled': True,
            'morphological': {
                'threshold': 10,
                'kernel_size': 5,
                'margin': 5
            },
            'contour_detection': {
                'threshold': 15,
                'blur_kernel': 5,
                'min_contour_area': 1000,
                'margin': 10
            },
            'adaptive_threshold': {
                'block_size': 11,
                'c_constant': 2,
                'kernel_size': 3,
                'margin': 8
            },
            'fixed_threshold': {
                'threshold': 20,
                'margin': 5
            }
        },
        'image_variants': {
            'rgb_clahe': {
                'clip_limit': 2.0,
                'tile_grid_size': [8, 8]
            },
            'min_pooling': {
                'kernel_size': 3
            },
            'lab_clahe': {
                'clip_limit': 3.0,
                'tile_grid_size': [8, 8]
            },
            'max_green_gsc': {
                'apply_histogram_equalization': True,
                'apply_contrast_enhancement': True,
                'contrast_alpha': 1.2,
                'contrast_beta': 10
            }
        },
        'quality_control': {
            'enabled': True,
            'input_validation': {
                'min_resolution': [224, 224],
                'max_resolution': [4096, 4096],
                'check_corruption': True
            },
            'final_validation': True
        },
        'performance': {
            'parallel_variants': True,
            'parallel_images': True,
            'max_workers': 4
        },
        'debug': {
            'verbose_logging': False,
            'save_intermediate_images': False,
            'log_processing_time': True,
            'intermediate_paths': {
                'black_border_clipped': './debug/clipped/',
                'rgb_clahe': './debug/rgb_clahe/',
                'min_pooling': './debug/min_pooling/',
                'lab_clahe': './debug/lab_clahe/',
                'max_green_gsc': './debug/max_green_gsc/'
            }
        }
    }

    with open(output_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)

    print(f"Default configuration saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Fundus Image Preprocessor')
    parser.add_argument('--create-config', type=str, help='Create default configuration file')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--input', type=str, help='Input image path')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--batch', action='store_true', help='Process directory of images in batch mode')

    args = parser.parse_args()

    if args.create_config:
        create_default_config(args.create_config)
    elif args.config and args.input:
        # Initialize preprocessor
        preprocessor = FundusPreprocessor(args.config)

        # Check debug settings from YAML config
        debug_enabled = preprocessor.config.get('debug', {}).get('save_intermediate_images', False)
        if debug_enabled:
            print("🔍 Debug mode enabled via YAML config")
        else:
            print("ℹ️ Debug mode disabled")

        # Process single image
        image = cv2.imread(args.input)
        if image is None:
            print(f"Failed to load image: {args.input}")
            exit(1)

        # Process image
        results = preprocessor.process_image(image, Path(args.input).stem)

        # Save results
        output_dir = args.output or './processed_images'
        os.makedirs(output_dir, exist_ok=True)

        for variant_name, variant_image in results.items():
            # Convert to BGR for saving
            if variant_image.dtype == np.float32:
                save_image = (variant_image * 255).astype(np.uint8)
            else:
                save_image = variant_image

            save_image_bgr = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)

            output_path = os.path.join(output_dir, f"{Path(args.input).stem}_{variant_name}.jpg")
            cv2.imwrite(output_path, save_image_bgr)
            print(f"Saved {variant_name} variant to: {output_path}")
    else:
        parser.print_help()
