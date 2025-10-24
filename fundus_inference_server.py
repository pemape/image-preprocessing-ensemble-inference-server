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

# Import our modules
from fundus_preprocessor import FundusPreprocessor
from diabetic_retinopathy_classifier import DiabeticRetinopathyClassifier


class FundusInferenceServer:
    """Main server class integrating preprocessing and classification."""
    
    def __init__(self, 
                 preprocessing_config: str,
                 classifier_config: Optional[str] = None,
                 host: str = '0.0.0.0',
                 port: int = 5000,
                 debug: bool = False):
        """
        Initialize the inference server.
        
        Args:
            preprocessing_config: Path to preprocessing configuration
            classifier_config: Path to classifier configuration (optional)
            host: Server host
            port: Server port
            debug: Debug mode
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
        
        # Setup Flask app
        self.app = self._create_flask_app()
        
        # Server statistics
        self.stats = {
            'total_requests': 0,
            'preprocessing_requests': 0,
            'classification_requests': 0,
            'errors': 0,
            'start_time': time.time()
        }
        
        self.logger.info("FundusInferenceServer initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('FundusInferenceServer')
        
        level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        return logger
    
    def _create_flask_app(self) -> Flask:
        """Create and configure Flask application."""
        app = Flask(__name__)
        app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
        
        # Register routes
        app.route('/health', methods=['GET'])(self.health_check)
        app.route('/info', methods=['GET'])(self.get_info)
        app.route('/stats', methods=['GET'])(self.get_stats)
        app.route('/config', methods=['GET'])(self.get_config)
        app.route('/models', methods=['GET'])(self.get_models)
        
        # Processing endpoints
        app.route('/preprocess', methods=['POST'])(self.preprocess_image)
        app.route('/classify', methods=['POST'])(self.classify_image)
        app.route('/process', methods=['POST'])(self.full_process)  # Preprocess + Classify
        
        # Batch processing
        app.route('/batch/preprocess', methods=['POST'])(self.batch_preprocess)
        app.route('/batch/process', methods=['POST'])(self.batch_process)
        
        return app
    
    def health_check(self):
        """Health check endpoint."""
        self.stats['total_requests'] += 1
        
        health_status = {
            'status': 'healthy',
            'timestamp': time.time(),
            'uptime': time.time() - self.stats['start_time'],
            'modules': {
                'preprocessor': True,
                'classifier': self.classifier is not None
            }
        }
        
        return jsonify(health_status)
    
    def get_info(self):
        """Get server information."""
        self.stats['total_requests'] += 1
        
        info = {
            'name': 'Fundus Image Processing and Classification Server',
            'version': '1.0.0',
            'modules': {
                'preprocessing': {
                    'enabled': True,
                    'variants': 5,
                    'clipping_methods': 4
                },
                'classification': {
                    'enabled': self.classifier is not None,
                    'models': len(self.classifier.ensemble.models) if self.classifier else 0,
                    'voting_strategy': self.classifier.ensemble.voting_strategy if self.classifier else None
                }
            },
            'endpoints': {
                'GET /health': 'Health check',
                'GET /info': 'Server information',
                'GET /stats': 'Server statistics', 
                'GET /config': 'Configuration details',
                'GET /models': 'Model information',
                'POST /preprocess': 'Image preprocessing only',
                'POST /classify': 'Classification from preprocessed images',
                'POST /process': 'Full pipeline (preprocess + classify)',
                'POST /batch/preprocess': 'Batch preprocessing',
                'POST /batch/process': 'Batch full processing'
            },
            'paper_reference': 'https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.12987'
        }
        
        return jsonify(info)
    
    def get_stats(self):
        """Get server statistics."""
        self.stats['total_requests'] += 1
        
        uptime = time.time() - self.stats['start_time']
        
        statistics = {
            'uptime_seconds': uptime,
            'uptime_formatted': f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s",
            'total_requests': self.stats['total_requests'],
            'preprocessing_requests': self.stats['preprocessing_requests'],
            'classification_requests': self.stats['classification_requests'],
            'errors': self.stats['errors'],
            'requests_per_minute': self.stats['total_requests'] / (uptime / 60) if uptime > 0 else 0
        }
        
        return jsonify(statistics)
    
    def get_config(self):
        """Get configuration details."""
        self.stats['total_requests'] += 1
        
        config_info = {
            'preprocessing': {
                'target_resolution': self.preprocessor.config.get('general', {}).get('target_resolution'),
                'normalize_pixels': self.preprocessor.config.get('general', {}).get('normalize_pixels'),
                'black_border_clipping': self.preprocessor.config.get('black_border_clipping', {}).get('enabled'),
                'parallel_processing': self.preprocessor.config.get('performance', {}).get('parallel_variants')
            }
        }
        
        if self.classifier:
            config_info['classification'] = {
                'voting_strategy': self.classifier.config.get('voting_strategy'),
                'force_cpu': self.classifier.config.get('force_cpu'),
                'confidence_threshold': self.classifier.config.get('confidence_threshold')
            }
        
        return jsonify(config_info)
    
    def get_models(self):
        """Get model information."""
        self.stats['total_requests'] += 1
        
        if not self.classifier:
            return jsonify({'error': 'Classification module not available'}), 400
        
        model_info = self.classifier.get_model_info()
        return jsonify(model_info)
    
    def preprocess_image(self):
        """Preprocess a single image."""
        try:
            self.stats['total_requests'] += 1
            self.stats['preprocessing_requests'] += 1
            
            # Get image from request
            image = self._get_image_from_request(request)
            if image is None:
                return jsonify({'error': 'No valid image provided'}), 400
            
            # Process image
            start_time = time.time()
            variants = self.preprocessor.process_image(image, 'uploaded_image')
            processing_time = time.time() - start_time
            
            # Convert to base64 for JSON response
            response_variants = {}
            for variant_name, variant_image in variants.items():
                encoded_image = self._encode_image(variant_image)
                response_variants[variant_name] = encoded_image
            
            response = {
                'status': 'success',
                'processing_time_seconds': processing_time,
                'variants': response_variants,
                'metadata': {
                    'original_size': list(image.shape[:2]),
                    'processed_size': list(variants['original'].shape[:2]),
                    'variant_count': len(variants)
                }
            }
            
            return jsonify(response)
            
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Preprocessing error: {e}")
            return jsonify({'error': str(e)}), 500
    
    def classify_image(self):
        """Classify from preprocessed images."""
        try:
            self.stats['total_requests'] += 1
            self.stats['classification_requests'] += 1
            
            if not self.classifier:
                return jsonify({'error': 'Classification module not available'}), 400
            
            # Get preprocessed images from request
            preprocessed_images = self._get_preprocessed_images_from_request(request)
            if not preprocessed_images:
                return jsonify({'error': 'No valid preprocessed images provided'}), 400
            
            # Classify
            start_time = time.time()
            results = self.classifier.classify(preprocessed_images)
            classification_time = time.time() - start_time
            
            response = {
                'status': 'success',
                'classification_time_seconds': classification_time,
                'prediction': results
            }
            
            return jsonify(response)
            
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Classification error: {e}")
            return jsonify({'error': str(e)}), 500
    
    def full_process(self):
        """Full pipeline: preprocess + classify."""
        try:
            self.stats['total_requests'] += 1
            self.stats['preprocessing_requests'] += 1
            self.stats['classification_requests'] += 1
            
            # Get image from request
            image = self._get_image_from_request(request)
            if image is None:
                return jsonify({'error': 'No valid image provided'}), 400
            
            # Preprocess
            preprocessing_start = time.time()
            variants = self.preprocessor.process_image(image, 'uploaded_image')
            preprocessing_time = time.time() - preprocessing_start
            
            # Classify if classifier available
            classification_results = None
            classification_time = 0
            
            if self.classifier:
                classification_start = time.time()
                classification_results = self.classifier.classify(variants)
                classification_time = time.time() - classification_start
            
            # Prepare response
            response = {
                'status': 'success',
                'processing_times': {
                    'preprocessing_seconds': preprocessing_time,
                    'classification_seconds': classification_time,
                    'total_seconds': preprocessing_time + classification_time
                },
                'metadata': {
                    'original_size': list(image.shape[:2]),
                    'processed_size': list(variants['original'].shape[:2]),
                    'variant_count': len(variants)
                }
            }
            
            # Add classification results if available
            if classification_results:
                response['prediction'] = classification_results
            else:
                response['warning'] = 'Classification module not available'
            
            # Optionally include preprocessed images
            include_images = request.form.get('include_images', 'false').lower() == 'true'
            if include_images:
                response_variants = {}
                for variant_name, variant_image in variants.items():
                    encoded_image = self._encode_image(variant_image)
                    response_variants[variant_name] = encoded_image
                response['variants'] = response_variants
            
            return jsonify(response)
            
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Full processing error: {e}")
            return jsonify({'error': str(e)}), 500
    
    def batch_preprocess(self):
        """Batch preprocessing endpoint."""
        try:
            self.stats['total_requests'] += 1
            
            # Get images from request
            images = self._get_images_from_request(request)
            if not images:
                return jsonify({'error': 'No valid images provided'}), 400
            
            # Process batch
            start_time = time.time()
            results = self.preprocessor.process_batch(images, [f'batch_image_{i}' for i in range(len(images))])
            processing_time = time.time() - start_time
            
            # Convert results
            response_results = []
            for i, variants in enumerate(results):
                if variants:  # Success
                    response_variants = {}
                    for variant_name, variant_image in variants.items():
                        encoded_image = self._encode_image(variant_image)
                        response_variants[variant_name] = encoded_image
                    
                    response_results.append({
                        'status': 'success',
                        'image_index': i,
                        'variants': response_variants
                    })
                else:  # Failed
                    response_results.append({
                        'status': 'failed',
                        'image_index': i,
                        'error': 'Processing failed'
                    })
            
            self.stats['preprocessing_requests'] += len(images)
            
            response = {
                'status': 'success',
                'batch_size': len(images),
                'successful': sum(1 for r in response_results if r['status'] == 'success'),
                'processing_time_seconds': processing_time,
                'results': response_results
            }
            
            return jsonify(response)
            
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Batch preprocessing error: {e}")
            return jsonify({'error': str(e)}), 500
    
    def batch_process(self):
        """Batch full processing endpoint."""
        try:
            self.stats['total_requests'] += 1
            
            if not self.classifier:
                return jsonify({'error': 'Classification module not available for batch processing'}), 400
            
            # Get images from request
            images = self._get_images_from_request(request)
            if not images:
                return jsonify({'error': 'No valid images provided'}), 400
            
            # Process batch
            start_time = time.time()
            preprocessing_results = self.preprocessor.process_batch(images, [f'batch_image_{i}' for i in range(len(images))])
            preprocessing_time = time.time() - start_time
            
            # Classify each result
            classification_start = time.time()
            response_results = []
            
            for i, variants in enumerate(preprocessing_results):
                if variants:  # Preprocessing succeeded
                    try:
                        classification_results = self.classifier.classify(variants)
                        response_results.append({
                            'status': 'success',
                            'image_index': i,
                            'prediction': classification_results
                        })
                    except Exception as e:
                        response_results.append({
                            'status': 'classification_failed',
                            'image_index': i,
                            'error': str(e)
                        })
                else:  # Preprocessing failed
                    response_results.append({
                        'status': 'preprocessing_failed',
                        'image_index': i,
                        'error': 'Preprocessing failed'
                    })
            
            classification_time = time.time() - classification_start
            
            self.stats['preprocessing_requests'] += len(images)
            self.stats['classification_requests'] += len(images)
            
            response = {
                'status': 'success',
                'batch_size': len(images),
                'successful': sum(1 for r in response_results if r['status'] == 'success'),
                'processing_times': {
                    'preprocessing_seconds': preprocessing_time,
                    'classification_seconds': classification_time,
                    'total_seconds': preprocessing_time + classification_time
                },
                'results': response_results
            }
            
            return jsonify(response)
            
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Batch processing error: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _get_image_from_request(self, req) -> Optional[np.ndarray]:
        """Extract image from Flask request."""
        try:
            if 'image' not in req.files:
                return None
            
            file = req.files['image']
            if file.filename == '':
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
    
    def _get_images_from_request(self, req) -> List[np.ndarray]:
        """Extract multiple images from Flask request."""
        images = []
        
        try:
            files = req.files.getlist('images')
            
            for file in files:
                if file.filename != '':
                    image_bytes = file.read()
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if image is not None:
                        images.append(image)
            
            return images
            
        except Exception as e:
            self.logger.error(f"Error extracting images from request: {e}")
            return []
    
    def _get_preprocessed_images_from_request(self, req) -> Dict[str, np.ndarray]:
        """Extract preprocessed images from request (base64 encoded)."""
        try:
            data = req.get_json()
            if not data or 'variants' not in data:
                return {}
            
            variants = {}
            for variant_name, encoded_image in data['variants'].items():
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
        _, buffer = cv2.imencode('.jpg', image_bgr)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        
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
    parser = argparse.ArgumentParser(description='Fundus Image Processing and Classification Server')
    parser.add_argument('--preprocessing-config', type=str, default='preprocessing_config.yaml',
                        help='Path to preprocessing configuration file')
    parser.add_argument('--classifier-config', type=str, default=None,
                        help='Path to classifier configuration file')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Create server and run
    server = FundusInferenceServer(
        preprocessing_config=args.preprocessing_config,
        classifier_config=args.classifier_config,
        host=args.host,
        port=args.port,
        debug=args.debug
    )
    
    server.run()


if __name__ == '__main__':
    main()