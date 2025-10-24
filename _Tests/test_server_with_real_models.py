import pytest
import sys
import os
from pathlib import Path
import numpy as np
import cv2
import yaml
import tempfile
import json
import torch

# Add parent directory to path to import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from diabetic_retinopathy_classifier import DiabeticRetinopathyClassifier, ModelEnsemble
from fundus_preprocessor import FundusPreprocessor

class TestDiabeticRetinopathyWithRealModels:
    
    @pytest.fixture
    def test_image_path(self):
        """Path to your actual test image"""
        return r"D:\FEI STU\ing\2roc\DATABASE\Aptos\train_images\00a8624548a9.png"
    
    @pytest.fixture
    def soft_voting_config_path(self):
        """Path to soft voting configuration"""
        return r"c:\ProgrammingProjects\python\dp\image-preprocessing-ensemble-inference-server\classifier_config.yaml"
    
    @pytest.fixture
    def hard_voting_config_path(self):
        """Path to hard voting configuration"""
        return r"c:\ProgrammingProjects\python\dp\image-preprocessing-ensemble-inference-server\classifier_config_hard_voting.yaml"
    
    @pytest.fixture
    def preprocessor_config_path(self):
        """Path to preprocessor configuration"""
        return r"c:\ProgrammingProjects\python\dp\image-preprocessing-ensemble-inference-server\preprocessing_config.yaml"
    
    def test_image_exists(self, test_image_path):
        """Test that the test image file exists"""
        assert Path(test_image_path).exists(), f"Test image not found at {test_image_path}"
        print(f"✓ Test image found: {test_image_path}")
    
    def test_model_files_exist(self, soft_voting_config_path):
        """Test that all model files specified in config exist"""
        with open(soft_voting_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        missing_models = []
        for model in config['models']:
            model_path = model['model_path']
            if not Path(model_path).exists():
                missing_models.append(model_path)
        
        if missing_models:
            pytest.fail(f"Missing model files: {missing_models}")
        
        print(f"✓ All {len(config['models'])} model files exist")
    
    def test_load_and_preprocess_image(self, test_image_path, preprocessor_config_path):
        """Test loading and preprocessing the actual image"""
        # Load image
        image = cv2.imread(test_image_path)
        assert image is not None, "Failed to load image"
        
        print(f"Original image shape: {image.shape}")
        print(f"Original image dtype: {image.dtype}")
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Test basic preprocessing
        preprocessor = FundusPreprocessor(preprocessor_config_path)
        preprocessed_images = preprocessor.preprocess_image(image_rgb)
        
        # Verify preprocessed images
        assert 'original' in preprocessed_images, "Original preprocessing variant missing"
        assert 'rgb_clahe' in preprocessed_images, "RGB-CLAHE preprocessing variant missing"
        
        for variant, processed_img in preprocessed_images.items():
            assert processed_img.shape == (500, 500, 3), f"Wrong shape for {variant}: {processed_img.shape}"
            assert processed_img.dtype == np.uint8, f"Wrong dtype for {variant}: {processed_img.dtype}"
            print(f"✓ {variant} preprocessing: {processed_img.shape}, {processed_img.dtype}")
        
        return preprocessed_images
    
    def test_soft_voting_classification(self, test_image_path, soft_voting_config_path, preprocessor_config_path):
        """Test classification with soft voting using real models"""
        if not Path(test_image_path).exists():
            pytest.skip(f"Test image not available: {test_image_path}")
        
        try:
            # Initialize classifier
            classifier = DiabeticRetinopathyClassifier(soft_voting_config_path)
            
            # Load and preprocess image
            image = cv2.imread(test_image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            preprocessor = FundusPreprocessor(preprocessor_config_path)
            preprocessed_images = preprocessor.preprocess_image(image_rgb)
            
            # Classify
            results = classifier.classify(preprocessed_images)
            
            # Verify results
            assert 'predicted_class' in results
            assert 'confidence' in results
            assert 'voting_strategy' in results
            assert results['voting_strategy'] == 'soft'
            
            # Check class probabilities
            class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
            for class_name in class_names:
                assert class_name in results
                assert isinstance(results[class_name], float)
                assert 0 <= results[class_name] <= 1
            
            # Print results
            print(f"\n=== SOFT VOTING RESULTS ===")
            print(f"Predicted class: {results['predicted_class']}")
            print(f"Confidence: {results['confidence']:.4f}")
            print(f"Ensemble size: {results.get('ensemble_size', 'unknown')}")
            print(f"Class probabilities:")
            for class_name in class_names:
                print(f"  {class_name}: {results[class_name]:.4f}")
            
            if 'model_info' in results:
                model_info = results['model_info']
                print(f"Models used: {model_info.get('ensemble_size', 0)}")
                print(f"Architectures: {model_info.get('architectures', [])}")
            
        except Exception as e:
            pytest.fail(f"Soft voting classification failed: {e}")
    
    def test_hard_voting_classification(self, test_image_path, hard_voting_config_path, preprocessor_config_path):
        """Test classification with hard voting using real models"""
        if not Path(test_image_path).exists():
            pytest.skip(f"Test image not available: {test_image_path}")
        
        try:
            # Initialize classifier
            classifier = DiabeticRetinopathyClassifier(hard_voting_config_path)
            
            # Load and preprocess image
            image = cv2.imread(test_image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            preprocessor = FundusPreprocessor(preprocessor_config_path)
            preprocessed_images = preprocessor.preprocess_image(image_rgb)
            
            # Classify
            results = classifier.classify(preprocessed_images)
            
            # Verify results
            assert 'predicted_class' in results
            assert 'confidence' in results
            assert 'voting_strategy' in results
            assert results['voting_strategy'] == 'hard'
            
            # For hard voting, confidence should be 1.0 for predicted class
            predicted_class = results['predicted_class']
            assert results[predicted_class] == 1.0, "Hard voting should give confidence 1.0 for predicted class"
            
            # Print results
            print(f"\n=== HARD VOTING RESULTS ===")
            print(f"Predicted class: {results['predicted_class']}")
            print(f"Confidence: {results['confidence']:.4f}")
            print(f"Ensemble size: {results.get('ensemble_size', 'unknown')}")
            
            if 'model_info' in results:
                model_info = results['model_info']
                print(f"Models used: {model_info.get('ensemble_size', 0)}")
                print(f"Architectures: {model_info.get('architectures', [])}")
            
        except Exception as e:
            pytest.fail(f"Hard voting classification failed: {e}")
    
    def test_compare_voting_strategies(self, test_image_path, soft_voting_config_path, hard_voting_config_path, preprocessor_config_path):
        """Compare soft vs hard voting results"""
        if not Path(test_image_path).exists():
            pytest.skip(f"Test image not available: {test_image_path}")
        
        try:
            # Load and preprocess image once
            image = cv2.imread(test_image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            preprocessor = FundusPreprocessor(preprocessor_config_path)
            preprocessed_images = preprocessor.preprocess_image(image_rgb)
            
            # Test both voting strategies
            soft_classifier = DiabeticRetinopathyClassifier(soft_voting_config_path)
            hard_classifier = DiabeticRetinopathyClassifier(hard_voting_config_path)
            
            soft_results = soft_classifier.classify(preprocessed_images)
            hard_results = hard_classifier.classify(preprocessed_images)
            
            print(f"\n=== VOTING STRATEGY COMPARISON ===")
            print(f"Soft voting prediction: {soft_results['predicted_class']} (confidence: {soft_results['confidence']:.4f})")
            print(f"Hard voting prediction: {hard_results['predicted_class']} (confidence: {hard_results['confidence']:.4f})")
            
            # Check if predictions agree
            if soft_results['predicted_class'] == hard_results['predicted_class']:
                print("✓ Both voting strategies agree on the prediction")
            else:
                print("⚠ Voting strategies disagree - ensemble uncertainty detected")
            
            # Show probability distributions for soft voting
            class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
            print(f"\nSoft voting probabilities:")
            for class_name in class_names:
                print(f"  {class_name}: {soft_results[class_name]:.4f}")
            
        except Exception as e:
            pytest.fail(f"Voting strategy comparison failed: {e}")
    
    def test_model_info(self, soft_voting_config_path):
        """Test getting model information"""
        try:
            classifier = DiabeticRetinopathyClassifier(soft_voting_config_path)
            model_info = classifier.get_model_info()
            
            assert 'models' in model_info
            assert 'voting_strategy' in model_info
            assert 'device' in model_info
            assert 'total_models' in model_info
            
            print(f"\n=== MODEL INFORMATION ===")
            print(f"Total models loaded: {model_info['total_models']}")
            print(f"Voting strategy: {model_info['voting_strategy']}")
            print(f"Device: {model_info['device']}")
            
            print(f"\nLoaded models:")
            for i, model in enumerate(model_info['models'], 1):
                print(f"  {i}. {model['architecture']} - {model['dataset']} - {model['preprocessing_variant']}")
                print(f"     Path: {Path(model['model_path']).name}")
            
            # Verify we have both architectures
            architectures = [m['architecture'] for m in model_info['models']]
            assert 'efficientnetb4' in architectures, "EfficientNetB4 not found"
            assert 'xception' in architectures, "Xception not found"
            
            # Verify we have both preprocessing variants
            preprocessing_variants = [m['preprocessing_variant'] for m in model_info['models']]
            assert 'original' in preprocessing_variants, "Original preprocessing not found"
            assert 'rgb_clahe' in preprocessing_variants, "RGB-CLAHE preprocessing not found"
            
        except Exception as e:
            pytest.fail(f"Model info test failed: {e}")
    
    def test_device_configuration(self, soft_voting_config_path):
        """Test device configuration (CPU/GPU)"""
        try:
            classifier = DiabeticRetinopathyClassifier(soft_voting_config_path)
            model_info = classifier.get_model_info()
            
            device = model_info['device']
            print(f"\n=== DEVICE CONFIGURATION ===")
            print(f"Using device: {device}")
            
            if torch.cuda.is_available():
                print(f"CUDA available: Yes")
                print(f"CUDA device count: {torch.cuda.device_count()}")
                if torch.cuda.device_count() > 0:
                    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            else:
                print(f"CUDA available: No")
            
            # Test with forced CPU
            print(f"\nTesting CPU-only mode...")
            # This would require modifying the config temporarily
            
        except Exception as e:
            pytest.fail(f"Device configuration test failed: {e}")

def test_integration_full_pipeline():
    """Full integration test of the entire pipeline"""
    image_path = r"D:\FEI STU\ing\2roc\DATABASE\Aptos\train_images\00a8624548a9.png"
    config_path = r"c:\ProgrammingProjects\python\dp\image-preprocessing-ensemble-inference-server\classifier_config.yaml"
    preprocessor_config_path = r"c:\ProgrammingProjects\python\dp\image-preprocessing-ensemble-inference-server\preprocessing_config.yaml"
    
    if not Path(image_path).exists():
        pytest.skip(f"Test image not available: {image_path}")
    
    if not Path(config_path).exists():
        pytest.skip(f"Config file not available: {config_path}")
    
    try:
        print(f"\n=== FULL PIPELINE INTEGRATION TEST ===")
        print(f"Image: {Path(image_path).name}")
        
        # 1. Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"✓ Image loaded: {image.shape}")
        
        # 2. Preprocess
        preprocessor = FundusPreprocessor(preprocessor_config_path)
        preprocessed_images = preprocessor.preprocess_image(image_rgb)
        print(f"✓ Preprocessing complete: {list(preprocessed_images.keys())}")
        
        # 3. Classify
        classifier = DiabeticRetinopathyClassifier(config_path)
        results = classifier.classify(preprocessed_images)
        print(f"✓ Classification complete")
        
        # 4. Display results
        print(f"\nFINAL RESULTS:")
        print(f"Predicted class: {results['predicted_class']}")
        print(f"Confidence: {results['confidence']:.4f}")
        print(f"Voting strategy: {results['voting_strategy']}")
        
        class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
        print(f"\nClass probabilities:")
        for class_name in class_names:
            prob = results.get(class_name, 0.0)
            bar = '█' * int(prob * 20)  # Simple bar chart
            print(f"  {class_name:15}: {prob:.4f} {bar}")
        
        # Success assertions
        assert results['predicted_class'] in class_names
        assert 0 <= results['confidence'] <= 1
        assert results['voting_strategy'] in ['soft', 'hard']
        
        print(f"\n✓ Full pipeline test completed successfully!")
        
    except Exception as e:
        pytest.fail(f"Full pipeline integration test failed: {e}")

if __name__ == "__main__":
    # Run specific tests
    pytest.main([__file__, "-v", "-s"])