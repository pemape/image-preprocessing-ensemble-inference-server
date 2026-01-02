"""
Diabetic Retinopathy Classifier Module
Supports multiple CNN architectures (Xception, EfficientNetB4) with ensemble voting
Based on trained models from APTOS5 and DDR6 datasets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b4
import numpy as np
import cv2
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pretrainedmodels
import subprocess


class XceptionModel(nn.Module):
    """Xception model for diabetic retinopathy classification."""

    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super(XceptionModel, self).__init__()

        # Load pretrained Xception
        if pretrained:
            self.xception = pretrainedmodels.__dict__["xception"](pretrained="imagenet")
        else:
            self.xception = pretrainedmodels.__dict__["xception"](pretrained=None)

        # Replace the final classifier
        self.xception.last_linear = nn.Linear(
            self.xception.last_linear.in_features, num_classes
        )

    def forward(self, x):
        return self.xception(x)

    def get_model_ref(self):
        return self.xception


class EfficientNetB4Model(nn.Module):
    """EfficientNetB4 model for diabetic retinopathy classification."""

    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super(EfficientNetB4Model, self).__init__()

        # Load pretrained EfficientNetB4
        self.efficientnet = efficientnet_b4()

        num_ftrs = self.efficientnet.classifier[1].in_features
        # Replace the final classifier
        self.efficientnet.classifier[1] = nn.Linear(
            in_features=num_ftrs, out_features=num_classes, bias=True
        )

    def forward(self, x):
        return self.efficientnet(x)

    def get_model_ref(self):
        return self.efficientnet


class ModelEnsemble:
    """Ensemble of multiple models with voting strategies."""

    def __init__(
        self, models: List[Dict], device: torch.device, voting_strategy: str = "soft"
    ):
        """
        Initialize model ensemble.

        Args:
            models: List of model configurations
            device: Torch device
            voting_strategy: 'soft' or 'hard' voting
        """
        self.models = []
        self.device = device
        self.voting_strategy = voting_strategy
        self.logger = logging.getLogger(self.__class__.__name__)

        for model_config in models:
            model = self._load_model(model_config)
            if model is not None:
                self.models.append({"model": model, "config": model_config})
            else:
                self.logger.warning(
                    f"Model {model_config.get('model_path', 'unknown')} could not be loaded and will be skipped."
                )

        self.logger.info(f"Loaded {len(self.models)} models for ensemble")

    def _load_model(self, config: Dict) -> Optional[nn.Module]:
        """Load a single model from configuration."""
        try:
            architecture = config["architecture"].lower()
            model_path = config["model_path"]
            num_classes = config.get("num_classes", 5)

            # Create model
            if architecture == "xception":
                model = XceptionModel(num_classes=num_classes, pretrained=False)
                model = model.get_model_ref()
            elif architecture == "efficientnetb4":
                model = EfficientNetB4Model(num_classes=num_classes, pretrained=False)
                model = model.get_model_ref()
            else:
                self.logger.error(f"Unknown architecture: {architecture}")
                return None

            # Load weights
            if Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location=self.device)

                # Handle different checkpoint formats
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                elif "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"])
                else:
                    model.load_state_dict(checkpoint)

                model.to(self.device)
                model.eval()

                self.logger.info(f"Loaded model: {architecture} from {model_path}")
                return model
            else:
                self.logger.error(f"Model file not found: {model_path}")
                return None

        except Exception as e:
            self.logger.error(
                f"Failed to load model {config.get('model_path', 'unknown')}: {e}"
            )
            return None

    def predict(self, images: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Predict using ensemble of models.

        Args:
            images: Dictionary of preprocessed image variants

        Returns:
            Dictionary with class probabilities
        """
        if not self.models:
            raise ValueError("No models loaded in ensemble")

        all_predictions = []

        for model_info in self.models:
            model = model_info["model"]
            config = model_info["config"]

            # Get the appropriate preprocessing variant for this model
            preprocessing_variant = config.get("preprocessing_variant", "original")

            if preprocessing_variant not in images:
                self.logger.warning(
                    f"Preprocessing variant '{preprocessing_variant}' not found, using 'original'"
                )
                preprocessing_variant = "original"

            if preprocessing_variant not in images:
                self.logger.error("No suitable image variant found for prediction")
                continue

            try:
                # Prepare image for this model
                image = images[preprocessing_variant]
                input_tensor = self._prepare_input(image, config)

                # Get prediction
                with torch.no_grad():
                    logits = model(input_tensor)
                    probabilities = F.softmax(logits, dim=1)
                    all_predictions.append(probabilities.cpu().numpy()[0])

            except Exception as e:
                self.logger.error(
                    f"Prediction failed for model {config.get('model_path', 'unknown')}: {e}"
                )

        if not all_predictions:
            raise ValueError("No successful predictions from ensemble")

        # Ensemble voting
        if self.voting_strategy == "soft":
            # Average probabilities
            ensemble_probs = np.mean(all_predictions, axis=0)
        elif self.voting_strategy == "hard":
            # Majority vote
            predictions = [np.argmax(pred) for pred in all_predictions]
            ensemble_pred = np.bincount(predictions).argmax()
            ensemble_probs = np.zeros(len(all_predictions[0]))
            ensemble_probs[ensemble_pred] = 1.0
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")

        # Convert to class probabilities dictionary
        class_names = [
            "No DR",
            "Mild DR",
            "Moderate DR",
            "Severe DR",
            "Proliferative DR",
        ]
        result = {class_names[i]: float(prob) for i, prob in enumerate(ensemble_probs)}

        # Add prediction metadata
        result["predicted_class"] = class_names[np.argmax(ensemble_probs)]
        result["confidence"] = float(np.max(ensemble_probs))
        result["ensemble_size"] = len(all_predictions)
        result["voting_strategy"] = self.voting_strategy

        return result

    def _prepare_input(self, image: np.ndarray, config: Dict) -> torch.Tensor:
        """Prepare input tensor for model."""
        # Image should already be preprocessed, just need to convert to tensor
        # target_size = config.get("input_size", (500, 500))

        # # Ensure image is in correct format
        # if image.dtype != np.uint8:
        #     if image.max() <= 1.0:
        #         image = (image * 255).astype(np.uint8)
        #     else:
        #         image = image.astype(np.uint8)

        # # Resize if needed
        # if image.shape[:2] != tuple(target_size):
        #     image = cv2.resize(image, target_size)

        # # Convert to tensor and normalize
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        tensor = transform(image).unsqueeze(0).to(self.device)
        return tensor


class DiabeticRetinopathyClassifier:
    """Main classifier class that combines preprocessing and classification."""

    def __init__(self, config_path: str):
        """
        Initialize the classifier.

        Args:
            config_path: Path to the classifier configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.device = self._setup_device()

        # Pull latest models from DVC/Azure Blob
        print("Syncing models from DVC (Azure Blob Storage)...")
        subprocess.run(["dvc", "pull", "-q"], check=False)

        # Initialize model ensemble
        self.ensemble = self._create_ensemble()

        self.logger.info("DiabeticRetinopathyClassifier initialized successfully")

    def _load_config(self, config_path: str) -> dict:
        """Load classifier configuration."""
        with open(config_path, "r") as f:
            if config_path.endswith(".json"):
                return json.load(f)
            else:
                import yaml

                return yaml.safe_load(f)

    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger(self.__class__.__name__)

        level = getattr(logging, self.config.get("log_level", "INFO").upper())
        logging.basicConfig(
            level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        return logger

    def _setup_device(self) -> torch.device:
        """Setup compute device."""
        if self.config.get("force_cpu", False):
            device = torch.device("cpu")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.logger.info(f"Using device: {device}")
        return device

    def _create_ensemble(self) -> ModelEnsemble:
        """Create model ensemble from configuration."""
        models_config = self.config.get("models", [])
        voting_strategy = self.config.get("voting_strategy", "soft")

        if not models_config:
            raise ValueError("No models configured")

        return ModelEnsemble(models_config, self.device, voting_strategy)

    def classify(self, preprocessed_images: Dict[str, np.ndarray]) -> Dict:
        """
        Classify diabetic retinopathy from preprocessed images.

        Args:
            preprocessed_images: Dictionary of preprocessed image variants

        Returns:
            Classification results with probabilities and metadata
        """
        try:
            results = self.ensemble.predict(preprocessed_images)

            # Add additional metadata
            results["model_info"] = {
                "ensemble_size": len(self.ensemble.models),
                "architectures": list(
                    set([m["config"]["architecture"] for m in self.ensemble.models])
                ),
                "datasets": list(
                    set(
                        [
                            m["config"].get("dataset", "unknown")
                            for m in self.ensemble.models
                        ]
                    )
                ),
                "voting_strategy": self.ensemble.voting_strategy,
            }

            return results

        except Exception as e:
            self.logger.error(f"Classification failed: {e}")
            raise

    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        model_info = []

        for model_data in self.ensemble.models:
            config = model_data["config"]
            info = {
                "architecture": config["architecture"],
                "dataset": config.get("dataset", "unknown"),
                "preprocessing_variant": config.get(
                    "preprocessing_variant", "original"
                ),
                "model_path": config["model_path"],
                "num_classes": config.get("num_classes", 5),
            }
            model_info.append(info)

        return {
            "models": model_info,
            "voting_strategy": self.ensemble.voting_strategy,
            "device": str(self.device),
            "total_models": len(model_info),
        }


def create_default_classifier_config(output_path: str, model_base_path: str):
    """Create a default classifier configuration file."""

    # Auto-discover models
    models = []
    base_path = Path(model_base_path)

    # Look for .pt files
    for model_file in base_path.rglob("*.pt"):
        path_parts = model_file.parts

        # Extract information from path structure
        architecture = None
        dataset = None
        preprocessing = None

        for part in path_parts:
            if part.lower() in ["xception", "efficientnetb4"]:
                architecture = part
            elif part.lower() in ["aptos5", "ddr6"]:
                dataset = part.upper()
            elif "original" in part.lower():
                preprocessing = "original"
            elif "rgbclahe" in part.lower():
                preprocessing = "rgb_clahe"

        if architecture and dataset:
            model_config = {
                "architecture": architecture,
                "dataset": dataset,
                "preprocessing_variant": preprocessing or "original",
                "model_path": str(model_file),
                "num_classes": 5,
                "input_size": [500, 500],
            }
            models.append(model_config)

    config = {
        "log_level": "INFO",
        "force_cpu": False,
        "voting_strategy": "soft",  # or 'hard'
        "models": models[:4],  # Limit to 4 models for example
        "class_names": [
            "No DR",
            "Mild DR",
            "Moderate DR",
            "Severe DR",
            "Proliferative DR",
        ],
        "confidence_threshold": 0.5,
    }

    with open(output_path, "w") as f:
        if output_path.endswith(".json"):
            json.dump(config, f, indent=2)
        else:
            import yaml

            yaml.dump(config, f, default_flow_style=False, indent=2)

    print(f"Classifier configuration created: {output_path}")
    print(f"Found {len(models)} model files")
    return config


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Diabetic Retinopathy Classifier")
    parser.add_argument(
        "--create-config", type=str, help="Create classifier configuration"
    )
    parser.add_argument("--model-path", type=str, help="Base path for model discovery")
    parser.add_argument("--config", type=str, help="Classifier configuration file")
    parser.add_argument("--test", action="store_true", help="Test classifier")

    args = parser.parse_args()

    if args.create_config and args.model_path:
        create_default_classifier_config(args.create_config, args.model_path)
    elif args.config and args.test:
        # Test classifier
        classifier = DiabeticRetinopathyClassifier(args.config)
        model_info = classifier.get_model_info()
        print("Loaded models:")
        for model in model_info["models"]:
            print(
                f"  - {model['architecture']} ({model['dataset']}) - {model['preprocessing_variant']}"
            )
    else:
        parser.print_help()
