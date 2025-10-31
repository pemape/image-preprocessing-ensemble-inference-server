import connexion
from typing import Dict
from typing import Tuple
from typing import Union

from ensemble_inference.models.classify_image_request import ClassifyImageRequest  # noqa: E501
from ensemble_inference.models.classify_response import ClassifyResponse  # noqa: E501
from ensemble_inference.models.error_response import ErrorResponse  # noqa: E501
from ensemble_inference.models.preprocess_response import PreprocessResponse  # noqa: E501
from ensemble_inference.models.process_response import ProcessResponse  # noqa: E501
from ensemble_inference.models.voting_strategy_enum import VotingStrategyEnum  # noqa: E501
from ensemble_inference import util


def classify_image(body, voting_strategy=None):  # noqa: E501
    """Classify from preprocessed images

    Classify diabetic retinopathy from preprocessed image variants # noqa: E501

    :param classify_image_request: 
    :type classify_image_request: dict | bytes
    :param voting_strategy: Ensemble voting strategy for classification
    :type voting_strategy: dict | bytes

    :rtype: Union[ClassifyResponse, Tuple[ClassifyResponse, int], Tuple[ClassifyResponse, int, Dict[str, str]]
    """
    classify_image_request = body
    if connexion.request.is_json:
        classify_image_request = ClassifyImageRequest.from_dict(connexion.request.get_json())  # noqa: E501
    if connexion.request.is_json:
        voting_strategy =  VotingStrategyEnum.from_dict(connexion.request.get_json())  # noqa: E501
    return 'do some magic!'


def full_process(image, voting_strategy=None, include_encoded_images=None):  # noqa: E501
    """Full pipeline (preprocess + classify)

    Complete pipeline from raw image to classification result.  **Single Image Only**: This endpoint accepts exactly ONE image. Use &#x60;/batch/process&#x60; for multiple images.  **Caching**: Results are cached with Redis based on image hash and model configuration. Cached responses return near-instant results with &#x60;cached&#x3D;true&#x60; indicator.  # noqa: E501

    :param image: Fundus image file (JPEG, PNG, TIFF) - **SINGLE IMAGE ONLY**
    :type image: str
    :param voting_strategy: Ensemble voting strategy for classification
    :type voting_strategy: dict | bytes
    :param include_encoded_images: Include preprocessed images in response (results not cached if true)
    :type include_encoded_images: bool

    :rtype: Union[ProcessResponse, Tuple[ProcessResponse, int], Tuple[ProcessResponse, int, Dict[str, str]]
    """
    if connexion.request.is_json:
        voting_strategy =  VotingStrategyEnum.from_dict(connexion.request.get_json())  # noqa: E501
    return 'do some magic!'


def preprocess_image(image, include_encoded_images=None):  # noqa: E501
    """Preprocess a single image

    Apply preprocessing pipeline to generate 5 image variants # noqa: E501

    :param image: Fundus image file (JPEG, PNG, TIFF)
    :type image: str
    :param include_encoded_images: Include preprocessed images in response (results not cached if true)
    :type include_encoded_images: bool

    :rtype: Union[PreprocessResponse, Tuple[PreprocessResponse, int], Tuple[PreprocessResponse, int, Dict[str, str]]
    """
    return 'do some magic!'
