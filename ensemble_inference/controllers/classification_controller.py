import connexion
from typing import Dict
from typing import Tuple
from typing import Union

from ensemble_inference.models.classify_image_request import ClassifyImageRequest  # noqa: E501
from ensemble_inference.models.classify_response import ClassifyResponse  # noqa: E501
from ensemble_inference.models.error_response import ErrorResponse  # noqa: E501
from ensemble_inference import util


def classify_image(body, voting_strategy=None):  # noqa: E501
    """Classify from preprocessed images

    Classify diabetic retinopathy from preprocessed image variants # noqa: E501

    :param classify_image_request: 
    :type classify_image_request: dict | bytes
    :param voting_strategy: Ensemble voting strategy for classification
    :type voting_strategy: str

    :rtype: Union[ClassifyResponse, Tuple[ClassifyResponse, int], Tuple[ClassifyResponse, int, Dict[str, str]]
    """
    classify_image_request = body
    if connexion.request.is_json:
        classify_image_request = ClassifyImageRequest.from_dict(connexion.request.get_json())  # noqa: E501
    return 'do some magic!'
