import connexion
from typing import Dict
from typing import Tuple
from typing import Union

from ensemble_inference.models.error_response import ErrorResponse  # noqa: E501
from ensemble_inference.models.preprocess_response import PreprocessResponse  # noqa: E501
from ensemble_inference import util


def preprocess_image(image):  # noqa: E501
    """Preprocess a single image

    Apply preprocessing pipeline to generate 5 image variants # noqa: E501

    :param image: Fundus image file (JPEG, PNG, TIFF)
    :type image: str

    :rtype: Union[PreprocessResponse, Tuple[PreprocessResponse, int], Tuple[PreprocessResponse, int, Dict[str, str]]
    """
    return 'do some magic!'
