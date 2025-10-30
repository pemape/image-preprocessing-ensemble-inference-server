import connexion
from typing import Dict
from typing import Tuple
from typing import Union

from ensemble_inference.models.error_response import ErrorResponse  # noqa: E501
from ensemble_inference.models.process_response import ProcessResponse  # noqa: E501
from ensemble_inference import util


def full_process(image, voting_strategy=None, include_images=None):  # noqa: E501
    """Full pipeline (preprocess + classify)

    Complete pipeline from raw image to classification result.  **Single Image Only**: This endpoint accepts exactly ONE image. Use &#x60;/batch/process&#x60; for multiple images.  **Caching**: Results are cached with Redis based on image hash and model configuration. Cached responses return near-instant results with &#x60;cached&#x3D;true&#x60; indicator.  # noqa: E501

    :param image: Fundus image file (JPEG, PNG, TIFF) - **SINGLE IMAGE ONLY**
    :type image: str
    :param voting_strategy: Ensemble voting strategy for classification
    :type voting_strategy: str
    :param include_images: Include preprocessed images in response (not cached if true)
    :type include_images: str

    :rtype: Union[ProcessResponse, Tuple[ProcessResponse, int], Tuple[ProcessResponse, int, Dict[str, str]]
    """
    return 'do some magic!'
