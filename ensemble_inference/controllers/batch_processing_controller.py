import connexion
from typing import Dict
from typing import Tuple
from typing import Union

from ensemble_inference.models.batch_preprocess_response import BatchPreprocessResponse  # noqa: E501
from ensemble_inference.models.batch_process_response import BatchProcessResponse  # noqa: E501
from ensemble_inference.models.error_response import ErrorResponse  # noqa: E501
from ensemble_inference import util


def batch_preprocess(images):  # noqa: E501
    """Batch preprocessing

    Preprocess multiple images in a single request # noqa: E501

    :param images: Multiple fundus image files
    :type images: List[str]

    :rtype: Union[BatchPreprocessResponse, Tuple[BatchPreprocessResponse, int], Tuple[BatchPreprocessResponse, int, Dict[str, str]]
    """
    return 'do some magic!'


def batch_process(images, voting_strategy=None):  # noqa: E501
    """Batch full processing

    Process multiple images through complete pipeline (preprocess + classify).  **Batch Size Limit**: Maximum 100 images per request (configurable).  **Caching**: Each image is checked against Redis cache. Cache hits skip processing. Response includes &#x60;cache_stats&#x60; showing hits/misses.  # noqa: E501

    :param images: Multiple fundus image files
    :type images: List[str]
    :param voting_strategy: Ensemble voting strategy for classification
    :type voting_strategy: str

    :rtype: Union[BatchProcessResponse, Tuple[BatchProcessResponse, int], Tuple[BatchProcessResponse, int, Dict[str, str]]
    """
    return 'do some magic!'
