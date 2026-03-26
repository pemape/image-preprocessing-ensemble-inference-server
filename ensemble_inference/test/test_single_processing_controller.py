import unittest

from flask import json

from ensemble_inference.models.classify_image_request import ClassifyImageRequest  # noqa: E501
from ensemble_inference.models.classify_response import ClassifyResponse  # noqa: E501
from ensemble_inference.models.error_response import ErrorResponse  # noqa: E501
from ensemble_inference.models.preprocess_response import PreprocessResponse  # noqa: E501
from ensemble_inference.models.process_response import ProcessResponse  # noqa: E501
from ensemble_inference.models.voting_strategy_enum import VotingStrategyEnum  # noqa: E501
from ensemble_inference.test import BaseTestCase


class TestSingleProcessingController(BaseTestCase):
    """SingleProcessingController integration test stubs"""

    def test_classify_image(self):
        """Test case for classify_image

        Classify from preprocessed images
        """
        classify_image_request = ensemble_inference.ClassifyImageRequest()
        query_string = [('voting_strategy', soft)]
        headers = { 
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        }
        response = self.client.open(
            '/classify',
            method='POST',
            headers=headers,
            data=json.dumps(classify_image_request),
            content_type='application/json',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    @unittest.skip("multipart/form-data not supported by Connexion")
    def test_full_process(self):
        """Test case for full_process

        Full pipeline (preprocess + classify)
        """
        query_string = [('voting_strategy', soft),
                        ('include_encoded_images', False)]
        headers = { 
            'Accept': 'application/json',
            'Content-Type': 'multipart/form-data',
        }
        data = dict(image='/path/to/file')
        response = self.client.open(
            '/process',
            method='POST',
            headers=headers,
            data=data,
            content_type='multipart/form-data',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    @unittest.skip("multipart/form-data not supported by Connexion")
    def test_preprocess_image(self):
        """Test case for preprocess_image

        Preprocess a single image
        """
        query_string = [('include_encoded_images', False)]
        headers = { 
            'Accept': 'application/json',
            'Content-Type': 'multipart/form-data',
        }
        data = dict(image='/path/to/file')
        response = self.client.open(
            '/preprocess',
            method='POST',
            headers=headers,
            data=data,
            content_type='multipart/form-data',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    unittest.main()
