import unittest

from flask import json

from ensemble_inference.models.classify_image_request import ClassifyImageRequest  # noqa: E501
from ensemble_inference.models.classify_response import ClassifyResponse  # noqa: E501
from ensemble_inference.models.error_response import ErrorResponse  # noqa: E501
from ensemble_inference.test import BaseTestCase


class TestClassificationController(BaseTestCase):
    """ClassificationController integration test stubs"""

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


if __name__ == '__main__':
    unittest.main()
