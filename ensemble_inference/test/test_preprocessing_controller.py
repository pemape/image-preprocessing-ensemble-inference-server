import unittest

from flask import json

from ensemble_inference.models.error_response import ErrorResponse  # noqa: E501
from ensemble_inference.models.preprocess_response import PreprocessResponse  # noqa: E501
from ensemble_inference.test import BaseTestCase


class TestPreprocessingController(BaseTestCase):
    """PreprocessingController integration test stubs"""

    @unittest.skip("multipart/form-data not supported by Connexion")
    def test_preprocess_image(self):
        """Test case for preprocess_image

        Preprocess a single image
        """
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
            content_type='multipart/form-data')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    unittest.main()
