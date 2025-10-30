import unittest

from flask import json

from ensemble_inference.models.error_response import ErrorResponse  # noqa: E501
from ensemble_inference.models.process_response import ProcessResponse  # noqa: E501
from ensemble_inference.test import BaseTestCase


class TestFullPipelineController(BaseTestCase):
    """FullPipelineController integration test stubs"""

    @unittest.skip("multipart/form-data not supported by Connexion")
    def test_full_process(self):
        """Test case for full_process

        Full pipeline (preprocess + classify)
        """
        query_string = [('voting_strategy', soft)]
        headers = { 
            'Accept': 'application/json',
            'Content-Type': 'multipart/form-data',
        }
        data = dict(image='/path/to/file',
                    include_images=false)
        response = self.client.open(
            '/process',
            method='POST',
            headers=headers,
            data=data,
            content_type='multipart/form-data',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    unittest.main()
