import unittest

from flask import json

from ensemble_inference.models.batch_preprocess_response import BatchPreprocessResponse  # noqa: E501
from ensemble_inference.models.batch_process_response import BatchProcessResponse  # noqa: E501
from ensemble_inference.models.error_response import ErrorResponse  # noqa: E501
from ensemble_inference.test import BaseTestCase


class TestBatchProcessingController(BaseTestCase):
    """BatchProcessingController integration test stubs"""

    @unittest.skip("multipart/form-data not supported by Connexion")
    def test_batch_preprocess(self):
        """Test case for batch_preprocess

        Batch preprocessing
        """
        headers = { 
            'Accept': 'application/json',
            'Content-Type': 'multipart/form-data',
        }
        data = dict(images=['/path/to/file'])
        response = self.client.open(
            '/batch/preprocess',
            method='POST',
            headers=headers,
            data=data,
            content_type='multipart/form-data')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    @unittest.skip("multipart/form-data not supported by Connexion")
    def test_batch_process(self):
        """Test case for batch_process

        Batch full processing
        """
        query_string = [('voting_strategy', soft)]
        headers = { 
            'Accept': 'application/json',
            'Content-Type': 'multipart/form-data',
        }
        data = dict(images=['/path/to/file'])
        response = self.client.open(
            '/batch/process',
            method='POST',
            headers=headers,
            data=data,
            content_type='multipart/form-data',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    unittest.main()
