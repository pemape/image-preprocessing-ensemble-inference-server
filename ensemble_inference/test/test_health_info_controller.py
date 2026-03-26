import unittest

from flask import json

from ensemble_inference.models.config_response import ConfigResponse  # noqa: E501
from ensemble_inference.models.error_response import ErrorResponse  # noqa: E501
from ensemble_inference.models.health_response import HealthResponse  # noqa: E501
from ensemble_inference.models.info_response import InfoResponse  # noqa: E501
from ensemble_inference.models.model_info_response import ModelInfoResponse  # noqa: E501
from ensemble_inference.test import BaseTestCase


class TestHealthInfoController(BaseTestCase):
    """HealthInfoController integration test stubs"""

    def test_get_config(self):
        """Test case for get_config

        Get configuration details
        """
        headers = { 
            'Accept': 'application/json',
        }
        response = self.client.open(
            '/config',
            method='GET',
            headers=headers)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_get_info(self):
        """Test case for get_info

        Get server information
        """
        headers = { 
            'Accept': 'application/json',
        }
        response = self.client.open(
            '/info',
            method='GET',
            headers=headers)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_get_models(self):
        """Test case for get_models

        Get model information
        """
        headers = { 
            'Accept': 'application/json',
        }
        response = self.client.open(
            '/models',
            method='GET',
            headers=headers)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_health_check(self):
        """Test case for health_check

        Health check
        """
        headers = { 
            'Accept': 'application/json',
        }
        response = self.client.open(
            '/health',
            method='GET',
            headers=headers)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    unittest.main()
