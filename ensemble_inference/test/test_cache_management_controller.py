import unittest

from flask import json

from ensemble_inference.models.cache_clear_response import CacheClearResponse  # noqa: E501
from ensemble_inference.models.cache_health_response import CacheHealthResponse  # noqa: E501
from ensemble_inference.models.cache_stats_response import CacheStatsResponse  # noqa: E501
from ensemble_inference.models.clear_cache_request import ClearCacheRequest  # noqa: E501
from ensemble_inference.models.error_response import ErrorResponse  # noqa: E501
from ensemble_inference.test import BaseTestCase


class TestCacheManagementController(BaseTestCase):
    """CacheManagementController integration test stubs"""

    def test_clear_cache(self):
        """Test case for clear_cache

        Clear cache
        """
        clear_cache_request = ensemble_inference.ClearCacheRequest()
        headers = { 
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        }
        response = self.client.open(
            '/cache/clear',
            method='POST',
            headers=headers,
            data=json.dumps(clear_cache_request),
            content_type='application/json')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_get_cache_health(self):
        """Test case for get_cache_health

        Get cache health
        """
        headers = { 
            'Accept': 'application/json',
        }
        response = self.client.open(
            '/cache/health',
            method='GET',
            headers=headers)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_get_cache_stats(self):
        """Test case for get_cache_stats

        Get cache statistics
        """
        headers = { 
            'Accept': 'application/json',
        }
        response = self.client.open(
            '/cache/stats',
            method='GET',
            headers=headers)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    unittest.main()
