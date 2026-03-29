# CacheManagementApi

All URIs are relative to *http://localhost:5000*

| Method | HTTP request | Description |
|------------- | ------------- | -------------|
| [**clearCache**](CacheManagementApi.md#clearCache) | **POST** /cache/clear | Clear cache |
| [**getCacheHealth**](CacheManagementApi.md#getCacheHealth) | **GET** /cache/health | Get cache health |
| [**getCacheStats**](CacheManagementApi.md#getCacheStats) | **GET** /cache/stats | Get cache statistics |


<a name="clearCache"></a>
# **clearCache**
> CacheClearResponse clearCache(clearCache\_request)

Clear cache

    Clear all or pattern-matched cached entries

### Parameters

|Name | Type | Description  | Notes |
|------------- | ------------- | ------------- | -------------|
| **clearCache\_request** | [**clearCache_request**](../Models/clearCache_request.md)|  | [optional] |

### Return type

[**CacheClearResponse**](../Models/CacheClearResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: application/json
- **Accept**: application/json

<a name="getCacheHealth"></a>
# **getCacheHealth**
> CacheHealthResponse getCacheHealth()

Get cache health

    Check Redis cache connection health

### Parameters
This endpoint does not need any parameter.

### Return type

[**CacheHealthResponse**](../Models/CacheHealthResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

<a name="getCacheStats"></a>
# **getCacheStats**
> CacheStatsResponse getCacheStats()

Get cache statistics

    Retrieve Redis cache statistics including hits, misses, and hit rate

### Parameters
This endpoint does not need any parameter.

### Return type

[**CacheStatsResponse**](../Models/CacheStatsResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

