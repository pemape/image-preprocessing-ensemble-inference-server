# HealthInfoApi

All URIs are relative to *http://localhost:5000*

| Method | HTTP request | Description |
|------------- | ------------- | -------------|
| [**getConfig**](HealthInfoApi.md#getConfig) | **GET** /config | Get configuration details |
| [**getInfo**](HealthInfoApi.md#getInfo) | **GET** /info | Get server information |
| [**getModels**](HealthInfoApi.md#getModels) | **GET** /models | Get model information |
| [**healthCheck**](HealthInfoApi.md#healthCheck) | **GET** /health | Health check |


<a name="getConfig"></a>
# **getConfig**
> ConfigResponse getConfig()

Get configuration details

    Retrieve current server configuration

### Parameters
This endpoint does not need any parameter.

### Return type

[**ConfigResponse**](../Models/ConfigResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

<a name="getInfo"></a>
# **getInfo**
> InfoResponse getInfo()

Get server information

    Retrieve detailed information about available modules and endpoints

### Parameters
This endpoint does not need any parameter.

### Return type

[**InfoResponse**](../Models/InfoResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

<a name="getModels"></a>
# **getModels**
> ModelInfoResponse getModels()

Get model information

    Retrieve information about loaded classification models

### Parameters
This endpoint does not need any parameter.

### Return type

[**ModelInfoResponse**](../Models/ModelInfoResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

<a name="healthCheck"></a>
# **healthCheck**
> HealthResponse healthCheck()

Health check

    Check server health status and uptime

### Parameters
This endpoint does not need any parameter.

### Return type

[**HealthResponse**](../Models/HealthResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

