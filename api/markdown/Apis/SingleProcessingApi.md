# SingleProcessingApi

All URIs are relative to *http://localhost:5000*

| Method | HTTP request | Description |
|------------- | ------------- | -------------|
| [**classifyImage**](SingleProcessingApi.md#classifyImage) | **POST** /classify | Classify from preprocessed images |
| [**fullProcess**](SingleProcessingApi.md#fullProcess) | **POST** /process | Full pipeline (preprocess + classify) |
| [**preprocessImage**](SingleProcessingApi.md#preprocessImage) | **POST** /preprocess | Preprocess a single image |


<a name="classifyImage"></a>
# **classifyImage**
> ClassifyResponse classifyImage(classifyImage\_request, voting\_strategy)

Classify from preprocessed images

    Classify diabetic retinopathy from preprocessed image variants.  Dynamic batching is handled transparently on the server side. Clients do not need to provide any batching-specific path parameter. 

### Parameters

|Name | Type | Description  | Notes |
|------------- | ------------- | ------------- | -------------|
| **classifyImage\_request** | [**classifyImage_request**](../Models/classifyImage_request.md)|  | |
| **voting\_strategy** | [**VotingStrategyEnum**](../Models/.md)| Ensemble voting strategy for classification | [optional] [default to null] [enum: soft, hard] |

### Return type

[**ClassifyResponse**](../Models/ClassifyResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: application/json
- **Accept**: application/json

<a name="fullProcess"></a>
# **fullProcess**
> ProcessResponse fullProcess(image, voting\_strategy, include\_encoded\_images)

Full pipeline (preprocess + classify)

    Complete pipeline from raw image to classification result.  **Single Image Only**: This endpoint accepts exactly ONE image.  **Caching**: Results are cached with Redis based on image hash and model configuration. Cached responses return near-instant results with &#x60;cached&#x3D;true&#x60; indicator.  **Dynamic batching**: Classification stage may be dynamically micro-batched server-side. This behavior is transparent to clients. 

### Parameters

|Name | Type | Description  | Notes |
|------------- | ------------- | ------------- | -------------|
| **image** | **File**| Fundus image file (JPEG, PNG, TIFF) - **SINGLE IMAGE ONLY** | [default to null] |
| **voting\_strategy** | [**VotingStrategyEnum**](../Models/.md)| Ensemble voting strategy for classification | [optional] [default to null] [enum: soft, hard] |
| **include\_encoded\_images** | **Boolean**| Include preprocessed images in response (results not cached if true) | [optional] [default to false] |

### Return type

[**ProcessResponse**](../Models/ProcessResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: multipart/form-data
- **Accept**: application/json

<a name="preprocessImage"></a>
# **preprocessImage**
> PreprocessResponse preprocessImage(image, include\_encoded\_images)

Preprocess a single image

    Apply preprocessing pipeline to generate 5 image variants

### Parameters

|Name | Type | Description  | Notes |
|------------- | ------------- | ------------- | -------------|
| **image** | **File**| Fundus image file (JPEG, PNG, TIFF) | [default to null] |
| **include\_encoded\_images** | **Boolean**| Include preprocessed images in response (results not cached if true) | [optional] [default to false] |

### Return type

[**PreprocessResponse**](../Models/PreprocessResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: multipart/form-data
- **Accept**: application/json

