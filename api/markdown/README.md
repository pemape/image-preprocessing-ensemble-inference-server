# Documentation for Fundus Image Processing and Classification API

<a name="documentation-for-api-endpoints"></a>
## Documentation for API Endpoints

All URIs are relative to *http://localhost:5000*

| Class | Method | HTTP request | Description |
|------------ | ------------- | ------------- | -------------|
| *CacheManagementApi* | [**clearCache**](Apis/CacheManagementApi.md#clearCache) | **POST** /cache/clear | Clear cache |
*CacheManagementApi* | [**getCacheHealth**](Apis/CacheManagementApi.md#getCacheHealth) | **GET** /cache/health | Get cache health |
*CacheManagementApi* | [**getCacheStats**](Apis/CacheManagementApi.md#getCacheStats) | **GET** /cache/stats | Get cache statistics |
| *HealthInfoApi* | [**getConfig**](Apis/HealthInfoApi.md#getConfig) | **GET** /config | Get configuration details |
*HealthInfoApi* | [**getInfo**](Apis/HealthInfoApi.md#getInfo) | **GET** /info | Get server information |
*HealthInfoApi* | [**getModels**](Apis/HealthInfoApi.md#getModels) | **GET** /models | Get model information |
*HealthInfoApi* | [**healthCheck**](Apis/HealthInfoApi.md#healthCheck) | **GET** /health | Health check |
| *SingleProcessingApi* | [**classifyImage**](Apis/SingleProcessingApi.md#classifyImage) | **POST** /classify | Classify from preprocessed images |
*SingleProcessingApi* | [**fullProcess**](Apis/SingleProcessingApi.md#fullProcess) | **POST** /process | Full pipeline (preprocess + classify) |
*SingleProcessingApi* | [**preprocessImage**](Apis/SingleProcessingApi.md#preprocessImage) | **POST** /preprocess | Preprocess a single image |


<a name="documentation-for-models"></a>
## Documentation for Models

 - [BatchProcessingMetrics](./Models/BatchProcessingMetrics.md)
 - [CacheClearResponse](./Models/CacheClearResponse.md)
 - [CacheHealthResponse](./Models/CacheHealthResponse.md)
 - [CacheStats](./Models/CacheStats.md)
 - [CacheStatsResponse](./Models/CacheStatsResponse.md)
 - [ClassProbability](./Models/ClassProbability.md)
 - [ClassificationResult](./Models/ClassificationResult.md)
 - [ClassifyResponse](./Models/ClassifyResponse.md)
 - [ConfigResponse](./Models/ConfigResponse.md)
 - [ConfigResponse_classification](./Models/ConfigResponse_classification.md)
 - [ConfigResponse_preprocessing](./Models/ConfigResponse_preprocessing.md)
 - [DynamicBatchingConfig](./Models/DynamicBatchingConfig.md)
 - [ErrorResponse](./Models/ErrorResponse.md)
 - [HealthResponse](./Models/HealthResponse.md)
 - [HealthResponse_modules](./Models/HealthResponse_modules.md)
 - [ImageProcessingTimes](./Models/ImageProcessingTimes.md)
 - [ImageProperties](./Models/ImageProperties.md)
 - [InfoResponse](./Models/InfoResponse.md)
 - [InfoResponse_modules](./Models/InfoResponse_modules.md)
 - [InfoResponse_modules_classification](./Models/InfoResponse_modules_classification.md)
 - [InfoResponse_modules_preprocessing](./Models/InfoResponse_modules_preprocessing.md)
 - [ModelConfiguration](./Models/ModelConfiguration.md)
 - [ModelInfoResponse](./Models/ModelInfoResponse.md)
 - [ModelInfoResponse_models_inner](./Models/ModelInfoResponse_models_inner.md)
 - [OperationStatus](./Models/OperationStatus.md)
 - [PreprocessResponse](./Models/PreprocessResponse.md)
 - [PreprocessResponse_metadata](./Models/PreprocessResponse_metadata.md)
 - [PreprocessedImages](./Models/PreprocessedImages.md)
 - [ProcessMetadata](./Models/ProcessMetadata.md)
 - [ProcessResponse](./Models/ProcessResponse.md)
 - [ProcessResult](./Models/ProcessResult.md)
 - [VersionInfo](./Models/VersionInfo.md)
 - [VotingStrategyEnum](./Models/VotingStrategyEnum.md)
 - [classifyImage_request](./Models/classifyImage_request.md)
 - [clearCache_request](./Models/clearCache_request.md)


<a name="documentation-for-authorization"></a>
## Documentation for Authorization

All endpoints do not require authorization.
