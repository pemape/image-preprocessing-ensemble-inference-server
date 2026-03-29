# BatchProcessingMetrics
## Properties

| Name | Type | Description | Notes |
|------------ | ------------- | ------------- | -------------|
| **batch\_id** | **String** | Unique identifier for the executed dynamic batch | [optional] [default to null] |
| **batch\_wait\_ms** | **Float** | Time spent waiting in the dynamic batch queue before inference starts | [optional] [default to null] |
| **batch\_execution\_ms** | **Float** | Time spent executing the grouped batch on the model | [optional] [default to null] |
| **batch\_total\_ms** | **Float** | Total time from queue enqueue to batch result availability | [optional] [default to null] |
| **batch\_size** | **Integer** | Number of requests grouped into the executed batch | [optional] [default to null] |

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

