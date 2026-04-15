---
hide:
  - navigation
---


# Service Startup Parameters

xLLM uses gflags to manage service startup parameters. The specific parameter meanings are as follows:

## Common Parameters
| Parameter Name | Data Type | Default Value | Other Values | Description | Notes |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| `master_node_addr` | `string` | "127.0.0.1:19888" | ip:port | The listening address of the master node's rpc server | [Details](./features/basics.md) |
| `host` | `string` | "" | The machine IP where the current device is located | The host IP used by the current device for communication. An rpc server is started on each device for multi-device communication. |  |
| `port` | `int32` | 8010 | Any available port | Used in conjunction with the `host` parameter. The combination is used for rpc communication between devices. |  |
| `model` | `string` | "" |  | Path to the model. |  |
| `devices` | `string` | "npu:0" |  | Specifies the NPU devices used by the current process. |  |
| `nnodes` | `int32` | 1 |  | The total number of devices used by the current service. |  |
| `node_rank` | `int32` | 0 | 0 ~ (total devices - 1) | The rank id of each device. |  |
| `max_memory_utilization` | `double` | 0.8 | Between 0-1 | The maximum proportion of device memory available for model weights and KV Cache combined. |  |
| `max_tokens_per_batch` | `int32` | 10240 |  | The maximum number of tokens that can be computed per step. |  |
| `max_seqs_per_batch` | `int32` | 1024 |  | The maximum number of sequences that can be computed per step. |  |
| `enable_chunked_prefill` | `bool` | true | false | Whether to enable chunked prefill. |  |
| `enable_prefill_sp` | `bool` | false | true | Whether to enable prefill-only sequence parallel. | `enable_chunked_prefill=true` is supported only for prefill-only batches (`PREFILL` / `CHUNKED_PREFILL`); `MIXED` and `DECODE` batches do not run with sequence parallel. |
| `enable_schedule_overlap` | `bool` | false | true | Whether to enable asynchronous scheduling. | [Details](./features/async_schedule.md) |
| `enable_prefix_cache` | `bool` | true | false | Whether to enable prefix cache (not supported by DeepSeek currently). |  |
| `communication_backend` | `string` | "hccl" | "lccl" | The backend used for communication operations. |  |
| `block_size` | `int32` | 128 |  | The block size for KV Cache storage. |  |
| `task` | `string` | "generate" | "embed", "mm_embed" | Service type: generation, embedding, or multimodal embedding. |  |
| `max_cache_size` | `int64` | 0 |  | The usable KV Cache size in bytes. |  |
| `kv_cache_dtype` | `string` | "auto" | "int8" | KV Cache data type. "auto" aligns with model dtype (no quantization), "int8" enables INT8 quantization to save ~50% memory. MLU backend only. |  |

## MoE Model Related Parameters
| Parameter Name | Type | Default Value | Other Values | Description | Notes |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| `dp_size` | `int32` | 1 | Power of 2 | The dp scale size for the Attention part. |  |
| `ep_size` | `int32` | 1 | Power of 2 | The ep scale size for the MoE part. |  |
| `expert_parallel_degree` | `int32` | 0 | 1,2 | Parameter related to ep parallelism. Defaults to 0 when ep is not used, and to 1 when ep is enabled. Can be set to 2 when `ep_size` equals the total number of devices (uses all2all communication). |  |


## P-D Separation Related Parameters
| Parameter Name | Type | Default Value | Other Values | Description | Notes |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| `enable_disagg_pd` | `bool` | false | true | Whether to enable P-D separation. | [Details](./features/disagg_pd.md) |
| `disagg_pd_port` | `int32` | 7777 | Any available port | Configuration when P-D separation is enabled. Corresponds to the listening port number of the pd separation rpc server started on each card. |  |
| `instance_role` | `string` | DEFAULT | PREFILL, DECODE, MIX | Defaults to DEFAULT. Must be configured as PREFILL, DECODE, or MIX when P-D separation is enabled. |  |
| `kv_cache_transfer_mode` | `string` | "PUSH" | "PULL" | The mode for transferring KV Cache in P-D separation. PUSH mode: Prefill transmits layer by layer to Decode; PULL mode: Decode pulls the KV Cache from Prefill in one go. |  |
| `transfer_listen_port` | `int32` | 26000 | Any available port | Configuration when P-D separation is enabled. Corresponds to the listening port for KV Cache Transfer on each card. |  |


## Speculative Decoding Parameters
| Parameter Name | Type | Default Value | Other Values | Description | Notes |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| `draft_model` | `string` | "" |  | Path to the MTP model. Not required when `speculative_algorithm` is `PLD` or `PromptLookup`. | [Details](./features/mtp.md) |
| `draft_devices` | `string` | "npu:0" | Same format as `devices`, e.g. `npu:0` or `npu:0,npu:1` | Should be set consistently with the `devices` parameter for draft-model algorithms. It is ignored/not required for `PLD` and `PromptLookup`. |  |
| `num_speculative_tokens` | `int32` | 0 | Any positive integer; PLD commonly uses `8` | The number of speculative tokens proposed per decode step. |  |
| `speculative_algorithm` | `string` | "MTP" | "MTP", "Eagle3", "Suffix", "PLD", "PromptLookup" | Selects the speculative decoding algorithm. `PLD` and `PromptLookup` use prompt lookup decoding and do not require a draft model. |  |
| `speculative_suffix_cache_max_depth` | `int32` | 64 | Any integer greater than 0 | For `Suffix`, this is the suffix cache depth. For `PLD`, it caps the maximum lookup n-gram size together with the internal limit of `8`. |  |
| `speculative_suffix_max_spec_factor` | `double` | 1.0 | Any non-negative floating-point number | Upper bound factor for suffix-based speculation length. | Used by `Suffix`. |
| `speculative_suffix_max_spec_offset` | `double` | 0.0 | Any non-negative floating-point number | Additive upper bound for suffix-based speculation length. | Used by `Suffix`. |
| `speculative_suffix_min_token_prob` | `double` | 0.1 | Any floating-point number in `[0, 1]` | Minimum token probability threshold for suffix-tree speculation. | Used by `Suffix`. |
| `speculative_suffix_max_cached_requests` | `int32` | -1 | `-1`, `0`, or any positive integer | Maximum number of globally cached requests in suffix speculation. | Used by `Suffix`. |
| `speculative_suffix_use_tree_spec` | `bool` | false | true | Enables tree-based suffix speculation instead of path speculation. | Used by `Suffix`. |


## Graph Execution Related Parameters
| Parameter Name | Type | Default Value | Other Values | Description | Notes |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| `enable_graph` | `bool` | false | true | Whether to enable graph execution mode to optimize decode phase performance. Only applied during decode phase and does not take effect during prefill phase. Supports ACL Graph (NPU), and MLU Graph. | [Details](./features/graph_mode.md) |
| `enable_graph_mode_decode_no_padding` | `bool` | false | true | Builds decode graphs with the actual `num_tokens` instead of the padded shape. |  |
| `enable_prefill_piecewise_graph` | `bool` | false | true | Whether to enable piecewise graph for prefill phase. Attention runs eagerly while other ops are captured into CUDA graphs. |  |
| `max_tokens_for_graph_mode` | `int32` | 2048 | Any integer greater than or equal to 0 | Maximum number of tokens for graph execution. If 0, no limit is applied. |  |


## Parameters for Use with xLLM-service
| Parameter Name | Type | Default Value | Other Values | Description | Notes |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| `etcd_addr` | `string` | "" | ip:port | The listening address of the etcd's rpc server. |  |
| `enable_service_routing` | `bool` | false | true | Whether the request from the xllm service, use this when enable the xllm service. |  |

## Other Parameters
| Parameter Name | Type | Default Value | Other Values | Description | Notes |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| `max_concurrent_requests` | `int32` | 200 | Any integer greater than or equal to 0 | For rate limiting, restricts the total number of requests being processed in the instance. Set to 0 for no limit. |  |
| `model_id` | `string` | "" |  | Model name, not a path. |  |
| `num_request_handling_threads` | `int32` | 4 | Any integer greater than 0 | The thread pool size for handling input requests. |  |
| `prefill_scheduling_memory_usage_threshold` | `double` | 0.95 | Value between 0-1 | When kv cache usage reaches this threshold, scheduling of prefill requests is paused. |  |
| `num_response_handling_threads` | `int32` | 4 | Any integer greater than 0 | The thread pool size for handling outputs. |  |
| `rank_tablefile` | `string` | "" |  | Configuration file for creating the communication domain. Required for multi-node scenarios. |  |
