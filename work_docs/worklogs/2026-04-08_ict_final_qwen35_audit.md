# ict_final 分支 Qwen3.5 适配与性能优化审计报告

## 0. 审计元信息

- 时间：2026-04-08 22:38:18 CST
- 分支：ict_final（对照 main）
- 任务目标：评估 Qwen3.5 适配完成度与 xLLM 关键优化支持度，并核查 main 是否已补齐缺口。
- 主要触达模块：
  - xllm/models/llm
  - xllm/core/layers/npu_torch
  - xllm/core/runtime
  - xllm/core/distributed_runtime
  - xllm/core/scheduler
  - xllm/core/framework/block
  - docs/zh/features
  - docs/zh/supported_models.md
  - run_qwen.sh
  - log/node_0.log
- 验证状态：
  - 已完成代码与文档证据核对。
  - 已完成 ict_final 与 main 的同位实现/门禁项对照。
  - 未完成系统化压测（TPS/TTFT/P99 全量曲线）。

## 1. 审计范围与方法

- 审计对象：当前分支 ict_final，对照分支 main。
- 目标：评估 Qwen3.5 在 xLLM 中的适配完整度，以及 xLLM 关键性能优化在 Qwen3.5 上的支持状态。
- 约束：按赛题说明，不考虑量化模型，聚焦文本 token 推理。
- 证据来源：源码实现、启动脚本、运行日志、文档声明、main 分支同位检索。

## 2. Qwen3.5 基础适配完成度

### 2.1 模型注册与参数解析：已适配

- 证据：xllm/models/llm/qwen3_5.h:L29
  - Qwen3_5ModelImpl 继承 Qwen3NextModelImpl，并替换为 Qwen3_5DecoderLayerImpl。
- 证据：xllm/models/llm/qwen3_5.h:L156, L165, L174, L183
  - 已注册 qwen3_5、qwen3_5_text、qwen3_5_moe、qwen3_5_moe_text。
- 证据：xllm/models/llm/qwen3_5.h:L68, L149
  - 参数加载兼容 text_config/root/rope/layer_types 等，适配 Qwen3.5 配置差异。

结论：模型入口、配置解析、模型类型注册链路完整。

### 2.2 Qwen3.5 专用 NPU Torch 层：已适配

- 证据：xllm/core/layers/npu_torch/qwen3_5_decoder_layer_impl.cpp:L21
  - Qwen3_5DecoderLayerImpl 注入 Qwen3_5GatedDeltaNetImpl。
- 证据：xllm/core/layers/npu_torch/qwen3_5_gated_delta_net.cpp:L30, L38, L46, L54
  - Qwen3.5 独立投影 in_proj_qkv/z/b/a。
- 证据：xllm/core/layers/npu_torch/qwen3_5_gated_delta_net.cpp:L64, L100
  - merge_qkvz_from_split_activations 与 merge_ba_from_split_activations 实现了 Qwen3.5 特有拼接逻辑。

结论：Qwen3.5 的线性注意力子模块不是简单复用 Qwen3，而是做了专用实现。

### 2.3 Hybrid 注意力路由与 KV 结构：已适配

- 证据：xllm/core/framework/model/model_args.h:L429, L444
  - 通过 layer_types/full_attention_interval 判断 full/linear attention。
- 证据：xllm/core/layers/npu_torch/qwen3_next_hybrid_decoder_layer_base.cpp:L31, L40
  - 每层按判断结果路由到 attention_ 或 linear_attention_。
- 证据：xllm/core/distributed_runtime/llm_engine.cpp:L496, L522, L535, L632-L635
  - KV 容量估算引入 linear_slot_size；分配时额外初始化 conv_cache/ssm_cache。
- 证据：xllm/core/runtime/worker_impl.cpp:L177, L252-L285
  - Worker 对 linear layer 仅分配 conv/ssm cache，对 full attention layer 分配 k/v cache。

结论：Qwen3.5 Hybrid 架构在运行时内存与执行路径上已被识别并处理。

### 2.4 运行验证（当前环境）：可启动，但长 prompt 受限

- 证据：log/node_0.log:L9
  - 启动时后端解析为 TORCH（model_type=qwen3_5）。
- 证据：log/node_0.log:L83-L87
  - GND attention 的 conv/ssm cache 成功初始化。
- 证据：log/node_0.log:L88
  - 服务启动成功。
- 证据：log/node_0.log:L89
  - 出现 Prompt is too long: 65536。
- 证据：xllm/core/distributed_runtime/llm_master.cpp:L316-L323
  - 当 enable_chunked_prefill=false 时，最大上下文被截断为 min(max_position_embeddings, max_tokens_per_batch)。

结论：Qwen3.5 能稳定启动并完成 KV 初始化，但当前配置下长输入被硬限制。

## 3. xLLM 关键性能优化在 Qwen3.5 的支持度

### 3.1 Prefix Cache

状态：代码支持，Qwen3.5 未见模型级禁用；当前脚本默认关闭。

- 证据：xllm/core/framework/block/block_manager_impl.cpp:L28-L31
  - enable_prefix_cache 打开时创建 PrefixCache。
- 证据：xllm/core/framework/block/block_manager_impl.cpp:L128-L137, L154-L165
  - 已实现 match/insert 路径。
- 证据：run_qwen.sh:L119
  - 默认 --enable_prefix_cache=false。
- 证据：log/node_0.log:L8
  - 实际运行配置 enable_prefix_cache: 0。

收益评估：
- 高度依赖前缀复用率。高复用业务（固定 system prompt、多轮会话）可带来明显 TTFT/吞吐改善。

难度评估：
- 低到中。开关层面低成本；若出现内存/命中率异常，需要中等调参成本。

main 对照：
- 未发现 main 已新增 Qwen3.5 专属 Prefix Cache 适配；逻辑同类存在但不特化。

### 3.2 Chunked Prefill

状态：框架实现存在，但在 Qwen3.5 赛题分支中被明确标注“不支持”，且默认关闭。

- 证据：xllm/core/common/global_flags.cpp:L161
  - enable_chunked_prefill 全局开关存在。
- 证据：xllm/core/scheduler/scheduler_factory.cpp:L47-L53
  - 开启后走 ChunkedPrefillScheduler/PrefillOnlyScheduler。
- 证据：README_ict_final.md:L157
  - 明确写“当前xLLM不支持Qwen3.5的chunked prefill”。
- 证据：run_qwen.sh:L120
  - 默认 --enable_chunked_prefill=false。
- 证据：log/node_0.log:L8, xllm/core/distributed_runtime/llm_master.cpp:L316-L323
  - 当前运行关闭 chunked prefill，导致长 prompt 上限受 max_tokens_per_batch 限制。

收益评估：
- 很高。若实现稳定支持，可显著提升长上下文场景可用性，并提升 prefill 阶段资源利用率。

难度评估：
- 中到高。需要验证 Hybrid attention + 多节点 TP + 内存规划下的稳定性。

main 对照：
- 未发现 main 对 Qwen3.5 chunked prefill 的专门补齐证据；README 主线也无“已支持”声明。

### 3.3 Graph Mode（ACLGraph）

状态：代码对 Hybrid 模型做了兼容处理，但 Qwen3.5 未在文档支持表中明确列出；当前脚本默认关闭。

- 证据：xllm/core/runtime/acl_graph_executor_impl.cpp:L51-L60
  - find_attention_plan_kv_cache 会挑选首个有效 full-attention kv cache。
- 证据：xllm/core/runtime/acl_graph_executor_impl.cpp:L805-L814, L911-L918
  - capture/replay 均使用上述机制处理 hybrid 层场景。
- 证据：docs/zh/features/graph_mode.md:L58, L66
  - 文档给出 Qwen3/Qwen3-MoE 支持，不含 Qwen3.5 明确条目。
- 证据：run_qwen.sh:L123 与 log/node_0.log:L8
  - 当前默认 enable_graph=false，运行配置也为 0。

收益评估：
- 中到高。官方文档在 Qwen3 系列上给出 decode 吞吐约 8%-10% 提升，Qwen3.5 预期有迁移价值。

难度评估：
- 中。重点在图捕获桶化、动态形状与 hybrid kv/tiling 正确性验证。

main 对照：
- main 文档同样未显式列 Qwen3.5 Graph 支持；未见“主线已补文档/补认证”的信号。

### 3.4 MTP 投机推理

状态：推理侧模型与运行时映射已实现；但文档导出链路未覆盖 Qwen3.5，默认脚本未启用。

- 证据：xllm/models/llm/qwen3_5_mtp.h:L50-L67
  - load_qwen3_5_mtp_model_args 将 draft 层强制 full_attention。
- 证据：xllm/models/llm/qwen3_5_mtp.h:L258-L269
  - 注册 qwen3_5_mtp、qwen3_5_moe_mtp。
- 证据：xllm/core/runtime/worker_impl.cpp:L939-L951
  - 当满足条件时，将 qwen3_5/qwen3_5_moe draft 自动映射为 mtp 变体。
- 证据：docs/zh/features/mtp.md:L22-L26
  - 文档只列 DeepSeek/GLM4 MoE 导出，不含 Qwen3.5。
- 证据：log/node_0.log:L8
  - 当前 num_speculative_tokens: 0，未进入投机解码实跑。

收益评估：
- 高潜力。若 draft 质量与 acceptance rate 合理，decode 吞吐改善通常显著。

难度评估：
- 中到高。关键不在框架开关，而在可用 draft 权重、导出链路与 acceptance 参数调优。

main 对照：
- Qwen3.5 MTP 代码在 main 已存在（文件同名同路径）；文档侧仍未覆盖 Qwen3.5。

### 3.5 Prefill SP

状态：对 Qwen3.5 明确不支持（硬门禁）。

- 证据：xllm/xllm.cpp:L50-L67
  - prefill_sp_supported_model_set 仅 deepseek_v32、glm_moe_dsa；Qwen3.5 不在白名单，开启即 FATAL。
- 证据：xllm/core/scheduler/scheduler_factory.cpp:L47-L53
  - 框架有 PrefillOnlyScheduler 路径，但受前述门禁约束。

收益评估：
- 中到高（多卡/多机场景 prefill 阶段可获益明显）。

难度评估：
- 高。需要 Qwen3.5 attention/kernel 并行策略专项适配，不是简单开白名单。

main 对照：
- main 同位代码一致（main:xllm/xllm.cpp:49-66），未见已放开。

### 3.6 Schedule Overlap

状态：框架实现存在，但当前脚本默认关闭，Qwen3.5 未形成可复用实践结论。

- 证据：xllm/core/scheduler/continuous_scheduler.cpp:L1081-L1085
  - 有 step_with_schedule_overlap 专用路径。
- 证据：run_qwen.sh:L121 与 log/node_0.log:L8
  - 当前默认 enable_schedule_overlap=false，运行配置也为 0。

收益评估：
- 中。理论上可减少调度与执行空隙，提升有效吞吐。

难度评估：
- 中。需要关注与 chunked prefill/speculative/并发回调路径的交互稳定性。

main 对照：
- 未发现 main 对 Qwen3.5 的 schedule overlap 给出专项增强。

### 3.7 Multi-stream Parallel

状态：当前不可用（代码直接 FATAL）。

- 证据：xllm/core/distributed_runtime/master.cpp:L167-L171
  - enable_multi_stream_parallel 打开后直接报错“refactoring now”。

收益评估：
- 中到高（通信与计算重叠潜力）。

难度评估：
- 高（功能仍在重构阶段）。

main 对照：
- main 同位代码一致（main:xllm/core/distributed_runtime/master.cpp:167-171），未见已恢复可用。

### 3.8 NPU Kernel Backend（ATB/TORCH）

状态：Qwen3.5 被强制 TORCH backend，ATB 不可选。

- 证据：xllm/models/model_registry.cpp:L68-L75, L111-L122
  - qwen3_5* 在 torch-only 集合；AUTO 时解析到 TORCH，显式 ATB 会拒绝。
- 证据：log/node_0.log:L9
  - 实际运行解析为 TORCH。

收益评估：
- 中到高。若未来具备 ATB 路径，可能带来更高算子融合/图执行收益。

难度评估：
- 高。涉及算子适配、精度验证、性能回归和异常场景覆盖。

main 对照：
- main 同位代码一致（main:xllm/models/model_registry.cpp:68-75, 111-122），未见放开。

## 4. ict_final 与 main 分支结论对照

### 4.1 核心 Qwen3.5 实现是否 main 已有

- 对比文件：
  - xllm/models/llm/qwen3_5.h
  - xllm/models/llm/qwen3_5_mtp.h
  - xllm/core/layers/npu_torch/qwen3_5_decoder_layer_impl.cpp
  - xllm/core/layers/npu_torch/qwen3_5_gated_delta_net.cpp
- 结论：git diff main..ict_final 在上述文件无差异，说明核心适配并非 ict_final 独有新增。

### 4.2 未实现/效果欠佳项是否在 main 已补

- Prefill SP：未补（main:xllm/xllm.cpp:49-66 仍是同一白名单门禁）。
- Multi-stream parallel：未补（main:xllm/core/distributed_runtime/master.cpp:167-171 仍 FATAL）。
- Qwen3.5 ATB backend：未补（main:xllm/models/model_registry.cpp:68-75, 111-122 仍 torch-only）。
- Graph/MTP 文档覆盖：未补（main 文档同样未明确 Qwen3.5 条目）。
- Chunked prefill 的 Qwen3.5 专项支持：未发现 main 明确补齐证据。

## 5. 优先级建议（按预期收益/落地难度）

### P0（高收益，且直接影响可用性）

1) Qwen3.5 Chunked Prefill 稳定化
- 预期收益：高（长上下文可用性 + prefill 吞吐）。
- 难度：中到高。
- 直接依据：README_ict_final 明确不支持，且当前长 prompt 已触发限制。

2) Qwen3.5 MTP 实跑闭环（导出 + 启动 + acceptance 调优）
- 预期收益：高（decode 吞吐潜力大）。
- 难度：中到高。
- 直接依据：代码已有 MTP 路径，但文档与默认脚本未形成可复现实践。

### P1（中高收益，工程复杂度较高）

3) Qwen3.5 Graph Mode 认证与参数化
- 预期收益：中到高（参考 Qwen3 系列 8%-10% decode 提升）。
- 难度：中。

4) Qwen3.5 Prefill SP 适配
- 预期收益：中到高（多机 prefill）。
- 难度：高（当前硬门禁，不是开关问题）。

### P2（长期潜力项）

5) Qwen3.5 ATB backend 适配
- 预期收益：中到高。
- 难度：高。

6) Multi-stream parallel 复用到 Qwen3.5
- 预期收益：中到高。
- 难度：高（当前功能整体重构中）。

## 6. 最终结论

- Qwen3.5 的“模型可跑”适配在 ict_final 已基本完成：模型注册、参数解析、Hybrid 层、KV 结构、服务启动链路均已打通。
- 但“性能优化可用度”与“默认可用配置”之间存在明显差距：
  - 多项优化（prefix/chunked/schedule overlap/graph/mtp）在框架上有路径，但当前默认配置与文档覆盖不足，导致实际未发挥。
  - 关键优化（prefill_sp、multi-stream、ATB backend）对 Qwen3.5 仍处于未开放或不可用状态。
- 对照 main 后，未发现 main 已系统性补齐上述短板。短期最有价值的路线是：先把 Qwen3.5 的 chunked prefill 与 MTP 做成稳定可复现，再推进 graph 与并行增强。

## 7. 验证边界说明

- 已验证：代码路径、配置门禁、启动日志与文档一致性、main 分支同位检索。
- 未验证：各优化项在 Qwen3.5 上的完整 benchmark 曲线（TPS/TTFT/P99）与长期稳定性（长压、故障恢复、多并发场景）。