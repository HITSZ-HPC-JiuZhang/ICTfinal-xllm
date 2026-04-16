# Qwen3.5 ACLGraph TP temporary gate

- Time: 2026-04-12 20:22:03 CST
- Branch: feat/ictfinal-qwen3.5
- Task goal: Add a temporary rollback gate for Qwen3.5 ACLGraph on TORCH backend with TP>1 to verify whether decode degeneration is caused by TP graph communication.
- Touched modules/files:
  - xllm/core/runtime/acl_graph_executor_impl.cpp
  - work_docs/worklogs/2026-04-12_qwen35_graph_tp_gate.md
- Key findings / suspected root cause:
  - Qwen3.5 is TORCH-only on NPU.
  - In TP mode, decode path uses HCCL collectives through ProcessGroupHCCL wait-based synchronization, which is not graph-safe for ACLGraph capture/replay semantics.
  - Single-card graph works while TP graph degenerates, which matches a communication-path correctness issue rather than a Qwen3.5-only projection bug.
- Change made:
  - Added a temporary eager fallback in AclGraphExecutorImpl::run() when:
    - model_type is qwen3_5/qwen3_5_* or qwen3_next
    - npu_kernel_backend == TORCH
    - tp_world_size > 1
- Commands run:
  - rg / sed code inspection for graph design, ACLGraph executor, Qwen3.5 hybrid layers, GDN kernels, and HCCL process group implementation
  - git branch --show-current
  - date
- Validation status:
  - Static review completed.
  - Runtime validation pending user-side verification.
- Next steps / risks:
  - User verifies that TP degeneration disappears after eager fallback.
  - If confirmed, next step is to make TP collectives graph-safe and then remove this temporary gate.

## Follow-up update

- Time: 2026-04-12 20:55:00 CST
- User validation result:
  - With the temporary eager fallback enabled, TP decode degeneration disappeared.
  - This confirms the issue is in TP graph communication semantics rather than Qwen3.5 projection packing.
- Follow-up change:
  - Removed the temporary gate from `xllm/core/runtime/acl_graph_executor_impl.cpp` after confirmation, to avoid carrying workaround logic into the real fix.
- Next task:
  - Produce a complete repair plan for TP-safe ACLGraph execution on TORCH backend.

## Native HCCL rework update

- Time: 2026-04-12 23:20:00 CST
- Task goal:
  - Replace the failed `HcomGetCommHandleByGroup` attempt with xLLM-owned native HCCL communicator initialization for TP graph capture, while keeping eager path behavior unchanged.
- Touched modules/files:
  - xllm/core/framework/parallel_state/npu_process_group.h
  - xllm/core/framework/parallel_state/npu_process_group.cpp
  - xllm/core/framework/parallel_state/graph_execution_context.h
  - xllm/core/layers/npu_torch/qwen3_next_hybrid_decoder_layer_base.cpp
  - xllm/core/layers/npu_torch/qwen3_next_hybrid_decoder_layer_base.h
  - xllm/core/layers/common/word_embedding_impl.cpp
  - xllm/core/common/global_flags.h
  - xllm/core/common/global_flags.cpp
- Key findings / root cause refinement:
  - Runtime logs confirmed that `HcomGetCommHandleByGroup("tp_group")` cannot recover a usable native `HcclComm` from the current `ProcessGroupHCCL` initialization path.
  - During ACL graph capture, TP collectives still fall back to `ProcessGroupHCCL Work` semantics (`path=work`), including the initial word embedding gather and every hybrid layer allreduce.
  - This means the correct long-term fix must create and own a native `HcclComm` per xLLM process group instead of trying to introspect one out of torch_npu.
- Change made:
  - Replaced the Hcom-based communicator acquisition with native `HcclGetRootInfo` + `TCPStore` distribution + `HcclCommInitRootInfo`.
  - Kept graph capture routing logic so graph-time TP collectives prefer native HCCL when capture is active.
  - Cleaned the temporary collective scope tracing so layer-level scope covers the whole hybrid layer and branch scopes only annotate attention/GDN subpaths.
- Commands run:
  - `rg` / `sed` inspection on `log/node_0.log`, `collective_communicator.cpp`, `collective_service.*`, `worker_server.cpp`, and Ascend HCCL headers
  - sub-agent reviews with `gpt-5.3-codex` and path comparison research with `gpt-5.4`
- Validation status:
  - Static implementation complete.
  - User-side compile/run validation still pending after this native HCCL rework.
- Next steps / risks:
  - Verify that native TP graph collectives now log `path=native_hccl` instead of `path=work`.
  - Re-run the previously degenerating Qwen3.5 requests under TP graph mode.
  - If capture still falls back or degrades, inspect whether `ProcessGroupHCCL` and xLLM-owned native HCCL create mismatched communicator topology for the same group.
