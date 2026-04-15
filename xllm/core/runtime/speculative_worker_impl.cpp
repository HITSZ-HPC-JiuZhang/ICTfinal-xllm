/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "speculative_worker_impl.h"

#include <algorithm>

#include "common/global_flags.h"
#include "common/metrics.h"
#include "spec_input_builder.h"
#include "util/env_var.h"
#include "util/slice.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {

namespace {
#define TENSOR_REPEAT(tensor_, repeats)                                       \
  do {                                                                        \
    tensor_ = tensor_.defined()                                               \
                  ? tensor_.repeat_interleave(/*repeats=*/repeats, /*dim=*/0) \
                  : tensor_;                                                  \
  } while (0)

void repeat_sampling_tensor_by_counts(torch::Tensor& tensor,
                                      const torch::Tensor& repeat_counts) {
  tensor = tensor.defined()
               ? tensor.repeat_interleave(/*repeats=*/repeat_counts, /*dim=*/0)
               : tensor;
}

}  // namespace

SpeculativeWorkerImpl::SpeculativeWorkerImpl(
    const ParallelArgs& parallel_args,
    const torch::Device& device,
    const runtime::Options& options,
    const runtime::Options& target_options)
    : WorkerImpl(parallel_args, device, options) {
  impl_ =
      std::make_unique<LLMWorkerImpl>(parallel_args, device, target_options);

  std::optional<double> fixed_acceptance_rate =
      util::get_fix_speculative_acceptance_rate();
  if (fixed_acceptance_rate.has_value()) {
    rate_controller_ = std::make_shared<RejectionSamplerRateController>(
        *fixed_acceptance_rate);
  }
}

bool SpeculativeWorkerImpl::init_model(const std::string& model_weights_path,
                                       int32_t random_seed,
                                       MasterStatus master_status) {
  // Base class only loads the target model.
  bool result = true;
  if (impl_->get_status() == WorkerImpl::Status::UNINITIALIZED) {
    result = impl_->WorkerImpl::init_model(
        model_weights_path, random_seed, master_status);
    if (result) {
      dtype_ = impl_->dtype();
      embedding_size_ = impl_->hidden_size();
    }
  }
  enable_fused_kernel_ =
      impl_->get_optimization_config().enable_fused_spec_kernel;
  return result;
}

bool SpeculativeWorkerImpl::allocate_kv_cache(
    const std::vector<std::vector<int64_t>>& kv_cache_shape) {
  return impl_->allocate_kv_cache(kv_cache_shape);
}

#if defined(USE_NPU)
bool SpeculativeWorkerImpl::allocate_kv_cache_with_transfer(
    const std::vector<std::vector<int64_t>>& kv_cache_shape) {
  return impl_->allocate_kv_cache_with_transfer(kv_cache_shape);
}
#endif

std::optional<ForwardOutput> SpeculativeWorkerImpl::step(
    const ForwardInput& input) {
  if (input.token_ids.numel() == 0) {
    return step_empty(input);
  }

  if (!input.input_params.batch_forward_type.is_decode()) {
    return step_prefill(input);
  } else {
    return step_decode(input);
  }
}

ForwardInput SpeculativeWorkerImpl::update_input_by_last_step_output(
    ForwardInput& inputs) {
  // only process decode batch, so prepare draft input here.
  ForwardInput& new_inputs = inputs;

  auto& input_params = new_inputs.input_params;
  const int32_t num_sequences = input_params.num_sequences;
  const int32_t block_size = options_.block_size();

  torch::Tensor token_ids = safe_to(inputs.token_ids, torch::kCPU);
  torch::Tensor positions = safe_to(inputs.positions, torch::kCPU);
  torch::Tensor block_tables = safe_to(input_params.block_tables, torch::kCPU);
  auto view = specBuilder::make_decode_cpu_view(
      token_ids, positions, block_tables, input_params.kv_seq_lens_vec);
  torch::Tensor last_token_ids = safe_to(
      last_step_output_.sample_output.next_tokens.flatten(), torch::kCPU);
  Slice<int64_t> last_tokens_ids_slice = {
      last_token_ids.data_ptr<int64_t>(),
      static_cast<size_t>(last_token_ids.numel())};

  // Determine how many tokens were decoded in the last step
  // If the output is 2D, it means multiple tokens were generated per sequence
  int32_t last_step_decode_num = 1;
  if (last_step_output_.sample_output.next_tokens.dim() == 2) {
    last_step_decode_num = last_step_output_.sample_output.next_tokens.size(1);
  }

  specBuilder::DecodeBuildBuffers buf;
  buf.out_token_ids.reserve(num_sequences);
  buf.out_positions.reserve(num_sequences);
  buf.out_kv_seq_lens.reserve(num_sequences);
  buf.out_new_cache_slots.reserve(num_sequences);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    specBuilder::append_decode_row_from_last_step(input_params,
                                                  view,
                                                  seq_id,
                                                  view.token_ids[seq_id],
                                                  last_tokens_ids_slice,
                                                  last_step_decode_num,
                                                  block_size,
                                                  buf);
  }

  CHECK_EQ(buf.out_new_cache_slots.size(), buf.out_token_ids.size())
      << "step-update kv slots/tokens mismatch";
  CHECK_EQ(buf.out_positions.size(), buf.out_token_ids.size())
      << "step-update positions/tokens mismatch";

  // Create new tensors with updated values
  torch::TensorOptions int_options = inputs.token_ids.options();
  new_inputs.token_ids = torch::tensor(buf.out_token_ids, int_options);
  new_inputs.positions = torch::tensor(buf.out_positions, int_options);
  // update the input_params
  input_params.kv_max_seq_len = buf.kv_max_seq_len;
  input_params.kv_seq_lens_vec = std::move(buf.out_kv_seq_lens);
  input_params.kv_seq_lens =
      torch::tensor(input_params.kv_seq_lens_vec, int_options);
  input_params.q_cu_seq_lens =
      specBuilder::build_q_cu_seq_lens_tensor(input_params);
  input_params.new_cache_slots =
      torch::tensor(buf.out_new_cache_slots, int_options);

  return new_inputs.to(device_, dtype_);
}

void SpeculativeWorkerImpl::update_sampling_params(
    SamplingParameters& sampling_params,
    std::span<const int32_t> validate_token_counts,
    const int32_t total_num_val_tokens) {
  std::vector<int32_t> selected_token_idxes_vec;
  selected_token_idxes_vec.reserve(total_num_val_tokens);
  for (int32_t i = 0; i < total_num_val_tokens; ++i) {
    selected_token_idxes_vec.emplace_back(i);
  }
  torch::Tensor selected_token_idxes = torch::tensor(selected_token_idxes_vec);

  // sample_idxes equals to selected_token_idxes since only process decode batch
  sampling_params.selected_token_idxes = selected_token_idxes.to(device_);
  sampling_params.sample_idxes = selected_token_idxes.to(device_);

  const int32_t repeated_num_val_tokens =
      validate_token_counts.empty() ? 1 : validate_token_counts.front();
  const bool all_equal_validate_token_counts =
      std::all_of(validate_token_counts.begin(),
                  validate_token_counts.end(),
                  [repeated_num_val_tokens](int32_t validate_token_count) {
                    return validate_token_count == repeated_num_val_tokens;
                  });
  if (all_equal_validate_token_counts) {
    TENSOR_REPEAT(sampling_params.frequency_penalties, repeated_num_val_tokens);
    TENSOR_REPEAT(sampling_params.presence_penalties, repeated_num_val_tokens);
    TENSOR_REPEAT(sampling_params.repetition_penalties,
                  repeated_num_val_tokens);
    TENSOR_REPEAT(sampling_params.temperatures, repeated_num_val_tokens);
    TENSOR_REPEAT(sampling_params.top_p, repeated_num_val_tokens);
    TENSOR_REPEAT(sampling_params.top_k, repeated_num_val_tokens);
    TENSOR_REPEAT(sampling_params.unique_token_ids, repeated_num_val_tokens);
    TENSOR_REPEAT(sampling_params.unique_token_counts, repeated_num_val_tokens);
    TENSOR_REPEAT(sampling_params.unique_token_ids_lens,
                  repeated_num_val_tokens);
    TENSOR_REPEAT(sampling_params.do_sample, repeated_num_val_tokens);
    return;
  }

  torch::Tensor repeat_counts =
      torch::tensor(std::vector<int32_t>(validate_token_counts.begin(),
                                         validate_token_counts.end()),
                    torch::TensorOptions().dtype(torch::kLong));
  repeat_sampling_tensor_by_counts(sampling_params.frequency_penalties,
                                   repeat_counts);
  repeat_sampling_tensor_by_counts(sampling_params.presence_penalties,
                                   repeat_counts);
  repeat_sampling_tensor_by_counts(sampling_params.repetition_penalties,
                                   repeat_counts);
  repeat_sampling_tensor_by_counts(sampling_params.temperatures, repeat_counts);
  repeat_sampling_tensor_by_counts(sampling_params.top_p, repeat_counts);
  repeat_sampling_tensor_by_counts(sampling_params.top_k, repeat_counts);
  repeat_sampling_tensor_by_counts(sampling_params.unique_token_ids,
                                   repeat_counts);
  repeat_sampling_tensor_by_counts(sampling_params.unique_token_counts,
                                   repeat_counts);
  repeat_sampling_tensor_by_counts(sampling_params.unique_token_ids_lens,
                                   repeat_counts);
  repeat_sampling_tensor_by_counts(sampling_params.do_sample, repeat_counts);
}

void SpeculativeWorkerImpl::prepare_validate_inputs(
    const ForwardInput& input,
    ForwardInput& validate_input,
    std::span<const int32_t> draft_token_counts) {
  validate_input = input.to(device_, dtype_);
  auto& input_params = validate_input.input_params;
  torch::TensorOptions int_options = validate_input.token_ids.options();

  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  const int32_t num_sequences = input_params.num_sequences;
  const int32_t block_size = options_.block_size();
  std::vector<int32_t> validate_token_counts_vec;
  validate_token_counts_vec.reserve(num_sequences);
  int32_t total_num_val_tokens = 0;
  int32_t max_num_val_tokens = 0;
  if (!draft_token_counts.empty()) {
    CHECK_EQ(static_cast<int32_t>(draft_token_counts.size()), num_sequences)
        << "draft_token_counts size mismatch, expected=" << num_sequences
        << ", actual=" << draft_token_counts.size();
  }
  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    const int32_t draft_count = draft_token_counts.empty()
                                    ? num_speculative_tokens
                                    : draft_token_counts[seq_id];
    CHECK_GE(draft_count, 0)
        << "invalid draft_count=" << draft_count << ", seq_id=" << seq_id;
    CHECK_LE(draft_count, num_speculative_tokens)
        << "draft_count exceeds num_speculative_tokens, draft_count="
        << draft_count << ", num_speculative_tokens=" << num_speculative_tokens
        << ", seq_id=" << seq_id;
    const int32_t validate_token_count = draft_count + 1;
    validate_token_counts_vec.emplace_back(validate_token_count);
    total_num_val_tokens += validate_token_count;
    if (validate_token_count > max_num_val_tokens) {
      max_num_val_tokens = validate_token_count;
    }
  }

  torch::Tensor token_ids = safe_to(input.token_ids, torch::kCPU);
  torch::Tensor positions = safe_to(input.positions, torch::kCPU);
  torch::Tensor block_tables = safe_to(input_params.block_tables, torch::kCPU);
  auto view = specBuilder::make_decode_cpu_view(
      token_ids, positions, block_tables, input_params.kv_seq_lens_vec);
  specBuilder::DecodeBuildBuffers buf;
  buf.out_token_ids.reserve(total_num_val_tokens);
  buf.out_positions.reserve(total_num_val_tokens);
  buf.out_new_cache_slots.reserve(total_num_val_tokens);
  if (!FLAGS_enable_atb_spec_kernel) {
    buf.out_kv_seq_lens.reserve(total_num_val_tokens);
    buf.out_q_seq_lens.reserve(total_num_val_tokens);
    buf.out_block_tables.reserve(total_num_val_tokens);
  }

  std::vector<int32_t> atb_kv_seq_lens_vec = {};
  std::vector<int32_t> atb_q_seq_lens_vec = {};
  int32_t atb_kv_max_seq_len = 0;
  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    const int32_t seq_num_val_tokens = validate_token_counts_vec[seq_id];
    int32_t start_position = view.positions[seq_id];
    int32_t kv_len =
        specBuilder::calc_kv_len(view.kv_seq_lens, seq_id, /*offset=*/0);
    CHECK_EQ(start_position + 1, kv_len)
        << "validate position/kv_len mismatch, seq_id=" << seq_id
        << ", start_position=" << start_position << ", kv_len=" << kv_len;

    for (int32_t val_idx = 0; val_idx < seq_num_val_tokens; ++val_idx) {
      specBuilder::RowSpec row;
      row.seq_id = seq_id;
      if (val_idx == 0) {
        row.use_input_token = true;
      } else {
        row.token_id = -val_idx;
      }
      row.position_offset = val_idx;
      row.append_kv_len = !FLAGS_enable_atb_spec_kernel;
      row.append_q_len_one = !FLAGS_enable_atb_spec_kernel;
      row.append_block_table = !FLAGS_enable_atb_spec_kernel;
      specBuilder::append_decode_row(input_params, view, row, block_size, buf);
    }

    if (FLAGS_enable_atb_spec_kernel) {
      const int32_t kv_len_after_validation = kv_len + seq_num_val_tokens - 1;
      specBuilder::update_kv_seq_lens_and_max(
          atb_kv_seq_lens_vec, kv_len_after_validation, atb_kv_max_seq_len);
      specBuilder::append_seq_len_by_layout(atb_q_seq_lens_vec,
                                            seq_num_val_tokens);
    }
  }

  CHECK_EQ(buf.out_new_cache_slots.size(), buf.out_token_ids.size())
      << "validate kv slots/tokens mismatch";
  CHECK_EQ(buf.out_positions.size(), buf.out_token_ids.size())
      << "validate positions/tokens mismatch";

  validate_input.token_ids = torch::tensor(buf.out_token_ids, int_options);
  validate_input.positions = torch::tensor(buf.out_positions, int_options);
  // update the input_params
  if (!FLAGS_enable_atb_spec_kernel) {
    input_params.num_sequences = total_num_val_tokens;
    input_params.q_max_seq_len = 1;
    input_params.batch_forward_type = BatchForwardType::DECODE;
  } else {
    input_params.q_max_seq_len = max_num_val_tokens;
    input_params.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  }
  if (FLAGS_enable_atb_spec_kernel) {
    input_params.q_seq_lens_vec = std::move(atb_q_seq_lens_vec);
  } else {
    input_params.q_seq_lens_vec = std::move(buf.out_q_seq_lens);
  }
  input_params.q_seq_lens =
      torch::tensor(input_params.q_seq_lens_vec, int_options);
  input_params.q_cu_seq_lens =
      specBuilder::build_q_cu_seq_lens_tensor(input_params);
  if (FLAGS_enable_atb_spec_kernel) {
    input_params.kv_max_seq_len = atb_kv_max_seq_len;
    input_params.kv_seq_lens_vec = std::move(atb_kv_seq_lens_vec);
  } else {
    input_params.kv_max_seq_len = buf.kv_max_seq_len;
    input_params.kv_seq_lens_vec = std::move(buf.out_kv_seq_lens);
  }
  input_params.kv_seq_lens =
      torch::tensor(input_params.kv_seq_lens_vec, int_options);
  input_params.new_cache_slots =
      torch::tensor(buf.out_new_cache_slots, int_options);
  if (!FLAGS_enable_atb_spec_kernel) {
    util::pad_2d_vector(buf.out_block_tables, /*pad_value=*/0);
    input_params.block_tables =
        create_2d_tensor(buf.out_block_tables, torch::kInt).to(device_);
  }

  // update the sampling_params
  update_sampling_params(
      validate_input.sampling_params,
      std::span<const int32_t>(validate_token_counts_vec.data(),
                               validate_token_counts_vec.size()),
      total_num_val_tokens);

  // update dp_global_token_nums for dp/ep parallel
  if (num_sequences > 0) {
    for (int32_t& it : input_params.dp_global_token_nums) {
      it = static_cast<int32_t>(static_cast<int64_t>(it) *
                                static_cast<int64_t>(total_num_val_tokens) /
                                static_cast<int64_t>(num_sequences));
    }
  }
}

void SpeculativeWorkerImpl::prepare_work_before_execute(
    const ForwardInput& input,
    ForwardInput& processed_input) {
  WorkerImpl::prepare_work_before_execute(input, processed_input);
  if (input.input_params.batch_forward_type.is_decode() &&
      enable_schedule_overlap()) {
    processed_input.token_ids = safe_to(processed_input.token_ids, torch::kCPU);
    processed_input.positions = safe_to(processed_input.positions, torch::kCPU);
  }
}
}  // namespace xllm
