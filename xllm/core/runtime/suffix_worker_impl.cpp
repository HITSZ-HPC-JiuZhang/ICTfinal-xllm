/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "suffix_worker_impl.h"

#include <algorithm>

#include "common/metrics.h"
#include "framework/sampling/rejection_sampler.h"
#include "util/slice.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {

namespace {

void append_tokens_with_limit(std::vector<int32_t>& history,
                              std::span<const int32_t> tokens,
                              size_t max_size) {
  history.insert(history.end(), tokens.begin(), tokens.end());
  if (history.size() > max_size) {
    history.erase(history.begin(), history.end() - max_size);
  }
}

std::string summarize_int32_span(std::span<const int32_t> values,
                                 size_t limit = 8) {
  std::string out = "[";
  const size_t n = std::min(values.size(), limit);
  for (size_t i = 0; i < n; ++i) {
    if (i > 0) {
      out += ",";
    }
    out += std::to_string(values[i]);
  }
  if (values.size() > n) {
    out += ",...";
  }
  out += "]";
  return out;
}

int32_t get_max_validate_token_count(
    const std::vector<int32_t>& validate_token_counts) {
  if (validate_token_counts.empty()) {
    return 0;
  }
  return *std::max_element(validate_token_counts.begin(),
                           validate_token_counts.end());
}

}  // namespace

namespace {
runtime::Options SuffixTargetOptions(const runtime::Options& options) {
  auto opts = options;
  opts.enable_schedule_overlap(false);
  return opts;
}

bool is_prompt_lookup_algorithm(const std::string& algo) {
  return algo == "PLD" || algo == "PromptLookup";
}

void log_pld_decode_summary(uint64_t decode_batches,
                            uint64_t no_draft_batches,
                            uint64_t requested_draft_tokens,
                            uint64_t drafted_tokens,
                            uint64_t accepted_draft_tokens,
                            int32_t log_interval) {
  if (log_interval <= 0 || decode_batches == 0 ||
      decode_batches % static_cast<uint64_t>(log_interval) != 0) {
    return;
  }

  const double draft_hit_rate =
      requested_draft_tokens == 0
          ? 0.0
          : static_cast<double>(drafted_tokens) /
                static_cast<double>(requested_draft_tokens);
  const double accept_rate = drafted_tokens == 0
                                 ? 0.0
                                 : static_cast<double>(accepted_draft_tokens) /
                                       static_cast<double>(drafted_tokens);
  LOG(INFO) << "PLD decode summary: batches=" << decode_batches
            << ", no_draft_batches=" << no_draft_batches
            << ", requested_draft_tokens=" << requested_draft_tokens
            << ", drafted_tokens=" << drafted_tokens
            << ", accepted_draft_tokens=" << accepted_draft_tokens
            << ", draft_hit_rate=" << draft_hit_rate
            << ", accept_rate=" << accept_rate;
}
}  // namespace

SuffixWorkerImpl::SuffixWorkerImpl(const ParallelArgs& parallel_args,
                                   const torch::Device& device,
                                   const runtime::Options& options)
    : SpeculativeWorkerImpl(parallel_args,
                            device,
                            options,
                            SuffixTargetOptions(options)) {
  use_prompt_lookup_cache_ =
      is_prompt_lookup_algorithm(options_.speculative_algorithm());
  recent_tokens_max_size_ = options_.speculative_suffix_cache_max_depth();
  if (use_prompt_lookup_cache_) {
    int32_t prompt_lookup_min_ngram_size =
        std::max(1, options_.pld_min_ngram_size());
    int32_t prompt_lookup_max_ngram_size =
        std::max(prompt_lookup_min_ngram_size, options_.pld_max_ngram_size());
    if (prompt_lookup_min_ngram_size != options_.pld_min_ngram_size()) {
      LOG(WARNING) << "PLD min ngram size " << options_.pld_min_ngram_size()
                   << " is invalid; clamping to "
                   << prompt_lookup_min_ngram_size;
    }
    if (prompt_lookup_max_ngram_size != options_.pld_max_ngram_size()) {
      LOG(WARNING) << "PLD max ngram size " << options_.pld_max_ngram_size()
                   << " is smaller than min ngram size "
                   << prompt_lookup_min_ngram_size << "; raising max to "
                   << prompt_lookup_max_ngram_size;
    }
    if (recent_tokens_max_size_ < prompt_lookup_min_ngram_size) {
      LOG(WARNING) << "PLD recent token history depth "
                   << recent_tokens_max_size_
                   << " is smaller than min ngram size "
                   << prompt_lookup_min_ngram_size
                   << "; clamping history depth to keep PLD effective.";
      recent_tokens_max_size_ = prompt_lookup_min_ngram_size;
    }
    prompt_lookup_max_ngram_size = std::max(
        prompt_lookup_min_ngram_size,
        std::min(prompt_lookup_max_ngram_size, recent_tokens_max_size_));
    prompt_lookup_cache_ = std::make_unique<PromptLookupCache>(
        prompt_lookup_max_ngram_size, prompt_lookup_min_ngram_size);
    LOG(INFO) << "Prompt lookup decoding enabled: rank=" << parallel_args.rank()
              << ", device=" << device << ", num_speculative_tokens="
              << options_.num_speculative_tokens() << ", ngram_size=["
              << prompt_lookup_min_ngram_size << ", "
              << prompt_lookup_max_ngram_size
              << "], recent_tokens_max_size=" << recent_tokens_max_size_;
  } else {
    suffix_cache_ = std::make_unique<SuffixDecodingCache>(
        options_.speculative_suffix_cache_max_depth(),
        options_.speculative_suffix_max_cached_requests());
    LOG(INFO) << "Suffix speculative decoding enabled: rank="
              << parallel_args.rank() << ", device=" << device
              << ", num_speculative_tokens="
              << options_.num_speculative_tokens() << ", cache_max_depth="
              << options_.speculative_suffix_cache_max_depth();
  }
}

SuffixWorkerImpl::~SuffixWorkerImpl() { flush_all_pld_requests(); }

void SuffixWorkerImpl::ensure_pld_request_stats(const std::string& req_id) {
  if (use_prompt_lookup_cache_ && options_.pld_enable_request_stats()) {
    pld_request_stats_.try_emplace(req_id);
  }
}

void SuffixWorkerImpl::log_pld_request_stats(const std::string& req_id) const {
  if (!use_prompt_lookup_cache_ || !options_.pld_enable_request_stats()) {
    return;
  }

  auto it = pld_request_stats_.find(req_id);
  if (it == pld_request_stats_.end()) {
    return;
  }

  const PldRequestStats& stats = it->second;
  const double draft_hit_rate =
      stats.requested_draft_tokens == 0
          ? 0.0
          : static_cast<double>(stats.drafted_tokens) /
                static_cast<double>(stats.requested_draft_tokens);
  const double accept_rate =
      stats.drafted_tokens == 0
          ? 0.0
          : static_cast<double>(stats.accepted_draft_tokens) /
                static_cast<double>(stats.drafted_tokens);
  const double avg_draft_len =
      stats.decode_sequence_steps == 0
          ? 0.0
          : static_cast<double>(stats.drafted_tokens) /
                static_cast<double>(stats.decode_sequence_steps);
  const double avg_accepted_len =
      stats.decode_sequence_steps == 0
          ? 0.0
          : static_cast<double>(stats.accepted_draft_tokens) /
                static_cast<double>(stats.decode_sequence_steps);

  LOG(INFO) << "PLD request summary: req_id=" << req_id
            << ", decode_sequence_steps=" << stats.decode_sequence_steps
            << ", no_draft_sequence_steps=" << stats.no_draft_sequence_steps
            << ", requested_draft_tokens=" << stats.requested_draft_tokens
            << ", drafted_tokens=" << stats.drafted_tokens
            << ", accepted_draft_tokens=" << stats.accepted_draft_tokens
            << ", draft_hit_rate=" << draft_hit_rate
            << ", accept_rate=" << accept_rate
            << ", avg_draft_len_per_sequence_step=" << avg_draft_len
            << ", avg_accepted_len_per_sequence_step=" << avg_accepted_len
            << ", validate_waste_tokens=" << stats.validate_waste_tokens;
}

void SuffixWorkerImpl::cleanup_inactive_requests(
    const std::unordered_set<std::string>& current_req_ids) {
  for (const std::string& req_id : suffix_active_decode_req_ids_) {
    if (current_req_ids.find(req_id) != current_req_ids.end()) {
      continue;
    }

    if (use_prompt_lookup_cache_) {
      log_pld_request_stats(req_id);
      prompt_lookup_cache_->stop_request(req_id);
      pld_request_stats_.erase(req_id);
    } else if (suffix_cache_->has_active_request(req_id)) {
      suffix_cache_->stop_request(req_id);
    }
    suffix_recent_tokens_.erase(req_id);
  }
}

void SuffixWorkerImpl::flush_all_pld_requests() {
  if (use_prompt_lookup_cache_) {
    for (const auto& entry : suffix_recent_tokens_) {
      const std::string& req_id = entry.first;
      log_pld_request_stats(req_id);
      if (prompt_lookup_cache_->has_request(req_id)) {
        prompt_lookup_cache_->stop_request(req_id);
      }
    }
    pld_request_stats_.clear();
  } else if (suffix_cache_ != nullptr) {
    for (const std::string& req_id : suffix_active_decode_req_ids_) {
      if (suffix_cache_->has_active_request(req_id)) {
        suffix_cache_->stop_request(req_id);
      }
    }
  }
  suffix_recent_tokens_.clear();
  suffix_active_decode_req_ids_.clear();
}

std::optional<ForwardOutput> SuffixWorkerImpl::step_empty(
    const ForwardInput& input) {
  if (!input.input_params.batch_forward_type.is_decode()) {
    auto output = impl_->step(input);
    output->sample_output.embeddings = torch::Tensor();
    return output;
  } else {
    ForwardInput new_input = input;
    for (auto& it : new_input.input_params.dp_global_token_nums) {
      it *= options_.num_speculative_tokens() + 1;
    }

    auto future = impl_->step_async(new_input);
    ForwardOutput output = std::move(future).get().value();
    output.sample_output.embeddings = torch::Tensor();
    return output;
  }
}

std::optional<ForwardOutput> SuffixWorkerImpl::step_prefill(
    const ForwardInput& input) {
  Timer timer;
  // run the target model to get first token and hidden states
  auto future = impl_->step_async(input);
  ForwardOutput output = std::move(future).get().value();
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              timer.elapsed_seconds());

  const auto& input_params = input.input_params;
  const int32_t num_sequences = input_params.num_sequences;
  const auto& request_ids = input_params.request_ids;

  if ((suffix_cache_ != nullptr || prompt_lookup_cache_ != nullptr) &&
      request_ids.size() == static_cast<size_t>(num_sequences)) {
    torch::Tensor token_ids = safe_to(input.token_ids, torch::kCPU);
    Slice<int32_t> tokens_ids_slice = {
        token_ids.data_ptr<int32_t>(),
        static_cast<size_t>(input.token_ids.numel())};

    int32_t start_idx = 0;
    for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
      int32_t q_len = input_params.get_q_seq_len(seq_id);
      Slice<int32_t> seq_tokens =
          tokens_ids_slice.slice(start_idx, start_idx + q_len);
      start_idx += q_len;

      const std::string req_id = request_ids[seq_id];
      if (req_id.empty()) {
        continue;
      }

      if (use_prompt_lookup_cache_) {
        if (!prompt_lookup_cache_->has_request(req_id)) {
          prompt_lookup_cache_->start_request(req_id, seq_tokens);
          suffix_recent_tokens_[req_id].clear();
          ensure_pld_request_stats(req_id);
        } else {
          prompt_lookup_cache_->add_prompt(req_id, seq_tokens);
        }
      } else if (!suffix_cache_->has_active_request(req_id)) {
        suffix_cache_->start_request(req_id, seq_tokens);
        suffix_recent_tokens_[req_id].clear();
      } else {
        suffix_cache_->add_active_prompt(req_id, seq_tokens);
      }
      append_tokens_with_limit(suffix_recent_tokens_[req_id],
                               seq_tokens,
                               static_cast<size_t>(recent_tokens_max_size_));
    }

    torch::Tensor next_tokens =
        safe_to(output.sample_output.next_tokens, torch::kCPU);
    if (next_tokens.defined() &&
        next_tokens.numel() == static_cast<int64_t>(num_sequences)) {
      next_tokens = next_tokens.view({-1}).to(torch::kInt);
      Slice<int32_t> next_tokens_slice = {
          next_tokens.data_ptr<int32_t>(),
          static_cast<size_t>(next_tokens.numel())};
      for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
        int32_t token = next_tokens_slice[seq_id];
        if (token < 0) {
          continue;
        }
        const std::string req_id = request_ids[seq_id];
        if (req_id.empty()) {
          continue;
        }
        if (!use_prompt_lookup_cache_) {
          suffix_cache_->add_active_response(
              req_id, std::span<const int32_t>(&token, 1));
        }
        append_tokens_with_limit(suffix_recent_tokens_[req_id],
                                 std::span<const int32_t>(&token, 1),
                                 static_cast<size_t>(recent_tokens_max_size_));
      }
    }
  }

  output.sample_output.embeddings = torch::Tensor();
  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    return std::nullopt;
  }
  return output;
}

std::optional<ForwardOutput> SuffixWorkerImpl::step_decode(
    const ForwardInput& input) {
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  const int32_t num_sequences = input.input_params.num_sequences;
  const int32_t num_val_tokens = num_speculative_tokens + 1;
  const auto& request_ids = input.input_params.request_ids;
  const bool enable_variable_pld_validate =
      use_prompt_lookup_cache_ && !input.sampling_params.logprobs &&
      input.sampling_params.max_top_logprobs <= 0 &&
      rate_controller_ == nullptr &&
      input.input_params.dp_global_token_nums.size() <= 1;

  const bool has_request_ids =
      (suffix_cache_ != nullptr || prompt_lookup_cache_ != nullptr) &&
      request_ids.size() == static_cast<size_t>(num_sequences);
  const bool enable_pld_request_stats = options_.pld_enable_request_stats();
  if (has_request_ids) {
    std::unordered_set<std::string> current_req_ids;
    current_req_ids.reserve(num_sequences);
    for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
      if (!request_ids[seq_id].empty()) {
        current_req_ids.insert(request_ids[seq_id]);
      }
    }
    cleanup_inactive_requests(current_req_ids);
    suffix_active_decode_req_ids_ = std::move(current_req_ids);
  }

  torch::Tensor input_token_ids = safe_to(input.token_ids, torch::kCPU);
  Slice<int32_t> input_tokens_slice = {
      input_token_ids.data_ptr<int32_t>(),
      static_cast<size_t>(input_token_ids.numel())};

  Timer timer;

  std::vector<int32_t> draft_tokens_flat;
  draft_tokens_flat.reserve(num_sequences * num_speculative_tokens);
  std::vector<int32_t> draft_token_counts(num_sequences, 0);
  std::vector<int32_t> validate_token_counts(num_sequences, num_val_tokens);
  std::vector<std::string> req_ids(num_sequences);
  max_accepted_tokens_per_seq_.assign(
      num_sequences, use_prompt_lookup_cache_ ? 1 : num_val_tokens);
  int32_t total_pld_draft_tokens = 0;
  int32_t pld_attempted_sequences = 0;

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    int32_t fallback_token = input_tokens_slice[seq_id];
    for (int32_t i = 0; i < num_speculative_tokens; ++i) {
      draft_tokens_flat.emplace_back(fallback_token);
    }

    if ((suffix_cache_ == nullptr && prompt_lookup_cache_ == nullptr) ||
        request_ids.size() != static_cast<size_t>(num_sequences)) {
      continue;
    }

    const std::string req_id = request_ids[seq_id];
    if (req_id.empty()) {
      continue;
    }
    req_ids[seq_id] = req_id;

    auto& history = suffix_recent_tokens_[req_id];
    if (history.empty()) {
      append_tokens_with_limit(history,
                               std::span<const int32_t>(&fallback_token, 1),
                               static_cast<size_t>(recent_tokens_max_size_));
    }

    if (use_prompt_lookup_cache_) {
      if (!prompt_lookup_cache_->has_request(req_id)) {
        if (!history.empty()) {
          prompt_lookup_cache_->start_request(
              req_id, std::span<const int32_t>(history.data(), history.size()));
          ensure_pld_request_stats(req_id);
          LOG_EVERY_N(WARNING, 100)
              << "PLD request cache missing during decode; rebuilt cache "
                 "from recent token history. req_id="
              << req_id << ", history_tokens=" << history.size();
        } else {
          LOG_EVERY_N(WARNING, 100)
              << "PLD request cache missing during decode, falling back to "
                 "one accepted token for this step, req_id="
              << req_id;
          max_accepted_tokens_per_seq_[seq_id] = 1;
          continue;
        }
      }
    } else if (!suffix_cache_->has_active_request(req_id)) {
      suffix_cache_->start_request(
          req_id, std::span<const int32_t>(&fallback_token, 1));
      suffix_recent_tokens_[req_id].clear();
      append_tokens_with_limit(suffix_recent_tokens_[req_id],
                               std::span<const int32_t>(&fallback_token, 1),
                               static_cast<size_t>(recent_tokens_max_size_));
    }

    std::vector<int32_t> draft_token_ids;
    if (use_prompt_lookup_cache_) {
      ++pld_attempted_sequences;
      PromptLookupDraft draft = prompt_lookup_cache_->speculate(
          req_id,
          std::span<const int32_t>(history.data(), history.size()),
          num_speculative_tokens,
          options_.pld_candidate_count(),
          options_.pld_prefer_recent_match());
      draft_token_ids = std::move(draft.token_ids);
    } else {
      SuffixDecodingDraft draft = suffix_cache_->speculate(
          req_id,
          std::span<const int32_t>(history.data(), history.size()),
          /*max_spec_tokens=*/num_speculative_tokens,
          options_.speculative_suffix_max_spec_factor(),
          options_.speculative_suffix_max_spec_offset(),
          options_.speculative_suffix_min_token_prob(),
          options_.speculative_suffix_use_tree_spec());
      draft_token_ids = std::move(draft.token_ids);
    }

    const int32_t fill_count =
        std::min<int32_t>(num_speculative_tokens, draft_token_ids.size());
    draft_token_counts[seq_id] = fill_count;
    if (use_prompt_lookup_cache_) {
      if (enable_variable_pld_validate) {
        validate_token_counts[seq_id] = fill_count + 1;
      }
      max_accepted_tokens_per_seq_[seq_id] = fill_count + 1;
      total_pld_draft_tokens += fill_count;
      if (enable_pld_request_stats) {
        PldRequestStats& stats = pld_request_stats_[req_id];
        ++stats.decode_sequence_steps;
        stats.requested_draft_tokens +=
            static_cast<uint64_t>(num_speculative_tokens);
        stats.drafted_tokens += static_cast<uint64_t>(fill_count);
        if (fill_count == 0) {
          ++stats.no_draft_sequence_steps;
        }
      }
    }
    for (int32_t i = 0; i < fill_count; ++i) {
      draft_tokens_flat[seq_id * num_speculative_tokens + i] =
          draft_token_ids[i];
    }
  }

  if (use_prompt_lookup_cache_) {
    ++pld_decode_batches_;
    pld_requested_draft_tokens_total_ +=
        static_cast<uint64_t>(pld_attempted_sequences) *
        static_cast<uint64_t>(num_speculative_tokens);

    if (total_pld_draft_tokens == 0) {
      ++pld_no_draft_batches_;
      COUNTER_ADD(speculative_execution_latency_seconds_draft,
                  timer.elapsed_seconds());

      timer.reset();
      auto future = impl_->step_async(input);
      ForwardOutput output = std::move(future).get().value();
      COUNTER_ADD(speculative_execution_latency_seconds_target,
                  timer.elapsed_seconds());

      if (request_ids.size() == static_cast<size_t>(num_sequences)) {
        torch::Tensor next_tokens =
            safe_to(output.sample_output.next_tokens, torch::kCPU);
        if (next_tokens.defined() &&
            next_tokens.numel() == static_cast<int64_t>(num_sequences)) {
          next_tokens = next_tokens.view({-1}).to(torch::kInt);
          Slice<int32_t> next_tokens_slice = {
              next_tokens.data_ptr<int32_t>(),
              static_cast<size_t>(next_tokens.numel())};
          for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
            const std::string req_id = request_ids[seq_id];
            if (req_id.empty()) {
              continue;
            }
            const int32_t token = next_tokens_slice[seq_id];
            if (token < 0) {
              continue;
            }
            append_tokens_with_limit(
                suffix_recent_tokens_[req_id],
                std::span<const int32_t>(&token, 1),
                static_cast<size_t>(recent_tokens_max_size_));
          }
        }
      }

      VLOG(1) << "PLD no draft tokens found for current decode batch; using "
                 "single-token target decode fallback. reqs="
              << num_sequences
              << ", recent_tokens_max_size=" << recent_tokens_max_size_;
      log_pld_decode_summary(pld_decode_batches_,
                             pld_no_draft_batches_,
                             pld_requested_draft_tokens_total_,
                             pld_draft_tokens_total_,
                             pld_accepted_draft_tokens_total_,
                             options_.pld_stats_log_interval());

      output.sample_output.embeddings = torch::Tensor();
      if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
        return std::nullopt;
      }
      return output;
    }

    pld_draft_tokens_total_ += static_cast<uint64_t>(total_pld_draft_tokens);
    if (enable_pld_request_stats) {
      for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
        const std::string& req_id = req_ids[seq_id];
        if (req_id.empty()) {
          continue;
        }
        pld_request_stats_[req_id].validate_waste_tokens +=
            static_cast<uint64_t>(num_speculative_tokens -
                                  draft_token_counts[seq_id]);
      }
    }
  }

  ForwardInput validate_input;
  if (enable_variable_pld_validate) {
    prepare_validate_inputs(
        input,
        validate_input,
        std::span<const int32_t>(draft_token_counts.data(),
                                 draft_token_counts.size()));
  } else {
    prepare_validate_inputs(input, validate_input);
  }
  validate_input.skip_sampling_for_logits_only = true;
  const int32_t max_validate_tokens =
      enable_variable_pld_validate
          ? get_max_validate_token_count(validate_token_counts)
          : num_val_tokens;

  auto draft_token_ids_cpu =
      torch::tensor(draft_tokens_flat,
                    torch::TensorOptions().dtype(torch::kInt))
          .view({num_sequences, num_speculative_tokens});
  auto draft_token_ids_int =
      draft_token_ids_cpu.to(validate_input.token_ids.device());

  auto& validate_token_ids = validate_input.token_ids;
  for (int32_t i = 0; i < num_speculative_tokens; ++i) {
    auto mask = (validate_token_ids == -1 * (i + 1));
    if (enable_variable_pld_validate) {
      std::vector<int32_t> draft_col_values;
      draft_col_values.reserve(num_sequences);
      for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
        if (draft_token_counts[seq_id] <= i) {
          continue;
        }
        draft_col_values.emplace_back(
            draft_tokens_flat[seq_id * num_speculative_tokens + i]);
      }
      if (!draft_col_values.empty()) {
        CHECK_EQ(mask.sum().item<int64_t>(),
                 static_cast<int64_t>(draft_col_values.size()))
            << "variable-length PLD placeholder count mismatch for draft idx="
            << i;
        torch::Tensor draft_col_tensor =
            torch::tensor(draft_col_values,
                          torch::TensorOptions()
                              .dtype(torch::kInt)
                              .device(validate_input.token_ids.device()));
        validate_token_ids.masked_scatter_(mask, draft_col_tensor);
      }
    } else {
      auto draft_col_tensor =
          draft_token_ids_int.select(/*dim=*/1, /*index=*/i);
      validate_token_ids.masked_scatter_(mask, draft_col_tensor);
    }
  }

  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());

  timer.reset();
  auto future = impl_->step_async(validate_input);
  ForwardOutput target_output = std::move(future).get().value();
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              timer.elapsed_seconds());

  torch::Tensor draft_token_ids =
      draft_token_ids_int.to(target_output.logits.device()).to(torch::kLong);

  // RejectionSampler::forward requires draft_probs tensor for interface
  // compatibility, but greedy-only validation does not use its values.
  // Use a minimal placeholder to avoid one-hot scatter over vocab.
  auto draft_probs = torch::empty({num_sequences, num_speculative_tokens, 1},
                                  torch::TensorOptions()
                                      .dtype(torch::kFloat32)
                                      .device(target_output.logits.device()));

  timer.reset();
  SampleOutput val_output = validate(input.sampling_params,
                                     draft_token_ids,
                                     draft_probs,
                                     enable_variable_pld_validate,
                                     validate_token_counts,
                                     target_output);
  COUNTER_ADD(speculative_execution_latency_seconds_validation,
              timer.elapsed_seconds());

  if ((suffix_cache_ != nullptr || prompt_lookup_cache_ != nullptr) &&
      request_ids.size() == static_cast<size_t>(num_sequences)) {
    torch::Tensor accepted_tokens =
        safe_to(val_output.next_tokens, torch::kCPU).to(torch::kInt);
    accepted_tokens =
        accepted_tokens.view({num_sequences, max_validate_tokens});
    Slice<int32_t> accepted_tokens_slice = {
        accepted_tokens.data_ptr<int32_t>(),
        static_cast<size_t>(accepted_tokens.numel())};

    int32_t accepted_pld_draft_tokens = 0;
    for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
      const std::string& req_id = req_ids[seq_id];
      if (req_id.empty()) {
        continue;
      }

      std::vector<int32_t> accepted;
      accepted.reserve(validate_token_counts[seq_id]);
      int32_t first_reject_idx = -1;
      std::vector<int32_t> row_tokens;
      row_tokens.reserve(validate_token_counts[seq_id]);
      for (int32_t j = 0; j < validate_token_counts[seq_id]; ++j) {
        int32_t token = accepted_tokens_slice[seq_id * max_validate_tokens + j];
        row_tokens.emplace_back(token);
        if (token < 0) {
          if (first_reject_idx < 0) {
            first_reject_idx = j;
          }
          break;
        }
        accepted.emplace_back(token);
      }

      if (use_prompt_lookup_cache_) {
        const int32_t accepted_draft_tokens =
            first_reject_idx < 0 ? num_speculative_tokens
                                 : std::max(0, first_reject_idx);
        const int32_t bounded_accepted_draft_tokens =
            std::min(accepted_draft_tokens, draft_token_counts[seq_id]);
        accepted_pld_draft_tokens += bounded_accepted_draft_tokens;
        if (enable_pld_request_stats) {
          pld_request_stats_[req_id].accepted_draft_tokens +=
              static_cast<uint64_t>(bounded_accepted_draft_tokens);
        }
      }

      if (seq_id < 8) {
        VLOG(3) << "[spec-validate-output] seq=" << seq_id
                << " req_id=" << req_id << " accepted_len=" << accepted.size()
                << " first_reject_idx=" << first_reject_idx << " accepted="
                << summarize_int32_span(std::span<const int32_t>(
                       accepted.data(), accepted.size()))
                << " row_tokens="
                << summarize_int32_span(std::span<const int32_t>(
                       row_tokens.data(), row_tokens.size()));
        VLOG(3) << "[spec-reject-handle] seq=" << seq_id
                << " accepted_prefix_len=" << accepted.size()
                << " first_reject_idx=" << first_reject_idx << " pos_offset = "
                << (static_cast<int32_t>(accepted.size()) - 1) << " row_tokens="
                << summarize_int32_span(std::span<const int32_t>(
                       row_tokens.data(), row_tokens.size()));
      }

      if (!accepted.empty()) {
        if (!use_prompt_lookup_cache_) {
          suffix_cache_->add_active_response(
              req_id,
              std::span<const int32_t>(accepted.data(), accepted.size()));
        }
        append_tokens_with_limit(
            suffix_recent_tokens_[req_id],
            std::span<const int32_t>(accepted.data(), accepted.size()),
            static_cast<size_t>(recent_tokens_max_size_));
      }
    }

    if (use_prompt_lookup_cache_) {
      pld_accepted_draft_tokens_total_ +=
          static_cast<uint64_t>(accepted_pld_draft_tokens);
      log_pld_decode_summary(pld_decode_batches_,
                             pld_no_draft_batches_,
                             pld_requested_draft_tokens_total_,
                             pld_draft_tokens_total_,
                             pld_accepted_draft_tokens_total_,
                             options_.pld_stats_log_interval());
    }
  }

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    return std::nullopt;
  }
  val_output.embeddings = torch::Tensor();
  target_output.sample_output = val_output;
  return target_output;
}

SampleOutput SuffixWorkerImpl::validate(
    const SamplingParameters& sampling_params,
    const torch::Tensor& draft_token_ids,
    const torch::Tensor& draft_probs,
    bool enable_variable_pld_validate,
    const std::vector<int32_t>& validate_token_counts,
    const ForwardOutput& target_output) {
  (void)sampling_params;
  const int32_t batch_size =
      static_cast<int32_t>(draft_token_ids.size(/*dim=*/0));
  const int32_t vocab_size =
      static_cast<int32_t>(target_output.logits.size(/*dim=*/-1));
  if (!use_prompt_lookup_cache_ || !enable_variable_pld_validate) {
    const int32_t num_val_tokens = options_.num_speculative_tokens() + 1;
    CHECK_EQ(target_output.logits.size(/*dim=*/0),
             static_cast<int64_t>(batch_size) * num_val_tokens)
        << "suffix validate logits shape mismatch";

    using ISlice = torch::indexing::Slice;
    auto target_logits =
        target_output.logits.view({batch_size, num_val_tokens, vocab_size});
    auto bonus_token_ids =
        target_logits.index({ISlice(), num_val_tokens - 1, ISlice()})
            .argmax(/*dim=*/-1, /*keepdim=*/true);

    auto greedy_do_sample = torch::zeros({batch_size}, torch::kBool);
    auto rejection_sampler =
        std::make_unique<RejectionSampler>(greedy_do_sample,
                                           /*all_random_sample=*/false,
                                           /*all_greedy_sample=*/true,
                                           target_output.logprobs,
                                           target_output.max_top_logprobs,
                                           rate_controller_,
                                           enable_fused_kernel_);

    SampleOutput sample_output =
        rejection_sampler->forward(draft_token_ids.to(bonus_token_ids),
                                   draft_probs.to(target_logits.device()),
                                   target_logits,
                                   bonus_token_ids,
                                   /*mask_out_rejected_tokens=*/true);

    auto embeddings = target_output.sample_output.embeddings;
    sample_output.embeddings =
        embeddings.view({batch_size, num_val_tokens, embeddings.size(-1)});

    for (int32_t seq_id = 0;
         seq_id < static_cast<int32_t>(max_accepted_tokens_per_seq_.size());
         ++seq_id) {
      const int32_t keep_tokens =
          std::clamp(max_accepted_tokens_per_seq_[seq_id], 1, num_val_tokens);
      if (keep_tokens < num_val_tokens) {
        sample_output.next_tokens[seq_id]
            .slice(/*dim=*/0, /*start=*/keep_tokens)
            .fill_(-1);
      }
    }

    torch::Tensor mask = (sample_output.next_tokens == -1).to(torch::kInt64);
    size_t count = mask.sum().item<int64_t>();
    size_t num_draft_tokens =
        static_cast<size_t>(batch_size) * options_.num_speculative_tokens();
    COUNTER_ADD(speculative_num_draft_tokens_total, num_draft_tokens);
    COUNTER_ADD(speculative_num_accepted_tokens_total,
                num_draft_tokens - count);
    return sample_output;
  }

  CHECK_EQ(static_cast<int32_t>(validate_token_counts.size()), batch_size)
      << "validate_token_counts size mismatch";
  const int32_t max_validate_tokens =
      get_max_validate_token_count(validate_token_counts);
  int32_t total_num_val_tokens = 0;
  for (int32_t validate_token_count : validate_token_counts) {
    total_num_val_tokens += validate_token_count;
  }
  CHECK_EQ(target_output.logits.size(/*dim=*/0),
           static_cast<int64_t>(total_num_val_tokens))
      << "suffix validate logits shape mismatch";

  SampleOutput sample_output;
  const bool use_pld_greedy_fast_path =
      use_prompt_lookup_cache_ && !target_output.logprobs &&
      target_output.max_top_logprobs <= 0 && rate_controller_ == nullptr;
  CHECK(use_pld_greedy_fast_path)
      << "variable-length PLD validate currently requires greedy-only output";

  torch::Tensor next_tokens =
      torch::full({batch_size, max_validate_tokens},
                  -1,
                  torch::TensorOptions()
                      .dtype(torch::kLong)
                      .device(target_output.logits.device()));
  int32_t row_offset = 0;
  int64_t accepted_draft_tokens_total = 0;
  for (int32_t seq_id = 0; seq_id < batch_size; ++seq_id) {
    const int32_t seq_num_val_tokens = validate_token_counts[seq_id];
    const int32_t seq_num_draft_tokens = seq_num_val_tokens - 1;
    torch::Tensor seq_logits = target_output.logits.slice(
        /*dim=*/0,
        /*start=*/row_offset,
        /*end=*/row_offset + seq_num_val_tokens);
    row_offset += seq_num_val_tokens;
    torch::Tensor seq_bonus_token_id =
        seq_logits.select(/*dim=*/0, /*index=*/seq_num_val_tokens - 1)
            .argmax(/*dim=*/-1, /*keepdim=*/true);
    torch::Tensor seq_target_token_ids =
        seq_num_draft_tokens == 0
            ? torch::empty({0},
                           torch::TensorOptions()
                               .dtype(torch::kLong)
                               .device(target_output.logits.device()))
            : seq_logits
                  .slice(/*dim=*/0,
                         /*start=*/0,
                         /*end=*/seq_num_draft_tokens)
                  .argmax(/*dim=*/-1);
    if (seq_num_draft_tokens == 0) {
      next_tokens[seq_id]
          .slice(/*dim=*/0, /*start=*/0, /*end=*/1)
          .copy_(seq_bonus_token_id);
      continue;
    }

    torch::Tensor seq_draft_token_ids =
        draft_token_ids[seq_id]
            .slice(/*dim=*/0, /*start=*/0, /*end=*/seq_num_draft_tokens)
            .to(seq_target_token_ids.device());
    torch::Tensor seq_accepted =
        seq_target_token_ids ==
        seq_draft_token_ids.to(seq_target_token_ids.device());
    torch::Tensor seq_accepted_mask =
        RejectionSampler::build_accepted_mask(seq_accepted.unsqueeze(0))
            .squeeze(0);
    torch::Tensor seq_accepted_token_ids =
        torch::cat({seq_target_token_ids, seq_bonus_token_id}, /*dim=*/0);
    torch::Tensor seq_output_tokens =
        torch::where(seq_accepted_mask,
                     seq_accepted_token_ids,
                     -torch::ones_like(seq_accepted_token_ids));
    next_tokens[seq_id]
        .slice(/*dim=*/0, /*start=*/0, /*end=*/seq_num_val_tokens)
        .copy_(seq_output_tokens);

    torch::Tensor seq_prefix_accept =
        seq_accepted.to(torch::kInt32).cumprod(/*dim=*/0);
    accepted_draft_tokens_total += seq_prefix_accept.sum().item<int64_t>();
  }

  sample_output.next_tokens = next_tokens;
  sample_output.embeddings = torch::Tensor();

  for (int32_t seq_id = 0;
       seq_id < static_cast<int32_t>(max_accepted_tokens_per_seq_.size());
       ++seq_id) {
    const int32_t keep_tokens = std::clamp(
        max_accepted_tokens_per_seq_[seq_id], 1, validate_token_counts[seq_id]);
    if (keep_tokens < validate_token_counts[seq_id]) {
      sample_output.next_tokens[seq_id]
          .slice(/*dim=*/0, /*start=*/keep_tokens)
          .fill_(-1);
    }
  }

  size_t num_draft_tokens = 0;
  for (int32_t validate_token_count : validate_token_counts) {
    num_draft_tokens += static_cast<size_t>(validate_token_count - 1);
  }
  COUNTER_ADD(speculative_num_draft_tokens_total, num_draft_tokens);
  COUNTER_ADD(speculative_num_accepted_tokens_total,
              static_cast<size_t>(accepted_draft_tokens_total));

  return sample_output;
}

}  // namespace xllm
