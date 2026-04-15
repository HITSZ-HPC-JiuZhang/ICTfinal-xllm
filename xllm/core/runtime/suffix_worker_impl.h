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

#pragma once

#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "runtime/speculative_worker_impl.h"
#include "util/prompt_lookup_cache.h"
#include "util/suffix_decoding_cache.h"

namespace xllm {

// Suffix-based speculative decoding worker.
// Uses a suffix tree cache to generate draft tokens from previously seen
// patterns, without requiring a separate draft model.
class SuffixWorkerImpl : public SpeculativeWorkerImpl {
 public:
  SuffixWorkerImpl(const ParallelArgs& parallel_args,
                   const torch::Device& device,
                   const runtime::Options& options);

  ~SuffixWorkerImpl() override;

 protected:
  std::optional<ForwardOutput> step_prefill(const ForwardInput& input) override;
  std::optional<ForwardOutput> step_decode(const ForwardInput& inputs) override;
  std::optional<ForwardOutput> step_empty(const ForwardInput& inputs) override;

 private:
  SampleOutput validate(const SamplingParameters& sampling_params,
                        const torch::Tensor& draft_token_ids,
                        const torch::Tensor& draft_probs,
                        bool enable_variable_pld_validate,
                        const std::vector<int32_t>& validate_token_counts,
                        const ForwardOutput& target_output);

  struct PldRequestStats {
    uint64_t decode_sequence_steps = 0;
    uint64_t no_draft_sequence_steps = 0;
    uint64_t requested_draft_tokens = 0;
    uint64_t drafted_tokens = 0;
    uint64_t accepted_draft_tokens = 0;
    uint64_t validate_waste_tokens = 0;
  };

  struct PldAdaptiveSample {
    int32_t requested_draft_tokens = 0;
    int32_t drafted_tokens = 0;
    int32_t accepted_draft_tokens = 0;
  };

  struct PldAdaptiveState {
    int32_t cooldown_steps = 0;
    uint64_t window_requested_draft_tokens = 0;
    uint64_t window_drafted_tokens = 0;
    uint64_t window_accepted_draft_tokens = 0;
    std::deque<PldAdaptiveSample> window_samples;
  };

  void ensure_pld_request_stats(const std::string& req_id);

  void log_pld_request_stats(const std::string& req_id) const;
  bool pld_adaptive_enabled() const;
  int32_t get_pld_adaptive_draft_budget(const std::string& req_id,
                                        int32_t max_spec_tokens);
  void update_pld_adaptive_state(const std::string& req_id,
                                 int32_t requested_draft_tokens,
                                 int32_t drafted_tokens,
                                 int32_t accepted_draft_tokens);
  void cleanup_inactive_requests(
      const std::unordered_set<std::string>& current_req_ids);
  void flush_all_pld_requests();

  std::unique_ptr<SuffixDecodingCache> suffix_cache_;
  std::unique_ptr<PromptLookupCache> prompt_lookup_cache_;
  std::unordered_map<std::string, std::vector<int32_t>> suffix_recent_tokens_;
  std::unordered_map<std::string, PldRequestStats> pld_request_stats_;
  std::unordered_map<std::string, PldAdaptiveState> pld_adaptive_states_;
  std::unordered_set<std::string> suffix_active_decode_req_ids_;
  std::vector<int32_t> max_accepted_tokens_per_seq_;
  uint64_t pld_decode_batches_ = 0;
  uint64_t pld_requested_draft_tokens_total_ = 0;
  uint64_t pld_draft_tokens_total_ = 0;
  uint64_t pld_accepted_draft_tokens_total_ = 0;
  uint64_t pld_no_draft_batches_ = 0;
  int32_t recent_tokens_max_size_ = 0;
  bool use_prompt_lookup_cache_ = false;
};
}  // namespace xllm
