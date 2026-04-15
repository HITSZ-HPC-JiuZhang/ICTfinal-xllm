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

#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace xllm {

struct PromptLookupDraft {
  std::vector<int32_t> token_ids;
  int32_t match_len = 0;
};

class PromptLookupCache {
 public:
  PromptLookupCache(int32_t max_ngram_size = 8, int32_t min_ngram_size = 3);

  bool has_request(const std::string& req_id) const;

  void start_request(const std::string& req_id,
                     std::span<const int32_t> prompt_token_ids);

  void stop_request(const std::string& req_id);

  void add_prompt(const std::string& req_id,
                  std::span<const int32_t> prompt_token_ids);

  PromptLookupDraft speculate(const std::string& req_id,
                              std::span<const int32_t> context,
                              int32_t max_spec_tokens) const;

 private:
  struct RequestCache {
    std::vector<int32_t> prompt_token_ids;
    std::unordered_map<uint64_t, std::vector<int32_t>> ngram_index;
  };

  uint64_t hash_tokens(std::span<const int32_t> tokens) const;

  uint64_t hash_tokens(const std::vector<int32_t>& tokens,
                       int32_t start,
                       int32_t len) const;

  void add_prompt_tokens(RequestCache& request,
                         std::span<const int32_t> prompt_token_ids);

  void index_new_prompt_tokens(RequestCache& request, int32_t old_size);

  bool matches_at(const RequestCache& request,
                  std::span<const int32_t> context,
                  int32_t context_start,
                  int32_t prompt_start,
                  int32_t len) const;

 private:
  int32_t max_ngram_size_;
  int32_t min_ngram_size_;
  std::unordered_map<std::string, RequestCache> requests_;
};

}  // namespace xllm
