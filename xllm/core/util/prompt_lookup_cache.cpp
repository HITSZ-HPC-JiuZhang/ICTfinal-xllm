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

#include "prompt_lookup_cache.h"

#include <algorithm>
#include <stdexcept>

namespace xllm {
namespace {

constexpr uint64_t kFnvOffsetBasis = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;
constexpr int32_t kMinPromptLookupNgramSize = 1;
constexpr int32_t kMaxPromptLookupCandidateCount = 16;

uint64_t mix_int32(uint64_t hash, int32_t value) {
  hash ^= static_cast<uint64_t>(static_cast<uint32_t>(value));
  hash *= kFnvPrime;
  return hash;
}

struct Candidate {
  int32_t prompt_start = 0;
  int32_t draft_start = 0;
  int32_t draft_len = 0;
  int32_t continuation_prefix_reuse = 0;
};

bool is_better_candidate(const Candidate& lhs,
                         const Candidate& rhs,
                         bool prefer_recent_match) {
  if (lhs.draft_len != rhs.draft_len) {
    return lhs.draft_len > rhs.draft_len;
  }
  if (prefer_recent_match && lhs.prompt_start != rhs.prompt_start) {
    return lhs.prompt_start > rhs.prompt_start;
  }
  if (lhs.continuation_prefix_reuse != rhs.continuation_prefix_reuse) {
    return lhs.continuation_prefix_reuse > rhs.continuation_prefix_reuse;
  }
  if (!prefer_recent_match && lhs.prompt_start != rhs.prompt_start) {
    return lhs.prompt_start < rhs.prompt_start;
  }
  return false;
}

}  // namespace

PromptLookupCache::PromptLookupCache(int32_t max_ngram_size,
                                     int32_t min_ngram_size)
    : max_ngram_size_(max_ngram_size), min_ngram_size_(min_ngram_size) {
  if (max_ngram_size_ < kMinPromptLookupNgramSize) {
    throw std::invalid_argument("max_ngram_size must be positive");
  }
  if (min_ngram_size_ < kMinPromptLookupNgramSize) {
    throw std::invalid_argument("min_ngram_size must be positive");
  }
  if (min_ngram_size_ > max_ngram_size_) {
    throw std::invalid_argument("min_ngram_size must be <= max_ngram_size");
  }
}

bool PromptLookupCache::has_request(const std::string& req_id) const {
  return requests_.find(req_id) != requests_.end();
}

void PromptLookupCache::start_request(
    const std::string& req_id,
    std::span<const int32_t> prompt_token_ids) {
  RequestCache request;
  add_prompt_tokens(request, prompt_token_ids);
  requests_[req_id] = std::move(request);
}

void PromptLookupCache::stop_request(const std::string& req_id) {
  requests_.erase(req_id);
}

void PromptLookupCache::add_prompt(const std::string& req_id,
                                   std::span<const int32_t> prompt_token_ids) {
  auto it = requests_.find(req_id);
  if (it == requests_.end()) {
    start_request(req_id, prompt_token_ids);
    return;
  }
  add_prompt_tokens(it->second, prompt_token_ids);
}

PromptLookupDraft PromptLookupCache::speculate(const std::string& req_id,
                                               std::span<const int32_t> context,
                                               int32_t max_spec_tokens,
                                               int32_t candidate_count,
                                               bool prefer_recent_match) const {
  PromptLookupDraft draft;
  if (max_spec_tokens <= 0) {
    return draft;
  }
  if (candidate_count <= 0) {
    return draft;
  }
  candidate_count = std::min(candidate_count, kMaxPromptLookupCandidateCount);

  auto req_it = requests_.find(req_id);
  if (req_it == requests_.end()) {
    return draft;
  }

  const RequestCache& request = req_it->second;
  const int32_t context_size = static_cast<int32_t>(context.size());
  const int32_t max_match_len =
      std::min({max_ngram_size_,
                context_size,
                static_cast<int32_t>(request.prompt_token_ids.size())});

  for (int32_t match_len = max_match_len; match_len >= min_ngram_size_;
       --match_len) {
    const int32_t context_start = context_size - match_len;
    const uint64_t key = hash_tokens(context.subspan(context_start, match_len));
    auto index_it = request.ngram_index.find(key);
    if (index_it == request.ngram_index.end()) {
      continue;
    }

    const std::vector<int32_t>& positions = index_it->second;
    std::vector<Candidate> candidates;
    candidates.reserve(candidate_count);
    for (int32_t prompt_start : positions) {
      if (!matches_at(
              request, context, context_start, prompt_start, match_len)) {
        continue;
      }

      const int32_t draft_start = prompt_start + match_len;
      const int32_t available_tokens =
          static_cast<int32_t>(request.prompt_token_ids.size()) - draft_start;
      const int32_t draft_len = std::min(max_spec_tokens, available_tokens);
      if (draft_len <= 0) {
        continue;
      }
      if (candidate_count == 1 && !prefer_recent_match) {
        draft.match_len = match_len;
        draft.token_ids.insert(
            draft.token_ids.end(),
            request.prompt_token_ids.begin() + draft_start,
            request.prompt_token_ids.begin() + draft_start + draft_len);
        return draft;
      }

      Candidate candidate;
      candidate.prompt_start = prompt_start;
      candidate.draft_start = draft_start;
      candidate.draft_len = draft_len;
      candidate.continuation_prefix_reuse =
          count_continuation_prefix_reuse(request, draft_start, draft_len);
      if (static_cast<int32_t>(candidates.size()) < candidate_count) {
        candidates.emplace_back(candidate);
        continue;
      }

      int32_t worst_idx = 0;
      for (int32_t idx = 1; idx < static_cast<int32_t>(candidates.size());
           ++idx) {
        if (is_better_candidate(
                candidates[worst_idx], candidates[idx], prefer_recent_match)) {
          worst_idx = idx;
        }
      }
      if (is_better_candidate(
              candidate, candidates[worst_idx], prefer_recent_match)) {
        candidates[worst_idx] = candidate;
      }
    }

    if (candidates.empty()) {
      continue;
    }

    auto best_it = candidates.begin();
    for (auto it = candidates.begin() + 1; it != candidates.end(); ++it) {
      if (is_better_candidate(*it, *best_it, prefer_recent_match)) {
        best_it = it;
      }
    }
    draft.match_len = match_len;
    draft.token_ids.insert(
        draft.token_ids.end(),
        request.prompt_token_ids.begin() + best_it->draft_start,
        request.prompt_token_ids.begin() + best_it->draft_start +
            best_it->draft_len);
    return draft;
  }

  return draft;
}

uint64_t PromptLookupCache::hash_tokens(std::span<const int32_t> tokens) const {
  uint64_t hash =
      mix_int32(kFnvOffsetBasis, static_cast<int32_t>(tokens.size()));
  for (int32_t token : tokens) {
    hash = mix_int32(hash, token);
  }
  return hash;
}

uint64_t PromptLookupCache::hash_tokens(const std::vector<int32_t>& tokens,
                                        int32_t start,
                                        int32_t len) const {
  uint64_t hash = mix_int32(kFnvOffsetBasis, len);
  for (int32_t idx = 0; idx < len; ++idx) {
    hash = mix_int32(hash, tokens[start + idx]);
  }
  return hash;
}

void PromptLookupCache::add_prompt_tokens(
    RequestCache& request,
    std::span<const int32_t> prompt_token_ids) {
  if (prompt_token_ids.empty()) {
    return;
  }

  const int32_t old_size =
      static_cast<int32_t>(request.prompt_token_ids.size());
  request.prompt_token_ids.reserve(request.prompt_token_ids.size() +
                                   prompt_token_ids.size());
  request.prompt_token_ids.insert(request.prompt_token_ids.end(),
                                  prompt_token_ids.begin(),
                                  prompt_token_ids.end());
  request.ngram_index.reserve(request.prompt_token_ids.size());
  request.continuation_prefix_counts.reserve(request.prompt_token_ids.size());
  index_new_prompt_tokens(request, old_size);
  index_new_continuation_prefixes(request, old_size);
}

void PromptLookupCache::index_new_prompt_tokens(RequestCache& request,
                                                int32_t old_size) {
  const int32_t prompt_size =
      static_cast<int32_t>(request.prompt_token_ids.size());
  for (int32_t len = min_ngram_size_; len <= max_ngram_size_; ++len) {
    const int32_t first_new_start = std::max(0, old_size - len);
    const int32_t last_start = prompt_size - len - 1;
    for (int32_t start = first_new_start; start <= last_start; ++start) {
      const uint64_t key = hash_tokens(request.prompt_token_ids, start, len);
      request.ngram_index[key].emplace_back(start);
    }
  }
}

void PromptLookupCache::index_new_continuation_prefixes(RequestCache& request,
                                                        int32_t old_size) {
  const int32_t prompt_size =
      static_cast<int32_t>(request.prompt_token_ids.size());
  constexpr int32_t kMaxReusePrefixLen = 2;
  for (int32_t len = 1; len <= kMaxReusePrefixLen; ++len) {
    const int32_t first_new_start = std::max(0, old_size - len + 1);
    const int32_t last_start = prompt_size - len;
    for (int32_t start = first_new_start; start <= last_start; ++start) {
      const uint64_t key = hash_tokens(request.prompt_token_ids, start, len);
      ++request.continuation_prefix_counts[key];
    }
  }
}

bool PromptLookupCache::matches_at(const RequestCache& request,
                                   std::span<const int32_t> context,
                                   int32_t context_start,
                                   int32_t prompt_start,
                                   int32_t len) const {
  if (prompt_start < 0 || context_start < 0 || len <= 0) {
    return false;
  }
  if (prompt_start + len >=
      static_cast<int32_t>(request.prompt_token_ids.size())) {
    return false;
  }
  if (context_start + len > static_cast<int32_t>(context.size())) {
    return false;
  }
  for (int32_t idx = 0; idx < len; ++idx) {
    if (request.prompt_token_ids[prompt_start + idx] !=
        context[context_start + idx]) {
      return false;
    }
  }
  return true;
}

int32_t PromptLookupCache::count_continuation_prefix_reuse(
    const RequestCache& request,
    int32_t draft_start,
    int32_t draft_len) const {
  const int32_t prefix_len = std::min(2, draft_len);
  if (prefix_len <= 0) {
    return 0;
  }

  const uint64_t key =
      hash_tokens(request.prompt_token_ids, draft_start, prefix_len);
  auto it = request.continuation_prefix_counts.find(key);
  if (it == request.continuation_prefix_counts.end()) {
    return 0;
  }
  return it->second;
}

}  // namespace xllm
