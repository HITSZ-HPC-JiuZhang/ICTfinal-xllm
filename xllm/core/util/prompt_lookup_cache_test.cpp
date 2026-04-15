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

#include <gtest/gtest.h>

#include <vector>

namespace xllm {

TEST(PromptLookupCacheTest, FindsRepeatedPromptContinuation) {
  PromptLookupCache cache(/*max_ngram_size=*/4, /*min_ngram_size=*/3);
  std::vector<int32_t> prompt = {1, 2, 3, 4, 5, 1, 2, 3, 4, 6};
  cache.start_request("req", std::span<const int32_t>(prompt));

  std::vector<int32_t> context = {9, 1, 2, 3, 4};
  PromptLookupDraft draft =
      cache.speculate("req", std::span<const int32_t>(context), 2);

  EXPECT_EQ(draft.match_len, 4);
  EXPECT_EQ(draft.token_ids, std::vector<int32_t>({6}));
}

TEST(PromptLookupCacheTest, UsesIncrementalPromptChunks) {
  PromptLookupCache cache(/*max_ngram_size=*/3, /*min_ngram_size=*/2);
  std::vector<int32_t> first_chunk = {10, 20, 30};
  std::vector<int32_t> second_chunk = {10, 20, 40};
  cache.start_request("req", std::span<const int32_t>(first_chunk));
  cache.add_prompt("req", std::span<const int32_t>(second_chunk));

  std::vector<int32_t> context = {10, 20};
  PromptLookupDraft draft =
      cache.speculate("req", std::span<const int32_t>(context), 1);

  EXPECT_EQ(draft.match_len, 2);
  EXPECT_EQ(draft.token_ids, std::vector<int32_t>({40}));
}

TEST(PromptLookupCacheTest, IgnoresPromptTailWithoutContinuation) {
  PromptLookupCache cache(/*max_ngram_size=*/4, /*min_ngram_size=*/3);
  std::vector<int32_t> prompt = {1, 2, 3, 4};
  cache.start_request("req", std::span<const int32_t>(prompt));

  std::vector<int32_t> context = {1, 2, 3, 4};
  PromptLookupDraft draft =
      cache.speculate("req", std::span<const int32_t>(context), 2);

  EXPECT_TRUE(draft.token_ids.empty());
}

TEST(PromptLookupCacheTest, PrefersLongestContinuationOverLatestMatch) {
  PromptLookupCache cache(/*max_ngram_size=*/3, /*min_ngram_size=*/3);
  std::vector<int32_t> prompt = {1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 9};
  cache.start_request("req", std::span<const int32_t>(prompt));

  std::vector<int32_t> context = {1, 2, 3};
  PromptLookupDraft draft =
      cache.speculate("req", std::span<const int32_t>(context), 4);

  EXPECT_EQ(draft.match_len, 3);
  EXPECT_EQ(draft.token_ids, std::vector<int32_t>({4, 5, 6, 7}));
}

TEST(PromptLookupCacheTest, PrefersRecentMatchWhenContinuationLengthTies) {
  PromptLookupCache cache(/*max_ngram_size=*/3, /*min_ngram_size=*/3);
  std::vector<int32_t> prompt = {1, 2, 3, 7, 8, 1, 2, 3, 9, 10};
  cache.start_request("req", std::span<const int32_t>(prompt));

  std::vector<int32_t> context = {1, 2, 3};
  PromptLookupDraft draft = cache.speculate("req",
                                            std::span<const int32_t>(context),
                                            /*max_spec_tokens=*/2,
                                            /*candidate_count=*/4,
                                            /*prefer_recent_match=*/true);

  EXPECT_EQ(draft.match_len, 3);
  EXPECT_EQ(draft.token_ids, std::vector<int32_t>({9, 10}));
}

TEST(PromptLookupCacheTest, FallsBackToEarlierMatchWhenRecentPreferenceOff) {
  PromptLookupCache cache(/*max_ngram_size=*/3, /*min_ngram_size=*/3);
  std::vector<int32_t> prompt = {1, 2, 3, 7, 8, 1, 2, 3, 9, 10};
  cache.start_request("req", std::span<const int32_t>(prompt));

  std::vector<int32_t> context = {1, 2, 3};
  PromptLookupDraft draft = cache.speculate("req",
                                            std::span<const int32_t>(context),
                                            /*max_spec_tokens=*/2,
                                            /*candidate_count=*/4,
                                            /*prefer_recent_match=*/false);

  EXPECT_EQ(draft.match_len, 3);
  EXPECT_EQ(draft.token_ids, std::vector<int32_t>({7, 8}));
}

TEST(PromptLookupCacheTest, CandidateCountOnePreservesLegacySelection) {
  PromptLookupCache cache(/*max_ngram_size=*/3, /*min_ngram_size=*/3);
  std::vector<int32_t> prompt = {1, 2, 3, 7, 8, 1, 2, 3, 9, 10};
  cache.start_request("req", std::span<const int32_t>(prompt));

  std::vector<int32_t> context = {1, 2, 3};
  PromptLookupDraft draft = cache.speculate("req",
                                            std::span<const int32_t>(context),
                                            /*max_spec_tokens=*/2,
                                            /*candidate_count=*/1,
                                            /*prefer_recent_match=*/false);

  EXPECT_EQ(draft.match_len, 3);
  EXPECT_EQ(draft.token_ids, std::vector<int32_t>({7, 8}));
}

TEST(PromptLookupCacheTest, ConsidersMatchesBeyondCandidateWindow) {
  PromptLookupCache cache(/*max_ngram_size=*/3, /*min_ngram_size=*/3);
  std::vector<int32_t> prompt = {1,  2,  3, 10, 11, 1,  2,  3, 12, 13, 1,  2, 3,
                                 14, 15, 1, 2,  3,  16, 17, 1, 2,  3,  18, 19};
  cache.start_request("req", std::span<const int32_t>(prompt));

  std::vector<int32_t> context = {1, 2, 3};
  PromptLookupDraft draft = cache.speculate("req",
                                            std::span<const int32_t>(context),
                                            /*max_spec_tokens=*/2,
                                            /*candidate_count=*/4,
                                            /*prefer_recent_match=*/true);

  EXPECT_EQ(draft.match_len, 3);
  EXPECT_EQ(draft.token_ids, std::vector<int32_t>({18, 19}));
}

TEST(PromptLookupCacheTest, ReturnsEmptyDraftForInvalidCandidateCount) {
  PromptLookupCache cache(/*max_ngram_size=*/3, /*min_ngram_size=*/3);
  std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
  cache.start_request("req", std::span<const int32_t>(prompt));

  std::vector<int32_t> context = {1, 2, 3};
  PromptLookupDraft draft = cache.speculate("req",
                                            std::span<const int32_t>(context),
                                            /*max_spec_tokens=*/2,
                                            /*candidate_count=*/0,
                                            /*prefer_recent_match=*/true);

  EXPECT_EQ(draft.match_len, 0);
  EXPECT_TRUE(draft.token_ids.empty());
}

TEST(PromptLookupCacheTest, ReuseScoreOnlyUsesDraftedTokens) {
  PromptLookupCache cache(/*max_ngram_size=*/3, /*min_ngram_size=*/3);
  std::vector<int32_t> prompt = {1, 2, 3, 4, 9, 4, 1, 2, 3, 5, 8, 5, 8};
  cache.start_request("req", std::span<const int32_t>(prompt));

  std::vector<int32_t> context = {1, 2, 3};
  PromptLookupDraft draft = cache.speculate("req",
                                            std::span<const int32_t>(context),
                                            /*max_spec_tokens=*/1,
                                            /*candidate_count=*/4,
                                            /*prefer_recent_match=*/false);

  EXPECT_EQ(draft.match_len, 3);
  EXPECT_EQ(draft.token_ids, std::vector<int32_t>({4}));
}

TEST(PromptLookupCacheTest, ReturnsEmptyDraftForNegativeCandidateCount) {
  PromptLookupCache cache(/*max_ngram_size=*/3, /*min_ngram_size=*/3);
  std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
  cache.start_request("req", std::span<const int32_t>(prompt));

  std::vector<int32_t> context = {1, 2, 3};
  PromptLookupDraft draft = cache.speculate("req",
                                            std::span<const int32_t>(context),
                                            /*max_spec_tokens=*/2,
                                            /*candidate_count=*/-1,
                                            /*prefer_recent_match=*/true);

  EXPECT_EQ(draft.match_len, 0);
  EXPECT_TRUE(draft.token_ids.empty());
}

}  // namespace xllm
