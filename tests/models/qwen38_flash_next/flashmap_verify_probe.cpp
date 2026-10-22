#include <algorithm>
#include <array>
#include <cstdio>
#include <string>
#include <vector>

#include <hip/hip_runtime_api.h>

#include "src/models/qwen38_flash_next/engine.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace sampling = gufo::sampling;

namespace {

bool EqualSuffix(std::span<const std::int32_t> actual,
                 std::span<const std::int32_t> prefix,
                 std::span<const std::int32_t> suffix) {
  return actual.size() == prefix.size() + suffix.size() &&
         std::equal(prefix.begin(), prefix.end(), actual.begin()) &&
         std::equal(suffix.begin(), suffix.end(),
                    actual.begin() + static_cast<std::ptrdiff_t>(prefix.size()));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s MODEL_GGUF\n", argv[0]);
    return 2;
  }
  if (const hipError_t status = hipInit(0); status != hipSuccess) {
    std::fprintf(stderr, "HIP initialization failed: %s\n",
                 hipGetErrorString(status));
    return 1;
  }

  std::string error;
  qfn::ModelOptions options;
  options.max_context = 256;
  options.max_draft_tokens = 7;
  auto model = qfn::Model::Load(argv[1], options, &error);
  if (!model) {
    std::fprintf(stderr, "model load failed: %s\n", error.c_str());
    return 1;
  }
  if (model->HasMtp()) {
    std::fprintf(stderr, "probe must validate verification without MTP\n");
    return 1;
  }
  if (model->ExternalDraftSessionBytes(256) <=
      model->SessionBytes(gufo::core::SessionMode::kAutoregressive, 256)) {
    std::fprintf(stderr, "external-draft admission omits rollback storage\n");
    return 1;
  }
  const auto prompt = model->Tokenize("Write one safe bounds check in C++.");
  std::vector<sampling::TokenId> history(prompt.begin(), prompt.end());
  sampling::SamplingConfig greedy_config;
  greedy_config.temperature = 0.0F;
  constexpr std::size_t kBlock = 8;
  auto reference_session = model->CreateSession(
      gufo::core::SessionMode::kAutoregressive, 256, &error);
  sampling::SamplerState reference_sampler(greedy_config, history);
  if (!reference_session || !reference_session->Sync(prompt, &error)) {
    std::fprintf(stderr, "reference prefill failed: %s\n", error.c_str());
    return 1;
  }
  std::array<std::int32_t, kBlock> reference{};
  for (auto& token : reference) {
    qfn::Session::DecodeResult result;
    if (!reference_session->DecodeStep(1, reference_sampler, &result, &error,
                                       false) ||
        result.tokens.size() != 1) {
      std::fprintf(stderr, "reference decode failed: %s\n", error.c_str());
      return 1;
    }
    token = result.tokens.front();
  }

  const auto make_drafts = [&] {
    std::array<std::int32_t, kBlock - 1> drafts{};
    std::copy(reference.begin() + 1, reference.end(), drafts.begin());
    return drafts;
  };
  for (std::size_t mismatch = 0; mismatch < kBlock - 1; ++mismatch) {
    auto session = model->CreateSession(
        gufo::core::SessionMode::kAutoregressive, 256, &error);
    sampling::SamplerState sampler(greedy_config, history);
    if (!session || !session->Sync(prompt, &error)) {
      std::fprintf(stderr, "mismatch prefill failed: %s\n", error.c_str());
      return 1;
    }
    auto drafts = make_drafts();
    drafts[mismatch] =
        (drafts[mismatch] + 1) % static_cast<std::int32_t>(model->VocabSize());
    qfn::Session::DecodeResult result;
    if (!session->DecodeDraftStep(kBlock, drafts, sampler, &result, &error,
                                  false)) {
      std::fprintf(stderr, "draft verification failed: %s\n", error.c_str());
      return 1;
    }
    const auto kept = mismatch + 1;
    if (result.tokens.size() != kept ||
        !std::equal(result.tokens.begin(), result.tokens.end(),
                    reference.begin()) ||
        !EqualSuffix(session->Tokens(), prompt,
                     std::span(reference).first(kept)) ||
        session->Statistics().drafted != kBlock - 1 ||
        session->Statistics().accepted != mismatch ||
        session->Statistics().lookup != 0 ||
        session->Statistics().lookup_accepted != 0) {
      std::fprintf(stderr,
                   "mismatch rollback differs position=%zu kept=%zu accepted=%llu\n",
                   mismatch, result.tokens.size(),
                   static_cast<unsigned long long>(
                       session->Statistics().accepted));
      return 1;
    }
    qfn::Session::DecodeResult next;
    if (!session->DecodeStep(1, sampler, &next, &error, false) ||
        next.tokens.size() != 1 || next.tokens.front() != reference[kept]) {
      std::fprintf(stderr,
                   "rollback frontier differs position=%zu error=%s\n",
                   mismatch, error.c_str());
      return 1;
    }
  }

  auto accepted = model->CreateSession(
      gufo::core::SessionMode::kAutoregressive, 256, &error);
  sampling::SamplerState accepted_sampler(greedy_config, history);
  const auto drafts = make_drafts();
  qfn::Session::DecodeResult result;
  if (!accepted || !accepted->Sync(prompt, &error) ||
      !accepted->DecodeDraftStep(kBlock, drafts, accepted_sampler, &result,
                                 &error, false) ||
      result.tokens.size() != kBlock ||
      !std::equal(result.tokens.begin(), result.tokens.end(),
                  reference.begin()) ||
      !EqualSuffix(accepted->Tokens(), prompt, reference) ||
      accepted->Statistics().drafted != kBlock - 1 ||
      accepted->Statistics().accepted != kBlock - 1 ||
      accepted->Statistics().lookup != 0 ||
      accepted->Statistics().lookup_accepted != 0) {
    std::fprintf(stderr, "full draft acceptance differs: %s\n", error.c_str());
    return 1;
  }

  std::printf("FLASHMAP_VERIFY_OK mismatches=7 full_accept=7 mtp=absent\n");
  return 0;
}
