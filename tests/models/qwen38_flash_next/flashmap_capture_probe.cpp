#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <hip/hip_runtime_api.h>

#include "src/models/qwen38_flash_next/engine.hpp"

namespace qfn = gufo::models::qwen38_flash_next;

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::fprintf(stderr, "usage: %s MODEL_GGUF [TOKENS]\n", argv[0]);
    return 2;
  }
  const unsigned long requested = argc == 3 ? std::strtoul(argv[2], nullptr, 10)
                                             : 96UL;
  if (requested == 0 || requested > 4096) {
    std::fprintf(stderr, "TOKENS must be between 1 and 4096\n");
    return 2;
  }
  if (const hipError_t status = hipInit(0); status != hipSuccess) {
    std::fprintf(stderr, "HIP initialization failed: %s\n",
                 hipGetErrorString(status));
    return 1;
  }
  std::string error;
  qfn::ModelOptions options;
  options.max_context = static_cast<std::uint32_t>(requested);
  auto model = qfn::Model::Load(argv[1], options, &error);
  if (!model) {
    std::fprintf(stderr, "model load failed: %s\n", error.c_str());
    return 1;
  }
  const auto pattern = model->Tokenize("Implement a bounded ring buffer.");
  std::vector<std::int32_t> tokens(requested);
  for (std::size_t index = 0; index < tokens.size(); ++index)
    tokens[index] = pattern[index % pattern.size()];
  constexpr std::array<std::uint32_t, 6> kLayers{1, 12, 23, 34, 45, 47};
  std::vector<float> first;
  std::vector<float> second;
  std::vector<float> captured_logits;
  auto ordinary = model->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                       tokens.size(), &error);
  if (!ordinary || !ordinary->Sync(tokens, &error)) {
    std::fprintf(stderr, "ordinary prefill failed: %s\n", error.c_str());
    return 1;
  }
  const std::vector<float> ordinary_logits(ordinary->Logits().begin(),
                                           ordinary->Logits().end());
  if (!model->CaptureFlashMapFeatures(tokens, kLayers, &first, &error,
                                      &captured_logits) ||
      !model->CaptureFlashMapFeatures(tokens, kLayers, &second, &error)) {
    std::fprintf(stderr, "feature capture failed: %s\n", error.c_str());
    return 1;
  }
  const std::size_t expected =
      tokens.size() * kLayers.size() * model->config().hidden_size;
  const bool sizes = first.size() == expected && second.size() == expected;
  const bool deterministic =
      sizes && std::memcmp(first.data(), second.data(),
                           first.size() * sizeof(float)) == 0;
  const bool finite = std::ranges::all_of(
      first, [](float value) { return std::isfinite(value); });
  const bool nonzero = !std::ranges::all_of(
      first, [](float value) { return value == 0.0F; });
  const bool logits = captured_logits.size() == ordinary_logits.size() &&
                      std::memcmp(captured_logits.data(), ordinary_logits.data(),
                                  ordinary_logits.size() * sizeof(float)) == 0;
  if (!sizes || !deterministic || !finite || !nonzero || !logits) {
    std::size_t mismatch = 0;
    if (sizes && !deterministic) {
      while (mismatch < first.size() && first[mismatch] == second[mismatch])
        ++mismatch;
    }
    std::fprintf(stderr,
                 "captured feature contract mismatch sizes=%d deterministic=%d "
                 "finite=%d nonzero=%d logits=%d first=%zu second=%zu "
                 "captured_logits=%zu ordinary_logits=%zu expected=%zu "
                 "mismatch=%zu first_value=%g second_value=%g\n",
                 sizes, deterministic, finite, nonzero, logits, first.size(),
                 second.size(), captured_logits.size(), ordinary_logits.size(),
                 expected, mismatch, mismatch < first.size() ? first[mismatch] : 0,
                 mismatch < second.size() ? second[mismatch] : 0);
    return 1;
  }
  const auto hidden = model->config().hidden_size;
  const float* early = first.data();
  const float* late = first.data() + (kLayers.size() - 1) * hidden;
  if (std::equal(early, early + hidden, late)) {
    std::fprintf(stderr, "early and late target taps are unexpectedly equal\n");
    return 1;
  }
  std::printf("FLASHMAP_CAPTURE_OK tokens=%zu taps=%zu hidden=%u values=%zu\n",
              tokens.size(), kLayers.size(), model->config().hidden_size,
              first.size());
  return 0;
}
