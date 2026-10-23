#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

#if GUFO_HIP_HAS_WMMA_WAVE32
#include "src/models/qwen/hip/kernels/prefill_quant_gemm.hpp"
#endif

namespace gufo::models::qwen38_flash_next::rocm {

hipError_t PrepareW8A8GemmWave64() {
#if GUFO_HIP_HAS_WMMA_WAVE32
  hipFuncAttributes attributes{};
  return hipFuncGetAttributes(
      &attributes, reinterpret_cast<const void*>(
                       gufo::hip::WKQuantA8BlockedWmmaGEMMKernel<
                           128, 128, 2, 4, 1, core::GgmlType::kQ8_0, 64>));
#else
  return hipErrorNotSupported;
#endif
}

void W8A8GemmWave64(const void* w, const void* x_tiled, float* out,
                    std::size_t batch, std::size_t m, std::size_t k,
                    hipStream_t stream) {
#if GUFO_HIP_HAS_WMMA_WAVE32
  const dim3 grid(static_cast<unsigned>((batch + 127) / 128),
                  static_cast<unsigned>((m + 127) / 128));
  // Keep Flash-Next's compilation and scale order. This geometry is distinct
  // from Qwen27B's Q8 instantiation, so each retains its qualified schedule.
  gufo::hip::WKQuantA8BlockedWmmaGEMMKernel<128, 128, 2, 4, 1,
                                            core::GgmlType::kQ8_0, 64>
      <<<grid, 256, 0, stream>>>(w, x_tiled, out, batch, m, k);
#else
  // This entry point is unreachable because CDNA3 disables the wide-row
  // route at compile time. Retain a fail-closed definition for link safety.
  (void)w;
  (void)x_tiled;
  (void)out;
  (void)batch;
  (void)m;
  (void)k;
  (void)stream;
#endif
}

}  // namespace gufo::models::qwen38_flash_next::rocm
