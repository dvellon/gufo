#pragma once

// rocPRIM 4.6 asks __builtin_amdgcn_processor_is about processor names added
// after the native Windows ROCm 7.2 Clang 21 build. Clang diagnoses an
// unknown name while parsing the unused branch (currently gfx1250). Force
// rocPRIM's existing target-macro fallback while its include guard is first
// established; __GFX11__ still selects the exact RDNA3 behavior for gfx1151.
#if defined(_WIN32) && defined(__clang_major__) && __clang_major__ <= 21 && \
    __has_include(<rocprim/config.hpp>)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wbuiltin-macro-redefined"
#pragma push_macro("__has_builtin")
#undef __has_builtin
#define __has_builtin(...) 0
#include <rocprim/config.hpp>
#pragma pop_macro("__has_builtin")
#pragma clang diagnostic pop
#endif
