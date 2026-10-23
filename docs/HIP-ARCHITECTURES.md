# HIP architecture capabilities

Gufo builds one HIP architecture at a time. Architecture selection is a
compile-time contract, not a runtime promise that every model supports every
AMD GPU.

| Architecture | Family | Wave size | Matrix capability | Status |
|---|---|---:|---|---|
| `gfx1151` | Strix Halo | 32 | WMMA wave32 | Production |
| `gfx942` | CDNA3 / MI300 | 64 | MFMA wave64 | Experimental, Qwen3.8 FlashMap only |

The default remains `gfx1151`. A gfx942 build must set both
`CMAKE_HIP_ARCHITECTURES=gfx942` and
`GUFO_EXPERIMENTAL_GFX942_FLASHMAP=ON`; otherwise configuration fails. Multiple
architectures in one build are rejected because wave-size and matrix-instruction
assumptions are compile-time properties of the kernels.

## Target boundaries

Portable model foundations, Qwen tokenization, shared HIP services, and model
runtimes are separate CMake targets. A model runtime declares the architecture
capabilities it uses instead of acquiring unrelated kernels through
`gufo_core`.

The experimental gfx942 proof target is
`qwen38_flash_next_flashmap_capture`. It compiles only the Qwen3.8 host model,
tokenization, shared sampling/hipBLASLt services, Qwen3.8 runtime, and its
model-scoped MMQ kernels. Other Qwen, MiniMax, image, and production server
kernels are not part of that target's gfx942 closure.

The gfx942 backend preserves exact-output behavior by routing away from the
gfx1151-only wide-row WMMA kernels. It currently uses Gufo's existing vector,
MMQ, and hipBLAS fallbacks. These are correctness fallbacks, not a performance
claim. CDNA3 MFMA implementations can replace operations one at a time after
each has its own parity and performance qualification.

## Configure and build

```sh
cmake -S . -B build/gfx942-flashmap \
  -DENGINE_ENABLE_HIP=ON \
  -DGUFO_EXPERIMENTAL_GFX942_FLASHMAP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DGUFO_BUILD_TOOLS=ON \
  -DBUILD_TESTING=OFF
cmake --build build/gfx942-flashmap \
  --target qwen38_flash_next_flashmap_capture
```

Building the executable proves source and link closure for gfx942. Runtime
qualification still requires an actual gfx942 device and exact-output checks
against the pinned producer configuration.
