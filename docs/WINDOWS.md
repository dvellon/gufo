# Windows (native, gfx1151)

Gufo's qualified Windows path builds and runs natively on Windows 11 x64
against AMD's TheRock ROCm distribution for gfx1151 (Ryzen AI Max+ 395 /
Radeon 8060S). No WSL. Linux builds are unaffected: the port lives in
`compat/win32`, Windows-only files and `_WIN32` branches. AMD's native Windows
ROCm 7.2 SDK is also recognized by the build script, but that path is currently
experimental; see [Native AMD ROCm 7.2 SDK](#native-amd-rocm-72-sdk).

## Prerequisites

| Piece | Default location | Notes |
| --- | --- | --- |
| TheRock ROCm 10.1.0 (Windows, gfx1151) | active Python environment | Install AMD's official `rocm[devel,libraries,device-gfx1151]==10.1.0` wheels and run `rocm-sdk init`; `rocm-sdk path --root` locates the toolchain. Its clang compiles C, C++ and HIP |
| AMD display driver | system | Adrenalin for ROCm 10.1, driver-store `32.0.31041.5005` or newer. Older 31041 drivers can crash in `hipProfilerRegisterChunkCallbackExt` before the first kernel runs |
| vcpkg | `C:\vcpkg` | `vcpkg install icu curl openssl libpng libjpeg-turbo libwebp --triplet x64-windows` |
| Visual Studio Build Tools | any | "Desktop development with C++": MSVC STL + Windows SDK only; tested with VS 18 (MSVC 14.51) |
| CMake 3.21-3.31 and Ninja | active Python environment or `PATH` | 3.31.6 is validated; CMake 4.x currently mis-escapes the wheel toolchain's Windows paths |
| Dedicated GPU memory | AMD Software > Performance > Tuning > Variable Graphics Memory | 96 GB for Qwen3.8-Flash-Next at 256K context; see [Memory](#memory) |

The port is validated on TheRock 10.1.0. Other releases usually build, but a
newer clang can round fused kernels differently (see [Test status](#test-status)).

One isolated setup is:

```powershell
py -3.12 -m venv C:\rocm-10.1
C:\rocm-10.1\Scripts\python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ `
  "rocm[devel,libraries,device-gfx1151]==10.1.0" "cmake==3.31.6" "ninja==1.13.2"
C:\rocm-10.1\Scripts\rocm-sdk init
C:\rocm-10.1\Scripts\Activate.ps1
```

```powershell
powershell -ExecutionPolicy Bypass -File tools\windows\check.ps1   # what is missing, and how to fix it
powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1   # release -> build\release\gufo.exe
powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1 -Preset gpu-test   # + tests and tools
```

`build.ps1` imports the MSVC environment, discovers the active wheel toolchain
(or accepts `-Rocm`), configures with TheRock clang and the vcpkg toolchain,
builds, and copies the ROCm and vcpkg runtime DLLs plus the
hipBLASLt/rocBLAS kernel libraries next to `gufo.exe`, so `build\release` runs
as it is.

### Native AMD ROCm 7.2 SDK

The same script recognizes AMD's native Windows SDK layout, including clang
and the device bitcode installed under `C:\Program Files\AMD\ROCm\7.2`. Pass
the Ninja executable explicitly: the script's default is
`C:\tools\ninja\ninja.exe`, while package managers such as WinGet commonly
install it elsewhere.

```powershell
$ninja = (Get-Command ninja.exe -ErrorAction Stop).Source
powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1 `
  -Preset release `
  -BuildDir build\release-rocm72-native `
  -Target gufo_qwen38_flash_next `
  -Rocm 'C:\Program Files\AMD\ROCm\7.2' `
  -Ninja $ninja `
  -Jobs 4 `
  -Reconfigure
```

If ROCm package CMake files or runtime DLLs live outside the SDK root, pass
their roots with `-RocmDependencyRoots`. The script adds them to
`CMAKE_PREFIX_PATH` and searches them when copying runtime files. Passing a
wheel `_rocm_sdk_devel` directory also discovers sibling `_rocm_sdk_core` and
`_rocm_sdk_libraries` directories automatically. For example, when invoking
the script from PowerShell so the array remains typed:

```powershell
& .\tools\windows\build.ps1 `
  -Preset release `
  -BuildDir build\release-rocm72-native `
  -Target gufo_qwen38_flash_next `
  -Rocm 'C:\Program Files\AMD\ROCm\7.2' `
  -RocmDependencyRoots @('C:\path\to\_rocm_sdk_devel') `
  -Ninja (Get-Command ninja.exe -ErrorAction Stop).Source `
  -Jobs 4 `
  -Reconfigure
```

The native 7.2 path configures with AMD clang 21 and builds the
`gufo_qwen38_flash_next` library plus the FlashMap capture and target-verifier
executables for gfx1151. Windows resolves WebP through vcpkg's native
`WebP::webp` CMake package, so a separate host `pkg-config` executable is not
required. Real-model operator, equivalence, and performance qualification is
still pending; TheRock 10.1.0 remains the qualified Windows toolchain
described by the rest of this document.

## Running Qwen3.8-Flash-Next

Download the qualified files with the Hugging Face CLI
(`pip install -U huggingface_hub`):

```powershell
hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "UD-Q4_K_XL/*" "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" "mmproj-BF16.gguf"
powershell -ExecutionPolicy Bypass -File tools\windows\run-flash-next.ps1
```

`run-flash-next.ps1` finds the files in the Hugging Face cache and serves an
OpenAI-compatible API on `http://127.0.0.1:8080/v1` (thinking on, adaptive MTP,
the model's own sampler). `-Think off`, `-Draft mtp3|off`, `-Context N` and
`-Mode bench` (pp2048/tg128 at depths 0..128K) are the common variations. The
server is ready when the log shows `event=load_completed`.

Gufo reads `GUFO_*` environment variables as diagnostic switches; the run
script clears any left in the calling shell.

## How the port works

- `compat/win32/include` shadows the POSIX headers Gufo uses (`unistd.h`,
  `sys/mman.h`, `fcntl.h` additions, sockets, `spawn.h`, ...) and
  `compat/win32/posix.cpp` implements them on Win32. The directory is on the
  include path only on Windows. `_CRT_DECLARE_NONSTDC_NAMES=0` keeps the
  UCRT's own POSIX aliases out of the way.
- Semantics that matter:
  - `O_DIRECT` is `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED`, so the
    weight uploader keeps its parallel aligned direct reads.
  - NTFS serializes direct reads of a file that is also mapped, and Gufo maps
    every model shard: 32 threads of 4 KiB direct reads fall from ~70K to ~5K
    reads/s. Large sequential reads barely notice, but the Flash-Next n-gram
    table reader does ~16 random row reads per prompt token. On Windows it
    opens the table with `O_CONCURRENT_RANDOM` (a cached overlapped handle,
    68K reads/s at 32 readers, 128K at 128) and runs 128 readers. That took
    real-text prefill from ~300 to ~680-840 tokens/s.
  - `open("/proc/self/fd/N")` is `ReOpenFile`: an independent handle to the
    same file, immune to path replacement, exactly as on Linux.
  - `flock` locks one byte far past EOF (Windows locks are mandatory) and
    `close` unlocks explicitly (Windows releases locks asynchronously).
  - `renameat` uses POSIX-semantics rename; `st_ctime` is NTFS ChangeTime.
  - `posix_spawnp` passes descriptors through the CRT `lpReserved2` block, so
    FFmpeg's `pipe:N` inputs work.
  - Uncaught C++ exceptions and crashes print a Linux-style message (with
    `what()`) or `module+offset` for `llvm-symbolizer`.
- `src/core/platform/socket.hpp` covers the socket differences call sites
  cannot hide: `closesocket`, Winsock errors to `errno`, millisecond
  `SO_RCVTIMEO`, lingering close (Windows discards data queued before a reset),
  and polled websocket reads (a Winsock `SD_RECEIVE` resets the connection).
- `compat/win32/hip_include/cmath` works around MSVC 14.5x declaring
  `isgreater` & co. `constexpr` (hence `__host__ __device__` under HIP), which
  collides with clang's HIP math declarations.
- Executables reserve an 8 MiB stack, Linux's default (Windows: 1 MiB).

## Memory

On Windows, HIP allocations are not limited to the dedicated carve-out that
the BIOS or AMD Software reserves; `hipMalloc` succeeds up to ~110 GiB on a
128 GB machine, with the rest backed by shared system memory. Measured on a
128 GB Strix Halo:

- `hipMemGetInfo` double-counts memory beyond the carve-out (free drops 2 GiB
  per GiB allocated), which made Gufo refuse to size its sessions.
  `src/core/platform/device_memory.cpp` reports the process's WDDM budget
  (DXGI `QueryVideoMemoryInfo`, adapter matched by the HIP device LUID)
  instead.
- Backing new GPU memory costs ~28 GiB/s inside the carve-out but only
  ~0.63 GiB/s beyond it (Windows commits and maps system RAM page by page),
  which also halves concurrent disk reads during the load.
- Weights beyond the carve-out are also slower to read: the decode
  matrix-vector kernels run 6-22% slower on them.

Qwen3.8-Flash-Next at 256K context needs ~89 GiB of device memory (weights,
KV cache, recurrent state, scratch). With a 96 GB carve-out all of it is
dedicated: the load takes ~30 s and decode runs at full speed. A 64 GB
carve-out works, but the load takes ~2 minutes and decode is slower.

## Test status

- All Flash-Next operator tests pass, plus 86/92 CPU-labelled tests.
- Not ported: Linux-only Python dev tools (`h3_profile*`, `qwen27b.tools`,
  TTS reference verification needs numpy), and `video_jobs_test` needs
  symlink privileges (Developer Mode).
- `qwen_ssm_ops_test` and `minimax_h3_dit_analytic` assert bit-identical
  fused vs. reference kernels; TheRock's newer clang rounds differently.
