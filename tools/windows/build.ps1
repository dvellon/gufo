# Windows build of Gufo against AMD TheRock ROCm (gfx1151).
#
# Prerequisites (paths overridable by parameter; tools\windows\check.ps1
# checks them all):
#   - ROCm 10.1.0 for Windows gfx1151 from AMD's official Python wheels;
#     activate that environment and run `rocm-sdk init` once.
#   - vcpkg at C:\vcpkg with: icu curl openssl libpng libjpeg-turbo
#     libwebp (x64-windows)
#   - Visual Studio Build Tools (MSVC STL + Windows SDK; the compiler is TheRock clang)
#   - CMake 3.21+, and ninja on PATH or at C:\tools\ninja\ninja.exe
#
#   powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1     # release -> build\release\gufo.exe
#   powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1 -Preset gpu-test -Target <target>
param(
  [string]$Preset = "release",
  [string]$BuildDir = "",
  [string]$Target = "",
  [string]$Rocm = "",
  [string[]]$RocmDependencyRoots = @(),
  [string]$Vcpkg = "C:\vcpkg",
  [string]$Ninja = "C:\tools\ninja\ninja.exe",
  [int]$Jobs = 0,
  [switch]$Reconfigure,
  [switch]$KeepGoing,
  [string[]]$CMakeArgs = @()
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..\..").Path

if (-not $Rocm) {
  $rocmSdk = Get-Command rocm-sdk -ErrorAction SilentlyContinue
  $Rocm = if ($rocmSdk) { (& $rocmSdk.Source path --root).Trim() } else { "C:\TheRock\build" }
}

# The port is validated on one TheRock release; a newer clang can round the
# fused kernels differently (see docs\WINDOWS.md).
$validatedRock = "10.1.0"
$rockVersionFile = Join-Path $Rocm ".info\version"
$rockVersion = if (Test-Path $rockVersionFile) {
  (Get-Content $rockVersionFile -TotalCount 1).Trim()
} elseif ((Test-Path (Join-Path $Rocm "bin\hipcc.exe")) -and
          (Test-Path (Join-Path $Rocm "bin\amdhip64_7.dll"))) {
  "native-windows-install"
} else {
  throw "ROCm SDK not found at $Rocm (see docs\WINDOWS.md, or run tools\windows\check.ps1)"
}
if ($rockVersion -ne $validatedRock) {
  Write-Warning "ROCm $rockVersion at $Rocm; this port is validated on TheRock $validatedRock"
}

# Import the MSVC developer environment (headers, libs, link.exe).
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "Visual Studio C++ build tools not found" }
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
$env:PATH = "$(Split-Path $vswhere);$env:PATH"
cmd /c "`"$vcvars`" >nul && set" | ForEach-Object {
  if ($_ -match "^([^=]+)=(.*)$") { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
}

$compilerDir = @((Join-Path $Rocm "lib\llvm\bin"), (Join-Path $Rocm "bin")) |
  Where-Object { Test-Path (Join-Path $_ "clang++.exe") } |
  Select-Object -First 1
$deviceLibDir = @((Join-Path $Rocm "lib\llvm\amdgcn\bitcode"),
                  (Join-Path $Rocm "amdgcn\bitcode")) |
  Where-Object { Test-Path (Join-Path $_ "ocml.bc") } |
  Select-Object -First 1
if (-not $compilerDir -or -not $deviceLibDir) {
  throw "ROCm compiler or device bitcode not found below $Rocm"
}

$env:HIP_PATH = $Rocm.Replace('\', '/')
$env:HIP_DEVICE_LIB_PATH = $deviceLibDir.Replace('\', '/')
$env:PATH = "$Rocm\bin;$compilerDir;$(Split-Path $Ninja);$env:PATH"
$clang = (Join-Path $compilerDir "clang.exe").Replace('\', '/')
$clangxx = (Join-Path $compilerDir "clang++.exe").Replace('\', '/')
$rocmPrefixPath = (@($Rocm) + $RocmDependencyRoots) -join ";"

$build = if ($BuildDir) {
  if ([System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir } else { Join-Path $root $BuildDir }
} else {
  Join-Path $root "build\$Preset"
}
if ($Reconfigure -and (Test-Path "$build\CMakeCache.txt")) {
  Remove-Item "$build\CMakeCache.txt"
}
if (-not (Test-Path "$build\CMakeCache.txt")) {
  cmake -S $root --preset $Preset -B $build `
    "-DCMAKE_MAKE_PROGRAM=$($Ninja.Replace('\', '/'))" `
    "-DCMAKE_C_COMPILER=$clang" `
    "-DCMAKE_CXX_COMPILER=$clangxx" `
    "-DCMAKE_HIP_COMPILER=$clangxx" `
    "-DCMAKE_HIP_COMPILER_ROCM_ROOT=$($Rocm.Replace('\', '/'))" `
    "-DCMAKE_TOOLCHAIN_FILE=$($Vcpkg.Replace('\', '/'))/scripts/buildsystems/vcpkg.cmake" `
    "-DVCPKG_TARGET_TRIPLET=x64-windows" `
    "-DCMAKE_PREFIX_PATH=$($rocmPrefixPath.Replace('\', '/'))" `
    @CMakeArgs
  if ($LASTEXITCODE -ne 0) { throw "configure failed" }
}

$buildArgs = @("--build", $build)
if ($Target) { $buildArgs += @("--target", $Target) }
if ($Jobs -gt 0) { $buildArgs += @("--parallel", $Jobs) }
if ($KeepGoing) { $buildArgs += @("--", "-k", "0") }
cmake @buildArgs
if ($LASTEXITCODE -ne 0) { throw "build failed" }

# Runtime DLLs next to gufo.exe: ROCm (amdhip64, hipblas, hipblaslt, rocblas
# and their kernel libraries) and vcpkg's (icu, curl, ssl, png, jpeg, zlib).
# The 10.1 wheel splits runtime DLLs between sibling core, libraries and devel
# packages, while extracted TheRock distributions keep them under one root.
$bin = $build
$rocmRoots = @()
foreach ($candidate in $RocmDependencyRoots) {
  $rocmRoots += $candidate
  if ((Split-Path $candidate -Leaf) -eq "_rocm_sdk_devel") {
    $packageRoot = Split-Path $candidate -Parent
    $rocmRoots += @("_rocm_sdk_core", "_rocm_sdk_libraries") |
      ForEach-Object { Join-Path $packageRoot $_ } |
      Where-Object { Test-Path $_ }
  }
}
$rocmRoots += $Rocm
if ((Split-Path $Rocm -Leaf) -eq "_rocm_sdk_devel") {
  $packageRoot = Split-Path $Rocm -Parent
  $rocmRoots += @("_rocm_sdk_core", "_rocm_sdk_libraries") |
    ForEach-Object { Join-Path $packageRoot $_ } |
    Where-Object { Test-Path $_ }
}
$runtimeBins = $rocmRoots | ForEach-Object {
  @((Join-Path $_ "bin"), (Join-Path $_ "lib\rocm_sysdeps\bin"))
} | Where-Object { Test-Path $_ } | Select-Object -Unique
foreach ($runtimeBin in $runtimeBins) {
  Get-ChildItem "$runtimeBin\*.dll" -ErrorAction SilentlyContinue | Copy-Item -Destination $bin -Force
  foreach ($dir in @("hipblaslt", "rocblas")) {
    $libraryDir = Join-Path $runtimeBin $dir
    if (Test-Path $libraryDir) {
      Copy-Item $libraryDir -Destination $bin -Recurse -Force
    }
  }
}
Get-ChildItem "$Vcpkg\installed\x64-windows\bin\*.dll" | Copy-Item -Destination $bin -Force
Write-Host "gufo: built $build"
