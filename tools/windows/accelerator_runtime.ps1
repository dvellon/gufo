# Shared Windows runtime preparation for Gufo HIP and XDNA2 work.
#
# This file is dot-sourced by the checked-in launchers.  Keep the policy here
# so a reboot or a manually started Ollama instance cannot silently change the
# accelerator environment used by a benchmark or serving run.

Set-StrictMode -Version Latest

function Get-GufoFileSha256 {
  param([Parameter(Mandatory = $true)] [string]$Path)
  return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Resolve-GufoApplication {
  param([Parameter(Mandatory = $true)] [string]$Executable)
  if (Test-Path -LiteralPath $Executable -PathType Leaf) {
    return (Resolve-Path -LiteralPath $Executable).Path
  }
  $command = Get-Command -Name $Executable -CommandType Application -ErrorAction Stop |
    Select-Object -First 1
  return $command.Source
}

function Test-GufoDirectHipImport {
  param([Parameter(Mandatory = $true)] [string]$Executable)
  # Import-table DLL names are stored as ASCII.  This conservative marker scan
  # avoids binding the launcher to Visual Studio's dumpbin or an SDK-specific
  # PE parser.  A false positive stages an executable unnecessarily; a direct
  # HIP import cannot be missed without corrupting the PE import name itself.
  $bytes = [System.IO.File]::ReadAllBytes($Executable)
  $text = [System.Text.Encoding]::ASCII.GetString($bytes)
  return $text.IndexOf("amdhip64_7.dll", [StringComparison]::OrdinalIgnoreCase) -ge 0
}

function New-GufoVerifiedHardLink {
  param(
    [Parameter(Mandatory = $true)] [string]$Source,
    [Parameter(Mandatory = $true)] [string]$Destination,
    [Parameter(Mandatory = $true)] [string]$ExpectedSha256
  )
  if (-not (Test-Path -LiteralPath $Destination)) {
    try {
      New-Item -ItemType HardLink -Path $Destination -Target $Source -ErrorAction Stop |
        Out-Null
    } catch {
      # File.Copy with overwrite=false retains the same no-replace contract on
      # filesystems that do not support hard links.
      [System.IO.File]::Copy($Source, $Destination, $false)
    }
  }
  if (-not (Test-Path -LiteralPath $Destination -PathType Leaf)) {
    throw "accelerator stage entry is not a regular file: $Destination"
  }
  $actual = Get-GufoFileSha256 -Path $Destination
  if ($actual -ne $ExpectedSha256) {
    throw "accelerator stage digest mismatch: $Destination"
  }
}

function New-GufoAcceleratorExecutionStage {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory = $true)] [string]$RepositoryRoot,
    [Parameter(Mandatory = $true)] [string]$TargetExecutable,
    [Parameter(Mandatory = $true)] [string]$HipProbePath,
    [string]$ReleaseRoot = ""
  )
  $root = (Resolve-Path -LiteralPath $RepositoryRoot).Path
  $target = Resolve-GufoApplication -Executable $TargetExecutable
  $probe = Resolve-GufoApplication -Executable $HipProbePath
  if (-not $ReleaseRoot) {
    $ReleaseRoot = Join-Path $root "build\release-rocm101-upstream"
  }
  $release = (Resolve-Path -LiteralPath $ReleaseRoot).Path
  $runtimeFiles = @(Get-ChildItem -LiteralPath $release -File -Filter "*.dll" |
    Sort-Object -Property Name)
  if ($runtimeFiles.Count -eq 0) {
    throw "accelerator release runtime has no DLLs: $release"
  }
  $hipRuntime = $runtimeFiles | Where-Object { $_.Name -ieq "amdhip64_7.dll" }
  if (@($hipRuntime).Count -ne 1) {
    throw "accelerator release runtime must contain exactly one amdhip64_7.dll"
  }

  $entries = @()
  foreach ($file in $runtimeFiles) {
    $entries += [pscustomobject]@{
      RelativePath = $file.Name
      Source = $file.FullName
      Sha256 = Get-GufoFileSha256 -Path $file.FullName
    }
  }
  # rocBLAS and hipBLASLt locate architecture-specific code objects relative
  # to their loaded DLL.  Stage those immutable resources with the DLLs or the
  # gate would prove only hipInit while the target fails during model load.
  foreach ($resourceRoot in @("hipblaslt", "rocblas")) {
    $resourcePath = Join-Path $release $resourceRoot
    if (Test-Path -LiteralPath $resourcePath -PathType Container) {
      foreach ($file in Get-ChildItem -LiteralPath $resourcePath -File -Recurse |
                 Sort-Object -Property FullName) {
        $relative = [System.IO.Path]::GetRelativePath($release, $file.FullName)
        $entries += [pscustomobject]@{
          RelativePath = $relative
          Source = $file.FullName
          Sha256 = Get-GufoFileSha256 -Path $file.FullName
        }
      }
    }
  }
  $entries = @($entries | Sort-Object -Property RelativePath)
  $runtimeManifest = ($entries | ForEach-Object {
    "{0}  {1}" -f $_.Sha256, $_.RelativePath
  }) -join "`n"
  $runtimeBytes = [System.Text.Encoding]::UTF8.GetBytes($runtimeManifest)
  $runtimeHasher = [System.Security.Cryptography.SHA256]::Create()
  try {
    $runtimeSha = ([BitConverter]::ToString(
      $runtimeHasher.ComputeHash($runtimeBytes))).Replace("-", "").ToLowerInvariant()
  } finally {
    $runtimeHasher.Dispose()
  }
  $targetSha = Get-GufoFileSha256 -Path $target
  $probeSha = Get-GufoFileSha256 -Path $probe
  $stage = Join-Path $root (
    "build\gufo-accelerator-staging\v1-{0}-{1}-{2}" -f `
      $runtimeSha.Substring(0, 16), $targetSha.Substring(0, 16),
      $probeSha.Substring(0, 16))
  [System.IO.Directory]::CreateDirectory($stage) | Out-Null

  foreach ($entry in $entries) {
    $destination = Join-Path $stage $entry.RelativePath
    [System.IO.Directory]::CreateDirectory(
      [System.IO.Path]::GetDirectoryName($destination)) | Out-Null
    New-GufoVerifiedHardLink -Source $entry.Source `
      -Destination $destination `
      -ExpectedSha256 $entry.Sha256
  }
  $stagedTarget = Join-Path $stage "target.exe"
  $stagedProbe = Join-Path $stage "hip-gate.exe"
  New-GufoVerifiedHardLink -Source $target -Destination $stagedTarget `
    -ExpectedSha256 $targetSha
  New-GufoVerifiedHardLink -Source $probe -Destination $stagedProbe `
    -ExpectedSha256 $probeSha

  $expectedNames = @($entries.RelativePath) + @("hip-gate.exe", "target.exe")
  $actualNames = @(Get-ChildItem -LiteralPath $stage -Force -File -Recurse |
    ForEach-Object { [System.IO.Path]::GetRelativePath($stage, $_.FullName) } |
    Sort-Object)
  $expectedNames = @($expectedNames | Sort-Object)
  if (($actualNames -join "`n") -cne ($expectedNames -join "`n")) {
    throw "accelerator stage inventory mismatch: $stage"
  }
  return [pscustomobject]@{
    Directory = $stage
    TargetExecutable = $stagedTarget
    HipProbePath = $stagedProbe
    RuntimeManifestSha256 = $runtimeSha
    TargetSha256 = $targetSha
    HipProbeSha256 = $probeSha
  }
}

function Get-GufoOllamaProcesses {
  return @(Get-Process -ErrorAction SilentlyContinue | Where-Object {
    $_.ProcessName -like "ollama*"
  })
}

function Stop-GufoOllamaProcesses {
  $processes = @(Get-GufoOllamaProcesses)
  if ($processes.Count -eq 0) {
    return $false
  }

  $processes | Stop-Process -Force -ErrorAction Stop
  $deadline = [DateTime]::UtcNow.AddSeconds(10)
  do {
    Start-Sleep -Milliseconds 100
    $remaining = @(Get-GufoOllamaProcesses)
    if ($remaining.Count -gt 0) {
      $remaining | Stop-Process -Force -ErrorAction Stop
    }
  } while ($remaining.Count -gt 0 -and [DateTime]::UtcNow -lt $deadline)

  $remaining = @(Get-GufoOllamaProcesses)

  if ($remaining.Count -gt 0) {
    $identities = $remaining | ForEach-Object { "$($_.ProcessName):$($_.Id)" }
    throw "Ollama processes did not quiesce: $($identities -join ', ')"
  }
  return $true
}

function Initialize-GufoAcceleratorRuntime {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory = $true)] [string]$RepositoryRoot,
    [string]$HipProbePath = "",
    [string]$TargetExecutable = "",
    [switch]$SkipHipProbe
  )

  $root = (Resolve-Path -LiteralPath $RepositoryRoot).Path
  $stoppedOllama = Stop-GufoOllamaProcesses

  $runtimeTemp = Join-Path $root "build\gufo-runtime-temp"
  New-Item -ItemType Directory -Path $runtimeTemp -Force | Out-Null
  $runtimeTemp = (Resolve-Path -LiteralPath $runtimeTemp).Path
  $env:TEMP = $runtimeTemp
  $env:TMP = $runtimeTemp
  $env:TMPDIR = $runtimeTemp
  $env:GUFO_RUNTIME_TEMP = $runtimeTemp

  $probeIdentity = $null
  $runtimeManifestIdentity = $null
  $staged = $false
  $resolvedTarget = $TargetExecutable
  if ($TargetExecutable -and
      (Test-GufoDirectHipImport -Executable (
        Resolve-GufoApplication -Executable $TargetExecutable))) {
    if (-not $HipProbePath) {
      $HipProbePath = Join-Path $root "build\release-rocm101-upstream\hip_first_kernel_repro.exe"
    }
    $execution = New-GufoAcceleratorExecutionStage `
      -RepositoryRoot $root `
      -TargetExecutable $TargetExecutable `
      -HipProbePath $HipProbePath
    $resolvedTarget = $execution.TargetExecutable
    $HipProbePath = $execution.HipProbePath
    $runtimeManifestIdentity = $execution.RuntimeManifestSha256
    $staged = $true
  }
  if (-not $SkipHipProbe) {
    if (-not $HipProbePath) {
      $HipProbePath = Join-Path $root "build\release-rocm101-upstream\hip_first_kernel_repro.exe"
    }
    if (-not (Test-Path -LiteralPath $HipProbePath -PathType Leaf)) {
      throw "HIP health probe not found: $HipProbePath"
    }
    $HipProbePath = (Resolve-Path -LiteralPath $HipProbePath).Path
    $probeOutput = @(& $HipProbePath 2>&1)
    if ($LASTEXITCODE -ne 0) {
      $probeOutput | ForEach-Object { Write-Host $_ }
      throw "HIP health probe failed with exit code $LASTEXITCODE"
    }
    $probeIdentity = Get-GufoFileSha256 -Path $HipProbePath
  }

  return [pscustomobject]@{
    RepositoryRoot = $root
    RuntimeTemp = $runtimeTemp
    OllamaStopped = $stoppedOllama
    HipProbeSha256 = $probeIdentity
    RuntimeManifestSha256 = $runtimeManifestIdentity
    RuntimeStaged = $staged
    TargetExecutable = $resolvedTarget
  }
}
