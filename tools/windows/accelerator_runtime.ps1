# Shared Windows runtime preparation for Gufo HIP and XDNA2 work.
#
# This file is dot-sourced by the checked-in launchers.  Keep the policy here
# so a reboot or a manually started Ollama instance cannot silently change the
# accelerator environment used by a benchmark or serving run.

Set-StrictMode -Version Latest

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
    $probeIdentity = (Get-FileHash -Algorithm SHA256 -LiteralPath $HipProbePath).Hash.ToLowerInvariant()
  }

  return [pscustomobject]@{
    RepositoryRoot = $root
    RuntimeTemp = $runtimeTemp
    OllamaStopped = $stoppedOllama
    HipProbeSha256 = $probeIdentity
  }
}
