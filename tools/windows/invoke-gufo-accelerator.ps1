# Run an arbitrary command under Gufo's conflict-free Windows accelerator
# policy. Example:
#
# powershell -ExecutionPolicy Bypass -File tools\windows\invoke-gufo-accelerator.ps1 `
#   -Executable python -ArgumentList @('tools/gufo/npu_vitisai_probe.py')

param(
  [Parameter(Mandatory = $true)] [string]$Executable,
  [string[]]$ArgumentList = @(),
  [string]$HipProbePath = "",
  [switch]$SkipHipProbe
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
. (Join-Path $PSScriptRoot "accelerator_runtime.ps1")
$runtime = Initialize-GufoAcceleratorRuntime `
  -RepositoryRoot $root `
  -HipProbePath $HipProbePath `
  -TargetExecutable $Executable `
  -SkipHipProbe:$SkipHipProbe

Write-Host (
  "GUFO_ACCELERATOR_RUNTIME_READY temp={0} ollama_stopped={1} hip_probe_sha256={2} runtime_staged={3} runtime_manifest_sha256={4}" -f `
    $runtime.RuntimeTemp, `
    $runtime.OllamaStopped.ToString().ToLowerInvariant(), `
    $(if ($runtime.HipProbeSha256) { $runtime.HipProbeSha256 } else { "skipped" }), `
    $runtime.RuntimeStaged.ToString().ToLowerInvariant(), `
    $(if ($runtime.RuntimeManifestSha256) { $runtime.RuntimeManifestSha256 } else { "not-applicable" })
)

# A non-zero native exit is data this wrapper must propagate verbatim, not a
# PowerShell terminating error that collapses to exit 1 under newer hosts.
$PSNativeCommandUseErrorActionPreference = $false
& $runtime.TargetExecutable @ArgumentList
$targetExitCode = $LASTEXITCODE
if ($null -ne $targetExitCode) {
  exit $targetExitCode
}
