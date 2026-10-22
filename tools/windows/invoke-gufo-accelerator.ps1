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
  -SkipHipProbe:$SkipHipProbe

Write-Host (
  "GUFO_ACCELERATOR_RUNTIME_READY temp={0} ollama_stopped={1} hip_probe_sha256={2}" -f `
    $runtime.RuntimeTemp, `
    $runtime.OllamaStopped.ToString().ToLowerInvariant(), `
    $(if ($runtime.HipProbeSha256) { $runtime.HipProbeSha256 } else { "skipped" })
)

& $Executable @ArgumentList
if ($null -ne $LASTEXITCODE) {
  exit $LASTEXITCODE
}
