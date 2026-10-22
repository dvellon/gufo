$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
. (Join-Path $PSScriptRoot "accelerator_runtime.ps1")
$temporary = Join-Path $root (
  "build\gufo-runtime-temp\accelerator-runtime-selftest-" + [Guid]::NewGuid().ToString("N"))
$release = Join-Path $temporary "release"
[System.IO.Directory]::CreateDirectory($release) | Out-Null
try {
  [System.IO.File]::WriteAllText(
    (Join-Path $release "amdhip64_7.dll"), "runtime-a")
  [System.IO.File]::WriteAllText((Join-Path $release "support.dll"), "runtime-b")
  $codeObjects = Join-Path $release "hipblaslt\library\gfx-test"
  [System.IO.Directory]::CreateDirectory($codeObjects) | Out-Null
  [System.IO.File]::WriteAllText(
    (Join-Path $codeObjects "kernel.hsaco"), "runtime-resource")
  $target = Join-Path $temporary "original-target.exe"
  $probe = Join-Path $temporary "original-probe.exe"
  [System.IO.File]::WriteAllText($target, "MZ amdhip64_7.dll target")
  [System.IO.File]::WriteAllText($probe, "MZ amdhip64_7.dll probe")
  if (-not (Test-GufoDirectHipImport -Executable $target)) {
    throw "self-test HIP import marker was not detected"
  }
  $first = New-GufoAcceleratorExecutionStage `
    -RepositoryRoot $temporary `
    -ReleaseRoot $release `
    -TargetExecutable $target `
    -HipProbePath $probe
  $second = New-GufoAcceleratorExecutionStage `
    -RepositoryRoot $temporary `
    -ReleaseRoot $release `
    -TargetExecutable $target `
    -HipProbePath $probe
  if ($first.Directory -cne $second.Directory -or
      $first.RuntimeManifestSha256 -cne $second.RuntimeManifestSha256) {
    throw "content-addressed stage was not stable"
  }
  foreach ($pair in @(
    @($target, $first.TargetExecutable),
    @($probe, $first.HipProbePath),
    @((Join-Path $release "amdhip64_7.dll"),
      (Join-Path $first.Directory "amdhip64_7.dll")))) {
    if ((Get-GufoFileSha256 -Path $pair[0]) -cne
        (Get-GufoFileSha256 -Path $pair[1])) {
      throw "staged bytes differ from source: $($pair[1])"
    }
  }
  $names = @(Get-ChildItem -LiteralPath $first.Directory -Force -File -Recurse |
    ForEach-Object {
      [System.IO.Path]::GetRelativePath($first.Directory, $_.FullName)
    } | Sort-Object)
  $expected = @("amdhip64_7.dll", "hip-gate.exe",
                "hipblaslt\library\gfx-test\kernel.hsaco", "support.dll",
                "target.exe") | Sort-Object
  if (($names -join "`n") -cne ($expected -join "`n")) {
    throw "staged inventory is not exact: actual=$($names -join ',') expected=$($expected -join ',')"
  }

  # Exercise the public wrapper in a child PowerShell so its deliberate
  # `exit` cannot terminate this self-test.  The helper is intentionally not
  # HIP-linked: argument and exit-code forwarding are shared by staged and
  # ordinary targets, while the checks above cover the staging selection.
  $helper = Join-Path $temporary "argument-helper.cmd"
  [System.IO.File]::WriteAllText(
    $helper,
    "@echo off`r`necho ARG1=[%~1] ARG2=[%~2]`r`nexit /b 23`r`n")
  $wrapper = Join-Path $PSScriptRoot "invoke-gufo-accelerator.ps1"
  $driver = Join-Path $temporary "wrapper-driver.ps1"
  $command = @"
& '$($wrapper.Replace("'", "''"))' -Executable '$($helper.Replace("'", "''"))' -ArgumentList @('alpha beta', 'gamma') -SkipHipProbe
exit `$LASTEXITCODE
"@
  [System.IO.File]::WriteAllText($driver, $command)
  $wrapperOutput = @(& pwsh -NoProfile -NonInteractive -File $driver 2>&1)
  $wrapperExit = $LASTEXITCODE
  if ($wrapperExit -ne 23) {
    throw "accelerator wrapper changed target exit code: $wrapperExit"
  }
  if (-not ($wrapperOutput -contains "ARG1=[alpha beta] ARG2=[gamma]")) {
    throw "accelerator wrapper changed target arguments: $($wrapperOutput -join ' | ')"
  }
  Write-Output "GUFO_ACCELERATOR_RUNTIME_SELFTEST_OK"
} finally {
  if (Test-Path -LiteralPath $temporary) {
    Remove-Item -LiteralPath $temporary -Recurse -Force
  }
}
