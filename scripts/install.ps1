#Requires -RunAsAdministrator
<#
.SYNOPSIS
  Installs DMS_Depth.aex and its runtime dependencies into the After Effects 2026 plug-ins folder.

  REQUIRES ELEVATION: the plug-ins folder is under C:\Program Files. Run from an elevated PowerShell:
      powershell -ExecutionPolicy Bypass -File scripts\install.ps1

  Layout created:
      <PluginsDir>\DMS\DMS_Depth.aex
      <PluginsDir>\DMS\DMS_deps\ort-cuda\...   (from scripts\fetch-deps.ps1)
      <PluginsDir>\DMS\DMS_deps\ort-dml\...

.PARAMETER PluginsDir
  Target plug-ins folder. Default: C:\Program Files\Adobe\Adobe After Effects 2026\Support Files\Plug-ins
.PARAMETER MediaCore
  Install into the shared MediaCore folder instead (picked up by After Effects and Premiere Pro):
  C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore
.PARAMETER Aex
  Path to the built plug-in. Default: <repo>\build\DMS_Depth.aex
.PARAMETER Deps
  Path to the dependency folder produced by fetch-deps.ps1. Default: <repo>\build\DMS_deps
.PARAMETER Uninstall
  Removes <PluginsDir>\DMS.
#>
[CmdletBinding()]
param(
    [string]$PluginsDir = "C:\Program Files\Adobe\Adobe After Effects 2026\Support Files\Plug-ins",
    [switch]$MediaCore,
    [string]$Aex = (Join-Path $PSScriptRoot "..\build\DMS_Depth.aex"),
    [string]$Deps = (Join-Path $PSScriptRoot "..\build\DMS_deps"),
    [switch]$Uninstall
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0

if ($MediaCore) { $PluginsDir = "C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore" }
$Target = Join-Path $PluginsDir "DMS"

# Files are locked while the host runs.
$running = Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -in @("AfterFX", "aerender", "Adobe Premiere Pro") }
if ($running) {
    throw "Close After Effects / Premiere Pro / aerender first (running: $($running.ProcessName -join ', '))."
}

if ($Uninstall) {
    if (Test-Path $Target) { Remove-Item -Recurse -Force $Target; Write-Host "Removed $Target" } else { Write-Host "Nothing to remove at $Target" }
    exit 0
}

if (-not (Test-Path $PluginsDir)) { throw "Plug-ins folder not found: $PluginsDir (is After Effects 2026 installed? use -PluginsDir)" }
if (-not (Test-Path $Aex)) { throw "Plug-in not found: $Aex (build the windows-release preset first)" }
if (-not (Test-Path $Deps)) { throw "Dependencies not found: $Deps (run scripts\fetch-deps.ps1 first)" }
$haveCuda = Test-Path (Join-Path $Deps "ort-cuda\onnxruntime.dll")
$haveDml = Test-Path (Join-Path $Deps "ort-dml\onnxruntime.dll")
if (-not ($haveCuda -or $haveDml)) { throw "$Deps contains neither ort-cuda nor ort-dml" }

New-Item -ItemType Directory -Force -Path $Target | Out-Null
Copy-Item -Force $Aex $Target
$pdb = [System.IO.Path]::ChangeExtension($Aex, ".pdb")
if (Test-Path $pdb) { Copy-Item -Force $pdb $Target }
$destDeps = Join-Path $Target "DMS_deps"
if (Test-Path $destDeps) { Remove-Item -Recurse -Force $destDeps }   # never mix dependency versions
Copy-Item -Recurse -Force $Deps $destDeps

Write-Host "Installed to $Target"
Write-Host ("  CUDA backend:     " + ($(if ($haveCuda) { "yes" } else { "no" })))
Write-Host ("  DirectML backend: " + ($(if ($haveDml) { "yes" } else { "no" })))
Write-Host "Models are downloaded on first use to %LOCALAPPDATA%\DMS\models (Effect Controls > Options)."
Write-Host "Log: %LOCALAPPDATA%\DMS\logs\dms.log"
