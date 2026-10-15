<#
.SYNOPSIS
  One-shot local setup and verification of DMS on Windows. Safe to re-run.

.DESCRIPTION
  1. Checks prerequisites (Visual Studio C++ tools, CMake, Git, NVIDIA driver, After Effects 2026,
     AE SDK) and prints exactly what to install if something is missing.
  2. Downloads the pinned runtime dependencies (ONNX Runtime CUDA + DirectML builds, CUDA/cuDNN DLLs)
     and checks that every DLL import resolves.
  3. Configures and builds: the .aex plug-in when the AE SDK is found, otherwise engine + CLI + tests.
  4. Runs the unit tests.
  5. GPU smoke test with the real model: downloads Depth Anything v2 Small (99 MB, SHA-256 verified,
     into %LOCALAPPDATA%\DMS\models where the plug-in also looks) and runs it on the GPU and on the CPU.
  6. Optionally installs the plug-in into After Effects (-Install, asks for elevation).

  Everything is logged to build\setup-log.txt. Send that file back if anything fails.

.PARAMETER AeSdk
  Folder of the unzipped After Effects SDK (any level above Examples\Headers\AE_Effect.h).
  Default: $env:AE_SDK_ROOT, then a search of common locations (Downloads, Documents, C:\SDKs).
.PARAMETER EngineOnly
  Skip the plug-in even if the SDK is found.
.PARAMETER SkipDeps
  Do not run fetch-deps.ps1 (use an existing build\DMS_deps).
.PARAMETER SkipGpuTest
  Skip the model download and GPU/CPU smoke test.
.PARAMETER Install
  After a successful build, run scripts\install.ps1 elevated (UAC prompt). After Effects must be closed.
.PARAMETER Generator
  CMake generator override, e.g. "Visual Studio 17 2022".
.PARAMETER Config
  Release (default) or Debug.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup-windows.ps1 -AeSdk "$HOME\Downloads\AfterEffectsSDK"
#>
[CmdletBinding()]
param(
    [string]$AeSdk = $env:AE_SDK_ROOT,
    [switch]$EngineOnly,
    [switch]$SkipDeps,
    [switch]$SkipGpuTest,
    [switch]$Install,
    [string]$Generator = "",
    [ValidateSet("Release", "Debug")][string]$Config = "Release"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0
$ProgressPreference = "SilentlyContinue"

$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BuildRoot = Join-Path $Repo "build"
New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null
$LogFile = Join-Path $BuildRoot "setup-log.txt"
try { Stop-Transcript | Out-Null } catch { }
Start-Transcript -Path $LogFile -Force | Out-Null

$Results = New-Object System.Collections.ArrayList
$Fatal = $false

function Section([string]$title) {
    Write-Host ""
    Write-Host "=== $title ===" -ForegroundColor Cyan
}
function Add-Result([string]$step, [string]$status, [string]$detail) {
    [void]$Results.Add([pscustomobject]@{ Step = $step; Status = $status; Detail = $detail })
    $color = switch ($status) { "OK" { "Green" } "WARN" { "Yellow" } "SKIP" { "DarkGray" } default { "Red" } }
    Write-Host ("  [{0}] {1}: {2}" -f $status, $step, $detail) -ForegroundColor $color
}
# Runs a native command, streams + tees its output, returns the exit code. stderr must not throw (PS 5.1).
function Invoke-Native([string]$exe, [string[]]$argList, [string]$logName) {
    $log = Join-Path $BuildRoot "setup-$logName.log"
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $exe @argList 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log | Out-Host
        return $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $old
    }
}
# Runs a native command and returns its output lines (no streaming).
function Get-NativeOutput([string]$exe, [string[]]$argList) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { return @(& $exe @argList 2>&1 | ForEach-Object { "$_" }) } finally { $ErrorActionPreference = $old }
}

Write-Host "DMS local setup - $(Get-Date -Format 'yyyy-MM-dd HH:mm')" -ForegroundColor Cyan
Write-Host "Repo: $Repo"
if (Get-Command git -ErrorAction SilentlyContinue) {
    $head = Get-NativeOutput "git" @("-C", $Repo, "log", "-1", "--format=%h %s")
    Write-Host "Commit: $($head -join ' ')"
}

# ------------------------------------------------------------------------------------------------
Section "1. Prerequisites"
# ------------------------------------------------------------------------------------------------
$os = [Environment]::OSVersion.Version
if ([Environment]::Is64BitOperatingSystem -and $os.Major -ge 10) {
    Add-Result "Windows" "OK" "$((Get-CimInstance Win32_OperatingSystem).Caption) build $($os.Build)"
} else {
    Add-Result "Windows" "FAIL" "64-bit Windows 10/11 required"
    $Fatal = $true
}
if ($Repo.Length -gt 80) {
    Add-Result "Repo path" "WARN" "long path ($($Repo.Length) chars); if the build fails on file paths, clone to e.g. C:\src\dms"
}

$git = Get-Command git -ErrorAction SilentlyContinue
if ($git) { Add-Result "Git" "OK" (Get-NativeOutput "git" @("--version") | Select-Object -First 1) }
else { Add-Result "Git" "FAIL" "install: winget install --id Git.Git -e"; $Fatal = $true }

# Visual Studio with the C++ x64 toolset (via vswhere, which ships with every VS 2017+ installer).
$vs = $null
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
# Report EVERY Visual Studio instance (incl. incomplete / Build Tools) and whether it has the C++ x64
# tools, so it is obvious why an installed version is not used.
function Get-VsInstances([string[]]$extra) {
    $raw = Get-NativeOutput $vswhere (@("-all", "-products", "*", "-format", "json") + $extra)
    $parsed = ($raw -join "`n") | ConvertFrom-Json  # PS 5.1 returns a JSON array as ONE object
    return @($parsed | ForEach-Object { $_ })
}
if (Test-Path $vswhere) {
    $allVs = @(Get-VsInstances @())  # @(): a single instance must stay an array (PS 5.1 StrictMode has no .Count on scalars)
    $withCpp = @((Get-VsInstances @("-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64")) | ForEach-Object { $_.instanceId })
    foreach ($i in $allVs) {
        $complete = (-not $i.PSObject.Properties["isComplete"]) -or $i.isComplete
        $launch = (-not $i.PSObject.Properties["isLaunchable"]) -or $i.isLaunchable
        $hasCpp = $withCpp -contains $i.instanceId
        $state = @()
        if ($hasCpp) { $state += "C++ x64 tools: yes" } else { $state += "C++ x64 tools: NO" }
        if (-not $complete) { $state += "installation INCOMPLETE (reboot, or Visual Studio Installer > Repair/Resume)" }
        if (-not $launch) { $state += "not launchable" }
        Write-Host ("    found {0} {1} at {2} - {3}" -f $i.displayName, $i.installationVersion, $i.installationPath, ($state -join "; "))
        if ($i.installationVersion -like "17.*" -and (-not $hasCpp -or -not $complete)) {
            Add-Result "VS 2022 not usable" "WARN" ("$($i.displayName): " + $(if (-not $complete) { "installation incomplete - reboot or open Visual Studio Installer and choose Resume/Repair" }
                else { "missing C++ tools - open Visual Studio Installer > Modify > tick 'Desktop development with C++' (and the 'MSVC v143 x64/x86' + 'Windows 11 SDK' components)" }))
        }
    }
    if (-not $allVs.Count) { Write-Host "    vswhere found no Visual Studio instances" }
}
if (Test-Path $vswhere) {
    $json = Get-NativeOutput $vswhere @("-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-format", "json")
    # Windows PowerShell 5.1 emits a JSON array as ONE object: re-enumerate it explicitly.
    $parsed = ($json -join "`n") | ConvertFrom-Json
    $all = @($parsed | ForEach-Object { $_ })
    $vs = $all | Where-Object { $_.installationVersion -like "17.*" } | Select-Object -First 1
    if (-not $vs) { $vs = $all | Sort-Object { [version]$_.installationVersion } -Descending | Select-Object -First 1 }
}
$vsMajor = 0
if ($vs) {
    $vsMajor = ([version]$vs.installationVersion).Major
    $note = ""
    if ($vsMajor -gt 17) { $note = " (project was written for VS 2022; newer MSVC should work)" }
    Add-Result "Visual Studio C++" "OK" "$($vs.displayName) $($vs.installationVersion)$note"
} else {
    Add-Result "Visual Studio C++" "FAIL" ("install VS 2022 with the C++ workload: winget install --id Microsoft.VisualStudio.2022.Community -e " +
        "--override `"--add Microsoft.VisualStudio.Workload.NativeDesktop --includeRecommended --passive`"")
    $Fatal = $true
}

$sdkInc = Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10\Include"
if (Test-Path $sdkInc) {
    $kits = @(Get-ChildItem $sdkInc -Directory | Where-Object { $_.Name -like "10.*" } | Sort-Object Name)
    if ($kits.Count) { Add-Result "Windows SDK" "OK" $kits[-1].Name } else { Add-Result "Windows SDK" "FAIL" "add 'Windows 11 SDK' in the VS Installer"; $Fatal = $true }
} else { Add-Result "Windows SDK" "FAIL" "add 'Windows 11 SDK' in the VS Installer"; $Fatal = $true }

# CMake >= 3.25: PATH first, then the copy bundled with Visual Studio.
$cmake = $null
$candidates = @()
$c = Get-Command cmake -ErrorAction SilentlyContinue
if ($c) { $candidates += $c.Source }
if ($vs) { $candidates += (Join-Path $vs.installationPath "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe") }
# Generator names come from `cmake --help` (e.g. "Visual Studio 17 2022", "Visual Studio 18 2026"),
# so a cmake that does not know the installed Visual Studio version is skipped.
function Get-VsGenerator([string]$cmakeExe, [int]$major) {
    foreach ($l in (Get-NativeOutput $cmakeExe @("--help"))) {
        if ($l -match "^\s*\*?\s*(Visual Studio $major \d{4})\b") { return $Matches[1] }
    }
    return $null
}
$tooOld = @()
foreach ($cand in ($candidates | Select-Object -Unique)) {
    if (-not (Test-Path $cand)) { continue }
    $line = Get-NativeOutput $cand @("--version") | Select-Object -First 1
    if (-not ($line -match "(\d+\.\d+(\.\d+)?)" -and [version]$Matches[1] -ge [version]"3.25")) { $tooOld += "$line ($cand)"; continue }
    if (-not $Generator -and $vsMajor -gt 0) {
        $g = Get-VsGenerator $cand $vsMajor
        if (-not $g) { $tooOld += "$line does not know Visual Studio $vsMajor ($cand)"; continue }
        $Generator = $g
    }
    $cmake = $cand
    Add-Result "CMake" "OK" "$line ($cand) -> generator '$Generator'"
    break
}
if (-not $cmake) {
    $why = if ($tooOld.Count) { " Rejected: $($tooOld -join '; ')." } else { "" }
    Add-Result "CMake" "FAIL" "need CMake >= 3.25 that supports your Visual Studio: winget install --id Kitware.CMake -e.$why"
    $Fatal = $true
}
$ctest = $null
if ($cmake) { $ctest = Join-Path (Split-Path $cmake) "ctest.exe" }

$python = $null
foreach ($p in @("python", "py")) {
    $cmd = Get-Command $p -ErrorAction SilentlyContinue
    if (-not $cmd -or $cmd.Source -like "*WindowsApps*") { continue }  # skip the Microsoft Store stub
    $v = Get-NativeOutput $cmd.Source @("--version") | Select-Object -First 1
    if ($v -match "Python 3") { $python = $cmd.Source; Add-Result "Python (optional)" "OK" $v; break }
}
if (-not $python) { Add-Result "Python (optional)" "WARN" "not found; the DLL closure check is skipped (winget install --id Python.Python.3.12 -e)" }

$nvsmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
$gpuName = ""
if ($nvsmi) {
    $q = Get-NativeOutput $nvsmi.Source @("--query-gpu=name,driver_version,memory.total", "--format=csv,noheader") | Select-Object -First 1
    $parts = @("$q" -split ",\s*")
    if ($parts.Count -ge 3) {
        $gpuName = $parts[0]
        $drv = [version]($parts[1] -replace "[^\d\.]", "")
        $st = "OK"
        $note = ""
        if ($drv -lt [version]"527.41") { $st = "WARN"; $note = " - older than 527.41: CUDA disabled, DirectML/CPU used. Update the driver." }
        elseif ($drv -lt [version]"560.0") { $note = " - CUDA Memory Sharing needs >= 560 (it will be ignored)" }
        Add-Result "NVIDIA GPU" $st "$($parts[0]), driver $($parts[1]), $($parts[2])$note"
    } else { Add-Result "NVIDIA GPU" "WARN" "nvidia-smi output not understood: $q" }
} else {
    $gpus = @(Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name })
    Add-Result "NVIDIA GPU" "WARN" "nvidia-smi not found (GPUs: $($gpus -join '; ')). DirectML or CPU will be used."
}

$aeExe = $null
$aeDirs = @(Get-ChildItem (Join-Path $env:ProgramFiles "Adobe") -Directory -Filter "Adobe After Effects*" -ErrorAction SilentlyContinue | Sort-Object Name -Descending)
foreach ($d in $aeDirs) {
    $exe = Join-Path $d.FullName "Support Files\AfterFX.exe"
    if (Test-Path $exe) { $aeExe = $exe; break }
}
if ($aeExe) {
    $st = "OK"
    if ($aeExe -notlike "*2026*") { $st = "WARN" }
    Add-Result "After Effects" $st "$aeExe$(if ($st -eq 'WARN') { ' (target is AE 2026; pass -PluginsDir to install.ps1 for this version)' })"
} else { Add-Result "After Effects" "WARN" "AfterFX.exe not found under $env:ProgramFiles\Adobe (the CLI and tests still work)" }

# After Effects SDK: find the folder that contains Headers\AE_Effect.h.
function Find-AeSdk([string]$start) {
    if (-not $start -or -not (Test-Path $start)) { return $null }
    $hit = Get-ChildItem -Path $start -Recurse -Filter "AE_Effect.h" -Depth 5 -ErrorAction SilentlyContinue |
        Where-Object { $_.Directory.Name -eq "Headers" } | Select-Object -First 1
    if ($hit) { return $hit.Directory.Parent.FullName }
    return $null
}
$sdkRoot = $null
$sdkNote = ""
if ($AeSdk) {
    if (-not (Test-Path $AeSdk)) { $sdkNote = "the -AeSdk folder '$AeSdk' does not exist. " }
    else {
        $sdkRoot = Find-AeSdk $AeSdk
        if (-not $sdkRoot) { $sdkNote = "no Headers\AE_Effect.h found under '$AeSdk' (is it unzipped?). " }
    }
}
if (-not $sdkRoot) {
    foreach ($base in @("C:\SDKs", "C:\AE_SDK", (Join-Path $HOME "Downloads"), (Join-Path $HOME "Documents"), (Join-Path $HOME "Desktop"))) {
        if (-not (Test-Path $base)) { continue }
        foreach ($d in @(Get-ChildItem $base -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "(?i)after.?effects.*sdk|ae.?sdk" })) {
            $sdkRoot = Find-AeSdk $d.FullName
            if ($sdkRoot) { break }
        }
        if ($sdkRoot) { break }
    }
}
$pipl = $null
if ($sdkRoot) { $pipl = Get-ChildItem -Path $sdkRoot -Recurse -Filter "PiPLtool.exe" -Depth 2 -ErrorAction SilentlyContinue | Select-Object -First 1 }
if ($sdkRoot -and $pipl) {
    $found = if ($sdkNote) { " (found by searching; $sdkNote" + "ignored -AeSdk)" } else { "" }
    Add-Result "After Effects SDK" "OK" "$sdkRoot$found"
} elseif ($sdkRoot) {
    Add-Result "After Effects SDK" "WARN" "$sdkRoot has headers but no PiPLtool.exe under Resources - plug-in build will fail"
} else {
    $zip = Get-ChildItem (Join-Path $HOME "Downloads") -Filter "*fter*ffects*SDK*.zip" -ErrorAction SilentlyContinue | Select-Object -First 1
    $hint = "download it from https://developer.adobe.com/after-effects/ (Adobe ID), unzip, re-run with -AeSdk <folder>"
    if ($zip) { $hint = "found $($zip.FullName) - unzip it, then re-run with -AeSdk <unzipped folder>" }
    $hint = $sdkNote + $hint
    Add-Result "After Effects SDK" "WARN" "$hint. Building engine + CLI + tests only."
}
$buildPlugin = [bool]($sdkRoot -and -not $EngineOnly)

if ($Fatal) {
    Section "Stopped: install the missing prerequisites above, then re-run this script"
} else {
    # --------------------------------------------------------------------------------------------
    Section "2. Runtime dependencies (pinned, SHA-256 verified)"
    # --------------------------------------------------------------------------------------------
    $deps = Join-Path $BuildRoot "DMS_deps"
    if ($SkipDeps) {
        Add-Result "fetch-deps" "SKIP" "-SkipDeps"
    } else {
        Write-Host "  First run: ~2.3 GB download + ~1.6 GB extraction (5-15 min, longer while antivirus scans the DLLs)."
        Write-Host "  Later runs finish in about a second once build\DMS_deps is complete."
        $t0 = Get-Date
        try {
            & (Join-Path $PSScriptRoot "fetch-deps.ps1") -DestDir $deps
            Write-Host ("  fetch-deps took {0:N0} s" -f ((Get-Date) - $t0).TotalSeconds)
            Add-Result "fetch-deps" "OK" $deps
        } catch {
            Add-Result "fetch-deps" "FAIL" "$_"
        }
    }
    if ($python -and (Test-Path $deps)) {
        $rc = Invoke-Native $python @("-I", (Join-Path $Repo "tools\check_deps.py"), $deps) "check-deps"
        if ($rc -eq 0) { Add-Result "DLL closure" "OK" "every import resolves" } else { Add-Result "DLL closure" "FAIL" "see build\setup-check-deps.log" }
    }

    # --------------------------------------------------------------------------------------------
    Section "3. Configure + build ($Config, $(if ($buildPlugin) { 'plug-in + engine + CLI + tests' } else { 'engine + CLI + tests' }))"
    # --------------------------------------------------------------------------------------------
    $bdir = Join-Path $BuildRoot ($(if ($buildPlugin) { "windows-release" } else { "windows-engine-only" }))
    $cache = Join-Path $bdir "CMakeCache.txt"
    if ((Test-Path $cache) -and -not (Select-String -Path $cache -Pattern "CMAKE_GENERATOR:INTERNAL=$([regex]::Escape($Generator))$" -Quiet)) {
        Write-Host "  build folder was configured with another generator; resetting its CMake cache" -ForegroundColor Yellow
        Remove-Item -Force $cache
        Remove-Item -Recurse -Force (Join-Path $bdir "CMakeFiles") -ErrorAction SilentlyContinue
    }
    $cfgArgs = @("-S", $Repo, "-B", $bdir, "-G", $Generator, "-A", "x64", "-DDMS_BUILD_CLI=ON", "-DDMS_BUILD_TESTS=ON",
                 "-DDMS_BUILD_PLUGIN=$(if ($buildPlugin) { 'ON' } else { 'OFF' })")
    if ($buildPlugin) { $cfgArgs += "-DAE_SDK_ROOT=$sdkRoot" }
    $built = $false
    $rc = Invoke-Native $cmake $cfgArgs "configure"
    if ($rc -ne 0) {
        Add-Result "Configure" "FAIL" "see build\setup-configure.log"
    } else {
        Add-Result "Configure" "OK" $bdir
        $rc = Invoke-Native $cmake @("--build", $bdir, "--config", $Config, "--parallel") "build"
        if ($rc -ne 0) {
            $errs = @(Select-String -Path (Join-Path $BuildRoot "setup-build.log") -Pattern ": (fatal )?error [A-Z]+\d+" | Select-Object -First 15 | ForEach-Object { $_.Line.Trim() })
            Add-Result "Build" "FAIL" "$($errs.Count) error line(s) shown below; full log build\setup-build.log"
            $errs | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
        } else {
            $built = $true
            Add-Result "Build" "OK" "no errors (warnings are errors: /W4 /WX)"
        }
    }
    $aex = Join-Path $BuildRoot $(if ($Config -eq "Debug") { "debug\DMS_Depth.aex" } else { "DMS_Depth.aex" })
    if ($buildPlugin -and $built) {
        if (Test-Path $aex) { Add-Result "Plug-in" "OK" "$aex ($([math]::Round((Get-Item $aex).Length / 1MB, 1)) MB)" }
        else { Add-Result "Plug-in" "FAIL" "build succeeded but $aex is missing" }
    }

    # --------------------------------------------------------------------------------------------
    Section "4. Unit tests"
    # --------------------------------------------------------------------------------------------
    if ($built) {
        $rc = Invoke-Native $ctest @("--test-dir", $bdir, "-C", $Config, "--output-on-failure", "--timeout", "600") "ctest"
        $sum = Select-String -Path (Join-Path $BuildRoot "setup-ctest.log") -Pattern "tests passed" | Select-Object -Last 1
        $summary = if ($sum) { $sum.Line.Trim() } else { "no summary line" }
        if ($rc -eq 0) { Add-Result "Unit tests" "OK" $summary }
        else {
            Add-Result "Unit tests" "FAIL" "$summary - see build\setup-ctest.log"
            Select-String -Path (Join-Path $BuildRoot "setup-ctest.log") -Pattern "\(Failed\)|\(SEGFAULT\)|\(Timeout\)|Exception" |
                Select-Object -First 15 | ForEach-Object { Write-Host "    $($_.Line.Trim())" -ForegroundColor Red }
        }
    } else { Add-Result "Unit tests" "SKIP" "build failed" }

    # --------------------------------------------------------------------------------------------
    Section "5. GPU smoke test with Depth Anything v2 Small"
    # --------------------------------------------------------------------------------------------
    $cli = Join-Path $bdir "bin\depthcli.exe"
    if (-not (Test-Path $cli)) { $cli = Join-Path $bdir "bin\$Config\depthcli.exe" }
    if ($SkipGpuTest) {
        Add-Result "GPU smoke test" "SKIP" "-SkipGpuTest"
    } elseif (-not $built -or -not (Test-Path $cli)) {
        Add-Result "GPU smoke test" "SKIP" "depthcli not built"
    } else {
        $savedDeps = $env:DMS_DEPS_DIR
        $savedLog = $env:DMS_LOG_DIR
        try {
            if (Test-Path (Join-Path $deps "ort-cuda\onnxruntime.dll") -PathType Leaf) { $env:DMS_DEPS_DIR = $deps }
            elseif (Test-Path (Join-Path $deps "ort-dml\onnxruntime.dll") -PathType Leaf) { $env:DMS_DEPS_DIR = $deps }
            $env:DMS_LOG_DIR = Join-Path $BuildRoot "setup-engine-logs"
            $out = Join-Path $BuildRoot "setup-out"
            $backendNames = @{ "0" = "cache"; "1" = "CPU"; "2" = "CUDA"; "3" = "DirectML"; "4" = "TensorRT" }

            Invoke-Native $cli @("--device") "device" | Out-Null
            $dev = Get-Content (Join-Path $BuildRoot "setup-device.log") -ErrorAction SilentlyContinue
            $cudaLine = $dev | Where-Object { $_ -match "CUDA=" } | Select-Object -First 1
            Add-Result "Engine device report" "OK" "$cudaLine"

            $rc = Invoke-Native $cli @("--download", "da2-small", "--quiet") "download"
            if ($rc -ne 0) { throw "model download failed (see build\setup-download.log)" }
            Add-Result "Model download" "OK" "Depth Anything v2 Small in $env:LOCALAPPDATA\DMS\models (SHA-256 verified)"

            $fixture = Join-Path $Repo "tests\fixtures\scene_floor.png"
            # 6 frames, cache off: frame 1 includes session creation, the rest show steady-state speed.
            $rc = Invoke-Native $cli @("-i", $fixture, "-o", (Join-Path $out "gpu"), "--no-cache", "--repeat", "6", "--log-level", "3") "gpu-run"
            $gpuLog = Get-Content (Join-Path $BuildRoot "setup-gpu-run.log")
            $last = $gpuLog | Where-Object { $_ -match "backend=(\d)" } | Select-Object -Last 1
            $avg = $gpuLog | Where-Object { $_ -match "avg=" } | Select-Object -Last 1
            $be = "?"
            if ($last -match "backend=(\d)") { $be = $backendNames[$Matches[1]] }
            $degraded = [bool]($gpuLog | Where-Object { $_ -match "\[degraded\]" })
            if ($rc -ne 0) { Add-Result "Inference (hardware acceleration)" "FAIL" "exit $rc - see build\setup-gpu-run.log" }
            elseif ($be -eq "CPU" -and $gpuName) { Add-Result "Inference (hardware acceleration)" "WARN" "ran on CPU despite an NVIDIA GPU$(if ($degraded) { ' (fallback, see message in log)' }); $avg" }
            else { Add-Result "Inference (hardware acceleration)" "OK" "backend $be; $avg" }

            $rc = Invoke-Native $cli @("-i", $fixture, "-o", (Join-Path $out "cpu"), "--no-cache", "--repeat", "2", "--ep", "cpu") "cpu-run"
            $cpuAvg = Get-Content (Join-Path $BuildRoot "setup-cpu-run.log") | Where-Object { $_ -match "avg=" } | Select-Object -Last 1
            if ($rc -eq 0) { Add-Result "Inference (CPU reference)" "OK" "$cpuAvg" } else { Add-Result "Inference (CPU reference)" "FAIL" "see build\setup-cpu-run.log" }

            $rc = Invoke-Native $cli @("-i", (Join-Path $Repo "tests\fixtures\scene_4k.png"), "-o", (Join-Path $out "gpu-4k"), "--no-cache",
                                       "--tiles", "4", "--colormap", "Turbo", "--edges", "50", "--smoothing", "20") "gpu-4k"
            $l4k = Get-Content (Join-Path $BuildRoot "setup-gpu-4k.log") | Where-Object { $_ -match "ms" } | Select-Object -First 1
            if ($rc -eq 0) { Add-Result "4K, 4 tiles, colormap" "OK" "$l4k" } else { Add-Result "4K, 4 tiles, colormap" "FAIL" "see build\setup-gpu-4k.log" }

            $fallbacks = @(Get-ChildItem $env:DMS_LOG_DIR -Filter "dms*.log" -ErrorAction SilentlyContinue |
                Select-String -Pattern "attempt .* failed|disabled for this session|CUDA dependencies incomplete|older than" | ForEach-Object { $_.Line })
            if ($fallbacks.Count) {
                Add-Result "Engine fallbacks" "WARN" "$($fallbacks.Count) fallback line(s) in build\setup-engine-logs\dms.log"
                $fallbacks | Select-Object -First 5 | ForEach-Object { Write-Host "    $_" -ForegroundColor Yellow }
            } else { Add-Result "Engine fallbacks" "OK" "none" }
            Write-Host "  Depth images: $out" -ForegroundColor Gray
        } catch {
            Add-Result "GPU smoke test" "FAIL" "$_"
        } finally {
            $env:DMS_DEPS_DIR = $savedDeps
            $env:DMS_LOG_DIR = $savedLog
        }
    }

    # --------------------------------------------------------------------------------------------
    Section "6. Install into After Effects"
    # --------------------------------------------------------------------------------------------
    $installScript = Join-Path $PSScriptRoot "install.ps1"
    $installCmd = "powershell -ExecutionPolicy Bypass -File `"$installScript`" -Aex `"$aex`" -Deps `"$deps`""
    if (-not ($buildPlugin -and $built -and (Test-Path $aex))) {
        Add-Result "Install" "SKIP" "no plug-in built"
    } elseif (-not $Install) {
        Add-Result "Install" "SKIP" "re-run with -Install, or from an elevated PowerShell: $installCmd"
    } elseif (Get-Process -Name "AfterFX", "aerender" -ErrorAction SilentlyContinue) {
        Add-Result "Install" "FAIL" "close After Effects / aerender first"
    } else {
        $p = Start-Process powershell -Verb RunAs -Wait -PassThru -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
            "`"$installScript`"", "-Aex", "`"$aex`"", "-Deps", "`"$deps`"")
        if ($p.ExitCode -eq 0) { Add-Result "Install" "OK" "open After Effects 2026: Effect > DMS > DMS Depth Map" }
        else { Add-Result "Install" "FAIL" "install.ps1 exited with $($p.ExitCode) (UAC declined?)" }
    }
}

# ------------------------------------------------------------------------------------------------
Section "SUMMARY"
# ------------------------------------------------------------------------------------------------
$Results | Format-Table -AutoSize -Wrap Step, Status, Detail | Out-String -Width 220 | Write-Host
$fails = @($Results | Where-Object { $_.Status -eq "FAIL" })
if ($fails.Count) {
    Write-Host "$($fails.Count) step(s) failed. Send build\setup-log.txt (and any build\setup-*.log it mentions)." -ForegroundColor Red
} else {
    Write-Host "All steps passed. Next: the After Effects checklist in docs\BUILD_WINDOWS.md. Log: build\setup-log.txt" -ForegroundColor Green
}
Stop-Transcript | Out-Null
if ($fails.Count) { exit 1 } else { exit 0 }
