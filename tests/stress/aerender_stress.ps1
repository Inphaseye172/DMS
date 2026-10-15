<#
.SYNOPSIS
  Headless After Effects stress test for DMS: renders 500 frames of a 4K comp for every
  model x mapping mode x tiles (1/4/16) with aerender, and FAILS if a render crashes, hangs longer
  than -TimeoutMin, or the process working set / GPU memory keeps growing after warm-up (leak).

.DESCRIPTION
  1. Builds tests\stress\build\stress.aep with make_stress_project.jsx (AfterFX.exe -noui -r).
  2. Renders each comp with aerender (-mem_usage keeps AE's own image cache small and bounded).
  3. Samples aerender + child AfterFX working set every second, and nvidia-smi memory.used.
  4. Leak rule: after the first 30% of the run (AE caches warm), a linear fit of memory over time
     must project less than -LeakLimitMB of growth over the remaining run.

  Prerequisites: plugin installed (scripts\install.ps1), models downloaded (Effect Controls > Options),
  a "PNG Sequence" output-module template (or pass -OMTemplate).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tests\stress\aerender_stress.ps1 -Models 1,2,3,6
#>
[CmdletBinding()]
param(
    [string]$AeDir = "C:\Program Files\Adobe\Adobe After Effects 2026\Support Files",
    [int[]]$Models = @(1),            # Model popup indices: 1 DA2-S, 2 DA2-B, 3 DA2-L, 4 DA3-S, 5 DA3-B, 6 Video
    [int]$Frames = 500,
    [int]$TimeoutMin = 10,
    [int]$LeakLimitMB = 256,
    [int]$VramLeakLimitMB = 256,
    [string]$OMTemplate = "PNG Sequence",
    [string]$Footage = (Join-Path $PSScriptRoot "..\fixtures\scene_4k.png"),
    [string]$WorkDir = (Join-Path $PSScriptRoot "build"),
    [switch]$KeepFrames
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0

$afterfx = Join-Path $AeDir "AfterFX.exe"
$aerender = Join-Path $AeDir "aerender.exe"
foreach ($p in @($afterfx, $aerender, $Footage)) { if (-not (Test-Path $p)) { throw "Not found: $p" } }
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
$WorkDir = (Resolve-Path $WorkDir).Path
$project = Join-Path $WorkDir "stress.aep"
$nvsmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue

# ---- 1. build the project -------------------------------------------------------------------
$env:DMS_STRESS_PROJECT = $project
$env:DMS_STRESS_FOOTAGE = (Resolve-Path $Footage).Path
$env:DMS_STRESS_FRAMES = "$Frames"
$env:DMS_STRESS_MODELS = ($Models -join ",")
if (Test-Path $project) { Remove-Item -Force $project }
Write-Host "Building $project ..."
$p = Start-Process -FilePath $afterfx -ArgumentList @("-noui", "-r", "`"$(Join-Path $PSScriptRoot 'make_stress_project.jsx')`"") -PassThru
if (-not $p.WaitForExit(15 * 60 * 1000)) { $p.Kill(); throw "AfterFX did not finish building the project" }
Get-Content "$project.log" -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  jsx: $_" }
if (-not (Test-Path $project)) { throw "Project was not created (see $project.log)" }

# ---- helpers -------------------------------------------------------------------------------------
function Get-TreeWorkingSet([int]$rootPid) {
    $all = Get-CimInstance Win32_Process -Property ProcessId, ParentProcessId, WorkingSetSize
    $ids = New-Object System.Collections.Generic.HashSet[int]
    [void]$ids.Add($rootPid)
    $changed = $true
    while ($changed) {
        $changed = $false
        foreach ($proc in $all) { if ($ids.Contains([int]$proc.ParentProcessId) -and $ids.Add([int]$proc.ProcessId)) { $changed = $true } }
    }
    $sum = 0
    foreach ($proc in $all) { if ($ids.Contains([int]$proc.ProcessId)) { $sum += [int64]$proc.WorkingSetSize } }
    return $sum
}
function Get-VramUsedMB {
    if (-not $nvsmi) { return $null }
    $v = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null | Select-Object -First 1
    if ($v) { return [double]$v } else { return $null }
}
function Get-TrendMB($samples, [double]$warmFraction) {
    # samples: array of @(t, valueMB). Least-squares slope after warm-up, projected over that window.
    $n = $samples.Count
    if ($n -lt 20) { return 0.0 }
    $tail = $samples[[int]($n * $warmFraction)..($n - 1)]
    $mx = ($tail | ForEach-Object { $_[0] } | Measure-Object -Average).Average
    $my = ($tail | ForEach-Object { $_[1] } | Measure-Object -Average).Average
    $num = 0.0; $den = 0.0
    foreach ($s in $tail) { $num += ($s[0] - $mx) * ($s[1] - $my); $den += ($s[0] - $mx) * ($s[0] - $mx) }
    if ($den -le 0) { return 0.0 }
    return ($num / $den) * ($tail[-1][0] - $tail[0][0])
}

# ---- 2. render every comp --------------------------------------------------------------------
$comps = @()
foreach ($m in $Models) { foreach ($map in @("raw", "normalize", "guided")) { foreach ($t in @("t1", "t4", "t16")) { $comps += "DMS_m${m}_${map}_${t}" } } }
$results = @()
foreach ($comp in $comps) {
    $out = Join-Path $WorkDir "frames\$comp"
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    $log = Join-Path $WorkDir "$comp.log"
    $aeArgs = @("-project", "`"$project`"", "-comp", "`"$comp`"", "-s", "0", "-e", "$($Frames - 1)", "-OMtemplate", "`"$OMTemplate`"",
              "-output", "`"$out\frame_[#####].png`"", "-mem_usage", "10", "60", "-continueOnMissingFootage")
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $proc = Start-Process -FilePath $aerender -ArgumentList $aeArgs -PassThru -NoNewWindow -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    $mem = New-Object System.Collections.ArrayList
    $vram = New-Object System.Collections.ArrayList
    $timedOut = $false
    while (-not $proc.HasExited) {
        Start-Sleep -Seconds 1
        $t = $sw.Elapsed.TotalSeconds
        [void]$mem.Add(@($t, (Get-TreeWorkingSet $proc.Id) / 1MB))
        $v = Get-VramUsedMB
        if ($null -ne $v) { [void]$vram.Add(@($t, $v)) }
        if ($sw.Elapsed.TotalMinutes -gt $TimeoutMin) {
            $timedOut = $true
            & taskkill /PID $proc.Id /T /F | Out-Null
            break
        }
    }
    $proc.WaitForExit()
    $exit = $proc.ExitCode
    $memTrend = Get-TrendMB $mem 0.3
    $vramTrend = if ($vram.Count) { Get-TrendMB $vram 0.3 } else { 0.0 }
    $peak = ($mem | ForEach-Object { $_[1] } | Measure-Object -Maximum).Maximum
    $rendered = (Get-ChildItem $out -Filter *.png -ErrorAction SilentlyContinue | Measure-Object).Count
    $dmsErrors = (Select-String -Path $log -Pattern "DMS:" -SimpleMatch -ErrorAction SilentlyContinue | Measure-Object).Count
    $ok = (-not $timedOut) -and ($exit -eq 0) -and ($rendered -ge $Frames) -and ($memTrend -lt $LeakLimitMB) -and ($vramTrend -lt $VramLeakLimitMB)
    $r = [pscustomobject]@{
        comp = $comp; ok = $ok; exit = $exit; timed_out = $timedOut; frames = $rendered; minutes = [math]::Round($sw.Elapsed.TotalMinutes, 2)
        peak_mb = [math]::Round($peak, 0); mem_trend_mb = [math]::Round($memTrend, 1); vram_trend_mb = [math]::Round($vramTrend, 1); dms_messages = $dmsErrors
    }
    $results += $r
    $tag = if ($ok) { "PASS" } else { "FAIL" }
    Write-Host ("[{0}] {1,-28} exit={2} frames={3} {4} min peak={5} MB trend={6} MB vram-trend={7} MB" -f $tag, $comp, $exit, $rendered, $r.minutes, $r.peak_mb, $r.mem_trend_mb, $r.vram_trend_mb)
    if (-not $KeepFrames) { Remove-Item -Recurse -Force $out -ErrorAction SilentlyContinue }
}
$results | ConvertTo-Json | Set-Content (Join-Path $WorkDir "aerender_stress_report.json")
$failed = @($results | Where-Object { -not $_.ok })
Write-Host "$($results.Count - $failed.Count)/$($results.Count) passed. Report: $(Join-Path $WorkDir 'aerender_stress_report.json')"
if ($failed.Count) { exit 1 } else { exit 0 }
