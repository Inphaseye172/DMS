<#
.SYNOPSIS
  Downloads the pinned runtime dependencies of the DMS plugin, verifies their SHA-256, and lays them
  out next to the plugin:

    <DestDir>\ort-cuda\   onnxruntime.dll, onnxruntime_providers_shared.dll, onnxruntime_providers_cuda.dll,
                          cudart64_12.dll, cublas64_12.dll, cublasLt64_12.dll, cufft64_11.dll, cudnn*64_9.dll
    <DestDir>\ort-dml\    onnxruntime.dll, onnxruntime_providers_shared.dll, DirectML.dll

  Everything in a folder is pre-loaded by full path by the engine, so nothing is resolved from PATH,
  System32 (old DirectML.dll) or the After Effects folder.

.PARAMETER DestDir
  Output folder. Default: <repo>\build\DMS_deps
.PARAMETER SkipCuda
  Do not fetch the CUDA build / NVIDIA libraries (~2.3 GB download, ~1.6 GB on disk).
.PARAMETER SkipDirectML
  Do not fetch the DirectML build.
.PARAMETER WithTensorRT
  Also copy onnxruntime_providers_tensorrt.dll (TensorRT itself must be installed separately).
.PARAMETER Force
  Re-verify and re-extract even if DestDir is already up to date.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\fetch-deps.ps1
#>
[CmdletBinding()]
param(
    [string]$DestDir = (Join-Path $PSScriptRoot "..\build\DMS_deps"),
    [switch]$SkipCuda,
    [switch]$SkipDirectML,
    [switch]$WithTensorRT,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is 10x slower with the progress bar

$Cache = Join-Path $PSScriptRoot "..\build\_downloads"
New-Item -ItemType Directory -Force -Path $Cache, $DestDir | Out-Null
$DestDir = (Resolve-Path $DestDir).Path

# name, url, sha256 (pinned; see THIRD_PARTY.md)
$Artifacts = @{
    "ort-gpu"  = @("onnxruntime-win-x64-gpu-1.22.0.zip",
                   "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-win-x64-gpu-1.22.0.zip",
                   "5b5241716b2628c1ab5e79ee620be767531021149ee68f30fc46c16263fb94dd")
    "ort-dml"  = @("Microsoft.ML.OnnxRuntime.DirectML.1.22.0.nupkg",
                   "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/Microsoft.ML.OnnxRuntime.DirectML.1.22.0.nupkg",
                   "b38d92c31d202d1dc772af9416fe30b6abae8527651e742c380c2d131e5f1817")
    "directml" = @("microsoft.ai.directml.1.15.4.nupkg",
                   "https://api.nuget.org/v3-flatcontainer/microsoft.ai.directml/1.15.4/microsoft.ai.directml.1.15.4.nupkg",
                   "4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9")
    "cudart"   = @("nvidia_cuda_runtime_cu12-12.9.79-py3-none-win_amd64.whl",
                   "https://files.pythonhosted.org/packages/py3/n/nvidia-cuda-runtime-cu12/nvidia_cuda_runtime_cu12-12.9.79-py3-none-win_amd64.whl",
                   "8e018af8fa02363876860388bd10ccb89eb9ab8fb0aa749aaf58430a9f7c4891")
    "cublas"   = @("nvidia_cublas_cu12-12.9.2.10-py3-none-win_amd64.whl",
                   "https://files.pythonhosted.org/packages/py3/n/nvidia-cublas-cu12/nvidia_cublas_cu12-12.9.2.10-py3-none-win_amd64.whl",
                   "623f43027d40d44ceadf0043f002bd25cf353e8f13ce90b9a87057019f560661")
    "cufft"    = @("nvidia_cufft_cu12-11.4.1.4-py3-none-win_amd64.whl",
                   "https://files.pythonhosted.org/packages/py3/n/nvidia-cufft-cu12/nvidia_cufft_cu12-11.4.1.4-py3-none-win_amd64.whl",
                   "8e5bfaac795e93f80611f807d42844e8e27e340e0cde270dcb6c65386d795b80")
    "cudnn"    = @("nvidia_cudnn_cu12-9.27.0.42-py3-none-win_amd64.whl",
                   "https://files.pythonhosted.org/packages/py3/n/nvidia-cudnn-cu12/nvidia_cudnn_cu12-9.27.0.42-py3-none-win_amd64.whl",
                   "06e9b0026f3bad97d2b58666330fabec04fe1672f776661ecb0ce0029c27f142")
}

# ---- fast path: an earlier run completed with exactly these pinned artifacts -------------------
# The stamp is deleted before any extraction and written only after everything succeeded, so an
# interrupted run (Ctrl+C) always redoes the work instead of trusting half-written files.
$selected = @()
if (-not $SkipCuda) { $selected += @("ort-gpu", "cudart", "cublas", "cufft", "cudnn") }
if (-not $SkipDirectML) { $selected += @("ort-dml", "directml") }
$stampFile = Join-Path $DestDir ".dms_deps_stamp"
$stamp = (($selected | Sort-Object | ForEach-Object { "$_=$($Artifacts[$_][2])" }) + "tensorrt=$([bool]$WithTensorRT)") -join "`n"
$keyFiles = @()
if (-not $SkipCuda) {
    $keyFiles += @("onnxruntime.dll", "onnxruntime_providers_shared.dll", "onnxruntime_providers_cuda.dll", "cudart64_12.dll", "cublas64_12.dll",
                   "cublasLt64_12.dll", "cufft64_11.dll", "cudnn64_9.dll") | ForEach-Object { Join-Path (Join-Path $DestDir "ort-cuda") $_ }
}
if (-not $SkipDirectML) {
    $keyFiles += @("onnxruntime.dll", "DirectML.dll") | ForEach-Object { Join-Path (Join-Path $DestDir "ort-dml") $_ }
}
# Stamp layout: the artifact lines above, a "---" line, then "relative\path|size" for every extracted file.
function Test-Stamp {
    if ($Force -or -not (Test-Path $stampFile)) { return $false }
    $parts = (Get-Content $stampFile -Raw) -split "`n---`n", 2
    if ($parts.Count -ne 2 -or $parts[0].Trim() -ne $stamp.Trim()) { return $false }
    foreach ($k in $keyFiles) { if (-not (Test-Path $k)) { return $false } }
    foreach ($line in ($parts[1] -split "`n")) {
        if (-not $line.Trim()) { continue }
        $rel, $len = $line.Trim() -split "\|", 2
        $f = Join-Path $DestDir $rel
        if (-not (Test-Path $f) -or (Get-Item $f).Length -ne [int64]$len) { return $false }
    }
    return $true
}
if (Test-Stamp) {
    Write-Host "Dependencies already up to date in $DestDir (use -Force to re-verify)."
    return
}
Remove-Item -Force $stampFile -ErrorAction SilentlyContinue

$Resolved = @{}
$Written = New-Object System.Collections.ArrayList   # "relative\path|size" of every file this run placed
function Get-Artifact([string]$key) {
    if ($Resolved.ContainsKey($key)) { return $Resolved[$key] }
    $name, $url, $sha = $Artifacts[$key]
    $path = Join-Path $Cache $name
    if (Test-Path $path) {
        $mb = [math]::Round((Get-Item $path).Length / 1MB)
        Write-Host "Verifying cached $name ($mb MB) ..."
        if ((Get-FileHash -Algorithm SHA256 $path).Hash.ToLower() -eq $sha) { $Resolved[$key] = $path; return $path }
        Write-Warning "$name in cache has a wrong hash; re-downloading"
        Remove-Item $path -Force
    }
    Write-Host "Downloading $name ..."
    $tmp = "$path.part"
    for ($i = 1; $i -le 4; $i++) {
        try {
            Invoke-WebRequest -Uri $url -OutFile $tmp -UseBasicParsing
            break
        } catch {
            if ($i -eq 4) { throw "Download failed for ${url}: $_" }
            Start-Sleep -Seconds ([math]::Pow(2, $i))
        }
    }
    $got = (Get-FileHash -Algorithm SHA256 $tmp).Hash.ToLower()
    if ($got -ne $sha) {
        Remove-Item $tmp -Force
        throw "SHA-256 mismatch for $name`n  expected $sha`n  got      $got"
    }
    Move-Item $tmp $path -Force
    $Resolved[$key] = $path
    return $path
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
# Extracts entries whose full name matches $pattern (regex) into $dest, flattening directories.
# Files already present with the right size are left alone; others are written to a temp file and
# renamed, so an interrupted run never leaves a truncated DLL under its final name.
function Expand-Matching([string]$archive, [string]$pattern, [string]$dest) {
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    $zip = [System.IO.Compression.ZipFile]::OpenRead($archive)
    $n = 0
    try {
        foreach ($e in $zip.Entries) {
            if (-not ($e.FullName -match $pattern -and $e.Name)) { continue }
            $n++
            $target = Join-Path $dest $e.Name
            $rel = $target.Substring($DestDir.Length).TrimStart([char]'\', [char]'/')
            [void]$Written.Add(("{0}|{1}" -f $rel, $e.Length))
            if ((Test-Path $target) -and (Get-Item $target).Length -eq $e.Length) { continue }
            Write-Host ("  extracting {0} ({1} MB)" -f $e.Name, [math]::Round($e.Length / 1MB))
            $tmp = "$target.tmp"
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile($e, $tmp, $true)
            Move-Item -Force $tmp $target
        }
    } finally { $zip.Dispose() }
    if ($n -eq 0) { throw "No entries matching '$pattern' in $archive" }
}

if (-not $SkipCuda) {
    $d = Join-Path $DestDir "ort-cuda"
    $ort = Get-Artifact "ort-gpu"
    $libs = "lib/(onnxruntime|onnxruntime_providers_shared|onnxruntime_providers_cuda)\.dll$"
    if ($WithTensorRT) { $libs = "lib/(onnxruntime|onnxruntime_providers_shared|onnxruntime_providers_cuda|onnxruntime_providers_tensorrt)\.dll$" }
    Expand-Matching $ort $libs $d
    Expand-Matching (Get-Artifact "cudart") "nvidia/cuda_runtime/bin/cudart64_12\.dll$" $d
    Expand-Matching (Get-Artifact "cublas") "nvidia/cublas/bin/(cublas64_12|cublasLt64_12)\.dll$" $d
    Expand-Matching (Get-Artifact "cufft")  "nvidia/cufft/bin/cufft64_11\.dll$" $d
    Expand-Matching (Get-Artifact "cudnn")  "nvidia/cudnn/bin/cudnn.*64_9\.dll$" $d
    Write-Host "CUDA backend ready in $d"
}
if (-not $SkipDirectML) {
    $d = Join-Path $DestDir "ort-dml"
    Expand-Matching (Get-Artifact "ort-dml") "runtimes/win-x64/native/(onnxruntime|onnxruntime_providers_shared)\.dll$" $d
    Expand-Matching (Get-Artifact "directml") "bin/x64-win/DirectML\.dll$" $d
    Write-Host "DirectML backend ready in $d"
}

# Licenses travel with the binaries.
$lic = Join-Path $DestDir "licenses"
New-Item -ItemType Directory -Force -Path $lic | Out-Null
if (-not $SkipCuda)     { Expand-Matching (Get-Artifact "ort-gpu") "/(LICENSE|ThirdPartyNotices\.txt)$" (Join-Path $lic "onnxruntime") }
if (-not $SkipDirectML) { Expand-Matching (Get-Artifact "directml") "^LICENSE.*\.txt$" (Join-Path $lic "directml") }
Set-Content -Path $stampFile -Value ($stamp + "`n---`n" + ($Written -join "`n")) -NoNewline
Write-Host "Done. Dependencies in $DestDir"
