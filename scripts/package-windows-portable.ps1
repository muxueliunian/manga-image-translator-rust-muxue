param(
    [switch]$Cuda,
    [switch]$NoZip
)

$ErrorActionPreference = "Stop"

if (-not $IsWindows -and $PSVersionTable.PSEdition -eq "Core") {
    Write-Warning "This package script is intended for Windows portable builds."
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
$ReleaseDir = Join-Path $RepoRoot "target\release"
$DistDir = Join-Path $RepoRoot "dist"
$PortableName = "manga-image-translator-rust-portable"
$PortableDir = Join-Path $DistDir $PortableName
$BinaryName = "simple-runtime.exe"
$BinaryPath = Join-Path $ReleaseDir $BinaryName
$ToolsRoot = Resolve-Path (Join-Path $RepoRoot "..\tools") -ErrorAction SilentlyContinue

function Write-Step {
    param([string]$Message)
    Write-Host "==> $Message"
}

function Use-IfExists {
    param([string]$Path)

    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $null
    }

    try {
        return (Resolve-Path -LiteralPath $Path -ErrorAction Stop).Path
    } catch {
        return $null
    }
}

function Initialize-LocalBuildEnvironment {
    if ($ToolsRoot) {
        $openCvRoot = Use-IfExists (Join-Path $ToolsRoot "opencv-4.11.0\opencv\build")
        if ($openCvRoot -and -not $env:OPENCV_LINK_PATHS) {
            $env:OPENCV_LINK_LIBS = "opencv_world4110"
            $env:OPENCV_LINK_PATHS = Join-Path $openCvRoot "x64\vc16\lib"
            $env:OPENCV_INCLUDE_PATHS = Join-Path $openCvRoot "include"
            $env:OPENCV_DISABLE_PROBES = "pkg_config,cmake,vcpkg_cmake,vcpkg"
            $env:OPENCV_BIN_DIR = Join-Path $openCvRoot "x64\vc16\bin"
            $env:PATH = "$env:OPENCV_BIN_DIR;$env:PATH"
            Write-Step "Using bundled OpenCV at $openCvRoot"
        }

        $llvmBin = Use-IfExists (Join-Path $ToolsRoot "LLVM-22.1.6\bin")
        if ($llvmBin -and -not $env:LIBCLANG_PATH) {
            $env:LIBCLANG_PATH = $llvmBin
            $env:PATH = "$llvmBin;$env:PATH"
            Write-Step "Using bundled LLVM/libclang at $llvmBin"
        }
    }
}

function Copy-IfExists {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    if (Test-Path -LiteralPath $Path) {
        Copy-Item -LiteralPath $Path -Destination $Destination -Force
        return $true
    }

    return $false
}

function Add-UniquePath {
    param(
        [System.Collections.Generic.List[string]]$List,
        [string]$Path
    )

    if ([string]::IsNullOrWhiteSpace($Path)) {
        return
    }

    try {
        $resolved = (Resolve-Path -LiteralPath $Path -ErrorAction Stop).Path
    } catch {
        return
    }

    if ((Test-Path -LiteralPath $resolved -PathType Container) -and -not $List.Contains($resolved)) {
        [void]$List.Add($resolved)
    }
}

function Get-OpenCvSearchDirs {
    $dirs = [System.Collections.Generic.List[string]]::new()

    Add-UniquePath $dirs $env:OPENCV_BIN_DIR
    Add-UniquePath $dirs $env:OPENCV_DIR

    if ($env:OPENCV_DIR) {
        Add-UniquePath $dirs (Join-Path $env:OPENCV_DIR "build\x64\vc16\bin")
        Add-UniquePath $dirs (Join-Path $env:OPENCV_DIR "build\x64\vc17\bin")
        Add-UniquePath $dirs (Join-Path $env:OPENCV_DIR "x64\vc16\bin")
        Add-UniquePath $dirs (Join-Path $env:OPENCV_DIR "x64\vc17\bin")
    }

    if ($env:OPENCV_LINK_PATHS) {
        foreach ($entry in ($env:OPENCV_LINK_PATHS -split ";")) {
            Add-UniquePath $dirs $entry
            Add-UniquePath $dirs (Join-Path (Split-Path -Parent $entry) "bin")
        }
    }

    foreach ($entry in ($env:Path -split ";")) {
        if ($entry -match "opencv") {
            Add-UniquePath $dirs $entry
        }
    }

    Add-UniquePath $dirs "C:\tools\opencv\build\x64\vc16\bin"
    Add-UniquePath $dirs "C:\tools\opencv\build\x64\vc17\bin"
    Add-UniquePath $dirs "C:\opencv\build\x64\vc16\bin"
    Add-UniquePath $dirs "C:\opencv\build\x64\vc17\bin"

    return $dirs
}

function Copy-OpenCvDlls {
    param([string]$Destination)

    $copied = 0
    foreach ($dir in (Get-OpenCvSearchDirs)) {
        $dlls = Get-ChildItem -LiteralPath $dir -Filter "opencv_world*.dll" -File -ErrorAction SilentlyContinue
        if (-not $dlls) {
            $dlls = Get-ChildItem -LiteralPath $dir -Filter "opencv_*.dll" -File -ErrorAction SilentlyContinue
        }

        foreach ($dll in $dlls) {
            if ($dll.BaseName -match "d$") {
                continue
            }
            Copy-Item -LiteralPath $dll.FullName -Destination $Destination -Force
            $copied++
        }
    }

    if ($copied -eq 0) {
        Write-Warning "OpenCV DLLs were not found. Set OPENCV_BIN_DIR, OPENCV_DIR, OPENCV_LINK_PATHS, or add OpenCV bin to PATH before packaging."
    }

    return $copied
}

function Write-Launcher {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string[]]$Lines
    )

    $content = @(
        "@echo off",
        "setlocal",
        "cd /d ""%~dp0""",
        "set ""PATH=%CD%;%PATH%"""
    ) + $Lines + @(
        "endlocal"
    )

    Set-Content -LiteralPath $Path -Value $content -Encoding ASCII
}

Write-Step "Building release binary"
Initialize-LocalBuildEnvironment
$cargoArgs = @("build", "--release", "--package", "simple-runtime")
if ($Cuda) {
    $cargoArgs += @("--features", "cuda")
}
& cargo @cargoArgs
if ($LASTEXITCODE -ne 0) {
    throw "cargo build failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path -LiteralPath $BinaryPath -PathType Leaf)) {
    throw "Expected binary was not found: $BinaryPath"
}

Write-Step "Preparing portable directory"
New-Item -ItemType Directory -Path $DistDir -Force | Out-Null
New-Item -ItemType Directory -Path $PortableDir -Force | Out-Null

foreach ($dir in @("config", "models", "uploads", "results")) {
    New-Item -ItemType Directory -Path (Join-Path $PortableDir $dir) -Force | Out-Null
}

Copy-Item -LiteralPath $BinaryPath -Destination $PortableDir -Force

Write-Step "Collecting runtime DLLs from target/release"
$runtimeDllPatterns = @(
    "*.dll"
)
foreach ($pattern in $runtimeDllPatterns) {
    Get-ChildItem -LiteralPath $ReleaseDir -Filter $pattern -File -ErrorAction SilentlyContinue |
        Where-Object { -not $_.Name.StartsWith("._") } |
        ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $PortableDir -Force }
}

Write-Step "Collecting OpenCV DLLs"
$openCvCopied = Copy-OpenCvDlls -Destination $PortableDir

$onnxDll = Join-Path $PortableDir "onnxruntime_providers_shared.dll"
if (-not (Test-Path -LiteralPath $onnxDll)) {
    Write-Warning "ONNX Runtime provider DLLs were not found in target/release. The portable build may need additional ONNX Runtime DLLs."
}

if ($Cuda) {
    $cudaDll = Join-Path $PortableDir "onnxruntime_providers_cuda.dll"
    if (-not (Test-Path -LiteralPath $cudaDll)) {
        Write-Warning "CUDA packaging was requested, but onnxruntime_providers_cuda.dll was not found in target/release."
    }
}

Write-Step "Writing launchers"
Write-Launcher -Path (Join-Path $PortableDir "run-ui.bat") -Lines @(
    "powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command ""Start-Process -FilePath '%CD%\$BinaryName' -ArgumentList 'ui-webview' -WorkingDirectory '%CD%' -WindowStyle Hidden"""
)

Write-Launcher -Path (Join-Path $PortableDir "run-ui-debug.bat") -Lines @(
    """%CD%\$BinaryName"" ui-webview",
    "pause"
)

Write-Launcher -Path (Join-Path $PortableDir "run-egui.bat") -Lines @(
    """%CD%\$BinaryName"" ui %*",
    "pause"
)

Write-Launcher -Path (Join-Path $PortableDir "run-webui.bat") -Lines @(
    "set ""MIT_HOST=127.0.0.1""",
    "set ""MIT_PORT=8080""",
    "if not ""%~1""=="""" set ""MIT_PORT=%~1""",
    "start ""manga-image-translator-rust webui"" http://%MIT_HOST%:%MIT_PORT%/",
    """%CD%\$BinaryName"" api --host %MIT_HOST% --port %MIT_PORT%"
)

Write-Launcher -Path (Join-Path $PortableDir "run-cli-example.bat") -Lines @(
    "if not exist ""uploads"" mkdir ""uploads""",
    "if not exist ""results"" mkdir ""results""",
    "echo Put input images in the uploads folder, then edit this example command if needed.",
    """%CD%\$BinaryName"" cli --input ""uploads"" --output ""results"" --config ""config\example.json"" --overwrite",
    "pause"
)

$readme = @"
Manga Image Translator Rust - Windows Portable
================================================

Contents:
- simple-runtime.exe
- Runtime DLLs copied from target\release
- OpenCV DLLs copied from OPENCV_BIN_DIR, OPENCV_DIR, OPENCV_LINK_PATHS, PATH, or common C:\tools\opencv paths when found
- config, models, uploads, and results directories

Launchers:
- run-ui.bat starts the WebView desktop UI.
- run-egui.bat starts the fallback egui desktop UI.
- run-webui.bat starts the local web API/UI at http://127.0.0.1:8080/.
  You can pass a port as the first argument, for example: run-webui.bat 8766
- run-cli-example.bat runs the CLI against the uploads folder and writes to results.
  Add or edit config\example.json before using that example command.

Packaging notes:
- If OpenCV DLLs were not copied, install OpenCV and set OPENCV_BIN_DIR to the folder containing opencv_world*.dll.
- CUDA packages also need compatible NVIDIA CUDA/cuDNN and ONNX Runtime CUDA provider DLLs.
- All launchers cd to their own directory before starting simple-runtime.exe.
"@
Set-Content -LiteralPath (Join-Path $PortableDir "README-portable.txt") -Value $readme -Encoding ASCII

if (-not $NoZip) {
    Write-Step "Creating zip archive"
    $zipPath = Join-Path $DistDir "$PortableName.zip"
    if (Test-Path -LiteralPath $zipPath) {
        Remove-Item -LiteralPath $zipPath -Force
    }
    Compress-Archive -LiteralPath $PortableDir -DestinationPath $zipPath -Force
}

Write-Host ""
Write-Host "Portable package created: $PortableDir"
if (-not $NoZip) {
    Write-Host "Zip archive created: $(Join-Path $DistDir "$PortableName.zip")"
}
Write-Host "OpenCV DLLs copied: $openCvCopied"
