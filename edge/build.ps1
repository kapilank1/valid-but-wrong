# Builds vbw_edge with MSVC + CMake + Ninja (all from VS Build Tools) and optionally runs the tests.
#
#   powershell -ExecutionPolicy Bypass -File edge\build.ps1            # configure + build (Release)
#   powershell -ExecutionPolicy Bypass -File edge\build.ps1 -Test      # ... then ctest
#   powershell -ExecutionPolicy Bypass -File edge\build.ps1 -Clean     # wipe edge\build first
#
# Works from any directory. If cl.exe is not already on PATH, the vcvars64 environment is imported into this
# PowerShell session (vcvars runs in cmd; its resulting environment is copied over).
param(
    [switch]$Test,
    [switch]$Clean,
    [ValidateSet("Release", "RelWithDebInfo", "Debug")][string]$BuildType = "Release"
)

$ErrorActionPreference = "Stop"
$EdgeDir = $PSScriptRoot
$BuildDir = Join-Path $EdgeDir "build"

function Find-VcVars {
    $installer = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer"
    $vswhere = Join-Path $installer "vswhere.exe"
    $candidates = @()
    if (Test-Path $vswhere) {
        $paths = & $vswhere -latest -prerelease -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        foreach ($p in @($paths)) {
            if ($p) { $candidates += (Join-Path $p "VC\Auxiliary\Build\vcvars64.bat") }
        }
    }
    foreach ($ver in @("18", "2022", "2019")) {
        foreach ($ed in @("BuildTools", "Community", "Professional", "Enterprise")) {
            foreach ($root in @(${env:ProgramFiles(x86)}, $env:ProgramFiles)) {
                $candidates += (Join-Path $root "Microsoft Visual Studio\$ver\$ed\VC\Auxiliary\Build\vcvars64.bat")
            }
        }
    }
    foreach ($c in $candidates) { if (Test-Path $c) { return $c } }
    return $null
}

function Import-VcVars([string]$vcvars) {
    # vcvars needs vswhere (in the Installer dir) on PATH.
    $installer = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer"
    $env:PATH = "$installer;$env:PATH"
    $tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("vbw_vcvars_" + [guid]::NewGuid().ToString("N") + ".cmd")
    Set-Content -Path $tmp -Encoding ASCII -Value @("@echo off", "call `"$vcvars`" >nul 2>&1", "if errorlevel 1 exit /b 1", "set")
    try {
        $lines = & cmd.exe /d /c $tmp
        if ($LASTEXITCODE -ne 0) { throw "vcvars64.bat failed (exit $LASTEXITCODE): $vcvars" }
    } finally {
        Remove-Item -Force $tmp -ErrorAction SilentlyContinue
    }
    foreach ($line in $lines) {
        if ($line -match '^([^=]+)=(.*)$') {
            [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
        }
    }
}

function Invoke-Native([string]$exe, [string[]]$arguments) {
    Write-Host "> $exe $($arguments -join ' ')"
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"   # native stderr must not become a terminating error in PowerShell 5.1
    try {
        & $exe @arguments
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prev
    }
    if ($code -ne 0) { throw "$exe failed with exit code $code" }
}

try {
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        $vcvars = Find-VcVars
        if (-not $vcvars) {
            throw "vcvars64.bat not found. Install Visual Studio Build Tools with the 'Desktop development with C++' workload and a Windows SDK."
        }
        Write-Host "Using $vcvars"
        Import-VcVars $vcvars
    }
    $cmake = (Get-Command cmake.exe -ErrorAction SilentlyContinue).Source
    $ctest = (Get-Command ctest.exe -ErrorAction SilentlyContinue).Source
    if (-not $cmake) { throw "cmake.exe not found on PATH (it ships with VS Build Tools: C++ CMake tools for Windows)." }
    if (-not $ctest) { $ctest = Join-Path (Split-Path $cmake -Parent) "ctest.exe" }
    if (-not (Get-Command ninja.exe -ErrorAction SilentlyContinue)) {
        throw "ninja.exe not found on PATH (it ships with VS Build Tools: C++ CMake tools for Windows)."
    }

    if ($Clean -and (Test-Path $BuildDir)) {
        Write-Host "Removing $BuildDir"
        Remove-Item -Recurse -Force $BuildDir
    }

    Invoke-Native $cmake @("-S", $EdgeDir, "-B", $BuildDir, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=$BuildType")
    Invoke-Native $cmake @("--build", $BuildDir)
    if ($Test) {
        Invoke-Native $ctest @("--test-dir", $BuildDir, "--output-on-failure")
    }
    Write-Host "OK: $BuildDir"
    exit 0
} catch {
    Write-Host "build.ps1: $_" -ForegroundColor Red
    exit 1
}
