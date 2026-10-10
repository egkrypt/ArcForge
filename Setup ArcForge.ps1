param([switch]$Recognition,[switch]$Cuda,[switch]$Inferno,[switch]$BuildFrontend)
$ErrorActionPreference = 'Stop'
$arcRoot = $PSScriptRoot
. (Join-Path $arcRoot 'tools\python_runtime.ps1')
$arcPython = Get-ArcForgePython -Root $arcRoot
$arcVenv = Join-Path $arcRoot '.venv'
if (-not (Test-Path -LiteralPath (Join-Path $arcVenv 'Scripts\python.exe'))) {
    & $arcPython -m venv $arcVenv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the app-local Python environment.' }
}
$arcPython = Join-Path $arcVenv 'Scripts\python.exe'
& $arcPython -m pip install -r (Join-Path $arcRoot 'requirements-server.txt')
if ($LASTEXITCODE -ne 0) { throw 'Server dependency installation failed.' }
if ($Inferno) {
    & $arcPython -m pip install -r (Join-Path $arcRoot 'requirements-inferno.txt')
    if ($LASTEXITCODE -ne 0) { throw 'InfernoSaber TensorFlow dependency installation failed.' }
}
if ($Recognition -or $Cuda) {
    & $arcPython -m pip install -r (Join-Path $arcRoot 'requirements-recognition.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Recognition dependency installation failed.' }
    if ($Cuda) {
        & $arcPython -m pip install -r (Join-Path $arcRoot 'requirements-gpu.txt')
        if ($LASTEXITCODE -ne 0) { throw 'CUDA dependency installation failed; CPU mode remains available.' }
    }
}
if ($BuildFrontend) {
    Push-Location (Join-Path $arcRoot 'frontend')
    try {
        & npm.cmd ci
        if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
        & npm.cmd run build
        if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
    } finally { Pop-Location }
}
Write-Host 'ArcForge is ready. Launch ArcForge.cmd opens the local app. Install optional recognition weights inside ArcAudio when needed.'
