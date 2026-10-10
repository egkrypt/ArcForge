$ErrorActionPreference = 'Stop'
$arcRoot = $PSScriptRoot
$arcUrl = 'http://127.0.0.1:8769'
. (Join-Path $arcRoot 'tools\python_runtime.ps1')
$arcPython = Get-ArcForgePython -Root $arcRoot
$arcReady = $false
try {
    $arcStatus = Invoke-RestMethod "$arcUrl/api/status" -TimeoutSec 2
    if ($arcStatus.name -ne 'ArcForge') { throw 'Port 8769 belongs to another application.' }
    $arcReady = $true
} catch {
    if ($_.Exception.Message -like '*belongs to another*') { throw }
}
if (-not $arcReady) {
    $arcProcess = Start-Process -FilePath $arcPython -ArgumentList '-X','utf8','server.py' -WorkingDirectory $arcRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $arcRoot 'server.out.log') -RedirectStandardError (Join-Path $arcRoot 'server.err.log') -PassThru
    $arcProcess.Id | Set-Content -LiteralPath (Join-Path $arcRoot 'server.pid')
    for ($arcAttempt = 0; $arcAttempt -lt 50; $arcAttempt++) {
        Start-Sleep -Milliseconds 200
        try {
            $arcStatus = Invoke-RestMethod "$arcUrl/api/status" -TimeoutSec 2
            if ($arcStatus.name -eq 'ArcForge') { $arcReady = $true; break }
        } catch { }
    }
    if (-not $arcReady) { throw "ArcForge did not start. See server.err.log in $arcRoot." }
}
Start-Process $arcUrl
