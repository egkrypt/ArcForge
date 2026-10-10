$ErrorActionPreference = 'Stop'
$arcUrl = 'http://127.0.0.1:8769'
try {
    $arcStatus = Invoke-RestMethod "$arcUrl/api/status" -Method Get -TimeoutSec 3
} catch {
    Write-Host 'ArcForge is not available. No process was stopped.'
    return
}
if ($arcStatus.name -ne 'ArcForge' -or $arcStatus.local -ne $true) {
    throw 'Port 8769 does not identify a local ArcForge service. No stop request was sent.'
}
try {
    # The service owns its lifecycle and refuses to interrupt generation/training.
    $arcStopped = Invoke-RestMethod "$arcUrl/api/system/stop" -Method Post -TimeoutSec 5
} catch {
    $arcDetail = $null
    try { $arcDetail = ($_.ErrorDetails.Message | ConvertFrom-Json).detail } catch { }
    if ($arcDetail) { throw "ArcForge did not stop: $arcDetail" }
    throw
}
if ($arcStopped.status -ne 'stopping') {
    throw 'ArcForge did not acknowledge the stop request. No process was killed.'
}
Write-Host $arcStopped.message
