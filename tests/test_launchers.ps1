$ErrorActionPreference = 'Stop'
$arcRoot = Split-Path -Parent $PSScriptRoot
foreach ($arcRelative in @('Setup ArcForge.ps1','Launch ArcForge.ps1','Stop ArcForge.ps1','tools\python_runtime.ps1')) {
    $arcErrors = $null
    $arcTokens = $null
    $null = [System.Management.Automation.Language.Parser]::ParseFile(
        (Join-Path $arcRoot $arcRelative), [ref]$arcTokens, [ref]$arcErrors)
    if ($arcErrors.Count) { throw ($arcErrors | Out-String) }
}
. (Join-Path $arcRoot 'tools\python_runtime.ps1')
$arcPython = Get-ArcForgePython -Root $arcRoot
$arcReported = & $arcPython -X utf8 -c 'import sys; print(sys.executable)'
if ($LASTEXITCODE -ne 0 -or $arcReported -ne $arcPython) { throw 'The resolver did not return the actual Python interpreter.' }
if ((Get-Content -Raw -LiteralPath (Join-Path $arcRoot 'Launch ArcForge.ps1')) -match 'codex-runtimes|Users\\') {
    throw 'The public launcher contains a machine-specific runtime assumption.'
}
$script:arcLaunchCalls = @()
function Invoke-RestMethod { param($Uri,$TimeoutSec); return @{ name='ArcForge' } }
function Start-Process {
    param($FilePath,$ArgumentList,$WorkingDirectory,$WindowStyle,$RedirectStandardOutput,$RedirectStandardError,[switch]$PassThru)
    $script:arcLaunchCalls += [string]$FilePath
}
. (Join-Path $arcRoot 'Launch ArcForge.ps1')
if ($script:arcLaunchCalls.Count -ne 1 -or $script:arcLaunchCalls[0] -ne 'http://127.0.0.1:8769') {
    throw 'Launching an already running ArcForge would start an extra service.'
}
Write-Host 'Launcher syntax, interpreter resolution and duplicate-service protection passed without starting a service or browser.'

$script:arcStopCalls = @()
$script:arcStopMode = 'normal'
$script:arcKilled = $false
function Stop-Process { $script:arcKilled = $true; throw 'The stop launcher must never kill a recorded PID.' }
function Invoke-RestMethod {
    param($Uri,$Method,$TimeoutSec)
    $script:arcStopCalls += @{ Uri=[string]$Uri; Method=[string]$Method }
    if ($Uri -eq 'http://127.0.0.1:8769/api/status' -and $Method -eq 'Get') {
        if ($script:arcStopMode -eq 'unavailable') { throw 'Service is unavailable.' }
        if ($script:arcStopMode -eq 'unrelated') { return @{ name='Another app'; local=$true } }
        return @{ name='ArcForge'; local=$true }
    }
    if ($Uri -eq 'http://127.0.0.1:8769/api/system/stop' -and $Method -eq 'Post') {
        if ($script:arcStopMode -eq 'busy') {
            $arcBusyError = [System.Management.Automation.ErrorRecord]::new(
                [System.InvalidOperationException]::new('HTTP 400'), 'ArcForgeBusy',
                [System.Management.Automation.ErrorCategory]::InvalidOperation, $null)
            $arcBusyError.ErrorDetails = [System.Management.Automation.ErrorDetails]::new(
                '{"detail":"Finish or cancel the active job before stopping the engine."}')
            throw $arcBusyError
        }
        return @{ status='stopping'; message='Mocked graceful stop acknowledged.' }
    }
    throw "Unexpected lifecycle request: $Method $Uri"
}
. (Join-Path $arcRoot 'Stop ArcForge.ps1')
if ($script:arcStopCalls.Count -ne 2 -or $script:arcStopCalls[1].Method -ne 'Post') {
    throw 'The stop launcher did not use the identified service graceful stop API.'
}
$script:arcStopCalls = @()
$script:arcStopMode = 'busy'
$arcBusyRejected = $false
try { . (Join-Path $arcRoot 'Stop ArcForge.ps1') } catch {
    $arcBusyRejected = $_.Exception.Message -like 'ArcForge did not stop:*active job*'
}
if (-not $arcBusyRejected -or $script:arcStopCalls.Count -ne 2) {
    throw 'The stop launcher did not preserve an active-job rejection.'
}
$script:arcStopCalls = @()
$script:arcStopMode = 'unrelated'
$arcOtherRejected = $false
try { . (Join-Path $arcRoot 'Stop ArcForge.ps1') } catch {
    $arcOtherRejected = $_.Exception.Message -like '*does not identify a local ArcForge*'
}
if (-not $arcOtherRejected -or $script:arcStopCalls.Count -ne 1) {
    throw 'The stop launcher sent a mutation to an unrelated service.'
}
$script:arcStopCalls = @()
$script:arcStopMode = 'unavailable'
. (Join-Path $arcRoot 'Stop ArcForge.ps1')
if ($script:arcStopCalls.Count -ne 1 -or $script:arcKilled) {
    throw 'An unavailable service must never trigger a PID kill or additional request.'
}
if ((Get-Content -Raw -LiteralPath (Join-Path $arcRoot 'Stop ArcForge.ps1')) -match 'Stop-Process|Get-Process|server\.pid') {
    throw 'The stop launcher must not depend on recorded process identifiers.'
}
Write-Host 'Graceful stop, active-job rejection, unrelated-service protection and no-PID-kill checks passed with mocked requests only.'
