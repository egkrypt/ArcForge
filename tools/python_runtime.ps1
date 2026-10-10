# Shared by setup and launch. Resolve an actual interpreter, never the py launcher PID.
function Get-ArcForgePython {
    param([Parameter(Mandatory=$true)][string]$Root)
    $arcVenvPython = Join-Path $Root '.venv\Scripts\python.exe'
    $arcCandidates = @()
    if (Test-Path -LiteralPath $arcVenvPython) {
        $arcCandidates += @{ Path=$arcVenvPython; Arguments=@() }
    } else {
        $arcPythonCommand = Get-Command python.exe -CommandType Application -ErrorAction SilentlyContinue
        if ($arcPythonCommand) { $arcCandidates += @{ Path=$arcPythonCommand.Source; Arguments=@() } }
        $arcLauncher = Get-Command py.exe -CommandType Application -ErrorAction SilentlyContinue
        if ($arcLauncher) {
            foreach ($arcTag in @('-3.12','-3.11','-3.10')) {
                $arcCandidates += @{ Path=$arcLauncher.Source; Arguments=@($arcTag) }
            }
        }
    }
    foreach ($arcCandidate in $arcCandidates) {
        try {
            $arcArguments = @($arcCandidate.Arguments) + @('-X','utf8','-c',
                "import sys; print(sys.executable); print('%s.%s' % sys.version_info[:2])")
            $arcOutput = @(& $arcCandidate.Path @arcArguments 2>$null)
            if ($LASTEXITCODE -eq 0 -and $arcOutput.Count -ge 2 -and
                $arcOutput[-1] -in @('3.10','3.11','3.12') -and (Test-Path -LiteralPath $arcOutput[0])) {
                return [string]$arcOutput[0]
            }
        } catch { }
    }
    if (Test-Path -LiteralPath $arcVenvPython) {
        throw 'The app-local .venv needs Python 3.10, 3.11 or 3.12. Recreate it with a supported interpreter before launching.'
    }
    throw 'Install Python 3.10, 3.11 or 3.12 and make python or the Windows py launcher available, then run Setup ArcForge.ps1.'
}
