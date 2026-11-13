# Stop the Phase-5 study and/or the RL runs (with all their worker processes).
#   powershell -File scripts\stop_runs.ps1            # RL only (safe: resumes from the last checkpoint)
#   powershell -File scripts\stop_runs.ps1 -Phase5    # also Phase 5 (cases in progress restart from zero)
param([switch]$Phase5)

$pattern = if ($Phase5) { 'run_rl\.py|phase5_optimal\.py' } else { 'run_rl\.py' }
$roots = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match $pattern -and $_.CommandLine -notmatch 'spawn_main' }
foreach ($p in $roots) {
    taskkill /PID $p.ProcessId /T /F 2>$null | Out-Null   # a child may already be gone with its parent
    Write-Output "stopped $($p.ProcessId): $($p.CommandLine.Substring(0, [Math]::Min(90, $p.CommandLine.Length)))"
}
if (-not $roots) { Write-Output "nothing running" }
