$ErrorActionPreference = 'Stop'
$taskRoot = 'E:\Aditya\robot_sim'
$trainingDir = Join-Path $taskRoot 'physical_v29\training'
$runDir = Join-Path $trainingDir 'runs\locomotion_20261003_r25_owned_observations'
$tests = Get-Content -LiteralPath (Join-Path $trainingDir 'diagnostics\walk_regression_20261003\phase_tests\tests.json') -Raw | ConvertFrom-Json
$core = Get-Content -LiteralPath (Join-Path $trainingDir 'diagnostics\walk_regression_20261003\unitree_core\counterfactual_rewards.json') -Raw | ConvertFrom-Json
if (-not $tests.passed -or -not $core.passed) { throw 'Counterfactual/core tests did not pass.' }
$snapshots = Get-Content -LiteralPath (Join-Path $trainingDir 'diagnostics\walk_regression_20261003\snapshot_tests\observation_snapshot_tests.json') -Raw | ConvertFrom-Json
if (-not $snapshots.passed) { throw 'Observation ownership regression did not pass.' }
foreach ($field in $tests.source_hashes.PSObject.Properties) {
    $current = (Get-FileHash -LiteralPath (Join-Path $trainingDir $field.Name) -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($current -ne $field.Value) { throw "Source changed since tested: $($field.Name)" }
}
$actor = $tests.selected_actor
if ((Get-FileHash -LiteralPath $actor -Algorithm SHA256).Hash.ToLowerInvariant() -ne $tests.selected_actor_sha256) { throw 'Standing actor bytes changed.' }
$old = Get-Content -LiteralPath (Join-Path $trainingDir 'ACTIVE_RUN.json') -Raw | ConvertFrom-Json
$priorState = Get-Content -LiteralPath (Join-Path $old.run 'state.json') -Raw | ConvertFrom-Json
if ($priorState.status -ne 'superseded_for_research_repair' -or $priorState.child_pid) { throw 'Previous chain has not drained.' }
$liveWorkers = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -match 'physical_v29.*(worker_unitree|supervisor_unitree)\.py'
})
if ($liveWorkers.Count) { throw 'A physical-v29 Isaac worker or supervisor still owns the GPU.' }
if (Test-Path -LiteralPath $runDir) { throw 'Preserve existing run; do not duplicate this launch.' }
New-Item -ItemType Directory -Path $runDir | Out-Null
$pythonExe = Join-Path $taskRoot 'env_isaaclab\Scripts\python.exe'
$supervisor = Join-Path $trainingDir 'supervisor_unitree.py'
$launchArgs = @('-u','-X','utf8',$supervisor,'--output',$runDir,'--train_envs','8192','--reserve_mb','300','--max_hours','18','--iterations_per_block','64','--actor_warm_start','--initial_checkpoint',$actor)
$started = Start-Process -FilePath $pythonExe -ArgumentList $launchArgs -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runDir 'supervisor_stdout.log') -RedirectStandardError (Join-Path $runDir 'supervisor_stderr.log') -PassThru
$deadline = [DateTime]::UtcNow.AddSeconds(30)
do {
    Start-Sleep -Milliseconds 250
    $statePath = Join-Path $runDir 'state.json'
    if (Test-Path -LiteralPath $statePath) {
        $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
        $actual = Get-CimInstance Win32_Process -Filter "ProcessId=$($state.supervisor_pid)"
        if ($actual -and $actual.CommandLine.Contains($supervisor) -and $actual.CommandLine.Contains($runDir)) { break }
    }
} while ([DateTime]::UtcNow -lt $deadline)
if (-not $actual) { throw 'New supervisor unverified; inspect logs before retrying.' }
$active = [ordered]@{
    run=$runDir; supervisor_pid=$state.supervisor_pid; started_unix=$state.started_unix
    prior_run=$old.run; initial_actor_checkpoint=$actor
    note='R25: own policy/critic observation snapshots across resets, eliminating reproduced stale-state PPO pairs; retain independently qualified R23 standing actor; restore phase/lift feedback before translation; latest-frame training-only standing retention; bounded PPO diagnostics and pre-update KL invariant; fresh physics/PPO/teacher/load/capacity checks and all nominal/pushed/retention gates; target8192/reserve300MiB/ramp180s'
}
$tmp = Join-Path $trainingDir 'ACTIVE_RUN.json.r25.tmp'
$active | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $tmp -Encoding utf8
Move-Item -LiteralPath $tmp -Destination (Join-Path $trainingDir 'ACTIVE_RUN.json') -Force
[ordered]@{run=$runDir;supervisor_pid=$state.supervisor_pid;wrapper_pid=$started.Id;status=$state.status;initial_actor=$actor;requested_environments=8192} | ConvertTo-Json
