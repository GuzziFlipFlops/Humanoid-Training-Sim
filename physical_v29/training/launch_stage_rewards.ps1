$ErrorActionPreference = 'Stop'
$taskRoot = 'E:\Aditya\robot_sim'
$trainingDir = Join-Path $taskRoot 'physical_v29\training'
$runDir = Join-Path $trainingDir 'runs\locomotion_20261002_r23_standing_curriculum'
$testPath = Join-Path $trainingDir 'diagnostics\stage_rewards_20261002\tests.json'
$corePath = Join-Path $trainingDir 'diagnostics\stage_rewards_20261002\unitree_core\counterfactual_rewards.json'
$tests = Get-Content -Raw -LiteralPath $testPath | ConvertFrom-Json
$core = Get-Content -Raw -LiteralPath $corePath | ConvertFrom-Json
if (-not $tests.passed -or -not $core.passed) { throw 'CPU reward tests did not pass.' }
foreach ($field in $tests.source_hashes.PSObject.Properties) {
    $current = (Get-FileHash -LiteralPath (Join-Path $trainingDir $field.Name) -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($current -ne $field.Value) { throw "Source changed since tested: $($field.Name)" }
}
$actor = $tests.selected_actor
if ((Get-FileHash -LiteralPath $actor -Algorithm SHA256).Hash.ToLowerInvariant() -ne $tests.selected_actor_sha256) { throw 'Selected actor changed.' }
$old = Get-Content -Raw -LiteralPath (Join-Path $trainingDir 'ACTIVE_RUN.json') | ConvertFrom-Json
$priorState = Get-Content -Raw -LiteralPath (Join-Path $old.run 'state.json') | ConvertFrom-Json
if ($priorState.status -ne 'superseded_for_stage_reward_repair' -or $priorState.child_pid) { throw 'Old chain has not drained.' }
$liveWorkers = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -match 'physical_v29.*(worker_unitree|supervisor_unitree)\.py'
})
if ($liveWorkers.Count) { throw 'An Isaac worker or supervisor still owns the GPU.' }
if (Test-Path -LiteralPath $runDir) { throw 'Preserve existing run; do not launch duplicates.' }
New-Item -ItemType Directory -Path $runDir | Out-Null
$pythonExe = Join-Path $taskRoot 'env_isaaclab\Scripts\python.exe'
$supervisor = Join-Path $trainingDir 'supervisor_unitree.py'
$argsForTraining = @('-u','-X','utf8',$supervisor,'--output',$runDir,'--train_envs','8192','--reserve_mb','300','--max_hours','18','--iterations_per_block','64','--actor_warm_start','--initial_checkpoint',$actor)
$started = Start-Process -FilePath $pythonExe -ArgumentList $argsForTraining -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runDir 'supervisor_stdout.log') -RedirectStandardError (Join-Path $runDir 'supervisor_stderr.log') -PassThru
$deadline = [DateTime]::UtcNow.AddSeconds(30)
do {
    Start-Sleep -Milliseconds 250
    $statePath = Join-Path $runDir 'state.json'
    if (Test-Path -LiteralPath $statePath) {
        $state = Get-Content -Raw -LiteralPath $statePath | ConvertFrom-Json
        $actual = Get-CimInstance Win32_Process -Filter "ProcessId=$($state.supervisor_pid)"
        if ($actual -and $actual.CommandLine.Contains($supervisor) -and $actual.CommandLine.Contains($runDir)) { break }
    }
} while ([DateTime]::UtcNow -lt $deadline)
if (-not $actual) { throw 'Could not verify new supervisor; inspect logs before retrying.' }
$active = [ordered]@{
    run=$runDir; supervisor_pid=$state.supervisor_pid; started_unix=$state.started_unix
    prior_run=$old.run; initial_actor_checkpoint=$actor
    note='V29 standing curriculum + declared pinned G1 locomotion adaptation; selected R22 150-update actor, std .35, fresh critic/Adam; horizontal support metric; fresh gates; target8192, reserve300MiB, ramp180s; no repeated automation'
}
$tmp = Join-Path $trainingDir 'ACTIVE_RUN.json.r23.tmp'
$active | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $tmp -Encoding utf8
Move-Item -LiteralPath $tmp -Destination (Join-Path $trainingDir 'ACTIVE_RUN.json') -Force
[ordered]@{run=$runDir;supervisor_pid=$state.supervisor_pid;wrapper_pid=$started.Id;status=$state.status;initial_actor=$actor;requested_environments=8192} | ConvertTo-Json
