$ErrorActionPreference = 'Stop'
$taskRoot = 'E:\Aditya\robot_sim'
$taskTraining = Join-Path $taskRoot 'physical_v29\training'
$taskPrior = Join-Path $taskTraining 'runs\locomotion_20261004_r30_exact_graph_execution'
$taskRun = Join-Path $taskTraining 'runs\locomotion_20261004_r31_exact_command_rehearsal'
$taskProof = Join-Path $PSScriptRoot 'continuation_proof.json'
$taskState = Get-Content -LiteralPath (Join-Path $taskPrior 'state.json') -Raw | ConvertFrom-Json
if ($taskState.status -ne 'superseded_for_diagnostic_repair' -or $null -ne $taskState.child_pid) { throw 'Prior chain is not drained.' }
if (Test-Path -LiteralPath $taskRun) { throw 'Preserve existing runs; use a fresh folder.' }
New-Item -ItemType Directory -Path $taskRun | Out-Null
$taskArgs = @('-u', '-X', 'utf8', (Join-Path $taskTraining 'supervisor_unitree.py'),
    '--output', $taskRun, '--reserve_mb', '300', '--max_hours', '18',
    '--iterations_per_block', '64', '--train_envs', '8192',
    '--migrate_run', $taskPrior, '--initial_checkpoint', $taskState.checkpoint,
    '--learning_rate', '0.0001', '--curriculum_proof', $taskProof)
$taskProcess = Start-Process -FilePath (Join-Path $taskRoot 'env_isaaclab\Scripts\python.exe') -ArgumentList $taskArgs -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskRun 'supervisor_stdout.log') -RedirectStandardError (Join-Path $taskRun 'supervisor_stderr.log')
$taskLaunch = @{ launcher_pid = $taskProcess.Id; arguments = $taskArgs; created_utc = [DateTime]::UtcNow.ToString('o') }
$taskLaunch | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $taskRun 'launch.json') -Encoding utf8
$taskDeadline = [DateTime]::UtcNow.AddSeconds(60)
do {
    Start-Sleep -Seconds 1
    $taskProcess.Refresh()
    if ($taskProcess.HasExited) { throw "New supervisor exited; inspect $taskRun" }
    $taskReady = (Test-Path -LiteralPath (Join-Path $taskRun 'chain_config.json')) -and (Test-Path -LiteralPath (Join-Path $taskRun 'state.json'))
    if ([DateTime]::UtcNow -gt $taskDeadline) { throw 'Supervisor did not finish freezing source in60 seconds.' }
} until ($taskReady)
$taskNewState = Get-Content -LiteralPath (Join-Path $taskRun 'state.json') -Raw | ConvertFrom-Json
$taskActive = @{ run = $taskRun; started_unix = $taskNewState.started_unix; supervisor_pid = $taskNewState.supervisor_pid;
    initial_full_checkpoint = $taskState.checkpoint; prior_run = $taskPrior; prior_main_updates = $taskState.total_training_iterations + $taskState.prior_training_iterations;
    status = 'starting'; started_utc = [DateTime]::UtcNow.ToString('o'); optimization_iteration = '04'; curriculum_proof = $taskProof;
    note = 'Training-only exact-command rehearsal; full actor/critic/Adam/exploration/adaptive LR retained. Fresh physical/PPO/load/capacity and current independent gates. Target8192, reserve300, fallback180s. Adaptive three-hour heartbeat active; endpoint actual5m/s or verified velocity ceiling, then pause and brief; computer stays on.' }
Copy-Item -LiteralPath (Join-Path $taskTraining 'ACTIVE_RUN.json') -Destination (Join-Path $PSScriptRoot 'ACTIVE_RUN_before_r31.json')
$taskActive | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $taskTraining 'ACTIVE_RUN.json') -Encoding utf8
$taskActive | ConvertTo-Json -Depth 6
