$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath 'E:\Aditya\robot_sim').Path
$trainingDir = Join-Path $taskRoot 'physical_v29\training'
$runDir = Join-Path $trainingDir 'runs\locomotion_20261002_r22_unitree_nominal'
if (Test-Path -LiteralPath $runDir) { throw 'Preserve the existing run; choose a fresh run directory.' }
$activeWorkers = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match 'supervisor(_unitree)?\.py|worker(_unitree)?\.py|probe_unitree_motion\.py'
})
if ($activeWorkers.Count -ne 0) { throw 'An owned training or diagnostic process is still alive; do not overlap CUDA workers.' }
$core = Get-Content -Raw -LiteralPath (Join-Path $trainingDir 'diagnostics\unitree_termination_20261002\counterfactual_rewards.json') | ConvertFrom-Json
$lifetime = Get-Content -Raw -LiteralPath (Join-Path $trainingDir 'diagnostics\unitree_termination_20261002\lifetime_tests.json') | ConvertFrom-Json
$nominal = Get-Content -Raw -LiteralPath (Join-Path $trainingDir 'diagnostics\unitree_termination_20261002\nominal_reset_regression.json') | ConvertFrom-Json
if (-not $core.passed -or -not $lifetime.passed -or -not $nominal.passed) { throw 'Reward/reference verification failed.' }
$testedHash = $core.port_sha256
$currentHash = (Get-FileHash -LiteralPath (Join-Path $trainingDir 'unitree_reward_port.py') -Algorithm SHA256).Hash.ToLowerInvariant()
if ($currentHash -ne $testedHash) { throw 'Reward source changed after the tests.' }
$old = Get-Content -Raw -LiteralPath (Join-Path $trainingDir 'ACTIVE_RUN.json') | ConvertFrom-Json
New-Item -ItemType Directory -Path $runDir | Out-Null
$pythonExe = Join-Path $taskRoot 'env_isaaclab\Scripts\python.exe'
$supervisor = Join-Path $trainingDir 'supervisor_unitree.py'
$argsForTraining = @('-u', '-X', 'utf8', $supervisor, '--output', $runDir, '--train_envs', '8192', '--reserve_mb', '300', '--max_hours', '18', '--iterations_per_block', '150')
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
if (-not $actual) { throw 'The launcher could not verify a live supervisor. Inspect the new logs before retrying.' }
$active = [ordered]@{
    run = $runDir; supervisor_pid = $state.supervisor_pid; started_unix = $state.started_unix
    prior_run = $old.run
    note = 'Fresh Unitree G1 core with RL Gym regular floor and H2 termination -200; corrected immutable nominal origin/action contract v2; no old actor/teacher; independent gates; target8192, reserve300MiB, ramp180s'
}
$tmp = Join-Path $trainingDir 'ACTIVE_RUN.json.r22.tmp'
$active | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $tmp -Encoding utf8
Move-Item -LiteralPath $tmp -Destination (Join-Path $trainingDir 'ACTIVE_RUN.json') -Force
[ordered]@{ run = $runDir; supervisor_pid = $state.supervisor_pid; wrapper_pid = $started.Id; status = $state.status; requested_environments = 8192 } | ConvertTo-Json
