$ErrorActionPreference = 'Stop'
$taskRoot = 'E:\Aditya\robot_sim'
$trainingRoot = Join-Path $taskRoot 'physical_v29\training'
$priorRoot = Join-Path $trainingRoot 'runs\locomotion_20261004_r28_heading_alignment_cache_fixed'
$nextRoot = Join-Path $trainingRoot 'runs\locomotion_20261004_r29_host_capacity_checked'
$diagnosticRoot = Join-Path $trainingRoot 'diagnostics\r27_review_20261003_2319'
$priorState = Get-Content -LiteralPath (Join-Path $priorRoot 'state.json') -Raw | ConvertFrom-Json
if ($priorState.status -ne 'superseded_for_diagnostic_repair' -or $null -ne $priorState.child_pid) { throw 'Prior worker has not drained' }
if (Test-Path -LiteralPath $nextRoot) { throw 'Preserve the existing next-run directory; do not launch twice' }
if (-not (Test-Path -LiteralPath $priorState.checkpoint)) { throw 'Missing continued checkpoint' }
$taskPython = Join-Path $taskRoot 'env_isaaclab\Scripts\python.exe'
$supervisorScript = Join-Path $trainingRoot 'supervisor_unitree.py'
$launchArguments = @('-u', '-X', 'utf8', $supervisorScript, '--output', $nextRoot,
    '--reserve_mb', '300', '--max_hours', '18', '--iterations_per_block', '64',
    '--train_envs', '8192', '--initial_checkpoint', $priorState.checkpoint,
    '--migrate_run', $priorRoot, '--prior_run', $priorRoot, '--learning_rate', '0.0001')
$launched = Start-Process -FilePath $taskPython -ArgumentList $launchArguments -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $diagnosticRoot 'r29_supervisor.stdout.log') `
    -RedirectStandardError (Join-Path $diagnosticRoot 'r29_supervisor.stderr.log')
Write-Output ($launched | Select-Object Id, ProcessName | ConvertTo-Json -Compress)
