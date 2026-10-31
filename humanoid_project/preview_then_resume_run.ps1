param(
    [double]$PreviewSpeed = 1.8,
    [switch]$SkipPreview
)

# Preview the saved NaturalRun policy, then continue this run only.
# No gait-repair rewards, sprint stage, or old checkpoint is substituted.
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$python = Join-Path (Split-Path -Parent $projectRoot) 'env_isaaclab\Scripts\python.exe'
$play = Join-Path $projectRoot 'scripts\rsl_rl\play.py'
$train = Join-Path $projectRoot 'scripts\rsl_rl\train.py'
$checkpoint = Join-Path $projectRoot 'logs\rsl_rl\humanoid5ft_natural_run_from_working_walk\2026-09-01_19-54-34\model_3800.pt'
$experiment = 'humanoid5ft_natural_run_resume3800'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$prefix = Join-Path $projectRoot "logs\background\natural_run_resume3800_$stamp"
$stdout = "$prefix.stdout.log"
$stderr = "$prefix.stderr.log"
$stateFile = "$prefix.state.json"
$gpuFile = "$prefix.gpu.csv"
$child = $null

foreach ($file in @($python, $play, $train, $checkpoint)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing file: $file" }
}
$active = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
    $_.CommandLine -match 'humanoid_project[\\/]scripts[\\/]rsl_rl[\\/](train|play)\.py'
})
if ($active.Count -gt 0) { throw 'A humanoid viewer or trainer is already running. Close it before using this launcher.' }

function Save-State([string]$status, [string]$message, [string]$savedCheckpoint = '') {
    [ordered]@{
        timestamp = (Get-Date).ToString('o')
        status = $status
        message = $message
        pid = $(if ($child) { $child.Id } else { 0 })
        task = 'Isaac-Humanoid5ft-NaturalRun-v0'
        checkpoint_source = $checkpoint
        checkpoint_result = $savedCheckpoint
        num_envs = 8192
        additional_iterations = 1000
        initial_speed_ceiling = 1.9
        final_speed_ceiling = 2.6
        curriculum_ramp_steps = 24000
        vram_reserve_mb = 512
        stdout = $stdout
        stderr = $stderr
        gpu_csv = $gpuFile
    } | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $stateFile -Encoding UTF8
}

Push-Location $projectRoot
try {
    & $python (Join-Path $projectRoot 'validate_checkpoint.py') $checkpoint
    if ($LASTEXITCODE -ne 0) { throw 'Checkpoint validation failed; training was not started.' }
    if (-not $SkipPreview) {
        Save-State 'preview' 'Close Isaac Sim normally to start the remaining training.'
        Write-Host "Previewing model_3800 at $PreviewSpeed m/s. Close the Isaac Sim window to resume training."
        $viewer = Start-Process -FilePath $python -ArgumentList @(
            $play, '--task', 'Isaac-Humanoid5ft-NaturalRun-Play-v0',
            '--checkpoint', $checkpoint, '--num_envs', '1',
            '--target_speed', $PreviewSpeed, '--real-time'
        ) -WorkingDirectory $projectRoot -NoNewWindow -Wait -PassThru
        if ($viewer.ExitCode -ne 0) {
            throw "Viewer exited with code $($viewer.ExitCode). Training was not started. Use -SkipPreview when ready."
        }
    }

    $arguments = @(
        $train, '--task', 'Isaac-Humanoid5ft-NaturalRun-v0', '--headless',
        '--num_envs', '8192', '--max_iterations', '1000',
        '--experiment_name', $experiment, '--resume_path', $checkpoint,
        'env.commands.base_velocity.ranges.lin_vel_x=[0.3,1.9]',
        'env.curriculum.speed.params.initial_max_speed=1.9',
        'env.curriculum.speed.params.ramp_steps=24000'
    )
    $started = Get-Date
    $child = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $projectRoot `
        -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    Save-State 'running' 'Resuming 1,000 iterations with the second half of the original speed curriculum.'
    Write-Host "Training started (PID $($child.Id)). Keep this terminal open."
    Write-Host "Log: $stdout"
    'timestamp,used_mb,total_mb,free_mb,gpu_util_percent,power_w,temperature_c' | Set-Content -LiteralPath $gpuFile
    $smi = (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue).Source
    $lowMemorySamples = 0
    while (-not $child.HasExited) {
        Start-Sleep -Seconds 10
        $child.Refresh()
        if ($child.HasExited) { break }
        if ($smi) {
            $sample = & $smi --query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits 2>$null | Select-Object -First 1
            if ($sample) {
                $fields = $sample -split ','
                $used = 0; $total = 0
                if ([int]::TryParse($fields[0].Trim(), [ref]$used) -and [int]::TryParse($fields[1].Trim(), [ref]$total)) {
                    $free = $total - $used
                    "$(Get-Date -Format o),$used,$total,$free,$($fields[2].Trim()),$($fields[3].Trim()),$($fields[4].Trim())" | Add-Content -LiteralPath $gpuFile
                    if ($free -lt 512) { $lowMemorySamples++ } else { $lowMemorySamples = 0 }
                    if ($lowMemorySamples -ge 3) {
                        & taskkill.exe /PID $child.Id /T /F | Out-Null
                        throw 'Stopped the owned trainer after three samples below the 512 MB VRAM reserve. Saved checkpoints remain intact.'
                    }
                }
            }
        }
        $lastIteration = Select-String -LiteralPath $stdout -Pattern 'Learning iteration' | Select-Object -Last 1
        if ($lastIteration) { Write-Host $lastIteration.Line.Trim() }
    }
    $child.WaitForExit()
    if (-not (Select-String -LiteralPath $stdout -Pattern '^Training time:' -Quiet)) {
        throw "Training stopped without its completion marker. Inspect $stderr and $stdout"
    }
    $run = Get-ChildItem (Join-Path $projectRoot "logs\rsl_rl\$experiment") -Directory |
        Where-Object { $_.CreationTime -ge $started.AddSeconds(-10) } |
        Sort-Object CreationTime -Descending | Select-Object -First 1
    $last = Get-ChildItem $run.FullName -Filter 'model_*.pt' -File |
        Sort-Object { [int]($_.BaseName -replace '^model_', '') } -Descending | Select-Object -First 1
    & $python (Join-Path $projectRoot 'validate_checkpoint.py') $last.FullName
    if ($LASTEXITCODE -ne 0) { throw 'Final checkpoint failed validation.' }
    Save-State 'training_complete_unassessed' 'Training and tensor validation finished; running behavior still needs evaluation.' $last.FullName
    Write-Host "Training finished: $($last.FullName)"
    Write-Host 'The policy is not behaviorally approved yet. No next stage will start.'
} catch {
    Save-State 'stopped' $_.Exception.Message
    throw
} finally {
    Pop-Location
}
