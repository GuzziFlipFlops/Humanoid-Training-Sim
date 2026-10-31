param(
    [int]$InitialNumEnvs = 8192,
    [int]$VramReserveMB = 768,
    [switch]$PreflightOnly,
    [string]$CompletedStandCheckpoint = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$workspaceRoot = Split-Path -Parent $projectRoot
$python = Join-Path $workspaceRoot "env_isaaclab\Scripts\python.exe"
$trainScript = Join-Path $projectRoot "scripts\rsl_rl\train.py"
$validator = Join-Path $projectRoot "validate_checkpoint.py"
$evaluator = Join-Path $projectRoot "evaluate_policy.py"
$logDir = Join-Path $projectRoot "logs\background"
$chainStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stateFile = Join-Path $logDir "humanoid_full_chain_$chainStamp.state.json"
$vramLog = Join-Path $logDir "humanoid_full_chain_$chainStamp.vram.csv"
$nvidiaSmi = (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue).Source

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Isaac Lab Python not found: $python" }
if (-not (Test-Path -LiteralPath $trainScript -PathType Leaf)) { throw "Training script not found: $trainScript" }

$stages = @(
    @{ Name="stand"; Task="Isaac-Humanoid5ft-Stand-v0"; Experiment="humanoid5ft_stand"; Iterations=500; ResumeFrom=$null; Checks=@(
        @{ Profile="stand"; Speed=0.0 }
    ) },
    @{ Name="walk"; Task="Isaac-Humanoid5ft-Walk-v0"; Experiment="humanoid5ft_walk"; Iterations=1600; ResumeFrom="stand"; Checks=@(
        @{ Profile="stand"; Speed=0.0 }, @{ Profile="walk"; Speed=0.5 }, @{ Profile="walk"; Speed=1.0 }
    ) },
    @{ Name="run_fast"; Task="Isaac-Humanoid5ft-RunFast-v0"; Experiment="humanoid5ft_run_fast"; Iterations=2500; ResumeFrom="walk"; Checks=@(
        @{ Profile="stand"; Speed=0.0 }, @{ Profile="walk"; Speed=1.0 },
        @{ Profile="run"; Speed=1.5 }, @{ Profile="run"; Speed=2.0 }, @{ Profile="run"; Speed=2.5 }
    ) }
)
$envCandidates = @(8192, 6144, 4096)
$candidateIndex = [Array]::IndexOf($envCandidates, $InitialNumEnvs)
if ($candidateIndex -lt 0) { throw "InitialNumEnvs must be 8192, 6144, or 4096." }
$selectedNumEnvs = $envCandidates[$candidateIndex]
$checkpoints = @{}
$completed = @()

if ($CompletedStandCheckpoint) {
    $CompletedStandCheckpoint = [System.IO.Path]::GetFullPath($CompletedStandCheckpoint)
    if (-not (Test-Path -LiteralPath $CompletedStandCheckpoint -PathType Leaf)) {
        throw "Completed standing checkpoint does not exist: $CompletedStandCheckpoint"
    }
    $checkpoints["stand"] = $CompletedStandCheckpoint
    $completed += "stand"
}

"timestamp,stage,num_envs,used_mb,total_mb,free_mb,gpu_util_percent,power_w,temperature_c" | Set-Content -LiteralPath $vramLog -Encoding UTF8

function Get-GpuMemory {
    if (-not $nvidiaSmi) { return $null }
    $line = & $nvidiaSmi --query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits 2>$null | Select-Object -First 1
    if (-not $line) { return $null }
    $parts = $line -split ','
    $used = [int]$parts[0].Trim()
    $total = [int]$parts[1].Trim()
    # Some Windows/NVIDIA driver states return "[N/A]" for power draw.  Power
    # telemetry is informational and must never terminate a training run.
    $gpuUtil = 0
    $powerW = 0.0
    $temperatureC = 0
    [void][int]::TryParse($parts[2].Trim(), [ref]$gpuUtil)
    [void][double]::TryParse($parts[3].Trim(), [ref]$powerW)
    [void][int]::TryParse($parts[4].Trim(), [ref]$temperatureC)
    return [pscustomobject]@{
        UsedMB=$used; TotalMB=$total; FreeMB=($total-$used)
        GpuUtil=$gpuUtil; PowerW=$powerW; TemperatureC=$temperatureC
    }
}

function Write-State {
    param([string]$Status,[string]$Stage,[string]$Message="",[int]$ProcessId=0,[string]$Stdout="",[string]$Stderr="")
    [ordered]@{
        timestamp=(Get-Date).ToString("o")
        status=$Status
        stage=$Stage
        message=$Message
        pid=$ProcessId
        selected_num_envs=$selectedNumEnvs
        vram_reserve_mb=$VramReserveMB
        stages=$stages
        completed_stages=$completed
        checkpoints=$checkpoints
        stdout=$Stdout
        stderr=$Stderr
        vram_log=$vramLog
    } | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath $stateFile -Encoding UTF8
}

function Get-LatestCheckpoint {
    param([string]$Experiment,[datetime]$NotBefore)
    $experimentDir = Join-Path $projectRoot "logs\rsl_rl\$Experiment"
    $run = Get-ChildItem -LiteralPath $experimentDir -Directory -ErrorAction Stop |
        Where-Object { $_.CreationTime -ge $NotBefore.AddSeconds(-10) } |
        Sort-Object CreationTime -Descending | Select-Object -First 1
    if (-not $run) { throw "No new run directory was created for $Experiment" }
    $checkpoint = Get-ChildItem -LiteralPath $run.FullName -Filter "model_*.pt" -File |
        Sort-Object { [int]($_.BaseName -replace '^model_', '') } -Descending | Select-Object -First 1
    if (-not $checkpoint) { throw "No checkpoint was produced in $($run.FullName)" }
    return $checkpoint.FullName
}

function Invoke-TrainingProcess {
    param([hashtable]$Stage,[int]$NumEnvs,[string]$ResumePath,[bool]$IsPreflight=$false)
    $suffix = if ($IsPreflight) { "preflight" } else { "train" }
    $stdout = Join-Path $logDir "$($Stage.Name)_${NumEnvs}_${suffix}_$chainStamp.stdout.log"
    $stderr = Join-Path $logDir "$($Stage.Name)_${NumEnvs}_${suffix}_$chainStamp.stderr.log"
    $iterations = if ($IsPreflight) { 8 } else { $Stage.Iterations }
    $arguments = @($trainScript,"--task",$Stage.Task,"--headless","--num_envs",$NumEnvs,"--max_iterations",$iterations)
    if ($ResumePath) { $arguments += @("--resume_path",$ResumePath) }
    $started = Get-Date
    $process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $projectRoot `
        -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    Write-State -Status $(if($IsPreflight){"preflight"}else{"running"}) -Stage $Stage.Name -ProcessId $process.Id -Stdout $stdout -Stderr $stderr

    $lowMemorySamples = 0
    $forcedFallback = $false
    while (-not $process.HasExited) {
        Start-Sleep -Seconds 5
        $process.Refresh()
        $memory = Get-GpuMemory
        if ($memory) {
            "$(Get-Date -Format o),$($Stage.Name),$NumEnvs,$($memory.UsedMB),$($memory.TotalMB),$($memory.FreeMB),$($memory.GpuUtil),$($memory.PowerW),$($memory.TemperatureC)" |
                Add-Content -LiteralPath $vramLog -Encoding UTF8
            if ($memory.FreeMB -lt $VramReserveMB) { $lowMemorySamples++ } else { $lowMemorySamples=0 }
            if ($lowMemorySamples -ge 3) {
                Stop-Process -Id $process.Id -Force
                $forcedFallback = $true
                break
            }
        }
    }
    $process.WaitForExit()
    # Isaac Sim can finish training and write its checkpoint successfully while
    # the Windows launcher reports an unreliable shutdown status.  Treat the
    # explicit RSL-RL completion marker as authoritative; a real Python/runtime
    # failure will not emit it and is still rejected.
    $trainingCompleted = $false
    if (Test-Path -LiteralPath $stdout -PathType Leaf) {
        $trainingCompleted = [bool](Select-String -LiteralPath $stdout -Pattern '^Training time:' -Quiet)
    }
    $success = (-not $forcedFallback) -and (($process.ExitCode -eq 0) -or $trainingCompleted)
    return [pscustomobject]@{
        ExitCode=$process.ExitCode; ForcedFallback=$forcedFallback; TrainingCompleted=$trainingCompleted
        Success=$success; Stdout=$stdout; Stderr=$stderr; Started=$started
    }
}

function Test-StagePromotion {
    param([hashtable]$Stage,[string]$Checkpoint)
    foreach ($check in $Stage.Checks) {
        $speedLabel = ([string]$check.Speed).Replace('.','p')
        $output = Join-Path $logDir "$($Stage.Name)_gate_$($check.Profile)_$speedLabel`_$chainStamp.json"
        $stdout = Join-Path $logDir "$($Stage.Name)_gate_$($check.Profile)_$speedLabel`_$chainStamp.stdout.log"
        $stderr = Join-Path $logDir "$($Stage.Name)_gate_$($check.Profile)_$speedLabel`_$chainStamp.stderr.log"
        $arguments = @(
            $evaluator, "--task", $Stage.Task, "--checkpoint", $Checkpoint,
            "--profile", $check.Profile, "--target_speed", $check.Speed,
            "--num_envs", 256, "--eval_steps", 500, "--output", $output, "--headless"
        )
        $gate = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $projectRoot `
            -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -Wait -PassThru
        $gatePassed = $false
        if (Test-Path -LiteralPath $output -PathType Leaf) {
            try { $gatePassed = [bool]((Get-Content $output -Raw | ConvertFrom-Json).passed) } catch { $gatePassed = $false }
        }
        if ((-not $gatePassed) -or ($gate.ExitCode -notin @(0,2))) {
            Write-State -Status "failed_gate" -Stage $Stage.Name -Message "Behavioral promotion gate failed for $($check.Profile) at $($check.Speed) m/s. See $output" -Stdout $stdout -Stderr $stderr
            return $false
        }
    }
    return $true
}

Write-State -Status "initializing" -Stage "preflight"
while ($true) {
    $preflight = Invoke-TrainingProcess -Stage $stages[0] -NumEnvs $selectedNumEnvs -ResumePath "" -IsPreflight $true
    if ($preflight.Success) { break }
    $candidateIndex++
    if ($candidateIndex -ge $envCandidates.Count) {
        Write-State -Status "failed" -Stage "preflight" -Message "8192, 6144, and 4096 environments all failed or violated the VRAM reserve."
        throw "No stable environment count was found. See $($preflight.Stderr)"
    }
    $selectedNumEnvs = $envCandidates[$candidateIndex]
    Write-State -Status "falling_back" -Stage "preflight" -Message "Trying $selectedNumEnvs environments."
}

if ($PreflightOnly) {
    Write-State -Status "preflight_passed" -Stage "stand" -Message "Training process, telemetry, completion detection, and VRAM checks passed."
    Write-Host "Preflight passed at $selectedNumEnvs environments."
    return
}

foreach ($stage in $stages) {
    if ($stage.Name -eq "stand" -and $CompletedStandCheckpoint) {
        Write-State -Status "completed_stage" -Stage "stand" -Message "Using previously trained and validated standing checkpoint."
        continue
    }
    $resumePath = ""
    if ($stage.ResumeFrom) {
        $resumePath = $checkpoints[$stage.ResumeFrom]
        if (-not $resumePath -or -not (Test-Path -LiteralPath $resumePath -PathType Leaf)) {
            throw "Required checkpoint '$($stage.ResumeFrom)' is unavailable before '$($stage.Name)'."
        }
    }

    while ($true) {
        $result = Invoke-TrainingProcess -Stage $stage -NumEnvs $selectedNumEnvs -ResumePath $resumePath
        if ($result.Success) { break }
        $candidateIndex++
        if ($candidateIndex -ge $envCandidates.Count) {
            Write-State -Status "failed" -Stage $stage.Name -Message "Stage failed at the minimum 4096 environments." -Stdout $result.Stdout -Stderr $result.Stderr
            throw "Stage '$($stage.Name)' failed. See $($result.Stderr)"
        }
        $selectedNumEnvs = $envCandidates[$candidateIndex]
        Write-State -Status "falling_back" -Stage $stage.Name -Message "Retrying at $selectedNumEnvs environments." -Stdout $result.Stdout -Stderr $result.Stderr
    }

    $checkpoint = Get-LatestCheckpoint -Experiment $stage.Experiment -NotBefore $result.Started
    & $python $validator $checkpoint
    if ($LASTEXITCODE -ne 0) {
        Write-State -Status "failed" -Stage $stage.Name -Message "Checkpoint validation failed: $checkpoint"
        throw "Checkpoint validation failed for '$($stage.Name)'."
    }
    if (-not (Test-StagePromotion -Stage $stage -Checkpoint $checkpoint)) {
        throw "Behavioral promotion gate failed for '$($stage.Name)'. Downstream training was intentionally stopped."
    }
    $checkpoints[$stage.Name] = $checkpoint
    $completed += $stage.Name
    Write-State -Status "completed_stage" -Stage $stage.Name -Message "Checkpoint passed finite-tensor validation."
}

Write-State -Status "completed" -Stage "run_fast" -Message "All foundational humanoid curriculum stages and checkpoint validations completed."
