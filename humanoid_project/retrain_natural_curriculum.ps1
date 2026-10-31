param(
    [int]$InitialNumEnvs = 8192,
    [int]$VramReserveMB = 768,
    [switch]$PreflightOnly,
    [switch]$SkipRecovery,
    [switch]$GaitOnly,
    [string]$ResumeStateFile = "",
    [string]$StartStage = "",
    [string]$StartCheckpoint = "",
    [int]$StartStageIterations = 0
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
$chainName = if ($GaitOnly) { "humanoid_gait_overnight" } else { "humanoid_natural_retrain" }
$stateFile = Join-Path $logDir "$chainName`_$chainStamp.state.json"
$vramLog = Join-Path $logDir "$chainName`_$chainStamp.vram.csv"
$nvidiaSmi = (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue).Source

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Isaac Lab Python not found: $python" }
if (-not (Test-Path -LiteralPath $trainScript -PathType Leaf)) { throw "Training script not found: $trainScript" }
$assetStatusPath = Join-Path $projectRoot "assets\urdf\asset_prepare_status.json"
$preparedUrdf = Join-Path $projectRoot "assets\urdf\humanoid_5ft_isaac.urdf"
$preparedUsd = Join-Path $projectRoot "assets\usd\humanoid_5ft.usd"
if (-not (Test-Path -LiteralPath $assetStatusPath -PathType Leaf)) { throw "Missing asset audit: $assetStatusPath" }
$assetStatus = Get-Content -LiteralPath $assetStatusPath -Raw | ConvertFrom-Json
$legPitchAxis = @($assetStatus.leg_pitch_axis_isaac | ForEach-Object { [double]$_ })
$axisIsPositiveY = (
    $legPitchAxis.Count -eq 3 -and
    [math]::Abs($legPitchAxis[0]) -lt 1.0e-6 -and
    [math]::Abs($legPitchAxis[1] - 1.0) -lt 1.0e-6 -and
    [math]::Abs($legPitchAxis[2]) -lt 1.0e-6
)
if (-not $axisIsPositiveY) {
    throw "Refusing to train: corrected leg pitch axis audit is not Isaac +Y. Re-run convert_asset.ps1."
}
if ((Get-Item -LiteralPath $preparedUsd).LastWriteTimeUtc -lt (Get-Item -LiteralPath $preparedUrdf).LastWriteTimeUtc) {
    throw "Refusing to train: USD is older than the corrected URDF. Re-run convert_asset.ps1."
}

$stages = @(
    @{ Name="natural_stand"; Task="Isaac-Humanoid5ft-NaturalStand-v0"; Experiment="humanoid5ft_natural_stand"; Iterations=500; ExtensionIterations=200; ResumeFrom=$null; Checks=@(
        @{ Profile="stand"; Speed=0.0 }
    ) },
    @{ Name="natural_walk"; Task="Isaac-Humanoid5ft-NaturalWalk-v0"; Experiment="humanoid5ft_natural_walk"; Iterations=1600; ExtensionIterations=400; ResumeFrom="natural_stand"; Checks=@(
        @{ Profile="stand"; Speed=0.0 }, @{ Profile="walk"; Speed=0.5 }, @{ Profile="walk"; Speed=1.0 }
    ) },
    @{ Name="natural_run"; Task="Isaac-Humanoid5ft-NaturalRun-v0"; Experiment="humanoid5ft_natural_run"; Iterations=2500; ExtensionIterations=500; ResumeFrom="natural_walk"; Checks=@(
        @{ Profile="stand"; Speed=0.0 }, @{ Profile="walk"; Speed=1.0 },
        @{ Profile="run"; Speed=2.0 }, @{ Profile="run"; Speed=2.5 }
    ) },
    @{ Name="natural_sprint4"; Task="Isaac-Humanoid5ft-NaturalSprint4-v0"; Experiment="humanoid5ft_natural_sprint4"; Iterations=3500; ExtensionIterations=700; ResumeFrom="natural_run"; Checks=@(
        @{ Profile="stand"; Speed=0.0 }, @{ Profile="walk"; Speed=1.0 },
        @{ Profile="run"; Speed=2.5 }, @{ Profile="run"; Speed=4.0 }
    ) },
    # Recovery begins from the stationary controller, not the sprint actor.
    # This removes a strong locomotion prior from the terminal hold behavior.
    @{ Name="getup_discovery"; Task="Isaac-Humanoid5ft-GetUpDiscovery-v0"; Experiment="humanoid5ft_natural_getup_discovery"; Iterations=1800; ExtensionIterations=500; ResumeFrom="natural_stand"; Recovery=$true; Checks=@(
        @{ Profile="getup"; Speed=0.0 }
    ) },
    @{ Name="getup_refine"; Task="Isaac-Humanoid5ft-GetUpRefine-v0"; Experiment="humanoid5ft_natural_getup_refine"; Iterations=1200; ExtensionIterations=500; ResumeFrom="getup_discovery"; Recovery=$true; Checks=@(
        @{ Profile="getup"; Speed=0.0 }
    ) }
)
if ($GaitOnly) {
    # Observation/action dimensions intentionally match the validated natural
    # walk policy, allowing the gait-quality stage to warm-start from it.
    $stages = @(
        @{ Name="gait_walk"; Task="Isaac-Humanoid5ft-GaitWalk-v0"; Experiment="humanoid5ft_gait_walk"; Iterations=2400; ExtensionIterations=600; ResumeFrom=$null; Checks=@(
            @{ Profile="stand"; Speed=0.0 }, @{ Profile="walk"; Speed=1.0 }
        ) },
        @{ Name="gait_sprint4"; Task="Isaac-Humanoid5ft-GaitSprint4-v0"; Experiment="humanoid5ft_gait_sprint4"; Iterations=4200; ExtensionIterations=900; ResumeFrom="gait_walk"; Checks=@(
            @{ Profile="stand"; Speed=0.0 }, @{ Profile="walk"; Speed=1.0 },
            @{ Profile="run"; Speed=2.5 }, @{ Profile="run"; Speed=4.0 }
        ) }
    )
}
if ($SkipRecovery) { $stages = @($stages | Where-Object { -not $_.Recovery }) }
$checkpoints = @{}
$completed = @()
if ($ResumeStateFile) {
    if (-not (Test-Path -LiteralPath $ResumeStateFile -PathType Leaf)) {
        throw "Resume state file not found: $ResumeStateFile"
    }
    $resumeState = Get-Content -LiteralPath $ResumeStateFile -Raw | ConvertFrom-Json
    foreach ($property in $resumeState.checkpoints.PSObject.Properties) {
        $checkpoints[$property.Name] = [string]$property.Value
    }
    $completed = @($resumeState.completed_stages)
}
if ($StartStage) {
    $stageNames = [string[]]@($stages | ForEach-Object { $_.Name })
    $startIndex = [Array]::IndexOf($stageNames, $StartStage)
    if ($startIndex -lt 0) { throw "Unknown start stage: $StartStage" }
    $stages = @($stages[$startIndex..($stages.Count - 1)])
    if ($StartStageIterations -gt 0) { $stages[0].Iterations = $StartStageIterations }
    if ($StartCheckpoint -and -not (Test-Path -LiteralPath $StartCheckpoint -PathType Leaf)) {
        throw "Start checkpoint not found: $StartCheckpoint"
    }
}
$envCandidates = @(8192, 6144, 4096)
$candidateIndex = [Array]::IndexOf($envCandidates, $InitialNumEnvs)
if ($candidateIndex -lt 0) { throw "InitialNumEnvs must be 8192, 6144, or 4096." }
$selectedNumEnvs = $envCandidates[$candidateIndex]

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
    $arguments = @(
        $trainScript,"--task",$Stage.Task,"--headless","--num_envs",$NumEnvs,
        "--max_iterations",$iterations,"--experiment_name",$Stage.Experiment
    )
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
    $resumePath = ""
    if ($StartStage -and $stage.Name -eq $StartStage -and $StartCheckpoint) {
        $resumePath = $StartCheckpoint
    } elseif ($stage.ResumeFrom) {
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
    $promotionPassed = Test-StagePromotion -Stage $stage -Checkpoint $checkpoint
    if ((-not $promotionPassed) -and $stage.ExtensionIterations -gt 0) {
        # One bounded continuation prevents an unattended chain from stopping
        # at a near-converged checkpoint while retaining a hard final gate.
        $extension = $stage.Clone()
        $extension.Name = "$($stage.Name)_extension"
        $extension.Iterations = $stage.ExtensionIterations
        Write-State -Status "extending" -Stage $stage.Name -Message "Promotion gate failed; training $($stage.ExtensionIterations) additional iterations."
        $extensionResult = Invoke-TrainingProcess -Stage $extension -NumEnvs $selectedNumEnvs -ResumePath $checkpoint
        if (-not $extensionResult.Success) {
            Write-State -Status "failed" -Stage $stage.Name -Message "Automatic extension failed." -Stdout $extensionResult.Stdout -Stderr $extensionResult.Stderr
            throw "Automatic extension failed for '$($stage.Name)'."
        }
        $checkpoint = Get-LatestCheckpoint -Experiment $stage.Experiment -NotBefore $extensionResult.Started
        & $python $validator $checkpoint
        if ($LASTEXITCODE -ne 0) { throw "Extended checkpoint validation failed for '$($stage.Name)'." }
        $promotionPassed = Test-StagePromotion -Stage $stage -Checkpoint $checkpoint
    }
    if (-not $promotionPassed) {
        Write-State -Status "failed_gate" -Stage $stage.Name -Message "Behavioral gate still failed after its bounded extension."
        throw "Behavioral promotion gate failed for '$($stage.Name)' after its bounded extension."
    }
    $checkpoints[$stage.Name] = $checkpoint
    $completed += $stage.Name
    Write-State -Status "completed_stage" -Stage $stage.Name -Message "Checkpoint passed finite-tensor validation."
}

Write-State -Status "completed" -Stage $stages[-1].Name -Message "All corrected-kinematics natural-motion stages passed their behavioral promotion gates."
