param(
    [int]$InitialNumEnvs = 8192,
    [int]$VramReserveMB = 512,
    [int]$SprintIterations = 3200,
    [int]$GetUpDiscoveryIterations = 1800,
    [int]$GetUpRefineIterations = 1000,
    [string]$RunCheckpoint = "E:\Aditya\robot_sim\humanoid_project\logs\rsl_rl\humanoid5ft_run_fast\2026-08-30_11-09-42\model_4597.pt",
    [string]$StandCheckpoint = "E:\Aditya\robot_sim\humanoid_project\logs\rsl_rl\humanoid5ft_stand\2026-08-30_09-12-00\model_499.pt",
    [string]$GetUpDiscoveryCheckpoint = "",
    [switch]$ResumeGetUp,
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$workspaceRoot = Split-Path -Parent $projectRoot
$python = Join-Path $workspaceRoot "env_isaaclab\Scripts\python.exe"
$trainScript = Join-Path $projectRoot "scripts\rsl_rl\train.py"
$validator = Join-Path $projectRoot "validate_checkpoint.py"
$evaluator = Join-Path $projectRoot "evaluate_policy.py"
$logDir = Join-Path $projectRoot "logs\background"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stateFile = Join-Path $logDir "sprint_getup_$stamp.state.json"
$telemetryFile = Join-Path $logDir "sprint_getup_$stamp.gpu.csv"
$nvidiaSmi = (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue).Source

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
foreach ($required in @($python, $trainScript, $validator, $evaluator, $RunCheckpoint, $StandCheckpoint)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Required file not found: $required" }
}

$allStages = @(
    @{ Name="sprint4"; Task="Isaac-Humanoid5ft-Sprint4-v0"; Experiment="humanoid5ft_sprint4"; Iterations=$SprintIterations; ResumeFrom="run_fast"; Checks=@(
        @{ Profile="run"; Speed=2.5 }, @{ Profile="run"; Speed=3.0 },
        @{ Profile="run"; Speed=3.5 }, @{ Profile="run"; Speed=4.0 }
    ) },
    @{ Name="getup_discovery"; Task="Isaac-Humanoid5ft-GetUpDiscovery-v0"; Experiment="humanoid5ft_getup_discovery"; Iterations=$GetUpDiscoveryIterations; ResumeFrom="stand"; Checks=@() },
    @{ Name="getup_refine"; Task="Isaac-Humanoid5ft-GetUpRefine-v0"; Experiment="humanoid5ft_getup_refine"; Iterations=$GetUpRefineIterations; ResumeFrom="getup_discovery"; Checks=@(
        @{ Profile="getup"; Speed=0.0 }
    ) }
)
$stages = $allStages
$envCandidates = @(8192, 6144, 4096)
$candidateIndex = [Array]::IndexOf($envCandidates, $InitialNumEnvs)
if ($candidateIndex -lt 0) { throw "InitialNumEnvs must be 8192, 6144, or 4096." }
$selectedNumEnvs = $envCandidates[$candidateIndex]
$checkpoints = @{ run_fast=$RunCheckpoint; stand=$StandCheckpoint }
$completed = @("run_fast", "stand")
if ($ResumeGetUp) {
    if (-not $GetUpDiscoveryCheckpoint) { throw "-GetUpDiscoveryCheckpoint is required with -ResumeGetUp." }
    if (-not (Test-Path -LiteralPath $GetUpDiscoveryCheckpoint -PathType Leaf)) { throw "Required file not found: $GetUpDiscoveryCheckpoint" }
    $stages = @($allStages | Where-Object { $_.Name -in @("getup_discovery", "getup_refine") })
    $stages[0].ResumeFrom = "getup_resume"
    $checkpoints.getup_resume = $GetUpDiscoveryCheckpoint
    $completed += "sprint4"
}

"timestamp,stage,num_envs,used_mb,total_mb,free_mb,gpu_util_percent,power_w,temperature_c" |
    Set-Content -LiteralPath $telemetryFile -Encoding UTF8

function Get-GpuTelemetry {
    if (-not $nvidiaSmi) { return $null }
    $line = & $nvidiaSmi --query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits 2>$null | Select-Object -First 1
    if (-not $line) { return $null }
    $parts = $line -split ','
    $used=0; $total=0; $util=0; $power=0.0; $temperature=0
    [void][int]::TryParse($parts[0].Trim(), [ref]$used)
    [void][int]::TryParse($parts[1].Trim(), [ref]$total)
    [void][int]::TryParse($parts[2].Trim(), [ref]$util)
    [void][double]::TryParse($parts[3].Trim(), [ref]$power)
    [void][int]::TryParse($parts[4].Trim(), [ref]$temperature)
    return [pscustomobject]@{UsedMB=$used;TotalMB=$total;FreeMB=($total-$used);Util=$util;Power=$power;Temperature=$temperature}
}

function Write-State {
    param([string]$Status,[string]$Stage,[string]$Message="",[int]$ProcessId=0,[string]$Stdout="",[string]$Stderr="")
    [ordered]@{
        timestamp=(Get-Date).ToString("o"); status=$Status; stage=$Stage; message=$Message; pid=$ProcessId
        selected_num_envs=$selectedNumEnvs; vram_reserve_mb=$VramReserveMB; stages=$stages
        completed_stages=$completed; checkpoints=$checkpoints; stdout=$Stdout; stderr=$Stderr
        telemetry=$telemetryFile
    } | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath $stateFile -Encoding UTF8
}

function Get-LatestCheckpoint {
    param([string]$Experiment,[datetime]$NotBefore)
    $root = Join-Path $projectRoot "logs\rsl_rl\$Experiment"
    $run = Get-ChildItem -LiteralPath $root -Directory -ErrorAction Stop |
        Where-Object { $_.CreationTime -ge $NotBefore.AddSeconds(-10) } |
        Sort-Object CreationTime -Descending | Select-Object -First 1
    if (-not $run) { throw "No new run directory was created for $Experiment" }
    $checkpoint = Get-ChildItem -LiteralPath $run.FullName -Filter "model_*.pt" -File |
        Sort-Object { [int]($_.BaseName -replace '^model_', '') } -Descending | Select-Object -First 1
    if (-not $checkpoint) { throw "No checkpoint was produced in $($run.FullName)" }
    return $checkpoint.FullName
}

function Invoke-Training {
    param([hashtable]$Stage,[int]$NumEnvs,[string]$ResumePath,[bool]$IsPreflight=$false)
    $kind = if($IsPreflight){"preflight"}else{"train"}
    $stdout = Join-Path $logDir "$($Stage.Name)_${NumEnvs}_${kind}_$stamp.stdout.log"
    $stderr = Join-Path $logDir "$($Stage.Name)_${NumEnvs}_${kind}_$stamp.stderr.log"
    $iterations = if($IsPreflight){8}else{$Stage.Iterations}
    $arguments = @($trainScript,"--task",$Stage.Task,"--headless","--num_envs",$NumEnvs,"--max_iterations",$iterations,"--resume_path",$ResumePath)
    $started = Get-Date
    $process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    Write-State -Status $(if($IsPreflight){"preflight"}else{"running"}) -Stage $Stage.Name -ProcessId $process.Id -Stdout $stdout -Stderr $stderr
    $lowMemorySamples=0; $forcedFallback=$false
    while(-not $process.HasExited){
        Start-Sleep -Seconds 5; $process.Refresh(); $gpu=Get-GpuTelemetry
        if($gpu){
            "$(Get-Date -Format o),$($Stage.Name),$NumEnvs,$($gpu.UsedMB),$($gpu.TotalMB),$($gpu.FreeMB),$($gpu.Util),$($gpu.Power),$($gpu.Temperature)" | Add-Content $telemetryFile -Encoding UTF8
            if($gpu.FreeMB -lt $VramReserveMB){$lowMemorySamples++}else{$lowMemorySamples=0}
            if($lowMemorySamples -ge 3){Stop-Process -Id $process.Id -Force;$forcedFallback=$true;break}
        }
    }
    $process.WaitForExit()
    $completedMarker = (Test-Path $stdout) -and [bool](Select-String $stdout -Pattern '^Training time:' -Quiet)
    return [pscustomobject]@{Success=((-not $forcedFallback)-and(($process.ExitCode -eq 0)-or$completedMarker));Started=$started;Stdout=$stdout;Stderr=$stderr}
}

function Test-Gates {
    param([hashtable]$Stage,[string]$Checkpoint)
    foreach($check in $Stage.Checks){
        $speedLabel=([string]$check.Speed).Replace('.','p')
        $output=Join-Path $logDir "$($Stage.Name)_gate_$($check.Profile)_${speedLabel}_$stamp.json"
        $stdout=Join-Path $logDir "$($Stage.Name)_gate_$($check.Profile)_${speedLabel}_$stamp.stdout.log"
        $stderr=Join-Path $logDir "$($Stage.Name)_gate_$($check.Profile)_${speedLabel}_$stamp.stderr.log"
        $args=@($evaluator,"--task",$Stage.Task,"--checkpoint",$Checkpoint,"--profile",$check.Profile,"--target_speed",$check.Speed,"--num_envs",256,"--eval_steps",500,"--output",$output,"--headless")
        $gate=Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -Wait -PassThru
        $gatePassed=$false
        if(Test-Path -LiteralPath $output -PathType Leaf){
            try{$gatePassed=[bool]((Get-Content $output -Raw|ConvertFrom-Json).passed)}catch{$gatePassed=$false}
        }
        if((-not $gatePassed)-or($gate.ExitCode -notin @(0,2))){Write-State -Status "failed_gate" -Stage $Stage.Name -Message "Gate failed for $($check.Profile) at $($check.Speed)." -Stdout $stdout -Stderr $stderr;return $false}
    }
    return $true
}

Write-State -Status "initializing" -Stage "preflight"
while($true){
    $preflightResume = $checkpoints[$stages[0].ResumeFrom]
    $preflight=Invoke-Training -Stage $stages[0] -NumEnvs $selectedNumEnvs -ResumePath $preflightResume -IsPreflight $true
    if($preflight.Success){break}
    $candidateIndex++
    if($candidateIndex -ge $envCandidates.Count){Write-State -Status "failed" -Stage "preflight" -Message "All environment counts failed.";throw "No stable environment count found."}
    $selectedNumEnvs=$envCandidates[$candidateIndex]
}
if($PreflightOnly){Write-State -Status "preflight_passed" -Stage "sprint4";return}

foreach($stage in $stages){
    $resume=$checkpoints[$stage.ResumeFrom]
    if(-not $resume){throw "Missing checkpoint $($stage.ResumeFrom)"}
    while($true){
        $result=Invoke-Training -Stage $stage -NumEnvs $selectedNumEnvs -ResumePath $resume
        if($result.Success){break}
        $candidateIndex++
        if($candidateIndex -ge $envCandidates.Count){Write-State -Status "failed" -Stage $stage.Name -Stdout $result.Stdout -Stderr $result.Stderr;throw "Stage $($stage.Name) failed."}
        $selectedNumEnvs=$envCandidates[$candidateIndex]
    }
    $checkpoint=Get-LatestCheckpoint -Experiment $stage.Experiment -NotBefore $result.Started
    & $python $validator $checkpoint
    if($LASTEXITCODE -ne 0){throw "Invalid checkpoint: $checkpoint"}
    if(-not(Test-Gates -Stage $stage -Checkpoint $checkpoint)){throw "Behavior gate failed for $($stage.Name)."}
    $checkpoints[$stage.Name]=$checkpoint; $completed+=$stage.Name
    Write-State -Status "completed_stage" -Stage $stage.Name -Message "Checkpoint and gates passed."
}
Write-State -Status "completed" -Stage "getup_refine" -Message "4 m/s sprint and lying-to-standing recovery workloads completed."
