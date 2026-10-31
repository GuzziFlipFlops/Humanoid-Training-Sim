param([switch]$Watch)
$trainingRoot = $PSScriptRoot
function Test-OwnedTrainingProcess([int]$trainingProcessId, [string]$runDirectory) {
    if (-not $trainingProcessId) { return $false }
    $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $trainingProcessId" -ErrorAction SilentlyContinue
    return [bool]($processInfo -and $processInfo.Name -like '*python*' -and
        $processInfo.CommandLine.Replace('/', '\').Contains($runDirectory.Replace('/', '\')))
}
do {
    $active = Get-Content -LiteralPath (Join-Path $trainingRoot 'ACTIVE_RUN.json') -Raw | ConvertFrom-Json
    $runState = Get-Content -LiteralPath (Join-Path $active.run 'state.json') -Raw | ConvertFrom-Json
    [PSCustomObject]@{
        Status = $runState.status
        Stage = $runState.stage
        Operation = $runState.operation
        Environments = $runState.environment_count
        NextTrainingEnvironments = $runState.next_training_environment_count
        PPOIteration = $runState.progress.iteration
        CompletedRunUpdates = $runState.total_training_iterations
        ContinuedPriorUpdates = $runState.prior_training_iterations
        CandidateQualified = [bool]$runState.candidate_qualified
        RetentionPassed = $runState.retention_passed
        VRAMFreeMB = $runState.gpu.free_mb
        RAMAvailableMB = $runState.system_memory.available_mb
        WorkerCommitMB = $runState.progress.gpu.process_private_mb
        CUDAAllocatedMB = $runState.progress.gpu.torch_allocated_mb
        ResourceRecycles = @($runState.resource_recycles).Where({ $null -ne $_ }).Count
        RampQualifyingSeconds = [math]::Round($runState.vram_ramp.qualifying_seconds, 1)
        RampNextEnvironments = $runState.vram_ramp.next_envs
        ForwardSpeed = $runState.progress.behavior.moving_forward_speed
        CompletedStepsPerSecond = $runState.progress.behavior.completed_steps_per_s
        SupervisorAlive = Test-OwnedTrainingProcess $runState.supervisor_pid $active.run
        WorkerAlive = Test-OwnedTrainingProcess $runState.child_pid $active.run
        Run = $active.run
    } | Format-List
    if ($Watch) { Start-Sleep -Seconds 10 }
} while ($Watch)
