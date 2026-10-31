$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$logDir = Join-Path $projectRoot "logs\background"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $logDir "sprint_getup_launcher_$stamp.stdout.log"
$stderr = Join-Path $logDir "sprint_getup_launcher_$stamp.stderr.log"
$process = Start-Process powershell.exe -ArgumentList @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $projectRoot "train_sprint_then_getup.ps1"),
    "-InitialNumEnvs", "8192", "-VramReserveMB", "512"
) -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
Write-Host "Sprint/get-up chain PID: $($process.Id)"
Write-Host "stdout: $stdout"
Write-Host "stderr: $stderr"
