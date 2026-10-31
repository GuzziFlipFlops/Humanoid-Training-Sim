$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$logDir = Join-Path $projectRoot "logs\background"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $logDir "launcher_$stamp.stdout.log"
$stderr = Join-Path $logDir "launcher_$stamp.stderr.log"
$chain = Join-Path $projectRoot "train_full_curriculum.ps1"
$process = Start-Process -FilePath "powershell.exe" -ArgumentList @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $chain,
    "-InitialNumEnvs", "8192", "-VramReserveMB", "768"
) -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr -PassThru
[ordered]@{
    timestamp=(Get-Date).ToString("o")
    pid=$process.Id
    stdout=$stdout
    stderr=$stderr
    chain=$chain
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logDir "launcher_$stamp.json") -Encoding UTF8
Write-Host "Humanoid curriculum launcher PID: $($process.Id)"
Write-Host "State files will appear under: $logDir"
