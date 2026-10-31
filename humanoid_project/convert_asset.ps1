$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$workspaceRoot = Split-Path -Parent $projectRoot
$python = Join-Path $workspaceRoot "env_isaaclab\Scripts\python.exe"
$converter = Join-Path $workspaceRoot "IsaacLab\scripts\tools\convert_urdf.py"
$prepare = Join-Path $projectRoot "prepare_asset.py"
$urdf = Join-Path $projectRoot "assets\urdf\humanoid_5ft_isaac.urdf"
$usd = Join-Path $projectRoot "assets\usd\humanoid_5ft.usd"

& $python $prepare
if ($LASTEXITCODE -ne 0) { throw "Isaac-coordinate URDF preparation failed with exit code $LASTEXITCODE" }

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $usd) | Out-Null
& $python $converter $urdf $usd `
    --joint-stiffness 0.0 `
    --joint-damping 0.0 `
    --joint-target-type none `
    --headless
if ($LASTEXITCODE -ne 0) { throw "URDF conversion failed with exit code $LASTEXITCODE" }

Write-Host "Converted humanoid asset: $usd"
