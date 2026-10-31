$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$workspaceRoot = Split-Path -Parent $projectRoot
$python = Join-Path $workspaceRoot "env_isaaclab\Scripts\python.exe"
$script = Join-Path $projectRoot "scripts\test_all_joints.py"

& $python $script --seconds_per_joint 0.55 --cycles 1.0 --hold_seconds 1.0
