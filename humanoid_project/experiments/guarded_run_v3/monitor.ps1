param(
    [string]$StatePath = 'E:\Aditya\robot_sim\humanoid_project\logs\guarded_run_v4_20260903_0305\state.json',
    [switch]$Once
)
$monitorArgs = @('-u', 'E:\Aditya\robot_sim\humanoid_project\experiments\guarded_run_v4\monitor.py', '--state', $StatePath)
if ($Once) { $monitorArgs += '--once' }
& 'E:\Aditya\robot_sim\env_isaaclab\Scripts\python.exe' @monitorArgs
