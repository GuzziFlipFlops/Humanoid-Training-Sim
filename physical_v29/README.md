# Physical v29 Isaac Lab handoff

Current package: `E:/Aditya/robot_sim/physical_v29`. Physical robot:1.6002 m,24 rigid links,23 controlled body axes,140 colored visual meshes. The older22-axis training asset remains separate.

## Run the fast GUI preview

```powershell
& 'E:\Aditya\robot_sim\env_isaaclab\Scripts\python.exe' -X utf8 `
  'E:\Aditya\robot_sim\physical_v29\test_joints.py' `
  --device cpu --hardware_actuators --visual_fast --seconds_per_joint 2.5 --repeat
```

Each joint makes a smooth visible sweep, then resets. Joint names and target/measured angles appear in the control panel. Front/side/three-quarter cameras, pause, repeat and stop are available. About2.5 simulated seconds per axis plus0.25 reset; rendering can extend wall time. Wrists and fingers remain fixed. Closing the window ends the preview; use `--exit_after_cycle` for one pass that closes automatically. No walking/training starts.

## Current configuration and evidence

[Actuator calculations](analysis/20260930_actuators/ACTUATOR_CALCULATIONS.md) describes `eff95_elbow7_20260930`:95% overall added transmissions/rod-linkages, elbow7 N·m sustained and operating cap at45 RPM, shoulder pitch120 RPM with480/500 RPM motor output. Factory8:1 outputs are not derated twice. `hardware_actuators.make_actuators()` applies the saved-CAD pose-dependent ankle/waist Jacobians and shared motor constraints. Independent URDF boxes alone cannot implement those constraints.

All23 axes passed strict tracking, companion-axis,24-link FK and reset checks in `validation/eff95_elbow7_20260930/joint_validation.json`. Three differential analytical/float32/shared-limit checks passed in `transmission_validation.json`. Large smooth preview evidence is separate in `validation/eff95_elbow7_20260930_visual`. Earlier ideal-efficiency actuator evidence remains in `validation/actuators_20260930`; original fixture evidence is `validation/joint_validation.json`.

Strict check:

```powershell
& 'E:\Aditya\robot_sim\env_isaaclab\Scripts\python.exe' -X utf8 `
  'E:\Aditya\robot_sim\physical_v29\test_joints.py' --headless --device cpu --hardware_actuators
```

## Scope and file ownership

Base fixed; gravity and contacts disabled. Masses are1 kg/link proxies; inertia,0.05 kg·m² armature and PD gains are test choices. Strict fixture is5 ms,120/15 on waist/hip/knee and35/4 elsewhere. Fast preview is10 ms and120/8 on heavy joints. This checks motion/control plumbing; it does not qualify load capability, collision-free travel, standing, thermal duty or battery current. Continuous torque is documented without a time/thermal model.

CAD front−Y maps to Isaac+X. Shoulder6° rest abduction is baked into geometry, q=0 is saved rest, knee flexion is positive/backward and elbow flexion negative. Hands are static official reference assemblies. Wrist hand meshes now include the missing rest rotation; eight affected visual STLs are synchronized. New full F3D independently reopens. Passive differential rods remain fixed packaging visuals; runtime mathematical closure supplies their varying ratios. Some combined waist corners are infeasible and are flagged.

`assets/actuator_limits.json` is current; `actuator_limits_pending.json` is historical. Preserve all140 `assets/meshes` files and the entire `assets/usd/configuration` directory. Use `--force_convert` after editing visuals/URDF. Exact source hashes and wrist provenance are in `assets/export_metadata.json`. Toolkit cleanup can hang after a completed report; stop only the owned test processes after confirming completion. No Git commit/push or locomotion training was performed.
