# Running torque: 99th percentile and maximum

Measured signed fixed-frame speed: **3.494529 m/s**; command 3.5 m/s.

32 independently initialized nominal-physics trials, 200 Hz. Statistics pool 115,200 valid samples per motor over 2–20 seconds; the first 2 seconds are settling and excluded. No falls, resets or invalid linkage samples occur. P99 is the 99th percentile of absolute applied torque, not a confidence limit. Maximum is the largest absolute applied torque in this capture. Pushed evaluations have no corresponding native torque population here.

The old table provided p95, not p99. Its 4.25 N·m left-knee value was the **1 m/s motor-shaft maximum**, not the running knee-pivot torque. Knee shaft values are before the 8:1 planetary × 2:1 chain, using the frozen model’s 95% total external efficiency: T_pivot = T_motor × 16 × 0.95. This mapping has now been checked against the raw named pivot-effort arrays, independently of the summary table.

| Knee | Motor shaft p99 (N·m) | Motor shaft max (N·m) | Knee pivot p99 (N·m) | Knee pivot max (N·m) |
|---|---:|---:|---:|---:|
| left_knee_pitch | 4.674 | 5.126 | 71.043 | 77.921 |
| right_knee_pitch | 4.313 | 4.844 | 65.554 | 73.632 |

## Installed motor shafts

DJI hip/knee values refer to bare shafts before their added reductions. Ankle/waist differentials use each exact CAD motor state at the factory-geared crank output; other geared/servo motors are reported at the installed output before any added transmission. These are simulation values, not measured hardware or continuous-duty ratings.

| Motor | p99 absolute torque (N·m) | Maximum absolute torque (N·m) |
|---|---:|---:|
| left_ankle_motor_1 | 14.845 | 20.937 |
| left_ankle_motor_2 | 9.631 | 19.529 |
| right_ankle_motor_1 | 6.597 | 7.885 |
| right_ankle_motor_2 | 16.686 | 21.505 |
| waist_motor_1 | 13.037 | 14.951 |
| waist_motor_2 | 13.525 | 17.921 |
| left_hip_pitch_motor | 8.756 | 9.266 |
| right_hip_pitch_motor | 8.962 | 9.487 |
| waist_yaw_motor | 45.000 | 45.000 |
| left_hip_roll_motor | 4.552 | 5.042 |
| right_hip_roll_motor | 5.114 | 5.989 |
| left_hip_yaw_motor | 43.221 | 45.000 |
| right_hip_yaw_motor | 45.000 | 45.000 |
| left_knee_pitch_motor | 4.674 | 5.126 |
| right_knee_pitch_motor | 4.313 | 4.844 |
| head_yaw_motor | 0.917 | 0.999 |
| left_shoulder_pitch_motor | 4.032 | 4.407 |
| right_shoulder_pitch_motor | 3.470 | 4.061 |
| head_pitch_motor | 0.519 | 0.705 |
| left_shoulder_roll_motor | 15.330 | 17.173 |
| right_shoulder_roll_motor | 9.579 | 11.544 |
| left_elbow_pitch_motor | 6.712 | 7.000 |
| right_elbow_pitch_motor | 7.000 | 7.000 |

## Single video episode, kept separate

The synchronized 3.5-command video has one robot and 2,000 analyzed native samples over its 10-second recorded window. Its torque results are stored separately in torque_p99.json; they are not pooled into the 32-trial population above.

Actor: `E:\Aditya\robot_sim\physical_v29\training\runs\locomotion_20261007_r40_goal_budget_full_state\1064_run_350_train_352\candidate.pt`; SHA256 `0ad6a1cc1c018951b82d05c67d02ca7a2663eb93845030e255a04503c60a61d0`.

[CSV](running_torque_p99.csv), [full statistics and validation](torque_p99.json), [reproducible CPU analysis](analyze.py).

Training and automation remain stopped. No sources, checkpoints, caps, rewards or raw captures were modified.
