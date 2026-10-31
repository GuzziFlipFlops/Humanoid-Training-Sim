# Installed motor shaft use at 1 m/s and maximum stable running command

Same exact preserved actor: `0ad6a1cc1c018951b82d05c67d02ca7a2663eb93845030e255a04503c60a61d0`. Simulation-derived installed-shaft values, native 5ms samples, 32 independently seeded states per command, 18 measured seconds after the first2s settling window. Complete-stage retention remains unqualified. Measured speeds and every failure are in the same-actor gate reports. Separate video episodes are not population averages.

Torque columns are RMS / absolute p95 / absolute peak (N·m). RPM columns are absolute p95 / absolute peak. Peaks can occur at different times. The CSV/JSON include signed mechanical power and simultaneous torque/RPM at peak absolute power. Mechanical power is not battery power.

## Command 1m/s; measured signed forward 1.0399m/s

| Installed motor | Torque RMS / p95 / peak N·m | RPM p95 / peak | Shaft / external transmission |
|---|---:|---:|---|
| left_ankle_motor_1 | 2.843 / 6.180 / 6.947 | 42.5 / 162.6 | CAD differential output/J(q) |
| left_ankle_motor_2 | 6.624 / 14.053 / 21.800 | 137.0 / 336.0 | CAD differential output/J(q) |
| right_ankle_motor_1 | 3.264 / 7.894 / 9.645 | 64.8 / 240.5 | CAD differential output/J(q) |
| right_ankle_motor_2 | 4.272 / 11.539 / 17.946 | 61.3 / 378.4 | CAD differential output/J(q) |
| waist_motor_1 | 5.754 / 11.693 / 14.660 | 37.2 / 63.5 | CAD differential output/J(q) |
| waist_motor_2 | 5.460 / 11.979 / 13.224 | 32.7 / 61.3 | CAD differential output/J(q) |
| left_hip_pitch_motor | 1.682 / 3.086 / 4.459 | 698.0 / 1049.8 | bare DJI motor shaft before total reduction, 15:1 external, η=0.95 |
| right_hip_pitch_motor | 1.729 / 3.634 / 4.045 | 483.5 / 573.5 | bare DJI motor shaft before total reduction, 15:1 external, η=0.95 |
| waist_yaw_motor | 10.321 / 19.349 / 31.358 | 18.4 / 25.8 | installed factory-geared or servo output before external transmission, 1:1 external, η=1 |
| left_hip_roll_motor | 1.449 / 3.214 / 3.374 | 142.5 / 261.5 | bare DJI motor shaft before total reduction, 15:1 external, η=0.95 |
| right_hip_roll_motor | 1.449 / 2.960 / 4.719 | 147.8 / 329.9 | bare DJI motor shaft before total reduction, 15:1 external, η=0.95 |
| left_hip_yaw_motor | 11.243 / 21.936 / 32.131 | 16.8 / 42.7 | installed factory-geared or servo output before external transmission, 1:1 external, η=1 |
| right_hip_yaw_motor | 9.542 / 20.815 / 33.280 | 21.2 / 48.1 | installed factory-geared or servo output before external transmission, 1:1 external, η=1 |
| left_knee_pitch_motor | 1.388 / 2.472 / 4.247 | 894.0 / 1137.3 | bare DJI motor shaft before total reduction, 16:1 external, η=0.95 |
| right_knee_pitch_motor | 1.335 / 2.475 / 3.685 | 765.5 / 1551.0 | bare DJI motor shaft before total reduction, 16:1 external, η=0.95 |
| head_yaw_motor | 0.189 / 0.432 / 0.526 | 11.5 / 16.3 | installed factory-geared or servo output before external transmission, 1:1 external, η=1 |
| left_shoulder_pitch_motor | 2.175 / 3.367 / 4.159 | 147.6 / 166.3 | installed factory-geared or servo output before external transmission, 4:1 external, η=0.95 |
| right_shoulder_pitch_motor | 1.796 / 3.061 / 3.775 | 118.2 / 145.1 | installed factory-geared or servo output before external transmission, 4:1 external, η=0.95 |
| head_pitch_motor | 0.153 / 0.312 / 0.487 | 9.9 / 22.3 | installed factory-geared or servo output before external transmission, 1:1 external, η=1 |
| left_shoulder_roll_motor | 4.984 / 10.286 / 11.157 | 21.7 / 23.7 | installed factory-geared or servo output before external transmission, 1:1 external, η=1 |
| right_shoulder_roll_motor | 4.615 / 7.482 / 8.891 | 20.0 / 22.4 | installed factory-geared or servo output before external transmission, 1:1 external, η=1 |
| left_elbow_pitch_motor | 4.653 / 7.000 / 7.000 | 28.1 / 38.3 | installed factory-geared or servo output before external transmission, 1:1 external, η=1 |
| right_elbow_pitch_motor | 3.830 / 6.352 / 7.000 | 19.7 / 25.0 | installed factory-geared or servo output before external transmission, 1:1 external, η=1 |

## Command 3.5m/s; measured signed forward 3.4945m/s

| Installed motor | Torque RMS / p95 / peak N·m | RPM p95 / peak | Shaft / external transmission |
|---|---:|---:|---|
| left_ankle_motor_1 | 4.325 / 11.869 / 20.937 | 115.7 / 584.6 | CAD differential output/J(q) |
| left_ankle_motor_2 | 2.718 / 6.363 / 19.529 | 170.3 / 653.2 | CAD differential output/J(q) |
| right_ankle_motor_1 | 1.775 / 4.320 / 7.885 | 133.4 / 397.5 | CAD differential output/J(q) |
| right_ankle_motor_2 | 3.950 / 12.168 / 21.505 | 120.5 / 593.1 | CAD differential output/J(q) |
| waist_motor_1 | 5.629 / 10.914 / 14.951 | 40.1 / 76.0 | CAD differential output/J(q) |
| waist_motor_2 | 6.841 / 11.532 / 17.921 | 56.1 / 84.8 | CAD differential output/J(q) |
| left_hip_pitch_motor | 5.192 / 7.949 / 9.266 | 1052.3 / 1412.9 | bare DJI motor shaft before total reduction, 15:1 external, η=0.95 |
| right_hip_pitch_motor | 5.368 / 8.031 / 9.487 | 1170.2 / 1567.0 | bare DJI motor shaft before total reduction, 15:1 external, η=0.95 |
| waist_yaw_motor | 33.179 / 45.000 / 45.000 | 36.9 / 49.2 | installed factory-geared or servo output before external transmission, 1:1 external, η=1 |
| left_hip_roll_motor | 2.280 / 4.104 / 5.042 | 947.5 / 1264.2 | bare DJI motor shaft before total reduction, 15:1 external, η=0.95 |
| right_hip_roll_motor | 2.113 / 4.058 / 5.989 | 697.3 / 887.0 | bare DJI motor shaft before total reduction, 15:1 external, η=0.95 |
| left_hip_yaw_motor | 18.646 / 35.294 / 45.000 | 68.2 / 98.9 | installed factory-geared or servo output before external transmission, 1:1 external, η=1 |
| right_hip_yaw_motor | 19.591 / 37.295 / 45.000 | 58.2 / 89.0 | installed factory-geared or servo output before external transmission, 1:1 external, η=1 |
| left_knee_pitch_motor | 1.746 / 3.804 / 5.126 | 2119.5 / 2485.8 | bare DJI motor shaft before total reduction, 16:1 external, η=0.95 |
| right_knee_pitch_motor | 1.680 / 3.356 / 4.844 | 1965.4 / 2333.9 | bare DJI motor shaft before total reduction, 16:1 external, η=0.95 |
| head_yaw_motor | 0.546 / 0.869 / 0.999 | 20.1 / 29.2 | installed factory-geared or servo output before external transmission, 1:1 external, η=1 |
| left_shoulder_pitch_motor | 2.444 / 3.752 / 4.407 | 113.6 / 188.7 | installed factory-geared or servo output before external transmission, 4:1 external, η=0.95 |
| right_shoulder_pitch_motor | 1.936 / 3.147 / 4.061 | 77.9 / 114.3 | installed factory-geared or servo output before external transmission, 4:1 external, η=0.95 |
| head_pitch_motor | 0.209 / 0.399 / 0.705 | 14.1 / 25.6 | installed factory-geared or servo output before external transmission, 1:1 external, η=1 |
| left_shoulder_roll_motor | 7.560 / 14.611 / 17.173 | 33.9 / 41.8 | installed factory-geared or servo output before external transmission, 1:1 external, η=1 |
| right_shoulder_roll_motor | 5.588 / 8.681 / 11.544 | 20.1 / 26.1 | installed factory-geared or servo output before external transmission, 1:1 external, η=1 |
| left_elbow_pitch_motor | 4.530 / 6.327 / 7.000 | 16.7 / 23.2 | installed factory-geared or servo output before external transmission, 1:1 external, η=1 |
| right_elbow_pitch_motor | 3.252 / 7.000 / 7.000 | 39.4 / 45.0 | installed factory-geared or servo output before external transmission, 1:1 external, η=1 |

Factory-geared motors and servos use the installed output shaft; inaccessible internal rotor torque is unavailable. DJI hips/knees use bare shaft before external15:1/16:1. Shoulder pitch uses factory output before external4:1 belt. Ankle and waist motors use each exact CAD crank shaft; no constant pitch/roll allocation. Relative angle zero is not encoder-calibrated.

[Raw population NPZ](selected_diverse_nominal/native_motor_200hz.npz), [1m/s CSV](selected_diverse_nominal/motor_samples_100.csv.gz), [maximum CSV](selected_diverse_nominal/motor_samples_350.csv.gz), [all motor power and simultaneous operating points](motor_tables.csv).
