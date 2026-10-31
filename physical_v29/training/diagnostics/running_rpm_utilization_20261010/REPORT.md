# Running RPM as percentage of configured maximum

Measured signed fixed-frame speed **3.494529m/s**.32 independently initialized nominal-physics trials,native200Hz,2–20s after settling,115200 valid samples per axis.No falls/resets/invalid transmissions.

Percentage=100×absolute observed RPM/configured maximum RPM at the same reference. P99 means99th percentile across all trial/time samples; peak is the observed maximum. These are modeled operating caps,not measured hardware loaded capability. No transmission-efficiency factor is applied to speed. Fixed reductions divide both observed motor RPM and its cap by the same ratio,so percentage is unchanged. The shoulder executable cap is120 outputRPM(480RPM before external4:1);metadata500RPM is not the executable cap.

## Final outputs after all reductions

Ankle/waist pitch+roll rows marked coupled compare against the configured per-axis PhysX bounding speed,NOT an independent attainable motor cap. Their true motor-cap percentages follow in the second table.

| Output | Configured RPM | P99 RPM | Peak RPM | P99 % | Peak % |
|---|---:|---:|---:|---:|---:|
| left_hip_pitch | 221.76 | 82.14 | 94.19 | 37.04% | 42.47% |
| right_hip_pitch | 221.76 | 86.32 | 104.47 | 38.93% | 47.11% |
| waist_yaw | 400.00 | 41.66 | 49.23 | 10.42% | 12.31% |
| left_hip_roll | 221.76 | 75.29 | 84.28 | 33.95% | 38.00% |
| right_hip_roll | 221.76 | 52.02 | 59.13 | 23.46% | 26.66% |
| waist_pitch (coupled bound) | 227.58 | 26.47 | 35.63 | 11.63% | 15.65% |
| left_hip_yaw | 400.00 | 82.22 | 98.86 | 20.56% | 24.71% |
| right_hip_yaw | 400.00 | 69.35 | 88.99 | 17.34% | 22.25% |
| waist_roll (coupled bound) | 347.68 | 30.71 | 38.12 | 8.83% | 10.96% |
| left_knee_pitch | 207.90 | 147.55 | 155.36 | 70.97% | 74.73% |
| right_knee_pitch | 207.90 | 135.81 | 145.87 | 65.33% | 70.16% |
| head_yaw | 45.05 | 24.11 | 29.16 | 53.53% | 64.74% |
| left_shoulder_pitch | 120.00 | 39.12 | 47.17 | 32.60% | 39.31% |
| right_shoulder_pitch | 120.00 | 22.58 | 28.58 | 18.82% | 23.81% |
| left_ankle_pitch (coupled bound) | 236.75 | 158.72 | 253.17 | 67.04% | 106.93% |
| right_ankle_pitch (coupled bound) | 236.75 | 146.66 | 186.73 | 61.95% | 78.87% |
| head_pitch | 45.05 | 19.37 | 25.59 | 43.00% | 56.81% |
| left_shoulder_roll | 400.00 | 36.25 | 41.85 | 9.06% | 10.46% |
| right_shoulder_roll | 400.00 | 22.89 | 26.06 | 5.72% | 6.52% |
| left_ankle_roll (coupled bound) | 400.29 | 143.88 | 191.77 | 35.94% | 47.91% |
| right_ankle_roll (coupled bound) | 400.29 | 117.93 | 154.99 | 29.46% | 38.72% |
| left_elbow_pitch | 45.00 | 19.89 | 23.20 | 44.21% | 51.55% |
| right_elbow_pitch | 45.00 | 42.98 | 45.00 | 95.52% | 100.00% |

## Actual differential motor-cap use

Each shaft below is the factory-geared installed output driving its CAD crank. Pose-dependent J,not a fixed ratio,maps these to final pitch/roll.

| Motor | Configured RPM | P99 RPM | Peak RPM | P99 % | Peak % | Above-cap fraction | Driving above cap samples |
|---|---:|---:|---:|---:|---:|---:|---:|
| left_ankle_motor_1 | 400 | 271.94 | 584.61 | 67.98% | 146.15% | 0.0651% | 0 |
| left_ankle_motor_2 | 400 | 337.40 | 653.25 | 84.35% | 163.31% | 0.5286% | 0 |
| right_ankle_motor_1 | 400 | 297.76 | 397.53 | 74.44% | 99.38% | 0.0000% | 0 |
| right_ankle_motor_2 | 400 | 434.88 | 593.11 | 108.72% | 148.28% | 1.3038% | 0 |
| waist_motor_1 | 400 | 50.48 | 75.97 | 12.62% | 18.99% | 0.0000% | 0 |
| waist_motor_2 | 400 | 69.97 | 84.78 | 17.49% | 21.19% | 0.0000% | 0 |

Above100% means actual observed overspeed,not additional authorized driving speed. Simulated mechanisms can move faster through momentum/external forces while their actuator brakes or applies zero torque. Native synchronized torque×velocity classification above records this;isolated overspeed does not establish a locomotion ceiling.

[Output CSV](output_rpm_utilization.csv),[motor CSV](motor_rpm_utilization.csv),[full statistics](rpm_utilization.json),[reproducible CPU analysis](analyze.py). Training and automation remain stopped;no simulator or learner restarted.
