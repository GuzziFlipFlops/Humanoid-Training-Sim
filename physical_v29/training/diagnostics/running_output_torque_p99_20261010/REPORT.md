# Final output torque after all reductions

Measured signed fixed-frame speed **3.494529 m/s** at command3.5.

The previous motor-shaft table was before added transmissions. This table instead reads the saved applied_pivot_torque_Nm array directly in its named runtime joint order. No reduction is applied again. All factory gearing, added gearing/chain and the modeled transmission efficiency are already represented in these final output efforts. Ankle and waist rows are the combined differential pitch/roll outputs, not either individual motor.

32 independently initialized nominal-physics trials; native200Hz; 2–20s after settling,115200 valid samples/output. No falls,resets or invalid transmissions. P99/max use absolute applied torque including braking. These are simulation applied actuator efforts,not measured hardware values. Configured torque limits cap available torque; they do not force the controller to use that torque. Some yaw/elbow outputs reach their caps,while hip/knee loads here are lower. This alone neither establishes a hardware ceiling nor validates simulation realism.

| Final output joint | P99 absolute N·m | Maximum absolute N·m |
|---|---:|---:|
| left_hip_pitch | 124.768 | 132.041 |
| right_hip_pitch | 127.708 | 135.189 |
| waist_yaw | 45.000 | 45.000 |
| left_hip_roll | 64.864 | 71.847 |
| right_hip_roll | 72.868 | 85.341 |
| waist_pitch | 23.958 | 26.249 |
| left_hip_yaw | 43.221 | 45.000 |
| right_hip_yaw | 45.000 | 45.000 |
| waist_roll | 21.869 | 26.519 |
| left_knee_pitch | 71.043 | 77.921 |
| right_knee_pitch | 65.554 | 73.632 |
| head_yaw | 0.917 | 0.999 |
| left_shoulder_pitch | 15.320 | 16.746 |
| right_shoulder_pitch | 13.188 | 15.432 |
| left_ankle_pitch | 35.235 | 50.168 |
| right_ankle_pitch | 30.989 | 38.169 |
| head_pitch | 0.519 | 0.705 |
| left_shoulder_roll | 15.330 | 17.173 |
| right_shoulder_roll | 9.579 | 11.544 |
| left_ankle_roll | 23.434 | 35.429 |
| right_ankle_roll | 21.328 | 27.634 |
| left_elbow_pitch | 6.712 | 7.000 |
| right_elbow_pitch | 7.000 | 7.000 |

Validation: all three differential outputs reconstruct from efficiency × CAD J-transpose × individual motor torque within recorded floating-point error; all17 non-differential p99/max outputs exactly match the prior independently converted pivot statistics. Original native capture and frozen sources are preserved. Actor1064/current running case passes; complete lower-speed retention remains unqualified.

[Full-precision CSV](output_torque_p99.csv), [statistics and validation](output_torque_p99.json), [CPU analysis](analyze.py).

Training and automation remain stopped; no new simulator worker,learning or production change.
