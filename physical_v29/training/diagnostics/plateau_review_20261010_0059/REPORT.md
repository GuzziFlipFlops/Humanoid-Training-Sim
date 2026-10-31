# Review of the 3.5 m/s plateau

Recorded 2026-10-10T01:01:17.287979+00:00. The user asks whether to end this iteration.

The current supervisor remains running at the 3.5 m/s curriculum stage. Its source advances only when both full evaluation reports pass. The stage list already includes 4, 4.5 and 5 m/s. Failure of any earlier command keeps the candidate in the same stage; more iterations do not by themselves permit promotion.

Exact checkpoint **1139_run_350_train_377**, SHA **b3a52c00e34a2ff922c96327c1746f9e226e974f88fe72d89d37e57a981116c4**, has matching completed nominal 1140 and pushed 1141 tests, 320 robots for 20 seconds and 32 robots per command. Both current 3.5 m/s cases pass: signed forward speeds **3.4506/3.4610 m/s**, lateral speeds **0.1230/0.1106 m/s**. Every other case passes except 0.15 m/s forward net travel and 1.5 m/s geometric flight. Both full reports therefore fail. The ordinary nominal resets are not independently seeded final-endpoint evidence. These gates do not certify newer partial learning or arm motion; the earlier full upper timeline review remains bound to actor 1115.

This is a reproduced failure to retain the full skill set, with intermittent direction regression in earlier matching pairs. The exact learning cause is not established. Earlier controlled reward, teacher, arm and sampling candidates did not preserve the whole contract and were rejected; no arbitrary replacement reward is deployed. The separate 4/4.5/5 m/s extrapolation tests fail gait/direction and prove no motor ceiling. Do not treat 3.5 m/s as a hardware maximum or successful 4 m/s fallback.

Recommendation: pause/end this unchanged production iteration, save its complete learned state and independent evidence, and compare the earlier neutral-head actor 267's complete state against current-state control in bounded tests. A revised reward/retention objective should be a separately versioned, controlled test if warranted. Preserve all useful actors, head/gait/teachers and native limits. Earlier-state CPU preparation is complete; its runtime continuation is not reported as executed.

No policy, reward, gain, cap, actor/critic/Adam/exploration or process change is made in this review. Training and the automation remain active because the user asked for advice, not an explicit stop. The recommendation is not achievement of the 5 m/s or conditional 4 m/s endpoint.

Sources: [exact evidence](evidence.json), [prior controlled comparisons](../adaptive_check_20261009_2048/REPORT.md), [higher-speed physical diagnostic](../adaptive_check_20261009_1047/high_speed_extrapolation/analysis.json), and the preserved supervisor at `E:/Aditya/robot_sim/physical_v29/training/runs/locomotion_20261007_r40_goal_budget_full_state/supervisor_unitree.py`, promotion loop beginning at line 725.
