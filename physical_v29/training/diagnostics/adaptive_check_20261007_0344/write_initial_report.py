import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
out=Path(__file__).resolve().parent;read=lambda p:json.loads(Path(p).read_text())
boundary=read(out/'stopped_boundary.json');account=read(out/'throughput_and_loads.json');cpu=read(out/'resume_cpu_tests.json');g=read(out/'initial_gate_review.json')['latest']
text=f'''# Five-hour training check — October 7, 03:44 UTC

Work in progress; final decision and recovery verification will replace this draft.

R39 stopped when the 18-hour wall-time guard expired during operation501. This is an expired supervisor budget, not qualified completion or evidence of a physical motor ceiling. Original logs,55 frozen hashes and the last full actor498 remain intact. Actor498 SHA `{boundary['checkpoint_sha256']}`. Thirty-two genuine PPO updates in interrupted501 have no saved learned checkpoint and are excluded from retained progress.

The last saved pair499/500 has100% survival. At3.5 command it travels {g['nominal_cases'][-1]['signed_speed']:.3f}/{g['pushed_cases'][-1]['signed_speed']:.3f} signed m/s with {g['nominal_cases'][-1]['lateral_speed']:.3f}/{g['pushed_cases'][-1]['lateral_speed']:.3f} lateral drift; current direction remains unqualified. Original267 and separately trusted349 are retained.

Recovery maintenance adds explicit `max_hours=0` for the user-authorized goal-driven loop. Finite budgets now finish the active worker at its saved boundary rather than killing an otherwise healthy block. The600-second no-progress guard, memory reserve/fallback/ramp, stages, reward and gate classifiers remain. Resume restores the exact saved full actor/critic/Adam/exploration, stopped teacher, qualified moving teacher and narrowly validated termination bridge. All53 authored worker/physics/teacher/PPO files must match byte for byte. {cpu['test_methods']} CPU test methods pass, including construction of the real fresh Chain snapshot and rejection of changed sources or live child state. This does not qualify locomotion.

Original R39 retained10538 completed-operation updates plus33 hash-proven recovered updates;16936 inherited are preserved. The fresh learner will declare27507 inherited actual main updates. At failure, retained-sample rates were{account['active_collection_PPO_samples_s']/1000:.2f}k active collection+PPO,{account['worker_effective_samples_s']/1000:.2f}k worker-effective and{account['whole_chain_to_failure_samples_s']/1000:.2f}k whole-chain samples/s. Documented pauses total{account['excluded_pause_seconds']:.1f}seconds and are excluded. The interrupted work and stopped downtime remain separately recorded.

Current bounded diagnostic: shoulder pitch stiffness70 versus35, damping1 unchanged, native motor caps unchanged. Both branches begin exact498 full state,4096 environments,64/128 PPO updates and unchanged320-robot20-second nominal/pushed gates. An additional seed1901 nominal invocation is recorded, but unchanged nominal reset clones do not constitute32 independently seeded initial-state trials. Original signatures do not certify the diagnostic physical gain override. No experimental actor or gain is deployed.

Actual upper timelines have been saved and visually inspected: [angles](initial_upper/upper_joint_angles.png), [velocities](initial_upper/upper_joint_velocities.png), [data](initial_upper/upper_joint_time_audit.json).32 robots/command are sampled at10Hz plus16 representatives/gate at50Hz, with reset segmentation. Fast shoulder bias/attenuation and gait-locked head oscillation persist. Native head zero is torso-relative; original face/camera optical axes remain uncalibrated. Exact speed-matched Unitree encoder trajectories are unavailable, so natural/Unitree likeness is not certified.

Primary references: [Isaac Lab actuator gains and drive limits](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.actuators.html), [official Unitree G1 velocity configuration](https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py), [human arm-swing study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136043). These supply actuator definitions and qualitative/morphology-limited comparison, not matching encoder curves or a convergence guarantee.
'''
(out/'REPORT.md').write_text(text,encoding='utf-8')
qa=[]
for f in ['initial_upper/upper_joint_angles.png','initial_upper/upper_joint_velocities.png']:
 p=out/f;qa.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),visually_inspected=True))
(out/'plot_qa.json').write_text(json.dumps(dict(recorded_utc=datetime.now(timezone.utc).isoformat(),files=qa),indent=2))
