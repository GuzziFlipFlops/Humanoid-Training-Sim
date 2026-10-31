"""Record completed smoke evidence without claiming long comparison success."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
import torch
OUT=Path(__file__).resolve().parent;TRAIN=OUT.parents[1];ROOT=OUT/'matched_command_advantage'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
names=['control_smoke','control_reload','candidate_smoke','candidate_reload']
assert all(read(ROOT/n/'training_validation.json')['passed'] and read(ROOT/n/'normalization_validation.json')['passed'] for n in names)
captures=[torch.load(ROOT/n/'first_rollout_normalization.pt',map_location='cpu',weights_only=False) for n in ('control_smoke','candidate_smoke')]
a,b=captures
parity={k:torch.equal(a[k],b[k]) for k in ('raw_advantage','old_global_normalized','labels','policy_observations','returns','values','rewards','dones','actions')}
assert all(parity.values())
assert not torch.equal(a['actual_advantage'],b['actual_advantage'])
summary=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),passed=True,
 actual_candidate_PPO_updates=8,actual_control_PPO_updates=8,environment_count=64,
 genuine_changed_update_executed=True,first_raw_rollout_bitwise_equal=parity,
 candidate_advantages_intentionally_changed=True,exact_actor_critic_Adam_exploration_load=True,
 zero_preupdate_KL=True,zero_command_mismatch=True,three_auxiliaries_batches_per_update=20,
 stable_memory=True,independent_64_128_two_seed_tests='in progress',
 gate_or_speed_breakthrough_certified=False,experimental_actor_deployed=False,
 pause=read(OUT/'diagnostic_pause.json'),frozen_source_manifest=str(OUT/'source_freeze.json'),
 failed_CPU_recording_helpers=['finalizer_failed_identity_stderr.log','finalizer_failed_shim_identity_stderr.log','finalizer_failed_serialization_stderr.log'],
 finalizer_repaired=True,final_check_due_utc='2026-10-10T09:03:00+00:00',remaining_automation_prompts=1)
(OUT/'smoke_validation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
paragraph='''
## Simulator smoke and reload verified

The new candidate completes4 genuine64-environment PPO updates and then reloads its complete checkpoint for4 more. The unchanged control does the same. Every load preserves actor,critic,Adam and exploration exactly. Pre-update KL and command mismatch are zero, and both teachers and upper refinement run20 batches/update. GPU/host guards and stable allocations pass.

The paired first rollout has bitwise-equal truthful observations,actions,rewards,dones,GAE returns,critic values and raw advantages. Only the declared candidate normalized advantages differ. This verifies that the implemented update is executed; it does not prove that it improves locomotion. The frozen-source baseline evaluation has all physical arrays and every case bitwise identical to the original.

The4096-environment64/128 tests in two learning seeds are now running sequentially with one actual CUDA worker. Full results and ordinary/diverse gates remain pending. Three CPU completion-helper setup failures (access to protected system process metadata, Windows virtualenv shim identification and JSON self-reference) were repaired; logs are preserved. They did not alter or interrupt simulator learning. The completion helper is CPU-only and cannot deploy a policy or extend the final-check budget.

The existing arm/head refinement remains active. The next check must inspect the new actual joint/target/velocity and slow-bias plots rather than infer natural motion from reward. No speed-matched mapped Unitree encoder curve or calibrated optical-gaze certification is claimed.
'''
with (OUT/'REPORT.md').open('a',encoding='utf-8') as f:f.write('\n'+paragraph)
entry='\n\n<!-- substantial_iteration_20261010_0405_started -->\n## October10,04:05UTC — substantial PPO update iteration started\n\n'+paragraph+f'\n[Report]({OUT.as_posix()}/REPORT.md). Final check remains09:03UTC; stop afterwards and complete new synchronized motor-shaft videos/tables if improvement is insufficient.\n'
paths=[TRAIN/'ITERATION_HISTORY.md',TRAIN.parents[1]/'docs/robot/TRAINING_JOURNAL_20261001.md',
 Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md'),
 Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),
 Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]
for i,p in enumerate(paths):
 assert p.exists(),p
 text=p.read_text(encoding='utf-8-sig')
 if 'substantial_iteration_20261010_0405_started' not in text:p.write_text(text+entry if i<3 else entry+'\n'+text,encoding='utf-8')
next=read(TRAIN/'automation/NEXT_CHECK.json')
next['substantial_iteration'].update(smoke_validation=str(OUT/'smoke_validation_summary.json'),
 actual_candidate_smoke_and_reload_updates=8,controlled_driver_identity=str(OUT/'controlled_driver_identity.json'),
 completion_helper=str(OUT/'finalize_iteration.py'),current_live_comparison=str(OUT/'live_comparison.json'))
next['focused_tests']=[
 'At the single final check09:03UTC pause this automation and stop owned learning at a full saved boundary, with no restart. Read the new R41 comparison/decision and exact actor hashes; diagnose real speed and all retained gates without treating the update as a breakthrough.',
 'Inspect new current/candidate/control head/shoulder/elbow angles,physical targets,velocities,arm-leg phase and noncancelling slow bias; preserve the exact useful actor and all rejected results.',
 'If progress is insufficient, fulfill the new normal-time full-CAD videos and synchronous native5ms installed-motor position/velocity/torque/mechanical-power exports,32 independent states and1m/s/max torque-RPM tables. Leave training and automation stopped and the computer on.']
next['next_tests']=next['focused_tests']
temp=TRAIN/'automation/NEXT_CHECK.tmp';temp.write_text(json.dumps(next,indent=2)+'\n');temp.replace(TRAIN/'automation/NEXT_CHECK.json')
(OUT/'started_records_receipt.json').write_text(json.dumps(dict(records=[dict(path=str(p),sha256=sha(p)) for p in paths],smoke_summary_sha256=sha(OUT/'smoke_validation_summary.json')),indent=2)+'\n')
print(json.dumps(summary,indent=2))
