"""New user-authorized iteration does not extend the one final-check budget."""
import json
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;TRAIN=OUT.parents[1];AUTO=TRAIN/'automation'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
request=read(AUTO/'FINAL_CHECK_REQUEST.json')
assert request['due_utc']=='2026-10-10T09:03:00+00:00' and request['remaining_scheduled_prompts']==1
request.update(no_change_to_current_healthy_training=False,
 current_iteration_change_authorized=True,current_change_request='Make a substantial iteration now for assessment at the single final automation.',
 current_iteration_report=str(OUT/'REPORT.md'),current_experiment=str(OUT/'matched_command_advantage'),
 current_iteration_version='command_conditioned_advantage_v1',final_deadline_unchanged=True)
write(AUTO/'FINAL_CHECK_REQUEST.json',request)
next=read(AUTO/'NEXT_CHECK.json')
next.update(user_budget_override=request,substantial_iteration_authorized_now=True,
 substantial_iteration=dict(status='controlled simulator comparison in progress; no experimental actor deployed',
 source_freeze=str(OUT/'source_freeze.json'),report=str(OUT/'REPORT.md'),comparison=str(OUT/'matched_command_advantage'),
 version='command_conditioned_advantage_v1',CPU_checks_passed=10,
 scope='Per-command advantage normalization replaces global normalization only for candidate learning. Raw rewards, GAE/critic targets, native physics/caps/inference/teachers and existing upper refinement remain.'),
 remaining_authorized_automation_prompts=1,next_scheduled_check_utc=request['due_utc'],
 stop_training_after_next_check=True,continue_until_speed_endpoint=False)
prior=read(TRAIN/'diagnostics/final_check_authorization_20261010/automation_update_args.json')
paragraph='''

NEW USER AUTHORIZATION: make a substantial iteration NOW for the single final check; this does not extend or reschedule the final 2026-10-10T09:03:00+00:00 deadline. Read diagnostics/substantial_iteration_20261010_0405/REPORT.md, source_freeze.json, matched_command_advantage and decision.json if present, then fresh ACTIVE_RUN/state. The new version command_conditioned_advantage_v1 changes PPO advantage scaling separately by truthful action-time command. This is a local optimization experiment, not CPO/PCGrad, not proof of unbiasedness for the original mixed objective, not a new physical reward or locomotion guarantee. Ten CPU checks pass; exact full-state control/candidate simulator smoke/reload and 64/128 tests in learning seeds347/359 are in progress. Candidate reward/GAE/critic targets, native limits/physics, teachers/head/upper20 and inference stay original. R40 remains immutable and paused only at a saved boundary during this one-worker comparison. No experimental actor is deployed merely from this dated paragraph. Read actual completion/decision/lineage and fresh owned state at the final check. If R41 is selected, stop its freshly owned processes; if the comparison failed, preserve evidence and stop whichever learner is actually active. No second automation or post-final learning is authorized.
'''
prior['prompt']=prior['prompt']+paragraph
assert prior['rrule']=='FREQ=DAILY;BYHOUR=5;BYMINUTE=3;BYSECOND=0'
write(OUT/'automation_update_args.json',prior)
next['adaptive_automation_prompt']=prior['prompt'];write(AUTO/'NEXT_CHECK.json',next)
report='''# Substantial iteration — October10,04:05UTC

The user authorizes a substantial change now, while keeping the single final check at09:03UTC and stopping this attempt afterwards. No breakthrough is promised.

The new frozen R41 source changes only the worker's declared training-update integration and adds command_advantage.py. Per-command normalized PPO advantages replace global normalization for the candidate. The original global method remains the matched control. This is experimental task scaling and may change the mixed-command clipped objective. It is not CPO, PCGrad, an unbiasedness proof or a retention guarantee.

The concrete lead is earlier paired evidence that global normalization transfers a slow-reward change into fast-command gradients even when fast returns are identical. Ten CPU fixtures pass, including exact latest-history command labeling, group separation, affine invariance, singleton/constant safety, disabled bitwise parity and enabled advantages-only mutation.

Frozen rewardv8, unchanged GAE/critic targets, truthful400-input/23-output inference, native CAD limits/physics/local ground, both teachers, forward head,35/1 arm gains and all three20-batch auxiliaries remain. Actor,critic,Adam and exploration load exactly. The changed training update has its own version/hash/manifest; the old reward signature does not certify it.

The simulator comparison is IN PROGRESS: exact-current full-state baseline replay,4-update64-env smoke/reload,4096-env64/128 controls and candidate in seeds347/359, ordinary and32-individually-seeded nominal/pushed tests with every original stopped/prior/current gate. No diagnostic actor has entered ordinary training. Read completion/decision rather than treating this note as success.

All checkpoint/source/log/telemetry evidence is preserved. At the final user-budget check stop training and this same automation, prevent restart, leave the computer on, and fulfill new real-time videos and installed motor-shaft telemetry/tables if improvement is insufficient. The earlier videos do not fulfill that new synchronized request.
'''
(OUT/'REPORT.md').write_text(report)
print(json.dumps(dict(due_utc=request['due_utc'],remaining_prompts=1,native_update_args=str(OUT/'automation_update_args.json'))))
