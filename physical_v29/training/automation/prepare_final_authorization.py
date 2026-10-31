"""Persist user's one-remaining-check budget; prepare native update arguments.

Does not pause current training or write native automation TOML.
"""
import json, math, shutil, hashlib, tomllib, subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

TRAIN = Path(__file__).resolve().parent.parent
OUT = TRAIN / 'diagnostics/final_check_authorization_20261010'
OUT.mkdir(exist_ok=True)
AUTO = Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml')
native = tomllib.loads(AUTO.read_text(encoding='utf-8'))
if not (OUT / 'previous_automation.toml').exists():
    shutil.copy2(AUTO, OUT / 'previous_automation.toml')
for name in ('FIVE_HOUR_CHECK.md', 'NEXT_CHECK.json'):
    backup = OUT / ('previous_' + name)
    if not backup.exists():
        shutil.copy2(TRAIN / 'automation' / name, backup)

stamp = datetime.now(timezone.utc)
due = datetime.fromtimestamp(math.ceil((stamp + timedelta(hours=5)).timestamp()/60)*60, timezone.utc)
# Resolve the requested zone with Windows' maintained timezone rules. The Isaac
# Python environment intentionally has no tzdata package; do not modify it.
zone_command = "[TimeZoneInfo]::ConvertTime([DateTimeOffset]::Parse('" + due.isoformat() + "'), [TimeZoneInfo]::FindSystemTimeZoneById('Eastern Standard Time')).ToString('o')"
local_due = datetime.fromisoformat(subprocess.check_output(
    ['powershell', '-NoProfile', '-Command', zone_command], text=True).strip())
request_id = 'final_training_check_user_budget_20261010'
protocol = TRAIN / 'automation/FINAL_CHECK_AND_MOTOR_CAPTURE.md'

def write(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)

request = dict(request_id=request_id, recorded_utc=stamp.isoformat(),
    due_utc=due.isoformat(), due_local=local_due.isoformat(), timezone='America/New_York',
    automation_id=native['id'], target_thread_id=native['target_thread_id'],
    remaining_scheduled_prompts=1, final_check_started=False,
    stop_training_after_final_check=True, pause_automation_at_final_check_start=True,
    resume_training_requires_new_user_authorization=True,
    overrides_historical_only_stop_at_5_or_actuator_limited_4_rule=True,
    classification='user-budget stop, not success or actuator-ceiling certification',
    progress_criterion='Reproducibly higher signed fixed-frame speed with matching full direction/gait/contact and stopped/all-prior retention gates; reward/command/transient alone does not count. Record smaller improvements separately.',
    motor_capture_if_insufficient_progress=True, protocol=str(protocol),
    motor_scope='installed motor shafts before external transmissions; differential motors separately through exact CAD states; inaccessible internal rotor torque is not fabricated',
    table_commands_m_s=[1.0, 'maximum demonstrated stable measured running speed'],
    playback='normal-time playback, synchronized original 5 ms motor physics samples',
    computer_on=True, preserve_frozen_sources=True, preserve_all_checkpoints=True,
    no_change_to_current_healthy_training=True)
write(TRAIN / 'automation/FINAL_CHECK_REQUEST.json', request)

workflow = (TRAIN / 'automation/FIVE_HOUR_CHECK.md').read_text(encoding='utf-8')
override = '''## Latest user authorization: one final check, then end this attempt

October 10 UTC: the user allows exactly ONE more automation prompt in five hours and ends this training attempt after that check. Read `FINAL_CHECK_REQUEST.json` and [FINAL_CHECK_AND_MOTOR_CAPTURE.md](FINAL_CHECK_AND_MOTOR_CAPTURE.md) FIRST. Pause this same automation at the final turn's start so it cannot trigger again. Save the exact learned state, stop only the owned supervisor/worker safely, prevent restart and leave the computer on. Do not resume learning after evaluation or recording. A final check that sees improvement also ends this attempt; preserve that actor for a later explicit user authorization. This budget supersedes all older instructions to keep training/automation active while 5/4 m/s qualification is incomplete. The historical 5/4 rules below remain qualification standards, not permission to ignore the new stop.

If progress is insufficient, record NEW real-time walking-to-running videos with synchronized installed-motor position/velocity/requested and applied torque/signed mechanical power at actual 5 ms steps. This is a NEW pending video/data request. Produce per-motor torque and RPM tables at 1 m/s and maximum demonstrated stable speed; keep anatomical pivot and installed motor shaft coordinates distinct. Follow the exact CAD differential/reduction and time-alignment protocol, decode/inspect videos and save raw data. All prior videos remain preserved but do not fulfill this new synchronized motor request. No speed endpoint or actuator ceiling is implied by the user-budget stop.

'''
if '## Latest user authorization: one final check' not in workflow:
    workflow = workflow.replace('## Identity and authority', override + '## Identity and authority', 1)
workflow = workflow.replace('While qualification is incomplete, keep the same heartbeat active and leave healthy training running.',
    'Until the final scheduled check begins, leave healthy training running. At that check, the new user budget requires stopping this attempt even if qualification is incomplete.')
workflow = workflow.replace('Full-body recordings already delivered remain complete; no repeat request is pending.',
    'Prior full-body recordings remain preserved. A NEW synchronized installed-motor recording/table request is pending under the final-check user budget.')
(TRAIN / 'automation/FIVE_HOUR_CHECK.md').write_text(workflow, encoding='utf-8')

next_path = TRAIN / 'automation/NEXT_CHECK.json'
handoff = json.loads(next_path.read_text(encoding='utf-8-sig'))
priorities = [
    'FINAL AUTHORIZED CHECK: read FINAL_CHECK_REQUEST.json and FINAL_CHECK_AND_MOTOR_CAPTURE.md first. Pause the SAME automation at entry, mark this single final request started, verify fresh owned state/hashes, preserve a saved full-state boundary, stop owned training/supervisor without restart and assess fresh exact current/prior gates. This user-budget stop supersedes the old keep-running-until-5/4 rule. No additional scheduled prompt or learning continuation is authorized.',
    'Compare actual signed fixed-frame speed and complete retention against preserved certified/passing actors; reward, command and isolated recovery do not count as a speed breakthrough. If insufficient progress, record NEW normal-speed full-CAD walking-to-running videos including 1 m/s and maximum demonstrated speed with synchronized 200 Hz installed-motor angle/velocity/torque/power. Validate mapping, frame/sample timing, normal physics/action parity and decoded video; inspect mandatory upper timelines. Use exact CAD motor states for differential pairs and external reductions for other motors, not pivot RPM/torque.',
    'Deliver per-motor tables at 1 m/s and measured maximum (torque RMS/p95/peak, RPM p95/peak, signed mechanical power, simultaneous peak-power torque/RPM), exact raw CSV/NPZ/video/checkpoint/source hashes and 32-independent-state population results. Record all stop/no-restart/automation receipts, failed gates and qualification limits, append history/journals/handoff, refresh CPU inventory and leave computer on. Capture failure must never resume training; no new automation.'
]
handoff.update(last_user_instruction_utc=stamp.isoformat(), user_budget_override=request,
    final_check_request=str(TRAIN / 'automation/FINAL_CHECK_REQUEST.json'), final_capture_protocol=str(protocol),
    automation_final_run_only=True, remaining_authorized_automation_prompts=1,
    next_scheduled_check_utc=due.isoformat(), next_scheduled_check_local=local_due.isoformat(),
    stop_training_after_next_check=True, continue_until_speed_endpoint=False,
    recordings_pending=True, video_request_status='new synchronized motor-side capture requested if progress insufficient',
    final_motor_table_commands=[1.0, 'maximum stable demonstrated speed'],
    focused_tests=priorities, next_tests=priorities, focus=priorities,
    automation_verified_active=False)

old_prompt = native['prompt']
prefix = old_prompt.split('Latest dated five-hour check', 1)[0]
prefix = prefix.replace('Keep it active while qualification is incomplete.',
    'Keep it active only until the single final check below; pause it at that turn\'s start.')
prefix = prefix.replace('Stop after actual fixed-frame 5 m/s locomotion passes',
    'Historical success qualification remains actual fixed-frame 5 m/s locomotion passing')
prefix = prefix.replace('This pause-and-brief endpoint replaces the historical shutdown instruction; leave the computer on.',
    'Leave the computer on. The newest one-final-check budget overrides historical continuation rules.')
final_intro = f'''FINAL AUTHORIZED AUTOMATION — exactly one remaining scheduled prompt. The user now ends this training attempt after one final check in five hours (scheduled {due.isoformat()}, {local_due.strftime('%I:%M %p %Z')} local). Read E:/Aditya/robot_sim/physical_v29/training/automation/FINAL_CHECK_REQUEST.json and FINAL_CHECK_AND_MOTOR_CAPTURE.md FIRST. These newest instructions override every older "continue until 5/4 m/s", "keep active while incomplete" or "no repeat video request" instruction. At entry pause this SAME native automation ID {native['id']}, preserving its thread/preferences; atomically mark the final request started so another scheduled prompt is not allowed. Save a full learned-state boundary and safely stop the freshly verified owned training supervisor/worker, prevent restart, leave the computer on, and use at most one Isaac/CUDA worker for evaluation/capture only. Do not resume learning after final tests or export failures. Stop this attempt even if improvement occurs, preserving any improved actor for a later explicit user authorization. A budget stop is not locomotion success or a motor ceiling.

Assess fresh exact-checkpoint nominal/pushed/stopped/all-prior gates and independently seeded states against the prior evidence. A speed breakthrough requires reproducibly higher actual signed fixed-frame speed with direction/contact/gait and retention intact; a reward rise, target, spike or survival alone is insufficient. Record smaller measured improvements separately. If progress is insufficient, fulfill the NEW real-time walking-to-running video and synchronized motor-data request in THIS final turn, including 1 m/s and maximum stable demonstrated speed. Record installed MOTOR shafts before external transmissions, not anatomical pivots: individual CAD differential theta/velocity/requested/applied torque, external reductions for other motors, native 5 ms (200 Hz) samples, signed mechanical P=tau*omega at the same shaft/time, raw CSV/NPZ, frame/sample alignment and normal-speed encoded video. Do not fabricate inaccessible internal rotor torque or call mechanical power battery power. Deliver per-motor torque RMS/p95/peak and RPM p95/peak tables at 1 m/s and measured maximum, simultaneous peak-power torque/RPM, 32 independently seeded population statistics and clearly separate the exact video episode. Inspect full-CAD preview/decode and required upper angle/target/velocity/phase/bias timelines. Preserve all old evidence, caps, truthful inference, head/gait and Fusion queued work. The old videos do not fulfill this new synchronized motor request.

'''
context = '''Latest preserved evidence: diagnostics/adaptive_check_20261010_0149/REPORT.md, decision.json, matched_full_state_review.json, experiment_validation.json, closing_gate_review.json and final_learning_verification.json. Both earlier-state learning seeds retained 1.5 flight but failed full direction/prior gates; no diagnostic actor/reward/gain/cap/optimizer was deployed. Exact reviewed1157 gave3.420/3.434 signed and0.451/0.422 lateral at command3.5, with0.15 path,1.5 geometric flight and3m direction failures. Gates and upper QA bind1157, not newer actors. R40/39000 resumed exact full state at8192; read fresh ACTIVE_RUN/state before ownership or qualification claims. Preserve267's fresh all-prior certificate and separately passing fast actors. Do not use another long reward sweep to postpone the user's stop.

Complete final stop/checkpoint/hash/gate/video/motor-table receipts, update local history/journals/handoff and CPU-only inventory, and brief the user with links. No further scheduled prompt, new automation, automatic restart or unapproved training continuation is authorized.'''
prompt = final_intro + prefix + context
handoff['adaptive_automation_prompt'] = prompt
write(next_path, handoff)

# One upcoming local-clock slot. This heartbeat is paused at that turn's entry;
# do not rely on a perpetual interval or COUNT anchored to an unknown old DTSTART.
rrule = f'FREQ=DAILY;BYHOUR={local_due.hour};BYMINUTE={local_due.minute};BYSECOND=0'
args = dict(mode='update', id=native['id'], kind='heartbeat', name='Humanoid training - final check and motor capture',
    prompt=prompt, rrule=rrule, status='ACTIVE', targetThreadId=native['target_thread_id'])
if 'notification_policy' in native:
    args['notificationPolicy'] = native['notification_policy']
write(OUT / 'automation_update_args.json', args)
write(OUT / 'request_evidence.json', dict(request=request, source_skill='humanoid-robot-training',
    prepared_native_update=True, native_update_verified=False, production_sources_changed=False,
    protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
    preserves_model_preference='same target thread; no model override',
    schedule_scope='single upcoming local-clock slot; same native heartbeat paused immediately on final entry'))

marker = '<!-- final_check_user_budget_20261010 -->'
record = f'''{marker}
## October 10 user budget: one final check and synchronized motor exports

At {stamp.isoformat()}, the user authorizes exactly one remaining scheduled check in five hours and ends this training attempt after it. Native target slot: {local_due.strftime('%Y-%m-%d %I:%M %p %Z')} ({due.isoformat()}). Read `training/automation/FINAL_CHECK_REQUEST.json` and `FINAL_CHECK_AND_MOTOR_CAPTURE.md` first. This instruction supersedes every older requirement to continue until qualified5/4m/s or to skip new videos. Pause the same automation at final entry, save complete learned state, stop only owned training/supervisor without restart and leave the computer on. Record a user-budget stop separately from locomotion success or an actuator ceiling.

If progress is insufficient, NEW normal-time full-CAD walking-to-running videos and synchronized actual5ms installed-motor position/velocity/requested+applied torque/signed mechanical power are pending. Provide per-motor torque/RPM tables at1m/s and maximum demonstrated stable speed. Use exact CAD differential motor states and external reductions before the anatomical pivots; do not fabricate internal rotor torque or electrical power. Save episode versus population scopes,32 independent initial states,frame/sample timing,video decode/visual QA and rawCSV/NPZ. Earlier fulfilled videos remain preserved but do not satisfy this new request. Current healthy training remains unchanged until this final check; no reward,cap,gain,actor,optimizer or production source change is made by scheduling. The unchanged iteration has not shown reliable progress beyond3.5m/s with complete retention; a breakthrough in five hours is uncertain and not promised.

[Final capture and stop protocol](E:/Aditya/robot_sim/physical_v29/training/automation/FINAL_CHECK_AND_MOTOR_CAPTURE.md).
'''
records = [(TRAIN / 'ITERATION_HISTORY.md', False),
    (Path('E:/Aditya/robot_sim/docs/robot/TRAINING_JOURNAL_20261001.md'), False),
    (Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md'), False),
    (Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'), True),
    (Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md'), True)]
for path, prepend in records:
    old = path.read_text(encoding='utf-8')
    if marker not in old:
        path.write_text(record + '\n' + old if prepend else old + '\n\n' + record, encoding='utf-8')
write(OUT / 'durable_records.json', dict(recorded_utc=stamp.isoformat(), marker=marker,
    paths=[str(x) for x,_ in records], no_commit_or_push=True))
print(json.dumps(dict(prepared=True, due_local=request['due_local'], due_utc=request['due_utc'],
                     one_final_prompt=True, stop_training_after_final_check=True,
                     native_update_args=str(OUT / 'automation_update_args.json'),
                     protocol=str(protocol)), indent=2))
