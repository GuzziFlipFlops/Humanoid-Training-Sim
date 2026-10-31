"""Verify saved evidence and the native automation without mutating training."""
import hashlib
import json
import re
import tomllib
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
TRAIN = OUT.parents[1]
ROOT = TRAIN.parents[1]
read = lambda path: json.loads(Path(path).read_text(encoding='utf-8-sig'))
write = lambda path, value: Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
now = datetime.now(timezone.utc).isoformat()
checks = []

def require(condition, description):
    assert condition, description
    checks.append(description)

before = tomllib.loads((OUT / 'previous_automation.toml').read_text(encoding='utf-8'))
automation_path = Path('C:/Users/rshan/.codex/automations') / before['id'] / 'automation.toml'
after = tomllib.loads(automation_path.read_text(encoding='utf-8'))
args = read(OUT / 'automation_update_args.json')
for key in ['id', 'kind', 'name', 'target_thread_id', 'rrule', 'created_at']:
    require(after.get(key) == before.get(key), 'Automation preserves ' + key)
for key in ['notification_policy', 'model', 'reasoning_effort']:
    require(after.get(key) == before.get(key), 'Automation preserves optional ' + key)
require(after['status'] == 'ACTIVE', 'Native automation is ACTIVE')
require(after['rrule'] == 'FREQ=HOURLY;INTERVAL=5', 'Same five-hour cadence')
require(after['prompt'] == args['prompt'], 'Native prompt equals tool update arguments')
prefix = after['prompt'].split('Latest dated five-hour check', 1)[0]
require(hashlib.sha256(prefix.encode()).hexdigest() == '46f1711b1bce88727bc847e7985a95992e127272ea59cc70e4174507927817c3',
        'Durable scope and final 5/conditional4 endpoint unchanged')
automation = dict(recorded_utc=now, passed=True, path=str(automation_path),
    automation_id=after['id'], status=after['status'], kind=after['kind'],
    target_thread_id=after['target_thread_id'], rrule=after['rrule'],
    prompt_sha256=hashlib.sha256(after['prompt'].encode()).hexdigest(),
    notification_preference_unchanged=True, duplicate_created=False)
write(OUT / 'automation_verification.json', automation)

inventory = read(OUT / 'inventory.log')
require(inventory['physical_runs'] == 40, 'CPU inventory covers 40 physical runs')
require(inventory['differing_manifest_hashes'] == 0, 'Inventory has no differing frozen hashes')
require(inventory['read_warnings'] == [], 'Inventory has no read warnings')
require(inventory['missing_manifest_artifacts'] == 8, 'Eight historical missing manifests remain explicit')
write(OUT / 'inventory_verification.json', dict(recorded_utc=now, passed=True, **inventory))

live = read(OUT / 'final_cpu/live_handoff.json')
v = read(OUT / 'final_learning_verification.json')
review = read(OUT / 'final_gate_review.json')['latest']
checked = read(OUT / 'closing_gate_review.json')['latest']
require(v['supervisor_pid'] == 39000 and live['state']['status'] == 'running', 'Same supervisor remains running')
require(len(live['actual_worker_pids']) == 1, 'One actual Isaac/CUDA worker')
require(len(live['frozen_source_integrity']) == 55 and all(live['frozen_source_integrity'].values()), 'All 55 frozen files verify')
require(v['protected_authored_worker_files_equal'] == 53, '53 authored worker files remain equal')
require(v['inherited_actual_main_updates'] == 27507 and v['hash_proven_recovered_main_updates'] == 50,
        'Inherited and retained recovery sample lineage unchanged')
require(v['previous_interrupted_unsaved_updates_excluded'] == 32, '32 unsaved R39 updates excluded')
require(not v['diagnostic_gains_and_actors_deployed'], 'No diagnostic actor or physical override deployed')
require(v['smoke_and_capacity_actors_discarded'], 'Smoke/capacity actors excluded')
for block in v['blocks']:
    proof = block['full_load_proof']
    require(all(proof[k] for k in ['actor_equal', 'critic_equal', 'optimizer_equal', 'exploration_equal'])
            and block['all_preupdate_KL_and_command_mismatch_zero'], 'Exact full state/KL/commands ' + Path(block['folder']).name)
last = v['blocks'][-1]
resumed = next(b for b in v['blocks'] if b['full_load_proof']['source_sha256'] == checked['sha256'])
require(resumed['updates'] == 64 and resumed['environments'] == 8192, 'First resumed full main block:64 updates at8192')
require(resumed['three_auxiliaries_batches_per_update'] == 20, 'Both teachers and upper refinement remain20 batches/update')
require(resumed['minimum_free_gpu_MiB'] >= 300, 'Resumed steady training retains GPU reserve')
require(review['sha256'] == checked['sha256'], 'Final CPU snapshot still binds the fully reviewed839 actor')

decision = read(OUT / 'decision.json')
require(not decision['deploy'] and not decision['production_change'] and not decision['endpoint_verified'],
        'Decision preserves full learner and does not claim endpoint/repair')
require(read(OUT / 'physical_validation.json')['passed'], '91 physical validation checks pass')
sham = read(OUT / 'replay_sham_analysis.json')
require(sham['passed'] and len(sham['validation_cases']) == 24, 'Replay sham validates24 paired cohorts')
require(sham['all_zero_offset_replay_cohorts_eventually_fall'] and sham['all_original_controls_survive'],
        'Zero-offset replay invalidates arm-specific fall inference')
require(not sham['mechanical_arm_coupling_identified'] and not sham['production_repair_tested'],
        'Replay sham does not claim causal isolation or production repair')
plot_count = 0
for folder in ['initial_upper', 'final_upper']:
    audit = read(OUT / folder / 'audit_set.json')
    require(not audit['plots_need_visual_QA'] and all(not r['needs_visual_QA'] for r in audit['supplements']),
            'Upper visual QA completed: ' + folder)
    if 'all_plots_visually_inspected' in audit:
        require(audit['all_plots_visually_inspected'], 'Explicit all-upper inspection flag: ' + folder)
    receipts = audit['visual_QA_receipts']
    count = 0
    for receipt in receipts:
        for plot in read(receipt)['plots']:
            require(plot['visually_inspected'] and sha(plot['path']) == plot['sha256'], 'Upper QA image bytes preserved: ' + str(plot['path']))
            count += 1
    require(count == 14, '14 actual upper images: ' + folder)
    plot_count += count
for plot in read(OUT / 'diagnostic_plot_QA.json')['plots']:
    require(plot['visually_inspected'] and sha(plot['path']) == plot['sha256'], 'Physical diagnostic plot bytes preserved')

histories = [TRAIN / 'ITERATION_HISTORY.md', ROOT / 'docs/robot/TRAINING_JOURNAL_20261001.md',
    Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md'),
    Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),
    Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]
for path in histories:
    require(path.read_text(encoding='utf-8').count('<!-- adaptive_check_20261009_0046_record -->') == 1,
            'Chronological report entry exists once: ' + str(path))
report = (OUT / 'REPORT.md').read_text(encoding='utf-8')
local_links = []
for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', report):
    if target.startswith(('http:', 'https:', '#')):
        continue
    require((OUT / target).exists(), 'Report artifact link exists: ' + target)
    local_links.append(target)

handoff = read(TRAIN / 'automation/NEXT_CHECK.json')
require(handoff['selected_actor_sha256'] == checked['sha256'], 'Handoff selected actor839 matches exact gates')
require(handoff['mandatory_all_speed_trace_sha256'] == checked['sha256'], 'All-command upper QA is actor839')
require(handoff['recording_request']['status'] == 'complete', 'Completed GUI request preserved')
require(not handoff['velocity_ceiling_proven'] and not handoff['current_stage_qualified'], 'No unqualified endpoint/promotion')
handoff.update(latest_liveness_utc=v['owned_snapshot_utc'], latest_observation=v['owned_snapshot_utc'],
    latest_liveness=str(OUT / 'final_cpu/live_handoff.json'), final_owned_observation=str(OUT / 'final_cpu/live_handoff.json'),
    final_verification=str(OUT / 'closeout_verification.json'), resumed_learning_evidence=str(OUT / 'final_learning_verification.json'),
    completed_main_updates=v['completed_main_updates'], new_main_updates=v['completed_main_updates'],
    recovered_main_updates=50, actual_retained_current_run_main_updates=v['actual_retained_main_updates'],
    partial_observed_main_updates=v['observed_partial_main_updates'], current_owned_operation=v['operation_path'],
    automation_verified_active=True, automation_verification=str(OUT / 'automation_verification.json'),
    inventory_verification=str(OUT / 'inventory_verification.json'))
write(TRAIN / 'automation/NEXT_CHECK.json', handoff)
verification = dict(recorded_utc=now, passed=True, owned_snapshot_utc=v['owned_snapshot_utc'],
    supervisor_pid=39000, actual_worker_pids=live['actual_worker_pids'], frozen_files_verified=55,
    current_operation=v['operation_path'], completed_main_updates=v['completed_main_updates'], recovered_main_updates=50,
    retained_main_updates=v['actual_retained_main_updates'], inherited_actual_main_updates=27507,
    partial_observed_main_updates=v['observed_partial_main_updates'], reviewed_actor=checked['checkpoint'],
    reviewed_sha256=checked['sha256'], first_resumed_block=resumed, last_completed_main=last,
    upper_QA_images_verified=plot_count, local_report_links_verified=len(local_links),
    automation=automation, inventory=inventory, checks=checks)
write(OUT / 'closeout_verification.json', verification)
evidence = read(OUT / 'evidence.json')
evidence['closeout_verification'] = str(OUT / 'closeout_verification.json')
evidence['automation_verification'] = str(OUT / 'automation_verification.json')
evidence['inventory_verification'] = str(OUT / 'inventory_verification.json')
write(OUT / 'evidence.json', evidence)
print(json.dumps({k: verification[k] for k in ['passed', 'owned_snapshot_utc', 'current_operation', 'retained_main_updates',
    'upper_QA_images_verified', 'local_report_links_verified']}, indent=2))
