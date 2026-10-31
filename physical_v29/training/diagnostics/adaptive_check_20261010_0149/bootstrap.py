"""Prepare CPU diagnostics from preserved tools; no production mutation."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent
TRAIN=OUT.parents[1]
OLD=OUT.parent/'adaptive_check_20261009_2048'
names=['review_gates.py','verify_recovery.py','refresh_evidence.py','upper_joint_time_audit.py',
 'upper_bias_audit.py','upper_target_bias_audit.py','run_upper_audit_set.py','summarize_upper.py',
 'record_upper_QA.py','record_branch_QA.py','gate_com_contact_audit.py','audit_current_motors.py',
 'compact_pair.py','record_diagnostic_memory.py','diverse_eval.py','runtime_joint_order.py',
 'prepare_and_pause.py','resume_after_diagnosis.py','monitor_check.py','older_branch_CPU_preflight.py']
copies=[]
for name in names:
    src=OLD/name;dst=OUT/name
    assert not dst.exists()
    shutil.copy2(src,dst)
    copies.append(dict(source=str(src),destination=str(dst),sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
p=OUT/'verify_recovery.py'
s=p.read_text().replace("OUT/'diagnostic_resume.json']:","OUT.parent/'adaptive_check_20261009_2048/diagnostic_resume.json',OUT/'diagnostic_resume.json']:")
assert "adaptive_check_20261009_2048/diagnostic_resume.json" in s
p.write_text(s)
p=OUT/'prepare_and_pause.py'
s=p.read_text().replace('Bounded fresh qualification of earlier neutral-head actor267 and current-state/upper-controller compatibility; truthful physical feedback, unchanged native limits and all gates; diagnostic composites never enter production',
 'Bounded complete-state continuation of earlier neutral-head267 versus unchanged current full state, two seeds and all native gates; no training-source/reward/physics modification or composite inference')
p.write_text(s)
shutil.copy2(TRAIN/'automation/NEXT_CHECK.json',OUT/'previous_NEXT_CHECK.json')
shutil.copy2(OLD/'final_learning_verification.json',OUT/'previous_end_verification.json')
shutil.copy2(Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml'),OUT/'previous_automation.toml')
shutil.copy2(OLD/'neutral_teacher_certificate.json',OUT/'neutral_teacher_certificate.json')
(OUT/'diagnostic_tool_lineage.json').write_text(json.dumps(copies,indent=2)+'\n')
with (OUT/'initial_refresh.log').open('w',encoding='utf-8') as log:
    subprocess.run([sys.executable,'-X','utf8',str(OUT/'refresh_evidence.py'),'--tag','initial'],check=True,stdout=log,stderr=subprocess.STDOUT)
v=json.loads((OUT/'initial_learning_verification.json').read_text());p=json.loads((OUT/'initial_gate_review.json').read_text())['latest']
print(json.dumps(dict(snapshot=v['owned_snapshot_utc'],supervisor=v['supervisor_pid'],workers=v['actual_worker_pids'],
 hashes=v['frozen_files_verified'],protected=v['protected_authored_worker_files_equal'],completed=v['completed_main_updates'],
 recovered=v['hash_proven_recovered_main_updates'],partial=v['observed_partial_main_updates'],stage=v['stage'],
 operation=v['operation_path'],environments=v['current_environment_count'],rates={k:v[k] for k in ['active_PPO_samples_s','worker_effective_samples_s','whole_chain_samples_s']},
 last_main={k:v['blocks'][-1][k] for k in ['folder','updates','minimum_free_gpu_MiB','minimum_host_available_MiB']},
 checkpoint=p['checkpoint'],sha256=p['sha256'],cases={t:[{k:c[k] for k in ['command','passed','failures','signed_speed','lateral_speed','flight']} for c in p[t+'_cases']] for t in ['nominal','pushed']}),indent=2))
