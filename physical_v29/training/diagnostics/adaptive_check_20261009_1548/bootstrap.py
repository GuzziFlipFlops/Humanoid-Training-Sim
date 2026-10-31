"""Start this bounded check with external CPU audits; never edit the live source."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent;TRAIN=OUT.parents[1]
OLD=OUT.parent/'adaptive_check_20261009_1047'
names=['review_gates.py','verify_recovery.py','refresh_evidence.py','upper_joint_time_audit.py',
 'upper_bias_audit.py','upper_target_bias_audit.py','run_upper_audit_set.py','summarize_upper.py',
 'record_upper_QA.py','action_projection.py','prepare_and_pause.py','resume_after_diagnosis.py',
 'capture_current_gradients.py','analyze_current_gradients.py','gate_com_contact_audit.py',
 'audit_current_motors.py','compact_pair.py','record_diagnostic_memory.py','diverse_eval.py']
copies=[]
for name in names:
    src=OLD/name;dst=OUT/name
    assert not dst.exists()
    shutil.copy2(src,dst)
    copies.append(dict(source=str(src),destination=str(dst),original_sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
p=OUT/'verify_recovery.py';s=p.read_text(encoding='utf-8')
s=s.replace("OUT/'diagnostic_resume.json'", "OUT.parent/'adaptive_check_20261009_1047/diagnostic_resume.json',OUT/'diagnostic_resume.json'")
s=s.replace("OUT/'high_speed_extrapolation/diagnostic_resume.json'", "OUT.parent/'adaptive_check_20261009_1047/high_speed_extrapolation/diagnostic_resume.json',OUT/'secondary/diagnostic_resume.json'")
p.write_text(s,encoding='utf-8')
p=OUT/'refresh_evidence.py';s=p.read_text(encoding='utf-8').replace('adaptive_check_20261009_0547/liveness_gate_review.json','adaptive_check_20261009_1047/closeout_gate_review.json');p.write_text(s,encoding='utf-8')
shutil.copy2(TRAIN/'automation/NEXT_CHECK.json',OUT/'previous_NEXT_CHECK.json')
shutil.copy2(Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml'),OUT/'previous_automation.toml')
(OUT/'diagnostic_tool_lineage.json').write_text(json.dumps(copies,indent=2)+'\n',encoding='utf-8')
with (OUT/'initial_refresh.log').open('w',encoding='utf-8') as log:
    subprocess.run([sys.executable,'-X','utf8',str(OUT/'refresh_evidence.py'),'--tag','initial'],check=True,stdout=log,stderr=subprocess.STDOUT)
v=json.loads((OUT/'initial_learning_verification.json').read_text());p=json.loads((OUT/'initial_gate_review.json').read_text())['latest']
print(json.dumps(dict(snapshot=v['owned_snapshot_utc'],supervisor=v['supervisor_pid'],workers=v['actual_worker_pids'],frozen=v['frozen_files_verified'],
    completed=v['completed_main_updates'],recovered=v['hash_proven_recovered_main_updates'],partial=v['observed_partial_main_updates'],stage=v['stage'],
    operation=v['operation_path'],environments=v['current_environment_count'],gpu=v['gpu'],host=v['system_memory'],
    rates={k:v[k] for k in ['active_PPO_samples_s','worker_effective_samples_s','whole_chain_samples_s']},
    checkpoint=p['checkpoint'],sha256=p['sha256'],cases={t:[{k:c[k] for k in ['command','passed','failures','signed_speed','lateral_speed','flight']} for c in p[t+'_cases']] for t in ['nominal','pushed']}),indent=2))
