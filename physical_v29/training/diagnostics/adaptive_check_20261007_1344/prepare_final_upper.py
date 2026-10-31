"""Bind final CPU upper diagnostics to the latest complete production pair."""
import json
from pathlib import Path
OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
review=read(OUT/'closing_gate_review.json');initial=read(OUT/'initial_gate_review.json')
for promotion in review['promotions']:
    if promotion['stage']=='run_300':
        promotion['matching_pair']=next(p for p in initial['promotions'] if p['stage']=='run_300')['matching_pair']
(OUT/'closing_gate_review.json').write_text(json.dumps(review,indent=2)+'\n')
learning=read(OUT/'final_learning_verification.json');live=read(OUT/'final_cpu/live_handoff.json')
pair=review['latest']
print(json.dumps(dict(observed=live['recorded_utc'],operation=live['state']['operation_path'],checkpoint=pair['checkpoint'],completed=learning['completed_main_updates'],recovered=learning['hash_proven_recovered_main_updates'],retained=learning['actual_retained_main_updates'],rates={k:learning[k] for k in ['active_PPO_samples_s','worker_effective_samples_s','whole_chain_samples_s']},cases={k:[dict(command=c['command'],signed=c['signed_speed'],lateral=c['lateral_speed'],flight=c['flight'],failed=c['failures']) for c in pair[k]] for k in ['nominal_cases','pushed_cases']}),indent=2))
notes=read(OUT/'diagnostic_accounting_notes.json')
notes['final_recorder_invocation_path_error']={'scope':'Diagnostic invocation only; no simulator or production mutation. Helpers resolve snapshot/output relative to their own diagnostic folder. Workspace-relative paths were accidentally doubled.', 'preserved_logs':['failed_final_learning_path.log','failed_final_gate_path.log'],'corrected':'Both helpers rerun with diagnostic-folder-relative paths; exit0.'}
(OUT/'diagnostic_accounting_notes.json').write_text(json.dumps(notes,indent=2)+'\n')
