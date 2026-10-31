import json
from pathlib import Path
OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
g=read(OUT/'initial_gate_review.json');previous=read(OUT.parent/'adaptive_check_20261007_1344/closing_gate_review.json')
for p in g['promotions']:
    if p['stage']=='run_300':p['matching_pair']=next(x for x in previous['promotions'] if x['stage']=='run_300')['matching_pair']
(OUT/'initial_gate_review.json').write_text(json.dumps(g,indent=2)+'\n')
l=read(OUT/'initial_learning_verification.json');live=read(OUT/'initial_cpu/live_handoff.json')
records=g.get('pairs',g.get('records',[]))
if not records:
    print('Gate keys:',list(g))
summary=dict(observed=live['recorded_utc'],stage=live['state']['stage'],operation=live['state']['operation_path'],count=live['state']['environment_count'],latest_checkpoint=g['latest']['checkpoint'],latest_cases={k:[dict(command=c['command'],passed=c['passed'],failures=c['failures'],signed=c['signed_speed'],lateral=c['lateral_speed'],flight=c['flight']) for c in g['latest'][k]] for k in ['nominal_cases','pushed_cases']},main={k:l[k] for k in ['completed_main_updates','observed_partial_main_updates','hash_proven_recovered_main_updates','actual_retained_main_updates','active_PPO_samples_s','worker_effective_samples_s','whole_chain_samples_s','documented_diagnostic_pause_seconds_excluded']},recent_main=[{k:b[k] for k in ['folder','updates','environments','minimum_free_gpu_MiB','minimum_host_available_MiB','private_memory_growth_MiB','torch_allocation_growth_MiB']} for b in l['blocks'][-5:]])
(OUT/'initial_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
