import json
from pathlib import Path
OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
r=read(OUT/'initial_cpu/live_handoff.json');n=read(OUT.parents[1]/'automation/NEXT_CHECK.json')
print(json.dumps({'recorded_utc':r['recorded_utc'],'state':{k:r['state'].get(k) for k in ['status','stage','checkpoint','operation_path','environment_count','total_training_iterations','supervisor_pid']},'actual_workers':r['actual_worker_pids'],'frozen':{'count':len(r['frozen_source_integrity']),'all_equal':all(r['frozen_source_integrity'].values())},'gpu':r['gpu'],'system_memory':r['system_memory'],'prior_focus':n['focused_tests']},indent=2))
