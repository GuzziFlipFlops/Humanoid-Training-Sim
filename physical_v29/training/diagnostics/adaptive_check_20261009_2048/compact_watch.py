"""Small owned-worker and exact-pair snapshot; diagnostics never count as main."""
import json,subprocess,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT/'matched_row_route'
r=subprocess.run([sys.executable,str(OUT/'monitor_check.py')],check=True,capture_output=True,text=True)
m=json.loads(r.stdout);w=m['actual_workers']
workers=[{k:v for k,v in x.items() if k in ['pid','script','output','observed_updates','KL','command_mismatch','free_GPU_MiB','available_host_MiB','three_auxiliary_batches']} for x in w]
pairs=[]
for seed in [251,263]:
 for branch in ['control','shared','rows']:
  for updates in [64,128]:
   fs=[ROOT/f'seed{seed}_{branch}_{updates:03d}_{label}' for label in ['nominal','push']]
   if not all((f/'operation_receipt.json').exists() for f in fs):continue
   tests=[]
   for f in fs:
    e=json.loads((f/'evaluation.json').read_text());a=json.loads((f/'run_config.json').read_text())['arguments']
    z=e['cases'][-1]['statistics'];q=next(c for c in e['cases'] if c['statistics']['command_m_s']==1.5)
    tests.append(dict(checkpoint=a['checkpoint'],all_pass=e['passed'],current_pass=e['cases'][-1]['passed'],signed=z['reset_frame_forward_speed_m_s'],lateral=z['reset_frame_lateral_speed_m_s'],
      flight1_5=q['statistics']['geometric_flight_fraction'],failures=[dict(command=c['statistics']['command_m_s'],failures=c['failures']) for c in e['cases'] if not c['passed']]))
   assert tests[0]['checkpoint']==tests[1]['checkpoint']
   pairs.append(dict(seed=seed,branch=branch,updates=updates,tests=tests))
expected=[ROOT/f'seed{seed}_{branch}_{updates:03d}_{label}/operation_receipt.json'
 for seed in [251,263] for branch in ['control','shared','rows'] for updates in [64,128]
 for label in ['train','nominal','push']+(['diverse_nominal','diverse_push'] if updates==128 else [])]
assert len(expected)==48
comparison_count=sum(p.exists() for p in expected)
all_count=len(list(ROOT.glob('*/operation_receipt.json')))
out=dict(recorded_utc=m['recorded_utc'],workers=workers,completed_pair_count=len(pairs),latest_pair=pairs[-1] if pairs else None,
 completed_operations=comparison_count,expected_comparison_operations=48,
 completed_setup_operations=all_count-comparison_count,
 completion=(ROOT/'completion.json').exists(),failure=json.loads((ROOT/'failure.json').read_text()) if (ROOT/'failure.json').exists() else None)
(OUT/'compact_watch.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
