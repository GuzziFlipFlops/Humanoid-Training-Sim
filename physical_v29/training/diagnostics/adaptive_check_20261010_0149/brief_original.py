"""Compact exact saved gates and owned learning evidence after a CPU snapshot."""
import argparse, json
from pathlib import Path
OUT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--tag',default='final');a=p.parse_args()
v=json.loads((OUT/(a.tag+'_learning_verification.json')).read_text())
r=json.loads((OUT/(a.tag+'_gate_review.json')).read_text())['latest']
b=v['blocks'][-1]
d=dict(utc=v['owned_snapshot_utc'],supervisor=v['supervisor_pid'],workers=v['actual_worker_pids'],
    hashes=v['frozen_files_verified'],protected=v['protected_authored_worker_files_equal'],
    completed=v['completed_main_updates'],recovered=v['hash_proven_recovered_main_updates'],
    retained=v['actual_retained_main_updates'],inherited=v['inherited_actual_main_updates'],partial=v['observed_partial_main_updates'],
    current_operation=v['operation_path'],current_environment_count=v['current_environment_count'],
    reviewed_actor=r['checkpoint'],reviewed_sha=r['sha256'],
    tests={t:dict(all_pass=r[t+'_all_pass'],failed={str(c['command']):c['failures'] for c in r[t+'_cases'] if not c['passed']},
        focal=[{k:c[k] for k in ['command','signed_speed','lateral_speed','flight','passed']} for c in r[t+'_cases'] if c['command'] in [.15,1.5,3.5]]) for t in ['nominal','pushed']},
    last_main={k:b[k] for k in ['folder','updates','environments','minimum_free_gpu_MiB','minimum_host_available_MiB','private_memory_growth_MiB','torch_allocation_growth_MiB']},
    last_full_load=b['full_load_proof'],
    rates={k:v[k] for k in ['active_PPO_samples_s','worker_effective_samples_s','whole_chain_samples_s']})
print(json.dumps(d,indent=2))
