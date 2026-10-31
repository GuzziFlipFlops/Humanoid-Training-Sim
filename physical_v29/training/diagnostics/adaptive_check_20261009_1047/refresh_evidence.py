"""Read fresh source/ownership, exact gates and retained main samples on CPU."""
import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
OUT = Path(__file__).resolve().parent
TRAIN = OUT.parents[1]
p = argparse.ArgumentParser(); p.add_argument('--tag', default='initial'); args = p.parse_args()
folder = OUT / (args.tag + '_cpu')
spec = importlib.util.spec_from_file_location('owned_recorder', TRAIN / 'record_owned_observation_handoff.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); m.OUT = folder; m.main()
for name, destination in [('review_gates.py', args.tag + '_gate_review.json'), ('verify_recovery.py', args.tag + '_learning_verification.json')]:
    with (OUT / (args.tag + '_' + name + '.log')).open('w', encoding='utf-8') as log:
        subprocess.run([sys.executable, '-X', 'utf8', str(OUT / name), '--snapshot', str(folder / 'live_handoff.json'),
                        '--output', str(OUT / destination)], check=True, stdout=log, stderr=subprocess.STDOUT)
review = json.loads((OUT / (args.tag + '_gate_review.json')).read_text())
older = json.loads((OUT.parent / 'adaptive_check_20261009_0547/liveness_gate_review.json').read_text())
for promotion in review['promotions']:
    if promotion['stage'] == 'run_300' and 'matching_pair' not in promotion:
        promotion['matching_pair'] = next(p['matching_pair'] for p in older['promotions'] if p['stage'] == 'run_300')
(OUT / (args.tag + '_gate_review.json')).write_text(json.dumps(review, indent=2) + '\n')
v = json.loads((OUT / (args.tag + '_learning_verification.json')).read_text())
print(json.dumps(dict(recorded_utc=v['owned_snapshot_utc'], checkpoint=review['latest']['checkpoint'], sha256=review['latest']['sha256'],
    completed=v['completed_main_updates'], recovered=v['hash_proven_recovered_main_updates'], inherited=v['inherited_actual_main_updates'],
    partial=v['observed_partial_main_updates'], actual_workers=v['actual_worker_pids'], frozen=v['frozen_files_verified'],
    resources=dict(environments=v['current_environment_count'], gpu=v['gpu'], ram=v['system_memory']),
    rates={k:v[k] for k in ['active_PPO_samples_s','worker_effective_samples_s','whole_chain_samples_s']},
    latest_main=v['blocks'][-1], cases={label:[{k:c[k] for k in ['command','passed','failures','signed_speed','lateral_speed','flight']} 
      for c in review['latest'][label + '_cases']] for label in ['nominal','pushed']}), indent=2))
