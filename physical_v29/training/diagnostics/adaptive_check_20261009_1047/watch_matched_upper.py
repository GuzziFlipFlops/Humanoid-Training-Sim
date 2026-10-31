"""Finite CPU audit watcher for the four saved128-update actors, no CUDA."""
import json,subprocess,sys,time
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT/'matched_slow_reward'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
pending={(seed,branch) for seed in [179,191] for branch in ['control','candidate']}
deadline=time.monotonic()+6900
while pending:
    assert time.monotonic()<deadline,'CPU watcher deadline; completed audits remain preserved'
    for seed,branch in sorted(pending.copy()):
        prefix=f'seed{seed}_{branch}_128'
        paths=[ROOT/(prefix+'_train'),ROOT/(prefix+'_nominal'),ROOT/(prefix+'_push')]
        if not all((p/'result.json').exists() and read(p/'result.json').get('completed') for p in paths):continue
        subprocess.run([sys.executable,'-X','utf8',str(OUT/'review_reward_experiment.py')],check=True,stdout=subprocess.DEVNULL)
        target=OUT/f'seed{seed}_{branch}_upper'
        assert not (target/'audit_set.json').exists()
        with (OUT/f'seed{seed}_{branch}_upper.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-X','utf8',str(OUT/'run_upper_audit_set.py'),
                '--gate-review',f'seed{seed}_{branch}_gate_review.json','--output',target.name],check=True,stdout=log,stderr=subprocess.STDOUT)
        print(json.dumps(dict(completed_cpu_upper=str(target),actor=str(paths[0]/'candidate.pt'),needs_visual_QA=True)),flush=True)
        pending.remove((seed,branch))
    if pending:time.sleep(5)
print('All four saved128-update upper motion audits prepared; visual QA still required.',flush=True)
