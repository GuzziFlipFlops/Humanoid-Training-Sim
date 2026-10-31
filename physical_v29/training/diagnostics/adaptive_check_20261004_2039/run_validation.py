"""Sequential owned physical tests: a single Isaac process at a time."""
import json, subprocess, sys, time, psutil
from pathlib import Path
OUT=Path(__file__).resolve().parent;WORKER=OUT/'validation_source/physical_v29/training/worker_unitree.py'
def read(p):return json.loads(p.read_text())
assert read(OUT/'candidate_cycle_tests.json')['passed']
assert read(OUT/'running_ppo_smoke/result.json')['completed']
for name,mode,count,iterations,duration,checkpoint,seed in [
    ('full_load_smoke','train',64,4,4.,OUT/'running_ppo_smoke/candidate.pt',71),
    ('checkpoint_reload','eval',64,0,4.,OUT/'full_load_smoke/candidate.pt',71),
    ('same_actor_nominal','eval',192,0,20.,OUT/'running_ppo_smoke/transferred_actor.pt',901)]:
    assert not (OUT/name).exists(),'Preserve earlier attempts'
    for p in psutil.process_iter(['cmdline']):
        assert not any(Path(x).name in ['worker_unitree.py','capture_gradient.py','physical_policy_comparison.py'] for x in (p.info['cmdline'] or [])),'Another Isaac worker is live'
    args=[sys.executable,'-u','-X','utf8',str(WORKER),'--headless','--device','cuda:0',
        '--mode',mode,'--num_envs',str(count),'--speed','1.5','--iterations',str(iterations),
        '--duration',str(duration),'--seed',str(seed),'--learning_rate','0.0001',
        '--checkpoint',str(checkpoint),'--output',str(OUT/name),'--reserve_mb','300']
    folder=OUT/name;folder.mkdir()
    with (folder/'stdout.log').open('w') as out,(folder/'stderr.log').open('w') as err:
        proc=subprocess.Popen(args,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        (folder/'command.json').write_text(json.dumps({'argv':args,'pid':proc.pid,'create_time':psutil.Process(proc.pid).create_time()},indent=2))
        print(json.dumps({'started':name,'pid':proc.pid}),flush=True)
        code=proc.wait(timeout=600)
    assert code==0 and (folder/'result.json').is_file(),name
    result=read(folder/'result.json');assert result['completed']
    print(json.dumps({'completed':name,'seconds':result['seconds']}),flush=True)
print('VALIDATION COMPLETE',flush=True)
