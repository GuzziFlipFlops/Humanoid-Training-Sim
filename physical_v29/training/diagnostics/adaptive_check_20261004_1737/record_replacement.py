"""CPU-only fresh replacement status; preserves the initial R31 evidence."""
import hashlib,json,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
run=HERE/'runs/locomotion_20261004_r33_qualified_walking_retention'
state=read(run/'state.json');config=read(run/'chain_config.json');seed=read(OUT/'retention_continuation_proof.json')
launcher=read(OUT/'deployment.json')['supervisor_launcher_pid'];pid=state['supervisor_pid'];proc=psutil.Process(pid)
assert launcher in [pid]+[p.pid for p in proc.parents()],'Supervisor not descended from owned launcher'
assert str(run).lower().replace('\\','/') in ' '.join(proc.cmdline()).lower().replace('\\','/')
hashes={}
for name,expected in config['sources_sha256'].items():
    frozen=run/'source/physical_v29'
    if name in ('supervisor_unitree.py','vram_ramp.py'):p=run/name
    elif name.startswith('physical_assets/'):p=frozen/'assets'/name.split('/',1)[1]
    elif name.startswith('physical_v29/'):p=frozen/name.split('/',1)[1]
    else:p=frozen/'training'/name
    assert sha(p)==expected,name
    hashes[name]=expected
workers=[]
for p in psutil.process_iter(['pid','ppid','name','cmdline']):
    try:
        args=p.info['cmdline'] or []
        if (p.info['name'] or '').lower()=='python.exe' and any(Path(a).name=='worker_unitree.py' for a in args):
            assert str(run).lower().replace('\\','/') in ' '.join(args).lower().replace('\\','/'),'Unowned simulation worker'
            workers.append(p.info)
    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
actual=[p for p in workers if any(c['ppid']==p['pid'] for c in workers)==False]
assert len(actual)<=1,'Multiple actual simulator workers'
now=datetime.now(timezone.utc).isoformat()
pointer={'run':str(run),'status':state['status'],'supervisor_pid':pid,'started_unix':state['started_unix'],
    'started_utc':datetime.fromtimestamp(state['started_unix'],timezone.utc).isoformat(),
    'prior_run':read(OUT/'diagnostic_pause.json')['run'],'initial_actor_checkpoint':seed['selected_checkpoint'],
    'initial_actor_sha256':seed['selected_checkpoint_sha256'],'latest_prior_preserved':read(OUT/'diagnostic_pause.json')['checkpoint'],
    'initial_checkpoint_scope':'Exact R32 checkpoint269 actor/exploration/critic/Adam retained; training-only qualified prior walking gradient repair; fresh current and retained qualification mandatory',
    'optimization_iteration':'04','repair_report':str(OUT/'REPORT.md'),
    'note':'Running reward v7 and geometric verifier v6 unchanged; added prior-command training retention, no teacher in inference'}
temp=HERE/'ACTIVE_RUN.json.tmp';temp.write_text(json.dumps(pointer,indent=2)+'\n');temp.replace(HERE/'ACTIVE_RUN.json')
gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
completed=[];gates=[]
for attempt in state['attempts']:
    folder=Path(attempt['folder']);p=folder/'result.json'
    if p.exists():
        result=read(p)
        if attempt['operation']=='eval':gates.append({'folder':str(folder),'result':result})
        elif 'benchmark_' in folder.name:completed.append({'folder':str(folder),'result':result})
snapshot={'recorded_utc':now,'active_pointer':pointer,'state':state,'hashes':hashes,'owned_workers':workers,
    'actual_worker_count':len(actual),'gpu_used_free_utilization_mb_pct':gpu,
    'host_available_mb':int(psutil.virtual_memory().available/1024**2),'completed_benchmarks':completed,'gates':gates}
if state.get('checkpoint') and Path(state['checkpoint']).is_file():snapshot['checkpoint_sha256']=sha(Path(state['checkpoint']))
(OUT/'replacement_status.json').write_text(json.dumps(snapshot,indent=2)+'\n')
print(json.dumps({'recorded_utc':now,'status':state['status'],'stage':state['stage'],'count':state.get('environment_count'),
    'attempt':state['attempts'][-1] if state['attempts'] else None,'worker_count':len(actual),'verified_hashes':len(hashes),
    'gpu':gpu,'host_available_mb':snapshot['host_available_mb'],'promotions':state['promotions']},indent=2))
