"""Sequential whole-state experiment with guaranteed original-supervisor resume."""
import hashlib,json,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent;ROOT=OUT/'matched_command_advantage'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
WORK_LIMIT=9000;START=time.time();WORK_START=None
NAMES={'worker_unitree.py','worker_entry.py','diverse_eval.py'}
def kill_owned(process,created):
    try:
        if process.is_running() and process.create_time()==created:
            tree=process.children(recursive=True)+[process]
            for p in tree:
                if p.is_running():p.terminate()
            _,alive=psutil.wait_procs(tree,timeout=15)
            for p in alive:p.kill()
            psutil.wait_procs(alive,timeout=15)
    except psutil.NoSuchProcess:pass

def launch(argv,folder,timeout=850):
    assert time.monotonic()-WORK_START<WORK_LIMIT,'Work-only budget exhausted before next operation'
    folder.mkdir(parents=True,exist_ok=False)
    assert not any(any(Path(x).name in NAMES for x in p.info['cmdline'] or []) for p in psutil.process_iter(['cmdline'])),'Another actual CUDA worker is active'
    beginning=time.time()
    with (folder/'stdout.log').open('w',encoding='utf-8') as so,(folder/'stderr.log').open('w',encoding='utf-8') as se:
        proc=subprocess.Popen([sys.executable,'-u','-X','utf8']+argv,stdout=so,stderr=se,creationflags=subprocess.CREATE_NO_WINDOW)
        ident=psutil.Process(proc.pid);created=ident.create_time()
        (folder/'process.json').write_text(json.dumps(dict(pid=proc.pid,create_time=created,argv=argv),indent=2),encoding='utf-8')
        deadline=time.monotonic()+timeout
        try:
            while proc.poll() is None and not (folder/'result.json').exists():
                assert time.monotonic()<deadline,'Bounded diagnostic worker exceeded operation budget'
                assert psutil.virtual_memory().available/2**20>=2048,'Diagnostic host memory guard'
                if (folder/'progress.jsonl').exists():
                    rows=[json.loads(v) for v in (folder/'progress.jsonl').read_text().splitlines()]
                    if len(rows)>=5:
                        assert rows[-1]['gpu']['torch_allocated_mb']-rows[2]['gpu']['torch_allocated_mb']<128,'Diagnostic CUDA graph growth exceeds allowance'
                time.sleep(2)
            if proc.poll() is None:
                assert read(folder/'result.json')['completed']
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    kill_owned(ident,created)
                    (folder/'exit_cleanup.json').write_text(json.dumps(dict(completed_saved_result=True,owned_PID=ident.pid,only_Kit_shutdown_cleanup=True)))
            assert (proc.returncode==0 or (folder/'exit_cleanup.json').exists()) and read(folder/'result.json')['completed'],(folder.name,proc.returncode)
        except BaseException:
            kill_owned(ident,created);raise
    result=dict(completed=str(folder),operation_seconds=round(time.time()-beginning,2),work_seconds=round(time.monotonic()-WORK_START,2))
    (folder/'operation_receipt.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

def validate_train(folder,count,source):
    proof=read(folder/'full_load_proof.json')
    assert proof['source_sha256']==sha(source)
    assert all(proof[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
    rows=[json.loads(x) for x in (folder/'progress.jsonl').read_text().splitlines()]
    assert len(rows)==count
    assert all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
    assert all(r[k]['batches']==20 for r in rows for k in ['retention','locomotion_retention','upper_refinement'])
    assert min(r['gpu']['free_mb'] for r in rows)>=300 and min(r['gpu']['ram_available_mb'] for r in rows)>=2048
    warm=rows[min(3,count-1):]
    span=max(r['gpu']['torch_allocated_mb'] for r in warm)-min(r['gpu']['torch_allocated_mb'] for r in warm)
    assert span<64
    receipt=dict(passed=True,actual_updates=count,actual_samples=sum(r['num_envs']*24 for r in rows),
        source_checkpoint=str(source),source_sha256=sha(source),exact_actor_critic_Adam_exploration=True,
        max_preupdate_KL=0,max_command_difference=0,three_original_auxiliaries_batches_per_update=20,
        min_GPU_MiB=min(r['gpu']['free_mb'] for r in rows),min_host_MiB=min(r['gpu']['ram_available_mb'] for r in rows),
        post_warmup_Torch_allocation_span_MiB=span)
    (folder/'training_validation.json').write_text(json.dumps(receipt,indent=2)+'\n')

def teacher_args(checkpoint):
    args=read(Path(checkpoint).parent/'run_config.json')['arguments'];args['termination_proof']=str(OUT/'termination_teacher_update_proof.json');result=[]
    for key in ['termination_proof','standing_prior','standing_prior_sha256','locomotion_prior_certificate']:
        result+=['--'+key,str(args[key])]
    return result

def evaluate(checkpoint,folder,seed,push=False,diverse=False):
    launch([str(OUT/('diverse_eval.py' if diverse else 'worker_entry.py')),
        '--mode','eval','--output',str(folder),'--checkpoint',str(checkpoint),
        '--num_envs','320','--duration','20','--seed',str(seed)]+common+(['--perturb'] if push else []),folder,360)
    ev=read(folder/'evaluation.json')
    assert len(ev['cases'])==10 and all(c['statistics']['trials']==32 for c in ev['cases'])
    return ev


def validate_normalization(folder,branch):
    import torch
    capture=torch.load(folder/'first_rollout_normalization.pt',weights_only=False,map_location='cpu')
    expected=capture['conditioned_advantage'] if branch=='candidate' else capture['old_global_normalized']
    assert torch.equal(capture['actual_advantage'],expected)
    assert torch.equal(capture['raw_advantage'],capture['returns']-capture['values'])
    rows=[json.loads(x) for x in (folder/'command_advantage.jsonl').read_text().splitlines()]
    assert all(x['applied']==(branch=='candidate') and x['original_GAE_returns_values_rewards_unchanged'] for x in rows)
    assert all(all(abs(g['conditioned_mean'])<1e-3 and (g['constant_or_singleton'] or abs(g['conditioned_std']-1)<1e-5) for g in x['groups']) for x in rows)
    assert all(sum(g['samples'] for g in x['groups'])==read(folder/'run_config.json')['arguments']['num_envs']*24 for x in rows)
    checkpoint=torch.load(folder/'candidate.pt',weights_only=False,map_location='cpu')
    assert checkpoint['infos']['command_advantage_applied']==(branch=='candidate')
    assert checkpoint['infos']['training_update_signature']==sha(Path(read(OUT/'source_freeze.json')['source'])/'training/command_advantage.py')
    (folder/'normalization_validation.json').write_text(json.dumps(dict(passed=True,branch=branch,updates=len(rows),
        raw_returns_values_unchanged=True,actual_normalizer_verified=True,saved_update_provenance=True),indent=2)+'\n')

ROOT.mkdir(exist_ok=False)
try:
    with (OUT/'pause.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-X','utf8',str(OUT/'prepare_and_pause.py')],check=True,stdout=log,stderr=subprocess.STDOUT)
    pause=read(OUT/'diagnostic_pause.json');WORK_START=time.monotonic()
    anchor=Path(pause['checkpoint'])
    common=['--headless','--device','cuda:0','--speed','3.5','--reserve_mb','300']
    manifest=read(OUT/'source_freeze.json');manifest.update(anchor_checkpoint=str(anchor),anchor_sha256=sha(anchor),
        pause=pause,work_limit_seconds=WORK_LIMIT,diagnostic_samples_excluded_from_main=True)
    (ROOT/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Disabled wrapper must reproduce the actual original deterministic gate bitwise.
    replay=ROOT/'baseline_replay';evaluate(anchor,replay,901)
    import numpy as np
    originals=[p.parent for p in Path(pause['run']).glob('*_gate_*_nominal/result.json')
        if read(p).get('completed') and Path(read(p.parent/'run_config.json')['arguments']['checkpoint']).resolve()==anchor.resolve()]
    assert len(originals)==1
    with np.load(originals[0]/'telemetry.npz') as a,np.load(replay/'telemetry.npz') as b:
        assert set(a.files)==set(b.files)
        arrays={k:np.array_equal(a[k],b[k],equal_nan=True) if a[k].dtype.kind in 'fc' else np.array_equal(a[k],b[k]) for k in a.files}
    assert all(arrays.values()) and read(originals[0]/'evaluation.json')==read(replay/'evaluation.json')
    (ROOT/'baseline_parity.json').write_text(json.dumps(dict(passed=True,arrays=arrays,original=str(originals[0]),checkpoint_sha256=sha(anchor)),indent=2)+'\n')
    for branch in ['control','candidate']:
        cp=anchor;flag=['--command_advantage'] if branch=='candidate' else []
        for label in ['smoke','reload']:
            folder=ROOT/f'{branch}_{label}'
            launch([str(OUT/'worker_entry.py'),'--mode','train','--output',str(folder),'--checkpoint',str(cp),
                '--num_envs','64','--iterations','4','--learning_rate','.0001','--seed','347']+teacher_args(anchor)+common+flag,folder,330)
            validate_train(folder,4,cp);validate_normalization(folder,branch)
            cp=folder/'candidate.pt'
    for seed in [347,359]:
        for branch in ['control','candidate']:
            checkpoint=anchor;flag=['--command_advantage'] if branch=='candidate' else []
            for updates in [64,128]:
                folder=ROOT/f'seed{seed}_{branch}_{updates:03d}_train'
                launch([str(OUT/'worker_entry.py'),'--mode','train','--output',str(folder),'--checkpoint',str(checkpoint),
                    '--num_envs','4096','--iterations','64','--learning_rate','.0001','--seed',str(seed)]+teacher_args(anchor)+common+flag,folder,1000)
                validate_train(folder,64,checkpoint);validate_normalization(folder,branch);checkpoint=folder/'candidate.pt'
                for label,evaluation_seed,push in [('nominal',901,False),('push',1301,True)]:
                    evaluate(checkpoint,ROOT/f'seed{seed}_{branch}_{updates:03d}_{label}',evaluation_seed,push)
                if updates==128:
                    for label,evaluation_seed,push in [('diverse_nominal',5011,False),('diverse_push',5039,True)]:
                        evaluate(checkpoint,ROOT/f'seed{seed}_{branch}_{updates:03d}_{label}',evaluation_seed,push,True)
    (ROOT/'completion.json').write_text(json.dumps(dict(completed=True,work_seconds=time.monotonic()-WORK_START,
        wall_seconds=time.time()-START,diagnostic_actor_deployed=False),indent=2)+'\n')
except BaseException as error:
    (ROOT/'failure.json').write_text(json.dumps(dict(error=repr(error),traceback=traceback.format_exc(),wall_seconds=time.time()-START),indent=2)+'\n')
    raise
finally:
    if (OUT/'diagnostic_pause.json').exists():
        with (OUT/'resume.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-X','utf8',str(OUT/'resume_after_diagnosis.py')],check=True,stdout=log,stderr=subprocess.STDOUT)
print('Versioned optimizer comparison finished; original learner resumed pending review.',flush=True)
