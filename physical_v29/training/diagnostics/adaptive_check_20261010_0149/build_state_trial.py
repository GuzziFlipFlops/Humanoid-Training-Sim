"""Stage a whole learned-state comparison; no target/reward/physics override."""
import ast,hashlib,json
from pathlib import Path
OUT=Path(__file__).resolve().parent
old=OUT.parent/'adaptive_check_20261009_2048/run_row_trial.py'
source=old.read_text();tree=ast.parse(source)
functions='\n\n'.join(ast.get_source_segment(source,node) for node in tree.body
    if isinstance(node,ast.FunctionDef) and node.name in ['kill_owned','launch'])
functions=functions.replace("'num_envs'", "'num_envs'")
header='''"""Sequential whole-state experiment with guaranteed original-supervisor resume."""
import hashlib,json,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent;ROOT=OUT/'matched_full_state'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
WORK_LIMIT=9000;START=time.time();WORK_START=None
NAMES={'worker_unitree.py','worker_entry.py','diverse_eval.py'}
'''
body='''
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
    (folder/'training_validation.json').write_text(json.dumps(receipt,indent=2)+'\\n')

def teacher_args(checkpoint):
    args=read(Path(checkpoint).parent/'run_config.json')['arguments'];result=[]
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

ROOT.mkdir(exist_ok=False)
try:
    with (OUT/'pause.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-X','utf8',str(OUT/'prepare_and_pause.py')],check=True,stdout=log,stderr=subprocess.STDOUT)
    pause=read(OUT/'diagnostic_pause.json');WORK_START=time.monotonic()
    with (OUT/'older_CPU_preflight.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-X','utf8',str(OUT/'older_branch_CPU_preflight.py')],check=True,stdout=log,stderr=subprocess.STDOUT)
    preflight=read(OUT/'older_branch_CPU_preflight.json');assert preflight['all_CPU_preflight_checks_pass']
    anchors={'control':Path(pause['checkpoint']),'earlier':Path(preflight['earlier_checkpoint'])}
    common=['--headless','--device','cuda:0','--speed','3.5','--reserve_mb','300']
    manifest=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),branches={b:dict(checkpoint=str(cp),sha256=sha(cp)) for b,cp in anchors.items()},
        wrapper_sha256=sha(OUT/'worker_entry.py'),diverse_eval_sha256=sha(OUT/'diverse_eval.py'),
        protected_source_copy=read(OUT/'protected_source_copy.json'),CPU_preflight=preflight,
        reward_physics_caps_observations_commands_PPO_and_auxiliaries_unchanged=True,
        teacher_actor_bytes_and_frozen_head_context_equal=True,whole_actor_critic_Adam_exploration_resumed=True,
        source_checkpoint_is_the_only_branch_difference=True,no_composite_controller_or_target_override=True,
        training_seeds=[317,331],training_environments=4096,updates=[64,128],
        nominal_seed=901,pushed_seed=1301,independent_initial_state_seeds=[5011,5039],
        work_budget_seconds=WORK_LIMIT,saved_boundary_and_drain_excluded=True,
        diagnostic_learning_not_main_progress=True,diagnostic_actor_deployed=False,
        scope='Earlier complete neutral-head learned state versus unchanged current complete state under identical protected runtime. This tests learned initialization including critic/Adam; it does not isolate one actor feature or reward term.')
    (ROOT/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2)+'\\n')
    replay=ROOT/'current_nominal_replay'
    evaluate(anchors['control'],replay,901)
    import numpy as np
    originals=[p.parent for p in Path(pause['run']).glob('*_gate_*_nominal/result.json')
        if read(p).get('completed') and Path(read(p.parent/'run_config.json')['arguments']['checkpoint']).resolve()==anchors['control'].resolve()]
    assert len(originals)==1
    with np.load(originals[0]/'telemetry.npz') as a,np.load(replay/'telemetry.npz') as b:
        assert set(a.files)==set(b.files)
        arrays={k:np.array_equal(a[k],b[k],equal_nan=True) if a[k].dtype.kind in 'fc' else np.array_equal(a[k],b[k]) for k in a.files}
    assert all(arrays.values()) and read(originals[0]/'evaluation.json')==read(replay/'evaluation.json')
    (ROOT/'current_baseline_parity.json').write_text(json.dumps(dict(passed=True,arrays=arrays,original=str(originals[0]),checkpoint_sha256=sha(anchors['control'])),indent=2)+'\\n')
    baselines=[]
    for label,seed,push,diverse in [('nominal',901,False,False),('push',1301,True,False),
                                     ('diverse_nominal',5011,False,True),('diverse_push',5039,True,True)]:
        folder=ROOT/('earlier_baseline_'+label)
        ev=evaluate(anchors['earlier'],folder,seed,push,diverse)
        baselines.append(dict(folder=str(folder),all_pass=ev['passed'],prior_through3_pass=all(c['passed'] for c in ev['cases'] if c['statistics']['command_m_s']<=3),
            failures=[dict(command=c['statistics']['command_m_s'],failures=c['failures']) for c in ev['cases'] if not c['passed']]))
    (ROOT/'earlier_baseline_review.json').write_text(json.dumps(dict(checkpoint=str(anchors['earlier']),tests=baselines,all_prior_through3_pass=all(x['prior_through3_pass'] for x in baselines)),indent=2)+'\\n')
    assert all(x['prior_through3_pass'] for x in baselines),'Fresh earlier-state prior contract fails; do not assume its dated certificate'
    for branch,cp in anchors.items():
        smoke=ROOT/(branch+'_smoke')
        launch([str(OUT/'worker_entry.py'),'--mode','train','--output',str(smoke),'--checkpoint',str(cp),
            '--num_envs','64','--iterations','4','--learning_rate','.0001','--seed','317']+teacher_args(cp)+common,smoke,300)
        validate_train(smoke,4,cp)
        reload=ROOT/(branch+'_reload');source=smoke/'candidate.pt'
        launch([str(OUT/'worker_entry.py'),'--mode','train','--output',str(reload),'--checkpoint',str(source),
            '--num_envs','64','--iterations','4','--learning_rate','.0001','--seed','317']+teacher_args(cp)+common,reload,300)
        validate_train(reload,4,source)
    for seed in [317,331]:
        for branch,anchor in anchors.items():
            checkpoint=anchor
            for updates in [64,128]:
                train=ROOT/f'seed{seed}_{branch}_{updates:03d}_train'
                launch([str(OUT/'worker_entry.py'),'--mode','train','--output',str(train),'--checkpoint',str(checkpoint),
                    '--num_envs','4096','--iterations','64','--learning_rate','.0001','--seed',str(seed)]+teacher_args(anchor)+common,train,900)
                validate_train(train,64,checkpoint);checkpoint=train/'candidate.pt'
                for label,evaluation_seed,push in [('nominal',901,False),('push',1301,True)]:
                    evaluate(checkpoint,ROOT/f'seed{seed}_{branch}_{updates:03d}_{label}',evaluation_seed,push)
                if updates==128:
                    for label,evaluation_seed,push in [('diverse_nominal',5011,False),('diverse_push',5039,True)]:
                        evaluate(checkpoint,ROOT/f'seed{seed}_{branch}_{updates:03d}_{label}',evaluation_seed,push,True)
    (ROOT/'completion.json').write_text(json.dumps(dict(completed=True,work_seconds=time.monotonic()-WORK_START,
        wall_seconds=time.time()-START,diagnostic_actor_deployed=False),indent=2)+'\\n')
except BaseException as error:
    (ROOT/'failure.json').write_text(json.dumps(dict(error=repr(error),traceback=traceback.format_exc(),wall_seconds=time.time()-START),indent=2)+'\\n')
    raise
finally:
    if (OUT/'diagnostic_pause.json').exists():
        with (OUT/'resume.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-X','utf8',str(OUT/'resume_after_diagnosis.py')],check=True,stdout=log,stderr=subprocess.STDOUT)
print('Complete-state comparison finished; original learner resumed.',flush=True)
'''
script=header+functions+'\n'+body
compile(script,str(OUT/'run_state_trial.py'),'exec')
(OUT/'run_state_trial.py').write_text(script)
(OUT/'state_trial_staging.json').write_text(json.dumps(dict(compiled=True,files={n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['run_state_trial.py','worker_entry.py','diverse_eval.py','prepare_and_pause.py','resume_after_diagnosis.py','older_branch_CPU_preflight.py']}),indent=2)+'\n')
print('Staged whole-state control/earlier comparison; no running source edit or learner reset.')
