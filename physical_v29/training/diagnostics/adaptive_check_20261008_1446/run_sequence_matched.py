"""Sequential, bounded diagnostic CUDA comparison; always resume owned training."""
import hashlib,json,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import psutil

OUT=Path(__file__).resolve().parent;ROOT=OUT/'matched_sequence'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
WORK_LIMIT=4200
START=time.time();WORK_START=None

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

def launch(argv,folder,timeout=650):
    assert time.monotonic()-WORK_START<WORK_LIMIT,'Work-only budget exhausted before next operation'
    folder.mkdir(parents=True,exist_ok=False)
    names={'worker_unitree.py','sequence_candidate.py','diverse_eval.py'}
    assert not any(any(Path(x).name in names for x in p.info['cmdline'] or []) for p in psutil.process_iter(['cmdline'])),'Another actual CUDA worker is active'
    beginning=time.time()
    with (folder/'stdout.log').open('w',encoding='utf-8') as so,(folder/'stderr.log').open('w',encoding='utf-8') as se:
        proc=subprocess.Popen([sys.executable,'-u','-X','utf8']+argv,stdout=so,stderr=se,creationflags=subprocess.CREATE_NO_WINDOW)
        ident=psutil.Process(proc.pid);created=ident.create_time()
        (folder/'process.json').write_text(json.dumps(dict(pid=proc.pid,create_time=created,argv=argv),indent=2))
        deadline=time.monotonic()+timeout
        try:
            while proc.poll() is None and not (folder/'result.json').exists():
                assert time.monotonic()<deadline,'Bounded diagnostic worker exceeded operation budget'
                if (folder/'progress.jsonl').exists():
                    rows=[json.loads(v) for v in (folder/'progress.jsonl').read_text().splitlines()]
                    if len(rows)>=5:
                        assert rows[-1]['gpu']['torch_allocated_mb']-rows[2]['gpu']['torch_allocated_mb']<128,'Diagnostic CUDA graph growth exceeds bounded allowance'
                time.sleep(2)
            if proc.poll() is None:
                assert read(folder/'result.json')['completed']
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    kill_owned(ident,created)
                    (folder/'exit_cleanup.json').write_text(json.dumps(dict(completed_saved_result=True,owned_PID=ident.pid,only_Kit_shutdown_cleanup=True)))
            assert (proc.returncode==0 or (folder/'exit_cleanup.json').exists()) and read(folder/'result.json')['completed'],(folder.name,proc.returncode)
        except BaseException:
            kill_owned(ident,created)
            raise
    result=dict(completed=folder.name,operation_seconds=round(time.time()-beginning,2),work_seconds=round(time.monotonic()-WORK_START,2))
    (folder/'operation_receipt.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

def validate_train(folder,count,applied):
    proof=read(folder/'full_load_proof.json')
    assert all(proof[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
    rows=[json.loads(v) for v in (folder/'progress.jsonl').read_text().splitlines()]
    assert len(rows)==count
    assert all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
    assert all(r[k]['batches']==20 for r in rows for k in ['retention','locomotion_retention','upper_refinement'])
    new=[r['upper_refinement']['experimental_sequence_mean'] for r in rows]
    assert all(r['applied']==applied for r in new)
    assert any(r['batches']==20 and r['completed_cycles_total']>0 for r in new),'No actual complete-cycle backward batches'
    assert all(r['statistics'].get('bounded_gradient_norm',0)<=.1*r['statistics'].get('actual_PPO_shoulder_norm',0)+1e-6 for r in new)
    warm=rows[3:]
    torch_span=max(r['gpu']['torch_allocated_mb'] for r in warm)-min(r['gpu']['torch_allocated_mb'] for r in warm)
    assert torch_span<64,'Diagnostic CUDA allocation fails steady-memory check'
    (folder/'training_validation.json').write_text(json.dumps(dict(passed=True,actual_updates=count,exact_full_state=True,
        max_pre_update_KL=0,max_action_command_difference=0,existing_auxiliary_batches=20,
        new_sequence_warmup_zero_batches=sum(r['batches']==0 for r in new),
        new_sequence_active_updates=sum(r['batches']>0 for r in new),applied=applied,
        post_warmup_torch_allocation_span_MiB=torch_span,minimum_free_GPU_MiB=min(r['gpu']['free_mb'] for r in rows)),indent=2))

ROOT.mkdir(exist_ok=False)
try:
    with (OUT/'pause.log').open('w') as log:
        subprocess.run([sys.executable,str(OUT/'prepare_and_pause.py')],check=True,stdout=log,stderr=subprocess.STDOUT)
    pause=read(OUT/'diagnostic_pause.json');cp=pause['checkpoint'];WORK_START=time.monotonic()
    args=read(Path(cp).parent/'run_config.json')['arguments'];teacher=[]
    for k in ['termination_proof','standing_prior','standing_prior_sha256','locomotion_prior_certificate']:
        teacher+=['--'+k,str(args[k])]
    common=['--headless','--device','cuda:0','--speed','3.5','--reserve_mb','300']
    wrapper=OUT/'sequence_candidate.py'
    (ROOT/'experiment_manifest.json').write_text(json.dumps(dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint=cp,checkpoint_sha256=pause['checkpoint_sha256'],exact_full_state=True,
        experimental_source_sha256=sha(OUT/'sequence_shoulder_refinement.py'),wrapper_sha256=sha(wrapper),
        original_signature_does_not_certify_new_training_gradient=True,
        original_worker_physics_rewards_caps_unchanged=True,original_head_upper_and_both_priors_retained=True,
        applies_only_two_final_shoulder_rows=True,no_shared_trunk_direct_derivative=True,
        mean_band_deg=5,phase_bins=8,weight=.025,max_new_to_actual_PPO_shoulder_norm=.1,
        no_imposed_swing_amplitude_or_phase_or_reference_curve=True,training_seed=149,num_envs=4096,
        updates=[64,128],independent_initial_states_seeds=[2309,1911],diagnostic_actors_deployed=False,
        work_budget_seconds=WORK_LIMIT,saved_boundary_wait_and_drain_excluded_from_work_budget=True),indent=2))
    f=ROOT/'unchanged_actor_nominal'
    launch([str(wrapper),'--mode','eval','--output',str(f),'--checkpoint',cp,'--num_envs','320','--duration','20','--seed','901']+common,f,240)
    smoke=ROOT/'candidate_smoke'
    launch([str(wrapper),'--apply-sequence-mean','--mode','train','--output',str(smoke),'--checkpoint',cp,
        '--num_envs','64','--iterations','8','--learning_rate','.0001','--seed','149']+teacher+common,smoke,240)
    validate_train(smoke,8,True)
    f=ROOT/'candidate_reload'
    launch([str(wrapper),'--mode','eval','--output',str(f),'--checkpoint',str(smoke/'candidate.pt'),
        '--num_envs','64','--duration','4','--seed','149']+common,f,240)
    f=ROOT/'candidate_memory_smoke'
    launch([str(wrapper),'--apply-sequence-mean','--mode','train','--output',str(f),'--checkpoint',cp,
        '--num_envs','1024','--iterations','16','--learning_rate','.0001','--seed','149']+teacher+common,f,240)
    validate_train(f,16,True)
    for branch in ['control','candidate']:
        checkpoint=cp
        for updates in [64,128]:
            f=ROOT/f'{branch}_{updates:03d}_train'
            launch([str(wrapper)]+(['--apply-sequence-mean'] if branch=='candidate' else [])+['--mode','train','--output',str(f),
                '--checkpoint',checkpoint,'--num_envs','4096','--iterations','64','--learning_rate','.0001','--seed','149']+teacher+common,f,850)
            validate_train(f,64,branch=='candidate')
            checkpoint=str(f/'candidate.pt')
            for label,seed,push in [('nominal',901,False),('push',1301,True)]:
                g=ROOT/f'{branch}_{updates:03d}_{label}'
                launch([str(wrapper),'--mode','eval','--output',str(g),'--checkpoint',checkpoint,'--num_envs','320',
                    '--duration','20','--seed',str(seed)]+common+(['--perturb'] if push else []),g,240)
            if updates==128:
                for label,seed,push in [('diverse_nominal',2309,False),('diverse_push',1911,True)]:
                    g=ROOT/f'{branch}_{updates:03d}_{label}'
                    launch([str(OUT/'diverse_eval.py'),'--mode','eval','--output',str(g),'--checkpoint',checkpoint,'--num_envs','320',
                        '--duration','20','--seed',str(seed)]+common+(['--perturb'] if push else []),g,240)
    (ROOT/'completion.json').write_text(json.dumps(dict(completed=True,wall_seconds=time.time()-START,
        work_seconds=time.monotonic()-WORK_START,diagnostic_actors_deployed=False),indent=2))
except BaseException as error:
    (ROOT/'failure.json').write_text(json.dumps(dict(error=repr(error),traceback=traceback.format_exc(),wall_seconds=time.time()-START),indent=2))
    raise
finally:
    if (OUT/'diagnostic_pause.json').exists():
        with (OUT/'resume.log').open('w') as log:
            subprocess.run([sys.executable,str(OUT/'resume_after_diagnosis.py')],check=True,stdout=log,stderr=subprocess.STDOUT)
print('Matched diagnostic finished; exact original full-state learner resumed.',flush=True)
