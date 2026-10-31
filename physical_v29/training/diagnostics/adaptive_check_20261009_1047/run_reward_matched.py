"""Sequential diagnostics on an owned saved boundary, with guaranteed resume."""
import hashlib,json,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import psutil

OUT=Path(__file__).resolve().parent;ROOT=OUT/'matched_slow_reward'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
WORK_LIMIT=6600;START=time.time();WORK_START=None
NAMES={'worker_unitree.py','slow_reward_candidate.py','motor_retention_probe.py','capture_current_gradients.py','diverse_eval.py'}

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
    result=dict(completed=str(folder.relative_to(OUT)),operation_seconds=round(time.time()-beginning,2),work_seconds=round(time.monotonic()-WORK_START,2))
    (folder/'operation_receipt.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

def validate_train(folder,count,applied=None):
    proof=read(folder/'full_load_proof.json')
    assert all(proof[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
    rows=[json.loads(v) for v in (folder/'progress.jsonl').read_text().splitlines()]
    assert len(rows)==count
    assert all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
    assert all(r[k]['batches']==20 for r in rows for k in ['retention','locomotion_retention','upper_refinement'])
    if applied is not None:assert read(folder/'experimental_reward.json')['applied']==applied
    warm=rows[min(3,count-1):]
    allocation_span=max(r['gpu']['torch_allocated_mb'] for r in warm)-min(r['gpu']['torch_allocated_mb'] for r in warm)
    assert allocation_span<64 and min(r['gpu']['free_mb'] for r in rows)>=300
    receipt=dict(passed=True,actual_updates=count,exact_full_state=True,max_pre_update_KL=0,max_action_command_difference=0,
        existing_auxiliary_batches=20,applied=applied,post_warmup_torch_allocation_span_MiB=allocation_span,
        minimum_free_GPU_MiB=min(r['gpu']['free_mb'] for r in rows),minimum_available_host_MiB=min(r['gpu']['ram_available_mb'] for r in rows))
    (folder/'training_validation.json').write_text(json.dumps(receipt,indent=2))

ROOT.mkdir(exist_ok=False)
try:
    with (OUT/'pause.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-X','utf8',str(OUT/'prepare_and_pause.py')],check=True,stdout=log,stderr=subprocess.STDOUT)
    pause=read(OUT/'diagnostic_pause.json');cp=pause['checkpoint'];WORK_START=time.monotonic()
    args=read(Path(cp).parent/'run_config.json')['arguments'];teacher=[]
    for k in ['termination_proof','standing_prior','standing_prior_sha256','locomotion_prior_certificate']:
        teacher+=['--'+k,str(args[k])]
    common=['--headless','--device','cuda:0','--speed','3.5','--reserve_mb','300']
    wrapper=OUT/'slow_reward_candidate.py'
    manifest=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),checkpoint=cp,checkpoint_sha256=pause['checkpoint_sha256'],
        exact_full_state=True,experimental_source_sha256=sha(OUT/'slow_tracking_kernel.py'),wrapper_sha256=sha(wrapper),
        original_signature_does_not_certify_changed_training_reward=True,original_physics_caps_teachers_upper_unchanged=True,
        changed_commands="0.1<speed<=1.0",old_tracking_std=.5,new_tracking_std="max(.10,.12*speed)",
        unchanged_sampler=True,unchanged_all_weights=True,unchanged_faster_reward=True,
        training_seeds=[179,191],num_envs=4096,updates=[64,128],independent_initial_state_seeds=[4531,4551],
        hypothesis='Derive slow Gaussian error width from existing classifier tolerances. Test precise retained tracking without weakening faster gait. Instantaneous gait oscillations are a measured risk, not a proof of repair.',
        diagnostic_actors_deployed=False,work_budget_seconds=WORK_LIMIT,saved_boundary_wait_and_drain_excluded=True)
    (ROOT/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    f=ROOT/'unchanged_actor_nominal'
    launch([str(wrapper),'--mode','eval','--output',str(f),'--checkpoint',cp,'--num_envs','320','--duration','20','--seed','901']+common,f,240)
    subprocess.run([sys.executable,'-X','utf8',str(OUT/'validate_physical_replay.py')],check=True)
    f=ROOT/'passive_PPO_capture'
    launch([str(OUT/'capture_current_gradients.py'),'--mode','train','--output',str(f),'--checkpoint',cp,
        '--num_envs','4096','--iterations','4','--learning_rate','.0001','--seed','179']+teacher+common,f,240)
    validate_train(f,4)
    launch([str(OUT/'motor_retention_probe.py'),'--headless','--device','cuda:0'],OUT/'motor_retention_response',600)
    subprocess.run([sys.executable,'-X','utf8',str(OUT/'analyze_physical_reward.py')],check=True)
    analysis=read(OUT/'physical_reward_counterfactual.json')
    assert analysis['safe_to_test_learning'], 'Physical counterfactual rejected this width before learning; keep original learner'
    smoke=ROOT/'candidate_smoke'
    launch([str(wrapper),'--slow-tracking-tolerance','--mode','train','--output',str(smoke),'--checkpoint',cp,
        '--num_envs','64','--iterations','8','--learning_rate','.0001','--seed','179']+teacher+common,smoke,240)
    validate_train(smoke,8,True)
    f=ROOT/'candidate_full_reload'
    launch([str(wrapper),'--slow-tracking-tolerance','--mode','train','--output',str(f),'--checkpoint',str(smoke/'candidate.pt'),
        '--num_envs','64','--iterations','4','--learning_rate','.0001','--seed','179']+teacher+common,f,240)
    validate_train(f,4,True)
    for seed in [179,191]:
        for branch in ['control','candidate']:
            checkpoint=cp
            for updates in [64,128]:
                f=ROOT/f'seed{seed}_{branch}_{updates:03d}_train'
                launch([str(wrapper)]+(['--slow-tracking-tolerance'] if branch=='candidate' else [])+['--mode','train','--output',str(f),
                    '--checkpoint',checkpoint,'--num_envs','4096','--iterations','64','--learning_rate','.0001','--seed',str(seed)]+teacher+common,f,850)
                validate_train(f,64,branch=='candidate');checkpoint=str(f/'candidate.pt')
                for label,evaluation_seed,push in [('nominal',901,False),('push',1301,True)]:
                    g=ROOT/f'seed{seed}_{branch}_{updates:03d}_{label}'
                    launch([str(wrapper),'--mode','eval','--output',str(g),'--checkpoint',checkpoint,'--num_envs','320',
                        '--duration','20','--seed',str(evaluation_seed)]+common+(['--perturb'] if push else []),g,240)
                if updates==128:
                    for label,evaluation_seed,push in [('diverse_nominal',4531,False),('diverse_push',4551,True)]:
                        g=ROOT/f'seed{seed}_{branch}_{updates:03d}_{label}'
                        launch([str(OUT/'diverse_eval.py'),'--mode','eval','--output',str(g),'--checkpoint',checkpoint,'--num_envs','320',
                            '--duration','20','--seed',str(evaluation_seed)]+common+(['--perturb'] if push else []),g,240)
    (ROOT/'completion.json').write_text(json.dumps(dict(completed=True,wall_seconds=time.time()-START,
        work_seconds=time.monotonic()-WORK_START,diagnostic_actors_deployed=False),indent=2))
except BaseException as error:
    (ROOT/'failure.json').write_text(json.dumps(dict(error=repr(error),traceback=traceback.format_exc(),wall_seconds=time.time()-START),indent=2))
    raise
finally:
    if (OUT/'diagnostic_pause.json').exists():
        with (OUT/'resume.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-X','utf8',str(OUT/'resume_after_diagnosis.py')],check=True,stdout=log,stderr=subprocess.STDOUT)
print('Diagnostic completed; exact original supervisor resumed.',flush=True)
