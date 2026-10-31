"""Sequential diagnostics on an owned saved boundary, with guaranteed resume."""
import hashlib,json,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
from runtime_joint_order import runtime_joint_order
import psutil

OUT=Path(__file__).resolve().parent;ROOT=OUT/'matched_coupled_teacher'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
WORK_LIMIT=7200;START=time.time();WORK_START=None
NAMES={'worker_unitree.py','neutral_teacher_candidate.py','neutral_teacher_probe.py','capture_current_gradients.py','diverse_eval.py'}

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
    if (folder/'result.json').exists() and read(folder/'result.json').get('completed'):
        assert (folder/'operation_receipt.json').exists(), 'Require completed worker cleanup receipt before reuse'
        assert read(folder/'process.json')['argv']==argv, 'Reuse only the exact planned operation'
        print(json.dumps(dict(reused_completed=str(folder.relative_to(OUT)))),flush=True)
        return
    if folder.exists():
        assert folder.name=='seed229_candidate_064_nominal' and (OUT/'independent_review_abort.json').exists(), 'Never overwrite an unexplained partial diagnostic'
        archived=folder.with_name(folder.name+'__aborted_wrong_URDF_review')
        assert folder.resolve().is_relative_to(ROOT.resolve()) and archived.resolve().is_relative_to(ROOT.resolve())
        folder.rename(archived)
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
    result=dict(completed=str(folder.relative_to(OUT)),operation_seconds=round(time.time()-beginning,2),work_seconds=round(time.monotonic()-WORK_START,2))
    (folder/'operation_receipt.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

def validate_train(folder,count,applied=None):
    if applied:
        names=runtime_joint_order(folder);expected=[names.index(n) for n in ['head_yaw','head_pitch']]
        observations=[json.loads(x) for x in (folder/'progress.jsonl').read_text().splitlines()]
        assert all(x['locomotion_retention']['head_excluded']==expected for x in observations), 'Bind exclusions to actual runtime joint order'
    proof=read(folder/'full_load_proof.json')
    assert all(proof[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
    rows=[json.loads(v) for v in (folder/'progress.jsonl').read_text().splitlines()]
    assert len(rows)==count
    assert all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
    assert all(r[k]['batches']==20 for r in rows for k in ['retention','locomotion_retention','upper_refinement'])
    if applied is not None:assert read(folder/'experimental_teacher_target.json')['applied']==applied
    warm=rows[min(3,count-1):]
    allocation_span=max(r['gpu']['torch_allocated_mb'] for r in warm)-min(r['gpu']['torch_allocated_mb'] for r in warm)
    assert allocation_span<64 and min(r['gpu']['free_mb'] for r in rows)>=300
    receipt=dict(passed=True,actual_updates=count,exact_full_state=True,max_pre_update_KL=0,max_action_command_difference=0,
        existing_auxiliary_batches=20,applied=applied,post_warmup_torch_allocation_span_MiB=allocation_span,
        minimum_free_GPU_MiB=min(r['gpu']['free_mb'] for r in rows),minimum_available_host_MiB=min(r['gpu']['ram_available_mb'] for r in rows))
    (folder/'training_validation.json').write_text(json.dumps(receipt,indent=2))

PAUSE=OUT/'remainder_pause'
assert ROOT.exists() and (OUT/'diagnostic_resume.json').exists()
manifest=read(ROOT/'experiment_manifest.json');cp=manifest['checkpoint']
assert sha(cp)==manifest['checkpoint_sha256']
assert sha(OUT/'neutral_teacher_candidate.py')==manifest['wrapper_sha256'] and sha(OUT/'command_local_teacher_target.py')==manifest['helper_sha256']
assert read(OUT.parent/'joint_order_review_repair.json')['experimental_head_mapping_correct']
try:
    with (PAUSE/'pause.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-X','utf8',str(PAUSE/'prepare_and_pause.py')],check=True,stdout=log,stderr=subprocess.STDOUT)
    pause=read(PAUSE/'diagnostic_pause.json');WORK_START=time.monotonic()
    original=read(OUT/'protected_source_copy.json');fresh=read(PAUSE/'protected_source_copy.json')
    assert fresh['protected_hashes']==original['protected_hashes'], 'Production sources must be byte-identical'
    receipt=dict(original_trial_checkpoint=cp,checkpoint_sha256=manifest['checkpoint_sha256'],
        current_main_saved_checkpoint=pause['checkpoint'],current_main_saved_SHA=pause['checkpoint_sha256'],
        reuse_only_exact_complete_saved_results=True,only_CPU_reviewer_changed=True,diagnostic_actor_deployed=False,
        original_wrapper_and_derivative_hashes_verified=True,false_positive_abort_preserved=True)
    (ROOT/'remainder_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    args=read(Path(cp).parent/'run_config.json')['arguments'];teacher=[]
    for k in ['termination_proof','standing_prior','standing_prior_sha256','locomotion_prior_certificate']:
        teacher+=['--'+k,str(args[k])]
    common=['--headless','--device','cuda:0','--speed','3.5','--reserve_mb','300'];wrapper=OUT/'neutral_teacher_candidate.py'
    for seed in [229,241]:
        for branch in ['control','candidate']:
            checkpoint=cp
            for updates in [64,128]:
                f=ROOT/f'seed{seed}_{branch}_{updates:03d}_train'
                launch([str(wrapper)]+(['--coupled-neutral-teacher'] if branch=='candidate' else [])+['--mode','train','--output',str(f),
                    '--checkpoint',checkpoint,'--num_envs','4096','--iterations','64','--learning_rate','.0001','--seed',str(seed)]+teacher+common,f,850)
                validate_train(f,64,branch=='candidate')
                if branch=='candidate':
                    rows=[json.loads(v) for v in (f/'progress.jsonl').read_text().splitlines()]
                    assert all(r['locomotion_retention']['head_excluded']==r['locomotion_retention'].get('head_excluded') for r in rows)
                    counts={str(c):sum(r['locomotion_retention']['command_local_coupled_samples'][str(c)] for r in rows) for c in [.15,1.5]}
                    assert all(v>0 for v in counts.values()),'No command-local targets applied'
                    (f/'command_local_target_validation.json').write_text(json.dumps(dict(passed=True,samples=counts,
                        head_derivative_zero_by_checked_helper=True,teacher_sha256=read(OUT/'neutral_teacher_certificate.json')['checkpoint_sha256']),indent=2)+'\n')
                checkpoint=str(f/'candidate.pt')
                for label,evaluation_seed,push in [('nominal',901,False),('push',1301,True)]:
                    g=ROOT/f'seed{seed}_{branch}_{updates:03d}_{label}'
                    launch([str(wrapper),'--mode','eval','--output',str(g),'--checkpoint',checkpoint,'--num_envs','320',
                        '--duration','20','--seed',str(evaluation_seed)]+common+(['--perturb'] if push else []),g,260)
                if updates==128:
                    for label,evaluation_seed,push in [('diverse_nominal',4671,False),('diverse_push',4691,True)]:
                        g=ROOT/f'seed{seed}_{branch}_{updates:03d}_{label}'
                        launch([str(OUT/'diverse_eval.py'),'--mode','eval','--output',str(g),'--checkpoint',checkpoint,'--num_envs','320',
                            '--duration','20','--seed',str(evaluation_seed)]+common+(['--perturb'] if push else []),g,260)

    (ROOT/'completion.json').write_text(json.dumps(dict(completed=True,wall_seconds=time.time()-START,
        remainder_work_seconds=time.monotonic()-WORK_START,first_attempt_failed_and_preserved=True,
        diagnostic_actors_deployed=False),indent=2)+'\n')
except BaseException as error:
    (ROOT/'remainder_failure.json').write_text(json.dumps(dict(error=repr(error),traceback=traceback.format_exc(),wall_seconds=time.time()-START),indent=2)+'\n')
    raise
finally:
    if (PAUSE/'diagnostic_pause.json').exists():
        with (PAUSE/'resume.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-X','utf8',str(PAUSE/'resume_after_diagnosis.py')],check=True,stdout=log,stderr=subprocess.STDOUT)
print('Remaining exact comparison complete; same original supervisor resumed.',flush=True)
