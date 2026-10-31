"""Sequential diagnostics on an owned saved boundary, with guaranteed resume."""
import hashlib,json,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import psutil

OUT=Path(__file__).resolve().parent;ROOT=OUT/'matched_row_route'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
WORK_LIMIT=10800;START=time.time();WORK_START=None
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
    proof=read(folder/'full_load_proof.json')
    assert all(proof[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
    rows=[json.loads(v) for v in (folder/'progress.jsonl').read_text().splitlines()]
    assert len(rows)==count
    assert all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
    assert all(r[k]['batches']==20 for r in rows for k in ['retention','locomotion_retention','upper_refinement'])
    if applied is not None:
        annotation=read(folder/'experimental_teacher_target.json');assert annotation['applied']==applied
        if annotation['final_output_rows_only']:
            assert all(r['locomotion_retention']['final_row_route']['batches']==20 and
                r['locomotion_retention']['final_row_route']['parameter_contributions']=={'weight':20,'bias':20} for r in rows)
            assert all(r['locomotion_retention']['final_row_route']['head_rows']==[11,16] for r in rows)

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
    common=['--headless','--device','cuda:0','--speed','3.5','--reserve_mb','300'];wrapper=OUT/'neutral_teacher_candidate.py'
    manifest=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),checkpoint=cp,checkpoint_sha256=pause['checkpoint_sha256'],
        exact_full_state=True,helper_sha256=sha(OUT/'command_local_teacher_target.py'),wrapper_sha256=sha(wrapper),
        original_signature_does_not_certify_experimental_training_targets=True,
        reward_weights_physics_caps_commands_PPO_critic_exploration_unchanged=True,
        other_command_teacher_and_context_unchanged=True,original_upper20_batches_unchanged=True,
        changed_commands=[.15,1.5],original_MSE_weight=.25,original_MSE_denominator_outputs=23,branches=['control','shared','rows'],route_helper_sha256=sha(OUT/'final_row_route.py'),CPU_fixtures=read(OUT/'final_row_CPU_checks.json'),
        neutral_teacher_certificate=str(OUT/'neutral_teacher_certificate.json'),student_input_truthful=True,
        head_direct_gradient_zero=True,command_history_phase_head_eligibility_required=True,
        no_teacher_in_inference=True,training_seeds=[251,263],num_envs=4096,updates=[64,128],
        independent_initial_state_seeds=[4751,4771],diagnostic_actors_deployed=False,
        hypothesis='The rejected coupled target altered shared features despite zero direct head rows. Compare only its selected derivative restricted to final21 output rows against the same shared target and unchanged full-state control. Original PPO, other teachers, upper, weights, inputs and physics remain. Final rows do not guarantee off-command or physical retention.',
        work_budget_seconds=WORK_LIMIT,saved_boundary_wait_and_drain_excluded=True)
    (ROOT/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Confirm unchanged inference on this exact current saved checkpoint.
    f=ROOT/'unchanged_actor_nominal'
    launch([str(wrapper),'--mode','eval','--output',str(f),'--checkpoint',cp,'--num_envs','320','--duration','20','--seed','901']+common,f,260)
    import numpy as np
    originals=[]
    for p in Path(pause['run']).glob('*_gate_*_nominal/result.json'):
        cfg=read(p.parent/'run_config.json')
        if Path(cfg['arguments']['checkpoint']).resolve()==Path(cp).resolve() and read(p).get('completed'):originals.append(p.parent)
    assert len(originals)==1
    with np.load(originals[0]/'telemetry.npz') as a,np.load(f/'telemetry.npz') as b:
        assert set(a.files)==set(b.files)
        arrays={k:np.array_equal(a[k],b[k],equal_nan=True) if a[k].dtype.kind in 'fc' else np.array_equal(a[k],b[k]) for k in a.files}
    assert all(arrays.values()) and read(originals[0]/'evaluation.json')==read(f/'evaluation.json')
    (ROOT/'unchanged_actor_parity.json').write_text(json.dumps(dict(passed=True,arrays=arrays,exact_checkpoint=cp,
        original=str(originals[0]),replay=str(f),all_cases_bitwise_equal=True),indent=2)+'\n')
    f=ROOT/'passive_PPO_capture'
    launch([str(OUT/'capture_current_gradients.py'),'--mode','train','--output',str(f),'--checkpoint',cp,
        '--num_envs','4096','--iterations','4','--learning_rate','.0001','--seed','251']+teacher+common,f,260)
    validate_train(f,4)
    with (OUT/'row_response_analysis.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-X','utf8',str(OUT/'analyze_row_response.py')],check=True,stdout=log,stderr=subprocess.STDOUT)
    # Recheck compatibility on this newer, exact trial checkpoint.
    launch([str(OUT/'neutral_teacher_probe.py'),'--headless','--device','cuda:0'],OUT/'teacher_compatibility_response',700)
    physical=read(OUT/'teacher_compatibility_response/result.json')
    assert physical['actor_loaded_exact'] and physical['checkpoint_sha256']==pause['checkpoint_sha256']
    assert physical['maximum_command_difference']==0 and physical['maximum_reward_term_sum_error']<1e-3
    selected=[g for g in physical['groups'] if g['actor']=='neutral_body_current_head']
    assert len(selected)==4 and all(g['survival']==1 and g['resets']==0 and g['mean_absolute_world_lateral_velocity']<.14
        and (g['command']!=1.5 or g['unloaded_geometric_flight_fraction']>.015)
        and (g['command']!=.15 or abs(g['mean_forward_world_speed']-.15)<.1) for g in selected),'Current-state physical compatibility no longer supports candidate'
    assert all(g['mean_reward_rate']>=next(c['mean_reward_rate'] for c in physical['groups'] if c['actor']=='current'
        and c['command']==g['command'] and c['name'].split('_')[0]==g['name'].split('_')[0]) for g in selected),'Current-state complete return no longer supports candidate'
    (ROOT/'current_physical_support.json').write_text(json.dumps(dict(passed=True,groups=selected,checkpoint=cp,
        same_exact_trial_checkpoint=True,diagnostic_composites_only=True),indent=2)+'\n')
    smoke=ROOT/'candidate_smoke'
    launch([str(wrapper),'--coupled-neutral-teacher','--final-row-route','--mode','train','--output',str(smoke),'--checkpoint',cp,
        '--num_envs','64','--iterations','8','--learning_rate','.0001','--seed','251']+teacher+common,smoke,260)
    validate_train(smoke,8,True)
    f=ROOT/'candidate_full_reload'
    launch([str(wrapper),'--coupled-neutral-teacher','--final-row-route','--mode','train','--output',str(f),'--checkpoint',str(smoke/'candidate.pt'),
        '--num_envs','64','--iterations','4','--learning_rate','.0001','--seed','251']+teacher+common,f,260)
    validate_train(f,4,True)
    (ROOT/'stage_ready.json').write_text(json.dumps(dict(physical_support=True,CPU_response=read(OUT/'current_row_response.json'),smoke_and_reload=True)))
    deadline=time.monotonic()+900
    while not (ROOT/'learning_decision.json').exists():
        assert time.monotonic()<deadline,'No diagnostic learning decision; safely resume original learner'
        time.sleep(2)
    assert read(ROOT/'learning_decision.json')['proceed'],'Diagnostic learning declined; safely resume original learner'
    for seed in [251,263]:
        for branch in ['control','shared','rows']:
            checkpoint=cp
            for updates in [64,128]:
                f=ROOT/f'seed{seed}_{branch}_{updates:03d}_train'
                launch([str(wrapper)]+(['--coupled-neutral-teacher']+(['--final-row-route'] if branch=='rows' else []) if branch!='control' else [])+['--mode','train','--output',str(f),
                    '--checkpoint',checkpoint,'--num_envs','4096','--iterations','64','--learning_rate','.0001','--seed',str(seed)]+teacher+common,f,850)
                validate_train(f,64,branch!='control')
                if branch!='control':
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
                    for label,evaluation_seed,push in [('diverse_nominal',4751,False),('diverse_push',4771,True)]:
                        g=ROOT/f'seed{seed}_{branch}_{updates:03d}_{label}'
                        launch([str(OUT/'diverse_eval.py'),'--mode','eval','--output',str(g),'--checkpoint',checkpoint,'--num_envs','320',
                            '--duration','20','--seed',str(evaluation_seed)]+common+(['--perturb'] if push else []),g,260)
    (ROOT/'completion.json').write_text(json.dumps(dict(completed=True,wall_seconds=time.time()-START,
        work_seconds=time.monotonic()-WORK_START,diagnostic_actors_deployed=False),indent=2)+'\n')
except BaseException as error:
    (ROOT/'failure.json').write_text(json.dumps(dict(error=repr(error),traceback=traceback.format_exc(),wall_seconds=time.time()-START),indent=2)+'\n')
    raise
finally:
    if (OUT/'diagnostic_pause.json').exists():
        with (OUT/'resume.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-X','utf8',str(OUT/'resume_after_diagnosis.py')],check=True,stdout=log,stderr=subprocess.STDOUT)
print('Diagnostic comparison completed; exact original supervisor resumed.',flush=True)
