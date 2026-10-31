"""Run only after matched training finishes, with exact owned pause/resume."""
from pathlib import Path
import hashlib,json,subprocess,sys,time,traceback
import psutil

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'high_speed_extrapolation'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
names={'worker_unitree.py','slow_reward_candidate.py','motor_retention_probe.py','high_speed_probe.py','diverse_eval.py','capture_current_gradients.py','capture_objective_advantages.py','inverse_arm_probe.py'}

def kill_owned(proc,identity):
    try:
        parent=psutil.Process(proc.pid)
        assert parent.create_time()==identity
        tree=parent.children(recursive=True)+[parent]
        for p in tree:
            if p.is_running():p.terminate()
        _,alive=psutil.wait_procs(tree,timeout=15)
        for p in alive:p.kill()
        psutil.wait_procs(alive,timeout=15)
    except psutil.NoSuchProcess:pass

def launch(argv,folder,limit):
    assert time.monotonic()-work_start<1400,'Work-only higher-speed budget exhausted'
    assert not any(any(Path(v).name in names for v in p.info['cmdline'] or []) for p in psutil.process_iter(['cmdline'])), 'Another CUDA worker is active'
    folder.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    with (folder/'stdout.log').open('w',encoding='utf-8') as so,(folder/'stderr.log').open('w',encoding='utf-8') as se:
        proc=subprocess.Popen([sys.executable,'-u','-X','utf8']+argv,stdout=so,stderr=se,creationflags=subprocess.CREATE_NO_WINDOW)
        identity=psutil.Process(proc.pid).create_time()
        (folder/'process.json').write_text(json.dumps(dict(pid=proc.pid,create_time=identity,argv=argv),indent=2))
        try:
            deadline=time.monotonic()+limit
            while proc.poll() is None and not (folder/'result.json').exists():
                assert time.monotonic()<deadline,'Higher-speed diagnostic worker timed out'
                time.sleep(2)
            if proc.poll() is None:
                assert read(folder/'result.json')['completed']
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    kill_owned(proc,identity)
                    (folder/'exit_cleanup.json').write_text(json.dumps(dict(completed_saved_result=True,only_owned_Kit_shutdown_cleanup=True)))
            assert (proc.returncode==0 or (folder/'exit_cleanup.json').exists()) and read(folder/'result.json')['completed']
        except BaseException:
            kill_owned(proc,identity);raise
    receipt=dict(folder=folder.name,completed=True,seconds=time.monotonic()-started,work_seconds=time.monotonic()-work_start)
    (folder/'operation_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)

assert (ROOT/'matched_slow_reward/completion.json').exists(), 'Do not overlap the matched comparison'
assert (ROOT/'diagnostic_resume.json').exists() and not (OUT/'diagnostic_pause.json').exists()
start=time.time();work_start=None
try:
    # Allow genuine original learning to finish before taking another boundary.
    # Never interrupt a partially trained worker to obtain a fresher probe.
    prior_pause=read(ROOT/'diagnostic_pause.json');deadline=time.monotonic()+900
    while True:
        state=read(Path(prior_pause['run'])/'state.json')
        assert state['status']=='running'
        supervisor=psutil.Process(prior_pause['supervisor_pid'])
        assert supervisor.create_time()==prior_pause['supervisor_create_time']
        checkpoint=Path(state['checkpoint'])
        if checkpoint.is_file() and hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=prior_pause['checkpoint_sha256']:
            break
        assert time.monotonic()<deadline,'No fresh saved original main block; preserve healthy training'
        time.sleep(3)
    with (OUT/'pause.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-X','utf8',str(OUT/'prepare_and_pause.py')],check=True,stdout=log,stderr=subprocess.STDOUT)
    work_start=time.monotonic();pause=read(OUT/'diagnostic_pause.json');checkpoint=pause['checkpoint']
    common=['--headless','--device','cuda:0','--speed','5','--reserve_mb','300','--checkpoint',checkpoint,'--num_envs','416','--duration','20']
    worker=OUT/'control_source/physical_v29/training/worker_unitree.py'
    for label,seed,push,diverse in [('nominal',901,False,False),('pushed',1301,True,False),('diverse_nominal',4591,False,True),('diverse_pushed',4611,True,True)]:
        folder=OUT/label
        launch([str(OUT/'diverse_eval.py' if diverse else worker),'--mode','eval','--output',str(folder),'--seed',str(seed)]+common+(['--perturb'] if push else []),folder,240)
        if diverse:
            receipt=read(folder/'initial_diversity.json')
            assert len(receipt['unique_initial_states_by_command'])==13
            assert all(n==32 for n in receipt['unique_initial_states_by_command'].values())
    launch([str(OUT/'high_speed_probe.py'),'--headless','--device','cuda:0'],OUT/'physical_response',500)
    subprocess.run([sys.executable,'-X','utf8',str(ROOT/'analyze_high_speed.py')],check=True)
    config=read(Path(checkpoint).parent/'run_config.json')['arguments'];teacher=[]
    for k in ['termination_proof','standing_prior','standing_prior_sha256','locomotion_prior_certificate']:
        teacher+=['--'+k,str(config[k])]
    for branch in ['original','narrow']:
        folder=OUT/('objective_'+branch)
        launch([str(OUT/'capture_objective_advantages.py')]+(['--slow-tracking-tolerance'] if branch=='narrow' else [])+
            ['--mode','train','--headless','--device','cuda:0','--speed',str(config['speed']),'--reserve_mb','300',
             '--checkpoint',checkpoint,'--output',str(folder),'--num_envs','4096','--iterations','4','--learning_rate','.0001','--seed','211']+teacher,folder,240)
        proof=read(folder/'full_load_proof.json');rows=[json.loads(r) for r in (folder/'progress.jsonl').read_text().splitlines()]
        assert all(proof[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
        assert len(rows)==4 and all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
        assert all(r[k]['batches']==20 for r in rows for k in ['retention','locomotion_retention','upper_refinement'])
    subprocess.run([sys.executable,'-X','utf8',str(ROOT/'analyze_objective_advantages.py')],check=True)
    launch([str(OUT/'inverse_arm_probe.py'),'--headless','--device','cuda:0'],OUT/'inverse_arm_response',500)
    subprocess.run([sys.executable,'-X','utf8',str(ROOT/'analyze_inverse_arm_response.py')],check=True)
    (OUT/'completion.json').write_text(json.dumps(dict(completed=True,checkpoint=checkpoint,checkpoint_sha256=pause['checkpoint_sha256'],
        work_seconds=time.monotonic()-work_start,wall_seconds=time.time()-start,diagnostic_actor_deployed=False,
        source_caps_gate_classifiers_unchanged=True,untrained_extrapolation_not_promotion=True),indent=2))
except BaseException as error:
    (OUT/'failure.json').write_text(json.dumps(dict(error=repr(error),traceback=traceback.format_exc(),wall_seconds=time.time()-start),indent=2))
    raise
finally:
    if (OUT/'diagnostic_pause.json').exists():
        with (OUT/'resume.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-X','utf8',str(OUT/'resume_after_diagnosis.py')],check=True,stdout=log,stderr=subprocess.STDOUT)
print('Higher-speed tests complete; original full learner resumed.',flush=True)
