"""One owned saved-boundary diagnosis, with cleanup and mandatory exact resume."""
import hashlib,json,subprocess,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import psutil
OUT=Path(__file__).resolve().parent;ROOT=OUT/'neutral_requalification'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
NAMES={'worker_unitree.py','neutral_teacher_probe.py','diverse_eval.py','capture_current_gradients.py','neutral_teacher_candidate.py'}
WORK_LIMIT=1600;WORK_START=None;START=time.time()

def cleanup(p,created):
    try:
        if p.is_running() and p.create_time()==created:
            tree=p.children(recursive=True)+[p]
            for q in tree:
                if q.is_running():q.terminate()
            _,alive=psutil.wait_procs(tree,timeout=15)
            for q in alive:q.kill()
            psutil.wait_procs(alive,timeout=15)
    except psutil.NoSuchProcess:pass

def launch(argv,folder,limit):
    assert time.monotonic()-WORK_START<WORK_LIMIT,'Bounded diagnostic work budget exhausted'
    assert not any(any(Path(x).name in NAMES for x in p.info['cmdline'] or []) for p in psutil.process_iter(['cmdline'])), 'Another CUDA worker is active'
    folder.mkdir(parents=True,exist_ok=False);beg=time.time()
    with (folder/'stdout.log').open('w',encoding='utf-8') as so,(folder/'stderr.log').open('w',encoding='utf-8') as se:
        process=subprocess.Popen([sys.executable,'-u','-X','utf8']+argv,stdout=so,stderr=se,creationflags=subprocess.CREATE_NO_WINDOW)
        p=psutil.Process(process.pid);created=p.create_time()
        (folder/'process.json').write_text(json.dumps(dict(pid=p.pid,create_time=created,argv=argv),indent=2))
        try:
            deadline=time.monotonic()+limit
            while process.poll() is None and not (folder/'result.json').exists():
                assert time.monotonic()<deadline,'Worker budget exceeded'
                assert psutil.virtual_memory().available/2**20>=2048,'Diagnostic host memory guard'
                time.sleep(2)
            if process.poll() is None:
                assert read(folder/'result.json')['completed']
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    cleanup(p,created)
                    (folder/'exit_cleanup.json').write_text(json.dumps(dict(completed_saved_result=True,owned_PID=p.pid,only_Kit_shutdown_cleanup=True)))
            assert (process.returncode==0 or (folder/'exit_cleanup.json').exists()) and read(folder/'result.json')['completed'], (folder,process.returncode)
        except BaseException:
            cleanup(p,created);raise
    receipt=dict(folder=str(folder),seconds=time.time()-beg,work_seconds=time.monotonic()-WORK_START,completed=True)
    (folder/'operation_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)


try:
    with (OUT/'pause.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-X','utf8',str(OUT/'prepare_and_pause.py')],check=True,stdout=log,stderr=subprocess.STDOUT)
    WORK_START=time.monotonic();pause=read(OUT/'diagnostic_pause.json');cp=pause['checkpoint']
    teacher=read(OUT/'teacher_diagnosis_preparation.json')['teacher_checkpoint']
    common=['--headless','--device','cuda:0','--reserve_mb','300','--mode','eval','--duration','20']
    for label,seed,push in [('diverse_nominal',4631,False),('diverse_push',4651,True)]:
        f=OUT/label
        launch([str(OUT/'diverse_eval.py'),'--speed','3','--num_envs','288','--checkpoint',teacher,'--seed',str(seed),'--output',str(f)]+common+(['--perturb'] if push else []),f,260)
        receipt=read(f/'initial_diversity.json')
        assert len(receipt['unique_initial_states_by_command'])==9 and all(n==32 for n in receipt['unique_initial_states_by_command'].values())
    launch([str(OUT/'neutral_teacher_probe.py'),'--headless','--device','cuda:0'],OUT/'teacher_compatibility_response',700)
    cfg=read(Path(cp).parent/'run_config.json')['arguments'];args=[]
    for k in ['termination_proof','standing_prior','standing_prior_sha256','locomotion_prior_certificate']:args+=['--'+k,str(cfg[k])]
    f=OUT/'passive_current_PPO'
    launch([str(OUT/'capture_current_gradients.py'),'--mode','train','--headless','--device','cuda:0','--reserve_mb','300',
        '--speed','3.5','--num_envs','4096','--iterations','4','--learning_rate','.0001','--seed','223','--checkpoint',cp,'--output',str(f)]+args,f,260)
    proof=read(f/'full_load_proof.json');rows=[json.loads(x) for x in (f/'progress.jsonl').read_text().splitlines()]
    assert len(rows)==4 and all(proof[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
    assert all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 and all(r[k]['batches']==20 for k in ['retention','locomotion_retention','upper_refinement']) for r in rows)
    (OUT/'teacher_diagnosis_completion.json').write_text(json.dumps(dict(completed=True,checkpoint=cp,checkpoint_sha256=pause['checkpoint_sha256'],
        teacher_certificate=str(OUT/'neutral_teacher_certificate.json'),work_seconds=time.monotonic()-WORK_START,wall_seconds=time.time()-START,
        diagnostic_PPO_updates_discarded=4,diagnostic_actors_deployed=False,composites_deployed=False),indent=2)+'\n')
except BaseException as error:
    (OUT/'teacher_diagnosis_failure.json').write_text(json.dumps(dict(error=repr(error),traceback=traceback.format_exc(),wall_seconds=time.time()-START),indent=2)+'\n')
    raise
finally:
    if (OUT/'diagnostic_pause.json').exists():
        with (OUT/'resume.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-X','utf8',str(OUT/'resume_after_diagnosis.py')],check=True,stdout=log,stderr=subprocess.STDOUT)
print('Fresh neutral-teacher compatibility diagnosis complete; exact original learner resumed.',flush=True)
