"""Separate saved-boundary wait from bounded work; always resume original."""
import json,subprocess,sys,time,hashlib
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def launch(script,args,folder,timeout):
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'harness_stdout.log').open('w',encoding='utf-8') as so,(folder/'harness_stderr.log').open('w',encoding='utf-8') as se:
        p=subprocess.Popen([sys.executable,'-u','-X','utf8',str(script)]+args,stdout=so,stderr=se,creationflags=subprocess.CREATE_NO_WINDOW)
        owned=psutil.Process(p.pid);identity=owned.create_time()
        (folder/'harness_process.json').write_text(json.dumps(dict(pid=p.pid,create_time=identity,args=args)))
        try:code=p.wait(timeout=timeout)
        except BaseException:
            if owned.is_running() and owned.create_time()==identity:
                tree=owned.children(recursive=True)+[owned]
                for proc in tree:
                    if proc.is_running():proc.terminate()
                psutil.wait_procs(tree,timeout=20)
            raise
    assert code==0,(str(script),code)
try:
    launch(OUT/'prepare_and_pause.py',[],OUT/'pause_receipt',850)
    started=time.time();pause=read(OUT/'diagnostic_pause.json');cp=Path(pause['checkpoint'])
    assert hashlib.sha256(cp.read_bytes()).hexdigest()==pause['checkpoint_sha256']
    launch(OUT/'capture_policy_states.py',['--mode','eval','--output',str(OUT/'truthful_nominal'),
        '--checkpoint',str(cp),'--num_envs','320','--duration','20','--seed','137',
        '--headless','--device','cuda:0','--speed','3.5','--reserve_mb','300'],OUT/'truthful_nominal',210)
    original=read(cp.parent/'run_config.json')['arguments'];teachers=[]
    for key in ['termination_proof','standing_prior','standing_prior_sha256','locomotion_prior_certificate']:
        teachers+=['--'+key,str(original[key])]
    f=OUT/'gradient_smoke'
    launch(OUT/'capture_current_gradients.py',['--mode','train','--output',str(f),
        '--checkpoint',str(cp),'--num_envs','4096','--iterations','4','--learning_rate','.0001',
        '--seed','137','--headless','--device','cuda:0','--speed','3.5','--reserve_mb','300']+teachers,f,210)
    proof=read(f/'full_load_proof.json');rows=[json.loads(v) for v in (f/'progress.jsonl').read_text().splitlines()]
    assert all(proof[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
    assert len(rows)==4 and all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
    assert all(r['upper_refinement']['batches']==20 and r['retention']['batches']==20 and r['locomotion_retention']['batches']==20 for r in rows)
    (OUT/'diagnostic_completion.json').write_text(json.dumps(dict(completed=True,active_diagnostic_seconds=time.time()-started,
        wait_and_drain_separate=True,passive_capture=True,exact_full_load=True,diagnostic_PPO_updates=4,
        diagnostic_actors_deployed=False,production_sources_changed=False),indent=2))
finally:
    if (OUT/'diagnostic_pause.json').exists() and not (OUT/'diagnostic_resume.json').exists():
        launch(OUT/'resume_after_diagnosis.py',[],OUT/'resume_receipt',45)
print('Passive truthful-state and gradient diagnostics finished; original learner resumed.')
