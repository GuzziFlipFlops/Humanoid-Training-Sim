"""Bounded sequential diagnostics; resume the same supervisor in finally."""
import json,subprocess,sys,time,hashlib
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
pause=read(OUT/'diagnostic_pause.json');cp=Path(pause['checkpoint'])
assert hashlib.sha256(cp.read_bytes()).hexdigest()==pause['checkpoint_sha256']
assert not (OUT/'diagnostic_resume.json').exists()
def launch(script,arguments,folder,timeout):
    folder.mkdir(exist_ok=True)
    with (folder/'harness_stdout.log').open('w',encoding='utf-8') as so,(folder/'harness_stderr.log').open('w',encoding='utf-8') as se:
        p=subprocess.Popen([sys.executable,'-u','-X','utf8',str(script)]+arguments,stdout=so,stderr=se,creationflags=subprocess.CREATE_NO_WINDOW)
        owned=psutil.Process(p.pid);identity=owned.create_time()
        (folder/'harness_process.json').write_text(json.dumps(dict(pid=p.pid,create_time=identity,arguments=arguments)))
        try:code=p.wait(timeout=timeout)
        except BaseException:
            if owned.is_running() and owned.create_time()==identity:
                tree=owned.children(recursive=True)+[owned]
                for proc in tree:
                    if proc.is_running():proc.terminate()
                psutil.wait_procs(tree,timeout=20)
            raise
    assert code==0,(str(script),code)
started=time.time()
try:
    launch(OUT/'run_heading_probe.py',[],OUT/'heading_response',350)
    args=read(cp.parent/'run_config.json')['arguments'];teacher=[]
    for key in ['termination_proof','standing_prior','standing_prior_sha256','locomotion_prior_certificate']:
        teacher+=['--'+key,str(args[key])]
    f=OUT/'gradient_smoke'
    launch(OUT/'capture_current_gradients.py',['--mode','train','--output',str(f),'--checkpoint',str(cp),
        '--num_envs','4096','--iterations','4','--learning_rate','.0001','--seed','97',
        '--headless','--device','cuda:0','--speed','3.5','--reserve_mb','300']+teacher,f,330)
    assert read(f/'result.json')['completed']
    proof=read(f/'full_load_proof.json')
    assert all(proof[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
    rows=[json.loads(v) for v in (f/'progress.jsonl').read_text().splitlines()]
    assert len(rows)==4 and all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
    (OUT/'diagnostic_completion.json').write_text(json.dumps(dict(completed=True,seconds=time.time()-started,
        full_state_equal=True,diagnostic_PPO_updates=4,diagnostic_actors_deployed=False,production_sources_changed=False),indent=2))
finally:
    launch(OUT/'resume_after_diagnosis.py',[],OUT/'resume_receipt',45)
print('Physical response and passive PPO audit complete; original training resumed.')
