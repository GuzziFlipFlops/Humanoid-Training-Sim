"""Sequential owned CUDA experiment with mandatory same-supervisor resume."""
import json,hashlib,subprocess,sys,time
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent;ROOT=OUT/'matched_shoulder'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def launch(argv,folder,timeout=500):
    folder.mkdir(parents=True,exist_ok=True)
    assert not any(any(Path(x).name in ['worker_unitree.py','shoulder_candidate.py','diverse_eval.py'] for x in p.info['cmdline'] or []) for p in psutil.process_iter(['cmdline']))
    with (folder/'stdout.log').open('w',encoding='utf-8') as so,(folder/'stderr.log').open('w',encoding='utf-8') as se:
        proc=subprocess.Popen([sys.executable,'-u','-X','utf8']+argv,stdout=so,stderr=se,creationflags=subprocess.CREATE_NO_WINDOW)
        ident=psutil.Process(proc.pid);created=ident.create_time()
        (folder/'process.json').write_text(json.dumps(dict(pid=proc.pid,create_time=created,argv=argv),indent=2))
        end=time.monotonic()+timeout
        try:
            while proc.poll() is None and not (folder/'result.json').exists():
                assert time.monotonic()<end,'Bounded diagnostic worker exceeded progress budget'
                time.sleep(2)
            if proc.poll() is None:
                assert read(folder/'result.json')['completed']
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    tree=ident.children(recursive=True)+[ident]
                    for p in tree:
                        if p.is_running():p.terminate()
                    psutil.wait_procs(tree,timeout=20)
                    (folder/'exit_cleanup.json').write_text(json.dumps(dict(completed_saved_result=True,owned_PID=ident.pid,only_Kit_shutdown_cleanup=True)))
            assert (proc.returncode==0 or (folder/'exit_cleanup.json').exists()) and read(folder/'result.json')['completed'],(folder.name,proc.returncode)
        except BaseException:
            if ident.is_running() and ident.create_time()==created:
                tree=ident.children(recursive=True)+[ident]
                for p in tree:
                    if p.is_running():p.terminate()
                psutil.wait_procs(tree,timeout=20)
            raise
    print(json.dumps(dict(completed=folder.name,wall_seconds=round(time.time()-START,1))),flush=True)
START=time.time()
subprocess.run([sys.executable,str(OUT/'prepare_and_pause.py')],check=True,stdout=(OUT/'pause.log').open('w'),stderr=subprocess.STDOUT)
pause=read(OUT/'diagnostic_pause.json');cp=pause['checkpoint'];TRAIN=OUT/'control_source/physical_v29/training'
args=read(Path(cp).parent/'run_config.json')['arguments'];teacher=[]
for k in ['termination_proof','standing_prior','standing_prior_sha256','locomotion_prior_certificate']:teacher+=['--'+k,str(args[k])]
common=['--headless','--device','cuda:0','--speed','3.5','--reserve_mb','300']
wrapper=OUT/'shoulder_candidate.py';ROOT.mkdir(exist_ok=True)
(ROOT/'experiment_manifest.json').write_text(json.dumps(dict(checkpoint=cp,sha256=pause['checkpoint_sha256'],
    exact_full_state=True,existing_head_upper_and_both_priors_retained=True,original_worker_physics_rewards_caps_unchanged=True,
    experimental_gradient_source_sha256=hashlib.sha256((OUT/'shoulder_mean_refinement.py').read_bytes()).hexdigest(),
    original_signature_does_not_certify_new_training_gradient=True,applies_only_two_final_shoulder_rows=True,
    no_shared_trunk_direct_derivative=True,mean_band_deg=5,phase_bins=8,weight=.025,max_new_to_actual_PPO_shoulder_norm=.1,
    no_imposed_swing_amplitude_or_phase_or_human_curve=True,training_seed=97,num_envs=4096,updates=[64,128],
    independent_initial_states_seeds=[2301,1901],diagnostic_actors_deployed=False),indent=2))
try:
    # Same actor full population inference: wrapper has no training hooks in eval.
    f=ROOT/'unchanged_actor_nominal'
    launch([str(wrapper),'--mode','eval','--output',str(f),'--checkpoint',cp,'--num_envs','320','--duration','20','--seed','901']+common,f)
    smoke=ROOT/'candidate_smoke'
    launch([str(wrapper),'--apply-shoulder-mean','--mode','train','--output',str(smoke),'--checkpoint',cp,'--num_envs','64','--iterations','4','--learning_rate','.0001','--seed','97']+teacher+common,smoke)
    f=ROOT/'candidate_reload'
    launch([str(wrapper),'--mode','eval','--output',str(f),'--checkpoint',str(smoke/'candidate.pt'),'--num_envs','64','--duration','4','--seed','97']+common,f)
    for branch in ['control','candidate']:
        checkpoint=cp
        for updates in [64,128]:
            f=ROOT/f'{branch}_{updates:03d}_train'
            launch([str(wrapper)]+(['--apply-shoulder-mean'] if branch=='candidate' else [])+['--mode','train','--output',str(f),'--checkpoint',checkpoint,'--num_envs','4096','--iterations','64','--learning_rate','.0001','--seed','97']+teacher+common,f)
            load=read(f/'full_load_proof.json');assert all(load[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
            rows=[json.loads(v) for v in (f/'progress.jsonl').read_text().splitlines()]
            assert len(rows)==64 and all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
            assert all(r['upper_refinement']['batches']==20 and r['upper_refinement']['experimental_shoulder_mean']['batches']==20 for r in rows)
            checkpoint=str(f/'candidate.pt')
            for label,seed,push in [('nominal',901,False),('push',1301,True)]:
                g=ROOT/f'{branch}_{updates:03d}_{label}'
                launch([str(wrapper),'--mode','eval','--output',str(g),'--checkpoint',checkpoint,'--num_envs','320','--duration','20','--seed',str(seed)]+common+(['--perturb'] if push else []),g)
            if updates==128:
                for label,seed,push in [('diverse_nominal',2301,False),('diverse_push',1901,True)]:
                    g=ROOT/f'{branch}_{updates:03d}_{label}'
                    launch([str(OUT/'diverse_eval.py'),'--mode','eval','--output',str(g),'--checkpoint',checkpoint,'--num_envs','320','--duration','20','--seed',str(seed)]+common+(['--perturb'] if push else []),g)
            assert time.time()-START<2600,'Overall bounded comparison budget exceeded'
    (ROOT/'completion.json').write_text(json.dumps(dict(completed=True,wall_seconds=time.time()-START,diagnostic_actors_deployed=False),indent=2))
finally:
    subprocess.run([sys.executable,str(OUT/'resume_after_diagnosis.py')],check=True,stdout=(OUT/'resume.log').open('w'),stderr=subprocess.STDOUT)
print('Matched diagnostic complete; same original full-state learner resumed.',flush=True)
