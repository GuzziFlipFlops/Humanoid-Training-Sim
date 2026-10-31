"""Sequential actual-physics gates and exact full-state PPO/load smoke."""
import json,hashlib,subprocess,sys,time
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
pause=read(OUT/'diagnostic_pause.json');RUN=Path(pause['run']);TRAIN=OUT/'candidate_source/physical_v29/training'
checkpoint=Path(pause['checkpoint']);state=read(OUT/'state_before_diagnosis.json')
assert state['stage']=='run_350' and hashlib.sha256(checkpoint.read_bytes()).hexdigest()==pause['checkpoint_sha256']
supervisor=psutil.Process(pause['supervisor_pid'])
assert supervisor.create_time()==pause['supervisor_create_time'] and supervisor.status()==psutil.STATUS_STOPPED
def ensure_free():
 for p in psutil.process_iter(['cmdline']):
  assert not any(Path(x).name in ['worker_unitree.py','physical_speed_probe.py'] for x in p.info['cmdline'] or []),'Another actual simulator is active'
def run(name,speed,cp,mode='eval',count=320,seed=901,perturb=False,duration=20,extra=None):
 ensure_free();folder=OUT/name;assert not folder.exists();folder.mkdir()
 cmd=[sys.executable,'-u','-X','utf8',str(TRAIN/'worker_unitree.py'),'--headless','--device','cuda:0',
  '--output',str(folder),'--mode',mode,'--speed',str(speed),'--num_envs',str(count),'--seed',str(seed),'--reserve_mb','300','--checkpoint',str(cp),
  '--termination_proof',str(OUT/'termination_load_proof.json')]
 if mode=='eval':cmd+=['--duration',str(duration)]+(['--perturb'] if perturb else [])
 else:cmd+=['--iterations','4','--learning_rate','0.0001']
 if extra:cmd+=extra
 (folder/'launch.json').write_text(json.dumps(cmd,indent=2)+'\n');start=time.time()
 print(json.dumps({'starting':name,'speed':speed,'mode':mode}),flush=True)
 with (folder/'stdout.log').open('w') as out,(folder/'stderr.log').open('w') as err:
  p=subprocess.Popen(cmd,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
  (folder/'owned_pid.json').write_text(json.dumps({'pid':p.pid,'create_time':psutil.Process(p.pid).create_time(),'started':start}))
  deadline=time.monotonic()+240
  while p.poll() is None:
   if (folder/'failure.json').exists() or time.monotonic()>deadline:
    owned=psutil.Process(p.pid);identity=read(folder/'owned_pid.json')
    assert owned.create_time()==identity['create_time']
    cleanup=[owned]+owned.children(recursive=True)
    (folder/'failed_worker_cleanup.json').write_text(json.dumps({'reason':'Recorded failure or bounded timeout','owned_pids':[x.pid for x in cleanup]}))
    for item in reversed(cleanup):
     if item.is_running():item.terminate()
    break
   time.sleep(1)
  rc=p.wait(timeout=30)
 (folder/'wrapper_result.json').write_text(json.dumps({'returncode':rc,'seconds':time.time()-start},indent=2)+'\n')
 assert rc==0,(name,rc)
 r=read(folder/'result.json');assert r['completed'],name
 print(json.dumps({'completed':name,'passed':r.get('passed'),'seconds':time.time()-start}),flush=True)
 return folder,r

run('101_current_run_350_gate_nominal',3.5,checkpoint)
run('102_current_run_350_gate_push',3.5,checkpoint,seed=1301,perturb=True)
teacher=Path(state['promotions'][-1]['checkpoint']);assert state['promotions'][-1]['speed']==3.
for label,seed,push in [('nominal',901,False),('push',1301,True)]:
 folder,r=run('103_teacher_run_300_gate_'+label,3.,teacher,count=288,seed=seed,perturb=push)
 assert r['passed'],'Trusted teacher needs every fresh matching prior gate'
sys.path.insert(0,str(TRAIN))
from standing_retention import qualification_certificate
from curriculum_provenance import training_reward_signature
GATE_VERSION=read(OUT/'103_teacher_run_300_gate_nominal/result.json')['gate_version']
certificate=qualification_certificate(OUT,teacher,3.,training_reward_signature(TRAIN),GATE_VERSION)
cert=OUT/'fresh_teacher_certificate.json';cert.write_text(json.dumps(certificate,indent=2)+'\n')
standing=state['promotions'][0]['checkpoint']
folder,r=run('104_full_state_ppo_smoke',3.5,checkpoint,mode='train',count=64,seed=71,
 extra=['--standing_prior',standing,'--standing_prior_sha256',hashlib.sha256(Path(standing).read_bytes()).hexdigest(),
        '--locomotion_prior_certificate',str(cert)])
proof=read(folder/'full_load_proof.json')
assert all(proof[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
rows=[json.loads(x) for x in (folder/'progress.jsonl').read_text().splitlines()]
assert len(rows)==4 and all(x['ppo']['pre_update_kl']==0 and x['behavior']['action_command_difference']==0 for x in rows)
run('105_full_state_checkpoint_reload',3.5,Path(r['checkpoint']),count=64,duration=4)
(OUT/'physical_full_load_smoke.json').write_text(json.dumps(dict(passed=True,checkpoint=str(checkpoint),
 checkpoint_sha256=pause['checkpoint_sha256'],actor_critic_Adam_exploration_exact=True,
 updates=4,zero_preupdate_KL_and_command_mismatch=True,smoke_actor_deployed=False,
 fresh_teacher_certificate=str(cert),scope='Same full-state learner retained; corrected termination only. No diagnostic actor is deployed.'),indent=2)+'\n')
print('All sequential physics/full-state/reload checks completed',flush=True)
