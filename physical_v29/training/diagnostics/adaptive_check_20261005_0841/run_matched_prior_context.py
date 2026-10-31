"""Sequential, source-pinned matched PPO/control experiment; actors discarded."""
import json,time,subprocess,sys,hashlib
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
pause=json.loads((OUT/'diagnostic_pause.json').read_text());run=Path(pause['run'])
assert pause['worker_drained']
supervisor=psutil.Process(pause['supervisor_pid'])
assert supervisor.create_time()==pause['supervisor_create_time'] and supervisor.status()==psutil.STATUS_STOPPED
checkpoint=Path(pause['checkpoint'])
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==pause['checkpoint_sha256']
prior=json.loads((checkpoint.parent/'standing_retention.json').read_text())
certificate=checkpoint.parent/'qualified_prior_certificate.json';assert certificate.is_file()
sources={'control':run/'source/physical_v29','candidate':OUT/'prior_context_source/physical_v29'}
def assert_idle():
    for p in psutil.process_iter(['cmdline']):
        args=p.info['cmdline'] or []
        assert not any(Path(a).name in ['worker_unitree.py','physical_prior_causal.py','physical_phase_causal.py'] for a in args),'Other actual Isaac worker'
def worker(source,folder,mode,cp,**kw):
    assert_idle();folder=OUT/folder;assert not folder.exists(),'Preserve completed and failed attempts'
    folder.mkdir();command=[sys.executable,'-u','-X','utf8',str(source/'training/worker_unitree.py'),
        '--headless','--device','cuda:0','--mode',mode,'--output',str(folder),'--checkpoint',str(cp),
        '--speed','1.5','--reserve_mb','300']
    for k,v in kw.items():
        command+=['--'+k]+([] if v is True else [str(v)])
    if mode=='train':command+=['--standing_prior',prior['teacher'],'--standing_prior_sha256',prior['sha256'],
                              '--locomotion_prior_certificate',str(certificate)]
    (folder/'launch.json').write_text(json.dumps(command,indent=2))
    print(json.dumps({'start':folder.name,'time':time.time()}),flush=True)
    with (folder/'stdout.log').open('w') as out,(folder/'stderr.log').open('w') as err:
        process=subprocess.run(command,stdout=out,stderr=err)
    assert process.returncode==0,(folder,process.returncode)
    assert (folder/'result.json').is_file(),folder
    result=json.loads((folder/'result.json').read_text());assert result['completed'],folder
    print(json.dumps({'completed':folder.name,'time':time.time()}),flush=True)
    return folder,result
records={}
folder,result=worker(sources['candidate'],'matched_prior_context','train',checkpoint,num_envs=4096,iterations=64,learning_rate=.0001,seed=71)
records['candidate']={'train':str(folder),'result':result,'evaluations':[]}
for perturb,seed in [(False,901),(True,1301)]:
    ef,er=worker(sources['candidate'],'matched_prior_context'+('_push' if perturb else '_nominal'),'eval',folder/'candidate.pt',
        **dict(num_envs=192,duration=20,seed=seed,**({'perturb':True} if perturb else {})))
    records['candidate']['evaluations'].append(str(ef))
reload,reload_result=worker(sources['candidate'],'prior_context_full_reload','train',folder/'candidate.pt',num_envs=64,iterations=4,learning_rate=.0001,seed=71)
records['reload']=str(reload);records['qualifies_locomotion']=False;records['diagnostic_actors_deployed']=False
(OUT/'matched_prior_context_completed.json').write_text(json.dumps(records,indent=2)+'\n')
