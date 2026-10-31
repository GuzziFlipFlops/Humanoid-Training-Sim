"""Preserve the failed CPU-import attempt; stage a separate bounded follow-up."""
import hashlib,json,shutil
from pathlib import Path
OUT=Path(__file__).resolve().parent
SECOND=OUT/'secondary_analysis';SECOND.mkdir(exist_ok=False)
TRAIN=OUT.parents[1]
names=['prepare_and_pause.py','resume_after_diagnosis.py','neutral_teacher_probe.py','diagnostic_teacher_rows.py',
       'diverse_eval.py','capture_current_gradients.py','analyze_current_gradients.py','action_projection.py']
records=[]
for name in names:
    source=OUT/name;text=source.read_text()
    if name in ['prepare_and_pause.py','resume_after_diagnosis.py','neutral_teacher_probe.py']:
        text=text.replace('HERE=OUT.parents[1]',f'HERE=Path({str(TRAIN)!r})')
    (SECOND/name).write_text(text)
    records.append(dict(path=name,original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        staged_sha256=hashlib.sha256((SECOND/name).read_bytes()).hexdigest()))
for name in ['neutral_teacher_certificate.json','teacher_diagnosis_preparation.json']:
    shutil.copy2(OUT/name,SECOND/name)
original=(OUT/'run_teacher_diagnosis.py').read_text()
header=original.split('def certify(checkpoint):')[0]
body='''
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
        diagnostic_PPO_updates_discarded=4,diagnostic_actors_deployed=False,composites_deployed=False),indent=2)+'\\n')
except BaseException as error:
    (OUT/'teacher_diagnosis_failure.json').write_text(json.dumps(dict(error=repr(error),traceback=traceback.format_exc(),wall_seconds=time.time()-START),indent=2)+'\\n')
    raise
finally:
    if (OUT/'diagnostic_pause.json').exists():
        with (OUT/'resume.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-X','utf8',str(OUT/'resume_after_diagnosis.py')],check=True,stdout=log,stderr=subprocess.STDOUT)
print('Fresh neutral-teacher compatibility diagnosis complete; exact original learner resumed.',flush=True)
'''
header=header.replace('WORK_LIMIT=2200','WORK_LIMIT=1600')
(SECOND/'run_followup.py').write_text(header+body)
(SECOND/'staging_receipt.json').write_text(json.dumps(dict(files=records,first_failed_attempt_preserved=True,
    parent_certificate=str(OUT/'neutral_teacher_certificate.json'),only_diagnostic_scripts_staged=True),indent=2)+'\n')
compile((SECOND/'run_followup.py').read_text(),str(SECOND/'run_followup.py'),'exec')
print(str(SECOND/'run_followup.py'))
