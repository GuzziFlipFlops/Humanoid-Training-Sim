"""Reuse the owned, saved-boundary comparison with a single declared update."""
from pathlib import Path
OUT=Path(__file__).resolve().parent;TRAIN=OUT.parents[1]
OLD=TRAIN/'diagnostics/adaptive_check_20261010_0149'
text=(OLD/'run_state_trial.py').read_text()
head=text[:text.index('ROOT.mkdir(exist_ok=False)')]
head=head.replace("ROOT=OUT/'matched_full_state'", "ROOT=OUT/'matched_command_advantage'")
head=head.replace("str(folder.relative_to(OUT))", "str(folder)")
head=head.replace("args=read(Path(checkpoint).parent/'run_config.json')['arguments'];result=[]", "args=read(Path(checkpoint).parent/'run_config.json')['arguments'];args['termination_proof']=str(OUT/'termination_teacher_update_proof.json');result=[]")
tail='''
ROOT.mkdir(exist_ok=False)
try:
    with (OUT/'pause.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-X','utf8',str(OUT/'prepare_and_pause.py')],check=True,stdout=log,stderr=subprocess.STDOUT)
    pause=read(OUT/'diagnostic_pause.json');WORK_START=time.monotonic()
    anchor=Path(pause['checkpoint'])
    common=['--headless','--device','cuda:0','--speed','3.5','--reserve_mb','300']
    manifest=read(OUT/'source_freeze.json');manifest.update(anchor_checkpoint=str(anchor),anchor_sha256=sha(anchor),
        pause=pause,work_limit_seconds=WORK_LIMIT,diagnostic_samples_excluded_from_main=True)
    (ROOT/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2)+'\\n')
    # Disabled wrapper must reproduce the actual original deterministic gate bitwise.
    replay=ROOT/'baseline_replay';evaluate(anchor,replay,901)
    import numpy as np
    originals=[p.parent for p in Path(pause['run']).glob('*_gate_*_nominal/result.json')
        if read(p).get('completed') and Path(read(p.parent/'run_config.json')['arguments']['checkpoint']).resolve()==anchor.resolve()]
    assert len(originals)==1
    with np.load(originals[0]/'telemetry.npz') as a,np.load(replay/'telemetry.npz') as b:
        assert set(a.files)==set(b.files)
        arrays={k:np.array_equal(a[k],b[k],equal_nan=True) if a[k].dtype.kind in 'fc' else np.array_equal(a[k],b[k]) for k in a.files}
    assert all(arrays.values()) and read(originals[0]/'evaluation.json')==read(replay/'evaluation.json')
    (ROOT/'baseline_parity.json').write_text(json.dumps(dict(passed=True,arrays=arrays,original=str(originals[0]),checkpoint_sha256=sha(anchor)),indent=2)+'\\n')
    for branch in ['control','candidate']:
        cp=anchor;flag=['--command_advantage'] if branch=='candidate' else []
        for label in ['smoke','reload']:
            folder=ROOT/f'{branch}_{label}'
            launch([str(OUT/'worker_entry.py'),'--mode','train','--output',str(folder),'--checkpoint',str(cp),
                '--num_envs','64','--iterations','4','--learning_rate','.0001','--seed','347']+teacher_args(anchor)+common+flag,folder,330)
            validate_train(folder,4,cp);validate_normalization(folder,branch)
            cp=folder/'candidate.pt'
    for seed in [347,359]:
        for branch in ['control','candidate']:
            checkpoint=anchor;flag=['--command_advantage'] if branch=='candidate' else []
            for updates in [64,128]:
                folder=ROOT/f'seed{seed}_{branch}_{updates:03d}_train'
                launch([str(OUT/'worker_entry.py'),'--mode','train','--output',str(folder),'--checkpoint',str(checkpoint),
                    '--num_envs','4096','--iterations','64','--learning_rate','.0001','--seed',str(seed)]+teacher_args(anchor)+common+flag,folder,1000)
                validate_train(folder,64,checkpoint);validate_normalization(folder,branch);checkpoint=folder/'candidate.pt'
                for label,evaluation_seed,push in [('nominal',901,False),('push',1301,True)]:
                    evaluate(checkpoint,ROOT/f'seed{seed}_{branch}_{updates:03d}_{label}',evaluation_seed,push)
                if updates==128:
                    for label,evaluation_seed,push in [('diverse_nominal',5011,False),('diverse_push',5039,True)]:
                        evaluate(checkpoint,ROOT/f'seed{seed}_{branch}_{updates:03d}_{label}',evaluation_seed,push,True)
    (ROOT/'completion.json').write_text(json.dumps(dict(completed=True,work_seconds=time.monotonic()-WORK_START,
        wall_seconds=time.time()-START,diagnostic_actor_deployed=False),indent=2)+'\\n')
except BaseException as error:
    (ROOT/'failure.json').write_text(json.dumps(dict(error=repr(error),traceback=traceback.format_exc(),wall_seconds=time.time()-START),indent=2)+'\\n')
    raise
finally:
    if (OUT/'diagnostic_pause.json').exists():
        with (OUT/'resume.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-X','utf8',str(OUT/'resume_after_diagnosis.py')],check=True,stdout=log,stderr=subprocess.STDOUT)
print('Versioned optimizer comparison finished; original learner resumed pending review.',flush=True)
'''
audit='''
def validate_normalization(folder,branch):
    import torch
    capture=torch.load(folder/'first_rollout_normalization.pt',weights_only=False,map_location='cpu')
    expected=capture['conditioned_advantage'] if branch=='candidate' else capture['old_global_normalized']
    assert torch.equal(capture['actual_advantage'],expected)
    assert torch.equal(capture['raw_advantage'],capture['returns']-capture['values'])
    rows=[json.loads(x) for x in (folder/'command_advantage.jsonl').read_text().splitlines()]
    assert all(x['applied']==(branch=='candidate') and x['original_GAE_returns_values_rewards_unchanged'] for x in rows)
    assert all(all(abs(g['conditioned_mean'])<1e-3 and (g['constant_or_singleton'] or abs(g['conditioned_std']-1)<1e-5) for g in x['groups']) for x in rows)
    assert all(sum(g['samples'] for g in x['groups'])==read(folder/'run_config.json')['arguments']['num_envs']*24 for x in rows)
    checkpoint=torch.load(folder/'candidate.pt',weights_only=False,map_location='cpu')
    assert checkpoint['infos']['command_advantage_applied']==(branch=='candidate')
    assert checkpoint['infos']['training_update_signature']==sha(Path(read(OUT/'source_freeze.json')['source'])/'training/command_advantage.py')
    (folder/'normalization_validation.json').write_text(json.dumps(dict(passed=True,branch=branch,updates=len(rows),
        raw_returns_values_unchanged=True,actual_normalizer_verified=True,saved_update_provenance=True),indent=2)+'\\n')
'''
(OUT/'run_comparison.py').write_text(head+audit+tail)
print('Prepared sequential controlled iteration with actual normalization/provenance validation.')
