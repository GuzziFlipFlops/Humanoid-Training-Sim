"""Preserve failed teacher-signature smoke; retry explicit teacher migration."""
from pathlib import Path
base=Path(__file__).with_name('run_guard_validation.py')
exec(compile(base.read_text().split("run('101_current_run_350_gate_nominal'")[0],str(base),'exec'))
cert=OUT/'fresh_teacher_certificate.json'
standing=state['promotions'][0]['checkpoint']
folder,r=run('106_full_state_ppo_smoke',3.5,checkpoint,mode='train',count=64,seed=71,
 extra=['--standing_prior',standing,'--standing_prior_sha256',hashlib.sha256(Path(standing).read_bytes()).hexdigest(),
        '--locomotion_prior_certificate',str(cert)])
proof=read(folder/'full_load_proof.json')
assert all(proof[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
rows=[json.loads(x) for x in (folder/'progress.jsonl').read_text().splitlines()]
assert len(rows)==4 and all(x['ppo']['pre_update_kl']==0 and x['behavior']['action_command_difference']==0 for x in rows)
run('107_full_state_checkpoint_reload',3.5,Path(r['checkpoint']),count=64,duration=4)
(OUT/'physical_full_load_smoke.json').write_text(json.dumps(dict(passed=True,checkpoint=str(checkpoint),
 checkpoint_sha256=pause['checkpoint_sha256'],actor_critic_Adam_exploration_exact=True,
 updates=4,zero_preupdate_KL_and_command_mismatch=True,smoke_actor_deployed=False,
 fresh_teacher_certificate=str(cert),teacher_explicit_source_migration=True,
 scope='Same full-state learner retained; corrected termination only. No diagnostic actor is deployed.'),indent=2)+'\n')
print('Physical/PPO/full-state/reload smoke passed',flush=True)
