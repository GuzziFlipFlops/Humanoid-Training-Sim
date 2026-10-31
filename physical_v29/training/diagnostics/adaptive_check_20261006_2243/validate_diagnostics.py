import json,hashlib,torch
from pathlib import Path
out=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
p=read(out/'diagnostic_pause.json');source=Path(p['run'])/'source/physical_v29';copy=out/'control_source/physical_v29'
protected=read(out/'protected_source_copy.json')['protected_files_identical']
checks=[]
for f in protected:assert (source/f).read_bytes()==(copy/f).read_bytes()
checks.append('53 authored original physical/CAD/reward/PPO sources are byte-identical in diagnostic copy')
load=read(out/'gradient_smoke/full_load_proof.json');assert all(load[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
assert load['source_sha256']==p['checkpoint_sha256'];checks.append('Exact original full actor/critic/Adam/exploration checkpoint loaded')
rows=[json.loads(x) for x in (out/'gradient_smoke/progress.jsonl').read_text().splitlines()]
assert len(rows)==4 and all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
checks.append('Four actual diagnostic PPO updates with zero pre-update KL/command difference')
saved=torch.load(out/'gradient_smoke/candidate.pt',map_location='cpu',weights_only=False)
assert all(torch.isfinite(t).all() for key in ['actor_state_dict','critic_state_dict'] for t in saved[key].values());checks.append('Saved diagnostic full checkpoint loads on CPU with finite actor and critic')
assert saved['optimizer_state_dict']['state'];checks.append('Saved Adam/exploration present; diagnostic actor is discarded from production')
a=read(out/'arm_response/result.json');assert a['completed'] and a['actor_loaded_exact'] and a['maximum_command_difference']==0
checks.append('Physical192-robot20s probe loads the exact original actor and uses truthful commands')
assert all(g['survival']==1 and g['resets']==0 for g in a['groups']);checks.append('All six normal-limit physical cohorts survive without resets')
assert a['maximum_reward_term_sum_error']<1e-5;checks.append('Complete physical reward term sum reconstructs')
h=read(out/'actual_head_axis_parity.json');assert h['maximum_rotation_error_deg']<.1;checks.append('URDF FK agrees with actual Isaac head orientation; optical axis not certified')
l=read(out/'prior_landing_event_analysis.json');assert l['unchanged_verifier_reproduces_every_saved_50hz_hop_event'];checks.append('Every saved representative hop event replays under unchanged verifier')
g=read(out/'current_gradient_analysis.json');assert len(g['batches'])==4 and all(b['moving_surrogate_derivative_cosine']>.999 for b in g['batches'])
checks.append('Four captured moving PPO score derivatives closely reproduce; CPU clipping boundary and failed stopped-hook subtraction explicitly retained')
r=read(out/'diagnostic_resume.json');assert r['same_supervisor'] and not r['training_source_changed'] and not r['diagnostic_actors_deployed'];checks.append('Same original supervisor resumed after bounded pause, no deployment')
(out/'diagnostic_validation.json').write_text(json.dumps(dict(passed=True,checks=checks,scope='Diagnostic evidence validation, not a new policy or gait qualification',unresolved='Old stopped-hook CPU subtraction does not reproduce capture; moving scores are separately validated. No production fault inferred.'),indent=2)+'\n')
print(len(checks),'diagnostic checks passed; no policy qualification inferred')
