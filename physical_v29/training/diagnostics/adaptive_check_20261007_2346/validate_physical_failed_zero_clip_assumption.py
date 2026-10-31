"""Independent CPU checks of completed physical evidence and decision."""
import json,hashlib
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;F=OUT/'timing_response'
r=json.loads((F/'result.json').read_text());a=json.loads((OUT/'timing_analysis.json').read_text());p=json.loads((OUT/'diagnostic_pause.json').read_text())
assert r['completed'] and r['actor_loaded_exact'] and r['checkpoint_sha256']==p['checkpoint_sha256']
assert hashlib.sha256(Path(r['checkpoint']).read_bytes()).hexdigest()==r['checkpoint_sha256']
assert len(r['groups'])==12 and all(g['survival']==1 and g['resets']==0 for g in r['groups'])
assert r['maximum_command_difference']==0 and r['maximum_reward_term_sum_error']<1e-5
with np.load(F/'telemetry.npz') as z:
    initial=np.concatenate([z['initial_q'],z['initial_qd']],axis=1);per=int(z['robots_per_group']);names=z['groups'].tolist()
assert per==32;seeds=[]
for lo,seed in [(0,4191),(6,4211)]:
    reference=initial[lo*per:(lo+1)*per]
    assert len({x.tobytes() for x in reference})==32
    assert all(np.array_equal(reference,initial[i*per:(i+1)*per]) for i in range(lo,lo+6))
    seeds.append(dict(seed=seed,independent_initial_states=32,all_six_command_law_cohorts_paired=True))
assert not np.array_equal(initial[:32],initial[6*32:7*32])
motor=json.loads((F/'substep_motor_audit.json').read_text())
clips=[v for g in motor['commands'] for v in g['motors']]
assert all(v['speed_clipped_fraction']==0 for v in clips)
assert a['synthetic_fixtures_passed']==5 and max(a['parity'].values())==0
comparison=[]
for seed in [4191,4211]:
    cohorts={g['name'].split('_3.5_')[-1]:g for g in r['groups'] if g['name'].startswith(f'seed{seed}_3.5_')}
    for law in ['instant','lowpass120ms','cycle380ms']:
        base=cohorts['baseline'];g=cohorts[law];instant=cohorts['instant']
        comparison.append(dict(seed=seed,law=law,lateral_m_s=g['mean_absolute_world_lateral_velocity'],
            reward_rate_change_vs_baseline=g['mean_reward_rate']-base['mean_reward_rate'],
            lateral_change_vs_instant=g['mean_absolute_world_lateral_velocity']-instant['mean_absolute_world_lateral_velocity'],
            reward_rate_change_vs_instant=g['mean_reward_rate']-instant['mean_reward_rate']))
assert all(x['lateral_change_vs_instant']>0 and x['reward_rate_change_vs_instant']<0 for x in comparison if x['law']!='instant')
result=dict(passed=True,completed=True,checkpoint=r['checkpoint'],sha256=r['checkpoint_sha256'],independent_initial_states=seeds,
    normal_limits_gravity_free_base_truthful_feedback=True,all_384_survive=True,driving_speed_clips=0,
    comparisons=comparison,qualification=False,
    decision='Reject120ms velocity low-pass and380ms trailing-cycle course variants for immediate learning/deployment: both seeds retain more lateral drift and lower complete mean reward than instantaneous course. No matched64/128 learning was run or claimed. Not proof averaging can never help; original rejected learned course1 remains rejected.',
    endpoint_proven=False,production_sources_or_actor_changed=False,
    interpretation='Diagnostic instantaneous COM velocity statistics after5s, not gate pose-derived fixed-frame speed. Small32-state-per-seed initial perturbations are verified; this is not a full final command/nominal/pushed/retention qualification protocol.')
(OUT/'physical_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
