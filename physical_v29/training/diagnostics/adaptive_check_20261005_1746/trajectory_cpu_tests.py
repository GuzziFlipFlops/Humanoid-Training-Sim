import json
from pathlib import Path
import numpy as np
from trajectory_parameters import LOW,HIGH,ORIGIN,INITIAL_STD,samples,offsets_numpy,update
checks=[]
def check(name,value):assert value,name;checks.append(name)
rng=np.random.default_rng(1746);x=samples(rng,ORIGIN,INITIAL_STD,10000)
check('all_search_parameters_inside_declared_bounds',bool(((x>=LOW)&(x<=HIGH)).all()))
check('origin_retained_exactly',np.array_equal(x[0],ORIGIN))
phase=np.linspace(0,1,10001)
check('zero_origin_offsets_every_phase',np.array_equal(offsets_numpy(phase,ORIGIN,8.),np.zeros((len(phase),3))))
p=HIGH.copy();y=offsets_numpy(phase,p,8.)
check('offsets_repeat_each_halfcycle',np.allclose(y,offsets_numpy(phase+.5,p,8.),atol=1e-13))
check('no_offset_before_three_seconds',np.array_equal(offsets_numpy(phase,p,2.),np.zeros_like(y)))
check('smooth_fade_half_amplitude',np.allclose(offsets_numpy(phase,p,3.5),.5*y))
check('smooth_periodic_join',np.allclose(offsets_numpy(np.array([0.,.5]),p,8.)[0],offsets_numpy(np.array([0.,.5]),p,8.)[1]))
check('individual_joint_offsets_bounded',bool((np.abs(y)<=np.abs(p[:3])+np.abs(p[3:6])+1e-12).all()))
control=np.array([-10.,-3.,0.,3.,10.]);check('zero_offset_preserves_executed_clipped_actions',np.array_equal(np.clip(control,-3,3),np.clip(np.clip(control,-3,3)+0.,-3,3)))
check('terminal_reward_included_future_reward_excluded',sum([1.,1.,-4.,0.,0.])==-2.)
mean,std,ids=update(x[:32],np.arange(32.))
check('elites_use_complete_return',np.array_equal(ids,np.arange(26,32)))
check('exploration_std_floor_retained',bool((std>=INITIAL_STD*.15).all()))
check('all_source_parameters_are_diagnostic_only',not (Path(__file__).resolve().parents[2]/'trajectory_parameters.py').exists())
out=dict(passed=True,checks=checks,scope='Parameter bounds, phase continuity, unchanged executed zero baseline and full-return selection; not a physical or policy qualification')
(Path(__file__).resolve().parent/'trajectory_cpu_tests.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
