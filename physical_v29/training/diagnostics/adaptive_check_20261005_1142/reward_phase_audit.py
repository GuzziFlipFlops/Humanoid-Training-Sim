"""Counterfactual v8 reward checks and phase plots from synchronized physics."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.set_num_threads(1)
OUT=Path(__file__).resolve().parent
HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
pause=read(OUT/'diagnostic_pause.json');run=Path(pause['run']);frozen=run/'source/physical_v29'
sys.path[:0]=[str(frozen/'training'),str(frozen)]
from stage_rewards import running_unload_quality, TOTAL_WEIGHT_N
z=np.load(OUT/'physical_response/telemetry.npz',allow_pickle=False)
data={k:z[k] for k in z.files};post=data['time']>=5.
names=list(data['joint_names']);per=int(data['robots_per_group'])
tests=[]
def check(name,condition,detail=None):
    tests.append(dict(name=name,passed=bool(condition),detail=detail))
    assert condition,name

force=torch.from_numpy(data['force'][post].reshape(-1,2))
phase=torch.from_numpy(data['reward_phase'][post].reshape(-1))
commands=torch.from_numpy(data['action_command'][post].reshape(-1,3))
running=commands[:,:2].norm(dim=1)>=1.5
duty=torch.where(running,.43,.55)
expected=torch.remainder(phase[:,None]+torch.tensor([0.,.5]),1.)>=duty[:,None]
healthy=torch.from_numpy(data['healthy'][post].reshape(-1))
actual=torch.from_numpy(data['term_running_unload'][post].reshape(-1))
pred=.5*running_unload_quality(force,expected)*healthy*running
err=float((pred-actual).abs().max())
check('Summed-load deployed reward reproduces saved running term',err<2e-6,err)
cl=torch.from_numpy(data['clearance'][post].reshape(-1,2))
geometric=(cl>.001).all(1)&(force.abs()<1.).all(1)
flight=torch.from_numpy(data['term_running_flight'][post].reshape(-1))
check('No flight credit without unloaded elevated physical soles',not bool((flight[~geometric]!=0).any()))
check('Walking and stopped commands have no running unload term',bool((actual[~running]==0).all()))
check('Observed command and action-time command match',bool(np.array_equal(data['observed_command'],data['action_command'])))
sum_terms=sum(data[k] for k in data if k.startswith('term_'))
sum_error=float(np.abs(sum_terms-data['reward_rate']).max())
check('Saved full objective reconstructs reward rate',sum_error<1e-4,sum_error)

loads=torch.tensor([600.,338.,180.,85.,18.,0.])
both=torch.ones((len(loads),2),dtype=torch.bool)
scores=running_unload_quality(torch.stack([loads,loads*0],1),both)
check('Progressive simultaneous unloading gives strictly increasing credit',bool((scores[1:]>scores[:-1]).all()))
check('Quality stays in [0,1]',bool(((scores>=0)&(scores<=1)).all()))
check('Zero force gets full quality',float(scores[-1])==1.)
for fraction in [0.,.25,.5,.75,1.]:
    redistributed=torch.tensor([[TOTAL_WEIGHT_N*fraction,TOTAL_WEIGHT_N*(1-fraction)]])
    score=running_unload_quality(redistributed,torch.ones((1,2),dtype=torch.bool))
    check(f'Double-swing load redistribution fraction {fraction} gives no extra credit',abs(float(score)-float(np.exp(-4.)))<1e-7)
single=torch.tensor([[True,False]])
check('Single swing ignores stance-foot load',bool(torch.equal(running_unload_quality(torch.tensor([[25.,0.]]),single),running_unload_quality(torch.tensor([[25.,500.]]),single))))
check('Neither foot intended to swing gives zero quality',float(running_unload_quality(torch.tensor([[0.,0.]]),torch.zeros((1,2),dtype=torch.bool)))==0.)
check('Partial unloading is credited before crossing binary flight threshold',bool((scores[:-1]>0).all()))

baseline=read(OUT/'physical_objective_analysis.json')['groups'][0]
audit=dict(checkpoint=pause['checkpoint'],tests=tests,all_passed=all(t['passed'] for t in tests),
           partial_force_curve_N=loads.tolist(),unload_quality=scores.tolist(),
           actual_double_swing_load_N=baseline['mean_total_force_in_double_swing_N'],
           actual_double_swing_running_unload_rate=baseline['actual_running_unload_in_double_swing'],
           conclusions=['No implementation discrepancy found in reconstructed v8 unloading or flight credit.',
                        'Continuous unloading signal is monotone at fixed state. This does not establish favorable full trajectory return.',
                        'Intended phase, target constraints, timing and costs require closed-loop comparison before a reward change.'],
           source_lead='https://arxiv.org/html/2011.01387v2',
           scope='CPU fixed-state reward counterexamples plus saved normal-limit physical trace. No deployed source or weight change, no policy qualification.')
(OUT/'reward_phase_audit.json').write_text(json.dumps(audit,indent=2)+'\n')

selected=[0,3,8]
colors=['#235789','#eb8c23','#2b8a63']
edges=np.linspace(0,1,41);centers=(edges[:-1]+edges[1:])/2
fig,axes=plt.subplots(3,2,figsize=(12,10),sharex=True)
def binned(values,phase):
    return [float(values[(phase>=a)&(phase<b)].mean()) for a,b in zip(edges[:-1],edges[1:])]
for k,color in zip(selected,colors):
    ids=slice(k*per,(k+1)*per);p=data['reward_phase'][post,ids]
    label=str(data['groups'][k])
    series=[data['force'][post,ids].sum(-1),data['clearance'][post,ids].min(-1)*1000,
            data['term_running_unload'][post,ids],data['reward_rate'][post,ids],
            data['target'][post,ids,names.index('left_knee_pitch')],data['q'][post,ids,names.index('left_knee_pitch')]]
    for ax,v in zip(axes.flat,series):ax.plot(centers,binned(v,p),label=label,color=color)
titles=['Total normal foot force (N)','Lower of the two sole clearances (mm)',
        'Actual v8 unloading reward rate','Complete reward rate',
        'Left knee target (rad)','Left knee measured position (rad)']
for ax,title in zip(axes.flat,titles):
    ax.set_title(title);ax.grid(alpha=.2)
    for start in [.43,.93]:ax.axvspan(start,start+.07,color='#c5d7e8',alpha=.4)
axes[0,0].legend(fontsize=8)
for ax in axes[-1]:ax.set_xlabel('Reward-time gait phase (cycle)')
fig.suptitle('R36 checkpoint 050: normal-limit interventions on truthful neutral-head states\nShaded intervals: both feet intended to swing; no cohort achieves physical flight',fontsize=13)
fig.tight_layout(rect=(0,0,1,.94))
fig.savefig(OUT/'phase_objective.png',dpi=150)
fig.savefig(OUT/'phase_objective.pdf')
print(json.dumps(dict(checks=len(tests),all_passed=audit['all_passed'],double_swing_unload=audit['actual_double_swing_running_unload_rate']),indent=2))
