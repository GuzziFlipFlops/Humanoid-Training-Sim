"""CPU-only exploration/score-function diagnostic; no production hook."""
import json,math
from pathlib import Path
import numpy as np
from scipy.special import ndtr,log_ndtr
from scipy.integrate import quad
import torch
OUT=Path(__file__).resolve().parent
pause=json.loads((OUT/'diagnostic_pause.json').read_text())
torch.set_num_threads(2)
saved=torch.load(pause['checkpoint'],map_location='cpu',weights_only=False)
stdkeys=[k for k in saved['actor_state_dict'] if k.endswith('log_std_param')]
assert len(stdkeys)==1
std=saved['actor_state_dict'][stdkeys[0]].exp().numpy().astype(float)

def tail_scores(mu,sigma):
    lo=(-3.-mu)/sigma;hi=(3.-mu)/sigma
    pdf=lambda z:np.exp(-.5*z*z-.5*np.log(2*np.pi))
    # log tails avoid underflow in the inverse Mills ratio.
    millslo=np.exp(-.5*lo*lo-.5*np.log(2*np.pi)-log_ndtr(lo))
    millshi=np.exp(-.5*hi*hi-.5*np.log(2*np.pi)-log_ndtr(-hi))
    return lo,hi,-millslo/sigma,millshi/sigma,-lo*millslo,hi*millshi

checks=[]
for mu in [-4.,0.,4.]:
    sigma=.7;lo,hi,lomu,himu,lost,hist=tail_scores(mu,sigma)
    def phi(z):return math.exp(-z*z/2)/math.sqrt(2*math.pi)
    # The total score integrates to zero for a constant return, both parameters.
    interior_mu=quad(lambda z:z/sigma*phi(z),lo,hi)[0]
    interior_std=quad(lambda z:(z*z-1)*phi(z),lo,hi)[0]
    assert abs(interior_mu+ndtr(lo)*lomu+ndtr(-hi)*himu)<1e-10
    assert abs(interior_std+ndtr(lo)*lost+ndtr(-hi)*hist)<1e-10
    # Test a clipped-action objective; derivative is compared with numeric integration.
    def objective(m,s):
        a=(-3-m)/s;b=(3-m)/s
        return math.tanh(-3)*ndtr(a)+quad(lambda z:math.tanh(m+s*z)*phi(z),a,b)[0]+math.tanh(3)*ndtr(-b)
    gmu=math.tanh(-3)*ndtr(lo)*lomu+quad(lambda z:math.tanh(mu+sigma*z)*(z/sigma)*phi(z),lo,hi)[0]+math.tanh(3)*ndtr(-hi)*himu
    gst=math.tanh(-3)*ndtr(lo)*lost+quad(lambda z:math.tanh(mu+sigma*z)*(z*z-1)*phi(z),lo,hi)[0]+math.tanh(3)*ndtr(-hi)*hist
    eps=1e-5
    assert abs(gmu-(objective(mu+eps,sigma)-objective(mu-eps,sigma))/(2*eps))<1e-6
    assert abs(gst-(objective(mu,sigma*math.exp(eps))-objective(mu,sigma*math.exp(-eps)))/(2*eps))<1e-6
    checks.extend([f'constant-return score expectation mu={mu}',f'bounded-action objective derivatives mu={mu}'])

z=np.load(OUT/'timing_probe/round_00.npz');per=int(z['robots_per_candidate']);window=z['time']>=5.
means=z['raw_action'][window,:per].astype(float);names=z['joint_names'].tolist()
lo,hi,lomu,himu,lost,hist=tail_scores(means,std)
outside=ndtr(lo)+ndtr(-hi)
# Exact score second moment from tails + the Gaussian interior moments.
phi=lambda v:np.exp(-v*v/2)/np.sqrt(2*np.pi)
interior_second=(ndtr(hi)-ndtr(lo)-hi*phi(hi)+lo*phi(lo))/(std**2)
reduced=(interior_second+ndtr(lo)*lomu*lomu+ndtr(-hi)*himu*himu).clip(0,1/(std**2))
fraction=1-reduced*(std**2)
assert np.isfinite(outside).all() and np.isfinite(fraction).all()
rows=[dict(joint=name,trained_std=float(std[i]),mean_outside_action_clamp_fraction=float((abs(means[...,i])>3).mean()),
    predicted_Gaussian_outside_probability=float(outside[...,i].mean()),
    constant_return_mean_score_second_moment_reduction_fraction=float(fraction[...,i].mean())) for i,name in enumerate(names)]
record=dict(checkpoint=pause['checkpoint'],sha256=pause['checkpoint_sha256'],cpu_tests_passed=True,checks=checks,
    current_actor_truthful_running_states=means.shape[0]*means.shape[1],mean_raw_output_clamp_fraction=float((abs(means)>3).mean()),
    Gaussian_predicted_outside_fraction=float(outside.mean()),joints=rows,
    deployed=False,actual_PPO_gradients_measured=False,actual_advantage_weighted_variance_measured=False,
    source='https://proceedings.mlr.press/v80/fujita18a.html',
    limitation='This computes analytic diagonal-Gaussian +/-3 tail score moments and synthetic bounded-objective checks only. It does not prove action clipping causes grounded locomotion or that a CAPG/PPO continuation improves flight. State-dependent soft limits, slew/CAD constraints, entropy, PPO ratios and real advantages still require validation. Earlier teacher-clipping-equivalence candidate is different and remains rejected. No native limit or policy distribution is changed.')
(OUT/'clipped_score_audit.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ['cpu_tests_passed','current_actor_truthful_running_states','mean_raw_output_clamp_fraction','Gaussian_predicted_outside_fraction']},indent=2))
