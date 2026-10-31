"""CPU-only candidate measurement. Never imported by production training."""
import hashlib,json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
RUN=Path(json.loads((OUT/'diagnostic_pause.json').read_text())['run'])
mass=json.loads((RUN/'source/physical_v29/training/mass_estimate.json').read_text())['total_mass_kg']
SCALE=mass*9.81*.25

def swing_probability(phase,duty,half_width):
    """Exact box-convolution of a circular swing interval, not Von Mises."""
    p=np.remainder(np.asarray(phase),1.)
    if half_width==0:return (p>=duty).astype(float)
    assert 0<half_width<.25
    left=p-half_width;right=p+half_width;length=np.zeros_like(p,dtype=float)
    for k in [-1,0,1]:
        length+=np.maximum(0,np.minimum(right,1.+k)-np.maximum(left,duty+k))
    return np.clip(length/(2*half_width),0,1)

def quality(force,phase,half_width):
    leg=np.remainder(np.asarray(phase)[...,None]+np.array([0.,.5]),1.)
    prob=swing_probability(leg,.43,half_width)
    return np.exp(-(np.maximum(force,0)*prob).sum(-1)/SCALE),prob

def tests():
    checks=[]
    # Midpoint samples avoid representing the binary discontinuity as a
    # floating-point periodicity failure at exactly duty or wrap.
    phi=(np.arange(10000)+.5)/10000
    for w in [0,.0165,.033,.066]:
        prob=swing_probability(phi,.43,w)
        assert np.isfinite(prob).all() and (prob>=0).all() and (prob<=1).all()
        assert np.allclose(prob,swing_probability(phi+1,.43,w),atol=1e-12)
        assert abs(prob.mean()-.57)<.00011
        q,c=quality(np.zeros((len(phi),2)),phi,w);assert (q==1).all()
        qloaded,_=quality(np.full((len(phi),2),300.),phi,w)
        assert (qloaded<q).all()
        lower,_=quality(np.full((len(phi),2),150.),phi,w)
        assert (lower>qloaded).all()
        mid=.465;f=np.array([[0.,676.],[338.,338.],[676.,0.]])
        qmid,_=quality(f,np.full(3,mid),w)
        if w<=.033:assert np.allclose(qmid,qmid[0],atol=1e-12)
        # Exponential of the summed weighted force; no average of exponentials.
        checks.extend([f'bounded/periodic w={w}',f'phase-integral preserved w={w}',
            f'monotone foot force w={w}',f'zero-force bounded w={w}'])
    # When probabilities differ, load transfer can change the smoothed score.
    # Quantify this explicitly; do not assert binary double-swing invariance near edges.
    tests=dict(passed=True,checks=checks,count=len(checks),walking_standing_unchanged_by_design=True,
        no_runtime_change=True,limitation='Force term alone cannot reject hopping, sliding or early exit. Full gait, clearance, direction, costs, terminal handling and independent gates remain mandatory. Edge probabilities differ; old binary double-swing redistribution invariance need not hold at a smoothed boundary.')
    (OUT/'smoothed_phase_cpu_tests.json').write_text(json.dumps(tests,indent=2)+'\n')

def analyze(path,labels,outname):
    z=np.load(path);force=z['force'];phase=z['reward_phase'];pre=z['phase'];time=z['time'];alive=z['alive'];per=int(z['robots_per_candidate'])
    healthy=(z['term_upright']/3.5)*(z['term_torso_upright']/.75)*(z['term_height']/2.5)
    old,oldcoef=quality(force,phase,0)
    max_reconstruction=float(abs(.5*old*healthy-z['term_running_unload']).max())
    assert max_reconstruction<2e-6
    window=time>=5.;oldreturn=z['returns'];reports=[]
    for w in [.0165,.033,.066]:
        new,coef=quality(force,phase,w);delta=.5*(new-old)*healthy
        # No failures in these cohorts. First terminal masking is required otherwise.
        assert alive.all(),'Implement terminal-specific delta masking before analyzing a fallen cohort'
        adjusted=oldreturn+(delta*.02).sum(0)
        rows=[]
        for i,label in enumerate(labels):
            ids=slice(i*per,(i+1)*per);f=force[:,ids];c=coef[:,ids]
            transfer=np.stack([np.zeros(f.shape[:-1]),f.sum(-1)],-1)
            redistributed=np.exp(-(transfer*c).sum(-1)/SCALE)
            true=((z['clearance'][:,ids]>.001).all(-1)&(f<1).all(-1))
            both=oldcoef[:,ids].all(-1);edge=both&(c.min(-1)<.999)
            rows.append(dict(label=label,complete_return=float(adjusted[ids].mean()),
                complete_return_delta_vs_same_control=float(adjusted[ids].mean()-adjusted[:per].mean()),
                original_return_delta_vs_same_control=float(oldreturn[ids].mean()-oldreturn[:per].mean()),
                changed_term_return_delta=float((adjusted[ids]-oldreturn[ids]).mean()),
                flight_fraction=float(true[window].mean()),
                mean_unload_rate_before=float((.5*old[:,ids]*healthy[:,ids])[window].mean()),
                mean_unload_rate_after=float((.5*new[:,ids]*healthy[:,ids])[window].mean()),
                edge_redistribution_quality_change_max=float(abs(redistributed-new[:,ids])[edge].max()) if edge.any() else 0.,
                flight_reward_phase_halfcycle=[float(x) for x in np.quantile(np.remainder(phase[:,ids][true],.5),[0,.5,1])] if true.any() else None,
                flight_preaction_phase_halfcycle=[float(x) for x in np.quantile(np.remainder(pre[:,ids][true],.5),[0,.5,1])] if true.any() else None))
        reports.append(dict(half_width_cycles=w,full_boundary_width_ms=2*w/1.65*1000,rows=rows))
    result=dict(recorded_file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        candidate='Uniform circular phase smoothing of existing summed-force term only; unchanged .5 coefficient/quarter-body-weight scale',
        deployed=False,new_learning_tested=False,reconstruction_max_abs=max_reconstruction,
        reports=reports,primary_source='https://arxiv.org/html/2011.01387v2',
        adaptation='Paper uses expected Von Mises phase intervals and force/velocity composition. This diagnostic instead uses a box convolution with widths defined by 20ms action ticks; it is not the exact paper algorithm.',
        limitation='Fixed-state reward recomputation changes no dynamics and proves no learning improvement. Edge redistribution and complete-return tradeoffs must pass before any candidate is considered.')
    (OUT/outname).write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    tests()
    old=OUT.parent/'adaptive_check_20261005_1746'
    labels=json.loads((old/'validation_selection.json').read_text())['labels']
    analyze(old/'trajectory_validation/round_00.npz',labels,'old_smoothed_counterfactual.json')
    if (OUT/'timing_probe/round_00.npz').exists():
        labels=json.loads((OUT/'timing_selection.json').read_text())['labels']
        r=analyze(OUT/'timing_probe/round_00.npz',labels,'current_smoothed_counterfactual.json')
        print(json.dumps({'completed':True,'reconstruction_error':r['reconstruction_max_abs'],
            'smoothing_rows':r['reports']},indent=2))
