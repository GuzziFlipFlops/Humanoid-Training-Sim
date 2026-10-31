"""Diagnostic training-only per-robot cycle mean; no inference/control override.

Unlike the rejected ensemble-phase mean, complete truthful cycles are grouped
within each robot and episode. Weight, mean band and 10% incoming PPO shoulder
gradient cap are identical to that prior candidate. Only aggregation changes.
"""
from collections import deque
import math
import torch

WEIGHT=.025
BAND_ACTION=math.radians(5)/.25
MAX_PPO_FRACTION=.1
PHASE_BINS=8

class CycleCache:
    def __init__(self,max_robots=128,max_cycles=256):
        self.max_robots=max_robots;self.max_cycles=max_cycles
        self.ids=None;self.current={};self.completed=deque(maxlen=max_cycles);self.tick=0
        self.resets=0;self.switches=0;self.invalid=0;self.total_completed=0

    def record(self,raw,targets=None):
        assert raw.ndim==2 and raw.shape[1]==400
        self.tick+=1
        if self.ids is None:
            stride=max(1,math.ceil(len(raw)/self.max_robots));self.ids=list(range(0,len(raw),stride))[:self.max_robots]
        assert self.ids[-1]<len(raw)
        selected=raw[self.ids].detach().cpu().clone()
        target=None if targets is None else targets[self.ids].detach().cpu().clone()
        for i,env_id in enumerate(self.ids):
            obs=selected[i];frames=obs.reshape(5,80);last=frames[-1]
            x,y=float(last[6]),float(last[7]);phase_norm=float(last[78:80].square().sum())
            eligible=x>.1 and abs(y)<1e-6 and bool((frames[:,6]-x).abs().max()<1e-6) and bool(frames[:,7].abs().max()<1e-6) and .99<phase_norm<1.01
            if not eligible:
                if env_id in self.current:
                    self.current.pop(env_id,None)
                    self.completed=deque((c for c in self.completed if c['env_id']!=env_id),maxlen=self.max_cycles)
                self.invalid+=1;continue
            phase=(math.atan2(float(last[78]),float(last[79]))/(2*math.pi))%1
            state=self.current.get(env_id)
            if state is None or abs(state['command']-x)>1e-6:
                if state is not None:
                    self.switches+=1
                    self.completed=deque((c for c in self.completed if c['env_id']!=env_id),maxlen=self.max_cycles)
                state=dict(command=x,previous=phase,started=False,bins={},start_tick=self.tick)
                self.current[env_id]=state
            wrapped=phase<state['previous']-.5
            if wrapped:
                if state['started'] and len(state['bins'])==PHASE_BINS:
                    entries=[state['bins'][b] for b in range(PHASE_BINS)]
                    self.completed.append(dict(env_id=env_id,command=x,end_tick=self.tick,start_tick=state['start_tick'],
                        observations=torch.stack([e[0] for e in entries]),
                        physical_targets=None if target is None else torch.stack([e[1] for e in entries])))
                    self.total_completed+=1
                state.update(started=True,bins={},start_tick=self.tick)
            state['previous']=phase
            if state['started']:
                index=min(PHASE_BINS-1,int(phase*PHASE_BINS))
                if index not in state['bins']:
                    state['bins'][index]=(obs.clone(),None if target is None else target[i].clone())

    def reset(self,dones):
        dead=set(torch.as_tensor(dones).detach().cpu().flatten().nonzero().flatten().tolist())
        self.resets+=sum(i in dead for i in (self.ids or []))
        for i in dead:self.current.pop(i,None)
        self.completed=deque((c for c in self.completed if c['env_id'] not in dead),maxlen=self.max_cycles)

    def batch(self):
        # Keep at most two recent cycles per robot. A 200-tick limit exceeds two
        # slow cycles but prevents indefinite stale replay after switches.
        counts={};chosen=[]
        for c in reversed(self.completed):
            if self.tick-c['end_tick']>200:continue
            if c['env_id'] not in self.current:continue
            if abs(c['command']-self.current[c['env_id']]['command'])>1e-6:continue
            if counts.get(c['env_id'],0)>=2:continue
            chosen.append(c);counts[c['env_id']]=counts.get(c['env_id'],0)+1
        chosen.reverse()
        if not chosen:return None,[]
        return torch.stack([c['observations'] for c in chosen]),chosen

def cycle_loss(effective):
    assert effective.ndim==3 and effective.shape[1:]==(8,2)
    means=effective.mean(1)
    excess=(means.abs()-BAND_ACTION).clamp_min(0.)
    huber=torch.where(excess<=1.,.5*excess.square(),excess-.5)
    return WEIGHT*huber.mean(),means

class SequenceShoulderRefinement:
    def __init__(self,actor,names,apply=False,env_getter=None):
        self.actor=actor;self.ids=[names.index(n) for n in ['left_shoulder_pitch','right_shoulder_pitch']]
        self.layer=actor.mlp[-1];self.apply=apply;self.env_getter=env_getter
        self.cache=CycleCache();self.features=None;self.pending={};self.batches=0;self.stats={};self.first=None
        self.handles=[self.layer.register_forward_pre_hook(self.capture),actor.mlp.register_forward_hook(self.attach),
            self.layer.weight.register_hook(lambda g:self.parameter(g,'weight')),
            self.layer.bias.register_hook(lambda g:self.parameter(g,'bias'))]
        actor._sequence_shoulder_diagnostic=self

    def record(self,observations):
        env=self.env_getter() if self.env_getter is not None else None
        targets=None if env is None else env.targets
        self.cache.record(observations['policy'],targets)

    def capture(self,module,args):
        self.features=args[0].detach() if module.training and torch.is_grad_enabled() else None

    def attach(self,module,inputs,mean):
        minibatch_features=self.features;self.features=None
        if minibatch_features is None or not mean.requires_grad:return
        sequences,cycles=self.cache.batch()
        if sequences is None:return
        raw=sequences.flatten(0,1).to(mean.device)
        with torch.no_grad():
            normalized=self.actor.obs_normalizer(raw)
            # RSL-RL's MLP subclasses Sequential with a dimensional constructor;
            # slicing attempts to construct that subclass without its dimensions.
            features=normalized
            for hidden_layer in list(self.actor.mlp.children())[:-1]:
                features=hidden_layer(features)
            features=features.detach()
            predicted=torch.nn.functional.linear(features,self.layer.weight.detach(),self.layer.bias.detach())[:,self.ids]
        with torch.enable_grad():
            leaf=predicted.detach().requires_grad_(True)
            loss,means=cycle_loss(leaf.clamp(-3.,3.).reshape(-1,8,2))
            derivative=torch.autograd.grad(loss,leaf)[0].detach()
        extra_w=derivative.T@features;extra_b=derivative.sum(0)
        extra_norm=torch.sqrt(extra_w.square().sum()+extra_b.square().sum())
        def remember(gradient):
            pw=gradient[:,self.ids].T@minibatch_features;pb=gradient[:,self.ids].sum(0)
            pnorm=torch.sqrt(pw.square().sum()+pb.square().sum())
            scale=(MAX_PPO_FRACTION*pnorm/extra_norm.clamp_min(1e-12)).clamp(max=1.)
            if self.apply:
                for k,value in [('weight',extra_w*scale),('bias',extra_b*scale)]:
                    self.pending[k]=value if k not in self.pending else self.pending[k]+value
            self.batches+=1
            quantities=dict(loss=loss.detach(),proposed_gradient_norm=extra_norm,actual_PPO_shoulder_norm=pnorm,
                bounded_gradient_norm=extra_norm*scale,scale=scale,complete_cycles=mean.new_tensor(len(cycles)),
                cycle_mean_abs_deg=means.detach().abs().mean()*.25*180/math.pi)
            for k,value in quantities.items():self.stats[k]=value if k not in self.stats else self.stats[k]+value
            if self.first is None:
                self.first=dict(commands=[c['command'] for c in cycles],env_ids=[c['env_id'] for c in cycles],
                    cycle_means_deg=(means.detach()*.25*180/math.pi).cpu().tolist(),
                    sample_clip_fraction=float((leaf.detach().abs()>3).float().mean()))
            return gradient
        mean.register_hook(remember)

    def parameter(self,gradient,key):
        extra=self.pending.pop(key,None)
        if extra is None:return gradient
        out=gradient.clone();out[self.ids]+=extra;return out

    def read_audit(self):
        assert not self.pending
        result=dict(version='experimental_individual_cycle_shoulder_mean_v2',applied=self.apply,batches=self.batches,
            weight=WEIGHT,mean_band_deg=5,max_PPO_shoulder_gradient_fraction=MAX_PPO_FRACTION,
            max_robots=self.cache.max_robots,max_cycles=self.cache.max_cycles,phase_bins=8,
            completed_cycles_total=self.cache.total_completed,resets_segmented=self.cache.resets,command_switches=self.cache.switches,
            truthful_collection=True,shared_trunk_direct_gradient=False,output_rows=['left_shoulder_pitch','right_shoulder_pitch'],
            statistics={k:float(v/max(1,self.batches)) for k,v in self.stats.items()},first_actual_batch=self.first)
        self.batches=0;self.stats={};self.first=None
        return result

    def close(self):
        for h in self.handles:h.remove()
        self.pending={};self.features=None
        if hasattr(self.actor,'_sequence_shoulder_diagnostic'):del self.actor._sequence_shoulder_diagnostic
