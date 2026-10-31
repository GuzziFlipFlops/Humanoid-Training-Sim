"""Training-only stopped-command behavior regularization for residual PPO.

Inject the exact gradient of a bounded-sample mean-action MSE into PPO's mean
output backward pass. This avoids forking the installed RSL-RL update routine.
The frozen teacher never sends commands to the simulator. Inference has no
hook, no action blending and no access to the teacher.
"""
import torch

LEARNER_VERSION = 'qualified_prior_retention_history5_v3'


def validate_teacher(infos, contract, expected_hashes):
    """Accept the already-qualified legacy actor with exact file-byte binding.

    Its original metadata predates optional motion/actuator fields. Asset and
    observation contract remain mandatory; present extra hashes must match.
    The chain independently requalifies this same teacher under current assets.
    """
    assert infos['contract']==contract,'Standing teacher observation contract mismatch'
    assert infos['asset_sha256']==expected_hashes['asset_sha256'],'Standing teacher asset mismatch'
    for key,value in expected_hashes.items():
        if key in infos:assert infos[key]==value,f'Standing teacher {key} mismatch'
    return sorted(set(expected_hashes)-set(infos))


class StandingRetention:
    def __init__(self, actor, teacher, weight=.25, max_samples=256, command_index=None, command_range=None):
        self.actor=actor;self.teacher=teacher.eval().requires_grad_(False)
        self.weight=weight;self.max_samples=max_samples;self.command_index=command_index
        self.command_range=command_range
        if command_range is not None:assert .1<=command_range[0]<command_range[1]
        self._observations=None;self._error=None;self._samples=0;self._batches=0
        self._handles=[actor.register_forward_pre_hook(self._capture),
                       actor.mlp.register_forward_hook(self._regularize)]

    def _capture(self, module, args):
        self._observations=args[0] if module.training and torch.is_grad_enabled() else None

    def _regularize(self, module, inputs, mean):
        observations=self._observations;self._observations=None
        if observations is None or not mean.requires_grad:return
        raw=observations['policy']
        # Five 80-value frames put the latest command at326, not the old
        # four-frame index246. Derive it, so command changes cannot regularize
        # a moving sample merely because its previous frame was stopped.
        index=self.command_index if self.command_index is not None else raw.shape[-1]-80+6
        if self.command_range is None:
            selected=raw[:,index:index+3].norm(dim=1)<=.1
        else:
            speed=raw[:,index:index+2].norm(dim=1)
            selected=(speed>self.command_range[0])&(speed<=self.command_range[1]+1e-6)
        ids=selected.nonzero(as_tuple=False).flatten()
        if not ids.numel():return
        stride=max(1,(ids.numel()+self.max_samples-1)//self.max_samples)
        ids=ids[::stride][:self.max_samples]
        with torch.no_grad():
            target=self.teacher(observations[ids]).clamp(-3.,3.)
            difference=mean[ids].detach()-target
            derivative=2.*self.weight*difference/difference.numel()
            error=difference.square().sum()
        self._error=error if self._error is None else self._error+error
        self._samples+=difference.numel();self._batches+=1
        # d(weight * mean((actor_mean - frozen_teacher_mean)^2))/d(mean).
        # Other rows receive only their original PPO gradient.
        mean.register_hook(lambda gradient:gradient.index_add(0,ids,derivative))

    def read_audit(self):
        mse=float(self._error/self._samples) if self._samples else 0.
        result=dict(standing_prior_mse=mse,standing_prior_loss=self.weight*mse,
            weight=self.weight,batches=self._batches,max_teacher_samples_per_batch=self.max_samples,
            learner_version=LEARNER_VERSION,command_range=self.command_range,scope='Training gradients only; no teacher in inference or promotion')
        self._error=None;self._samples=0;self._batches=0
        return result

    def close(self):
        for handle in self._handles:handle.remove()
        self._observations=None


def _sha(path):
    import hashlib
    from pathlib import Path
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_qualification(certificate, checkpoint, signature, gate_version):
    """Bind teacher use to exact full nominal/pushed and retained skill gates."""
    import json
    from pathlib import Path
    assert certificate['schema']=='qualified_prior_retention_v1'
    assert Path(certificate['checkpoint']).resolve()==Path(checkpoint).resolve()
    assert _sha(checkpoint)==certificate['checkpoint_sha256']
    assert certificate['reward_signature']==signature
    assert certificate['gate_version']==gate_version
    flags=[]
    for receipt in certificate['gates']:
        assert _sha(receipt['result'])==receipt['result_sha256']
        assert _sha(receipt['config'])==receipt['config_sha256']
        cfg=json.loads(Path(receipt['config']).read_text())
        r=json.loads(Path(receipt['result']).read_text())
        assert Path(cfg['arguments']['checkpoint']).resolve()==Path(checkpoint).resolve()
        assert cfg['reward_version']==certificate['reward_version'] and cfg['gate_version']==gate_version
        assert cfg['arguments']['speed']==certificate['maximum_command_m_s']
        assert cfg['arguments']['duration']>=20 and r['completed'] and r['passed']
        assert all(c['passed'] and all(c['checks'].values()) and c['statistics']['trials']>=32 for c in r['cases'])
        expected=[0.,.15,.35,.65,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.]
        expected=[x for x in expected if x<=certificate['maximum_command_m_s']+1e-6]
        assert [c['statistics']['command_m_s'] for c in r['cases']]==expected
        flags.append(bool(r['perturb']))
    assert sorted(flags)==[False,True]
    return certificate['maximum_command_m_s']


def qualification_certificate(run,checkpoint,speed,signature,gate_version):
    import json
    from pathlib import Path
    found={}
    for p in sorted(Path(run).glob('*_gate_*/result.json')):
        r=json.loads(p.read_text());c=p.parent/'run_config.json';cfg=json.loads(c.read_text())
        if not r.get('completed') or not r.get('passed'):continue
        if Path(cfg['arguments']['checkpoint']).resolve()!=Path(checkpoint).resolve():continue
        if cfg['arguments']['speed']!=speed:continue
        found[bool(r['perturb'])]=dict(result=str(p),result_sha256=_sha(p),config=str(c),config_sha256=_sha(c))
    assert set(found)=={False,True},'Missing matching full teacher qualification'
    result=dict(schema='qualified_prior_retention_v1',checkpoint=str(checkpoint),checkpoint_sha256=_sha(checkpoint),
        reward_signature=signature,gate_version=gate_version,reward_version=cfg['reward_version'],
        maximum_command_m_s=speed,gates=[found[False],found[True]])
    validate_qualification(result,checkpoint,signature,gate_version)
    return result


def verify_retention_migration(prior,new_source,proof_path):
    import json
    from pathlib import Path
    prior=Path(prior);new_source=Path(new_source);proof_path=Path(proof_path).resolve()
    proof=json.loads(proof_path.read_text());state=json.loads((prior/'state.json').read_text())
    assert state['status']=='superseded_for_diagnostic_repair' and state['child_pid'] is None
    assert proof['schema']=='prior_command_retention_continuation_v1'
    assert Path(proof['prior_run']).resolve()==prior.resolve()
    assert _sha(state['checkpoint'])==proof['drained_checkpoint_sha256']
    assert _sha(proof['selected_checkpoint'])==proof['selected_checkpoint_sha256']
    assert Path(proof['selected_checkpoint']).resolve().is_relative_to(prior.resolve())
    def inventory(root):
        return {p.relative_to(root).as_posix():p for p in root.rglob('*') if p.is_file()
                and '__pycache__' not in p.parts and p.suffix!='.pyc' and 'usd' not in p.parts}
    old,new=inventory(prior/'source/physical_v29'),inventory(new_source)
    changed={k for k in old.keys()|new.keys() if k not in old or k not in new or old[k].read_bytes()!=new[k].read_bytes()}
    assert changed=={'training/standing_retention.py','training/worker_unitree.py'}
    assert set(proof['implementation_hashes'])==changed
    for k,h in proof['implementation_hashes'].items():assert _sha(new[k])==h
    report=proof_path.parent/proof['cpu_tests']['path']
    assert _sha(report)==proof['cpu_tests']['sha256'] and json.loads(report.read_text())['passed']
    certificate=proof['qualified_teacher']
    validate_qualification(certificate,certificate['checkpoint'],proof['reward_signature'],proof['gate_version'])
    state=dict(state);state['checkpoint']=proof['selected_checkpoint']
    state['preserved_drained_iterations']=state['total_training_iterations']
    state['total_training_iterations']=proof['selected_main_updates']
    return state
