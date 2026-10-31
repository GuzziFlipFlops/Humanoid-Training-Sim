"""Pinned training-only curriculum migration; physics and old gates stay protected."""
import hashlib,json
from pathlib import Path
EVALUATION_SPEEDS = (0.,.15,.35,.65,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.)
ENV_BASE_SHA = 'c594ec70821cf1068c2ca14ac788a66ce4472b2ef64774d7a09a7f113da6c76b'
ENV_IMPORT = b'from command_curriculum import sample_training_speeds\n'
ENV_OVERRIDE = b" # Training-only exact-command rehearsal; evaluation delegates unchanged.\n def _sample_commands(self,ids):\n  if self.cfg.evaluation:return super()._sample_commands(ids)\n  upper=self.cfg.stage_speed;self.commands[ids]=0.\n  if upper>0:\n   r=torch.rand(len(ids),device=self.device);v=torch.rand(len(ids),device=self.device)\n   self.commands[ids,0]=sample_training_speeds(upper,ids,r,v)\n  if hasattr(self,'step_tracker'):self.step_tracker.reset(ids)\n  self.command_ticks[ids]=torch.randint(200,400,(len(ids),),device=self.device)\n\n"
WORKER_IMPORT = b'from curriculum_provenance import training_reward_signature\r\nfrom command_curriculum import TRAINING_COMMAND_VERSION,command_coverage\r\n'
OLD_SIGNATURE = b" return hashlib.sha256(b''.join((HERE/n).read_bytes() for n in ['stage_rewards.py','support_geometry.py','unitree_reward_port.py','environment_unitree.py','heading_control.py','environment.py','unitree_vendor/sources.json'])).hexdigest()\n"
NEW_SIGNATURE = b' return training_reward_signature(HERE)\n'
WORKER_METADATA = b"   details['training_command_version']=TRAINING_COMMAND_VERSION\r\n"
WORKER_COVERAGE = b"    state['command_coverage']=command_coverage(envraw.commands,args.speed)\r\n"
EVAL_IMPORT = b'from curriculum_provenance import evaluation_speeds\n'
OLD_SPEEDS = b' speeds=sorted(set([0.,speed]+[v for v in [.15,.35,.65,1.,1.5,2.,2.5] if v<=speed+1e-5]))\n'
NEW_SPEEDS = b' speeds=evaluation_speeds(speed)\n'
GOAL_CHECK = b"  if speed>=5.:checks['actual_goal_speed']=stats['reset_frame_forward_speed_m_s']>=4.95 # 1% measurement tolerance; a 5m/s command alone never qualifies\n"

def evaluation_speeds(speed):
    return sorted(set([0.,speed]+[v for v in EVALUATION_SPEEDS if v<=speed+1e-5]))

def _bytes(text,data):
    return text if isinstance(text,bytes) else text.encode()

def remove_environment_rehearsal(data):
    for text in (ENV_IMPORT,ENV_OVERRIDE):
        token=_bytes(text,data);assert data.count(token)==1
        data=data.replace(token,b'')
    assert hashlib.sha256(data).hexdigest()==ENV_BASE_SHA, 'Non-curriculum environment changed'
    return data

def training_reward_signature(here):
    # Preserve the reward/observation/action signature only after reconstructing
    # the exact old environment bytes. The added sampler is training-only and
    # has its own version and frozen hash, not a new inference/reward interface.
    data=[]
    for name in ['stage_rewards.py','support_geometry.py','unitree_reward_port.py','environment_unitree.py','heading_control.py','environment.py','unitree_vendor/sources.json']:
        content=(Path(here)/name).read_bytes()
        if name=='environment_unitree.py':content=remove_environment_rehearsal(content)
        data.append(content)
    return hashlib.sha256(b''.join(data)).hexdigest()

def verify_curriculum_migration(prior,new_source,proof_path):
    prior=Path(prior);new_source=Path(new_source);proof_path=Path(proof_path).resolve()
    proof=json.loads(proof_path.read_text());previous=json.loads((prior/'state.json').read_text())
    assert previous['status']=='superseded_for_diagnostic_repair' and previous.get('child_pid') is None
    assert proof['schema']=='exact_command_rehearsal_v1' and not proof['qualifies_locomotion']
    assert Path(proof['prior_run']).resolve()==prior.resolve()
    assert hashlib.sha256(Path(previous['checkpoint']).read_bytes()).hexdigest()==proof['checkpoint_sha256']
    old_source=prior/'source/physical_v29'
    def files(root):
        return {p.relative_to(root).as_posix():p for p in root.rglob('*') if p.is_file() and p.suffix!='.pyc'
            and p.relative_to(root).parts[:3]!=('training','assets','usd')}
    old,new=files(old_source),files(new_source)
    allowed={'training/environment_unitree.py','training/worker_unitree.py','training/evaluation.py',
             'training/command_curriculum.py','training/curriculum_provenance.py'}
    difference={n for n in old.keys()|new.keys() if n not in old or n not in new or old[n].read_bytes()!=new[n].read_bytes()}
    assert difference==allowed,('Unexpected runtime modification',difference)
    assert set(proof['implementation_hashes'])==allowed
    for name,digest in proof['implementation_hashes'].items():assert hashlib.sha256(new[name].read_bytes()).hexdigest()==digest,name
    assert remove_environment_rehearsal(new['training/environment_unitree.py'].read_bytes())==old['training/environment_unitree.py'].read_bytes()
    worker=new['training/worker_unitree.py'].read_bytes()
    for a,b in [(WORKER_IMPORT,''),(NEW_SIGNATURE,OLD_SIGNATURE),(WORKER_METADATA,''),(WORKER_COVERAGE,'')]:
        a=_bytes(a,worker);b=_bytes(b,worker);assert worker.count(a)==1;worker=worker.replace(a,b)
    assert worker==old['training/worker_unitree.py'].read_bytes(),'PPO/load/action logic changed'
    evaluation=new['training/evaluation.py'].read_bytes()
    for a,b in [(EVAL_IMPORT,''),(NEW_SPEEDS,OLD_SPEEDS),(GOAL_CHECK,'')]:
        a=_bytes(a,evaluation);b=_bytes(b,evaluation);assert evaluation.count(a)==1;evaluation=evaluation.replace(a,b)
    assert evaluation==old['training/evaluation.py'].read_bytes(),'Prior independent classifier changed'
    test_path=(proof_path.parent/proof['cpu_test']['path']).resolve()
    assert test_path.is_relative_to(proof_path.parent)
    assert hashlib.sha256(test_path.read_bytes()).hexdigest()==proof['cpu_test']['sha256']
    test=json.loads(test_path.read_text());assert test['passed'] and test['signature_identical'] and test['evaluation_sampler_bitwise_equal']
    assert test['all_prior_speeds_unchanged'] and test['protected_source_negative_checks']>=8
    return previous
