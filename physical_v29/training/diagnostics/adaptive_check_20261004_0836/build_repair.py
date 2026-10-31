"""Apply narrowly reversible command-curriculum changes outside the live frozen run."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = ROOT/'physical_v29/training'
OUT = Path(__file__).resolve().parent
BACKUP = OUT/'canonical_before'
BACKUP.mkdir(exist_ok=True)
pointer = json.loads((HERE/'ACTIVE_RUN.json').read_text())
prior = Path(pointer['run'])
assert prior.name == 'locomotion_20261004_r30_exact_graph_execution'

IMPORT_ENV = 'from command_curriculum import sample_training_speeds\n'
METHOD_ENV = ''' # Training-only exact-command rehearsal; evaluation delegates unchanged.
 def _sample_commands(self,ids):
  if self.cfg.evaluation:return super()._sample_commands(ids)
  upper=self.cfg.stage_speed;self.commands[ids]=0.
  if upper>0:
   r=torch.rand(len(ids),device=self.device);v=torch.rand(len(ids),device=self.device)
   self.commands[ids,0]=sample_training_speeds(upper,ids,r,v)
  if hasattr(self,'step_tracker'):self.step_tracker.reset(ids)
  self.command_ticks[ids]=torch.randint(200,400,(len(ids),),device=self.device)

'''
IMPORT_WORKER = 'from curriculum_provenance import training_reward_signature\nfrom command_curriculum import TRAINING_COMMAND_VERSION,command_coverage\n'
OLD_SIGNATURE = " return hashlib.sha256(b''.join((HERE/n).read_bytes() for n in ['stage_rewards.py','support_geometry.py','unitree_reward_port.py','environment_unitree.py','heading_control.py','environment.py','unitree_vendor/sources.json'])).hexdigest()\n"
NEW_SIGNATURE = ' return training_reward_signature(HERE)\n'
WORKER_METADATA = "   details['training_command_version']=TRAINING_COMMAND_VERSION\n"
WORKER_COVERAGE = "    state['command_coverage']=command_coverage(envraw.commands,args.speed)\n"
IMPORT_EVAL = 'from curriculum_provenance import evaluation_speeds\n'
OLD_SPEEDS = ' speeds=sorted(set([0.,speed]+[v for v in [.15,.35,.65,1.,1.5,2.,2.5] if v<=speed+1e-5]))\n'
NEW_SPEEDS = ' speeds=evaluation_speeds(speed)\n'
GOAL_CHECK = "  if speed>=5.:checks['actual_goal_speed']=stats['reset_frame_forward_speed_m_s']>=4.95 # 1% measurement tolerance; a 5m/s command alone never qualifies\n"

def change(name, replacements):
    path=HERE/name;current=path.read_bytes()
    old=(BACKUP/name).read_bytes() if (BACKUP/name).exists() else current
    if not (BACKUP/name).exists():(BACKUP/name).write_bytes(old)
    data=old
    for a,b in replacements:
        candidates=[(a.encode(),b.encode()),(a.replace('\n','\r\n').encode(),b.replace('\n','\r\n').encode())]
        found=[(a,b) for a,b in candidates if data.count(a)==1]
        assert found,(name,a[:120])
        a,b=found[0]
        data=data.replace(a,b)
    assert current in (old,data),'Unexpected edit since preserved backup'
    path.write_bytes(data)
    return old

env_old=change('environment_unitree.py',[
    ('from types import SimpleNamespace\n','from types import SimpleNamespace\n'+IMPORT_ENV),
    (' def configure_fixed_commands(self,speeds,heading):\n',METHOD_ENV+' def configure_fixed_commands(self,speeds,heading):\n')])
worker_old=change('worker_unitree.py',[
    ('from status_io import atomic_json\n','from status_io import atomic_json\n'+IMPORT_WORKER),
    (OLD_SIGNATURE,NEW_SIGNATURE),
    ('   details.update(infos or {})\n',WORKER_METADATA+'   details.update(infos or {})\n'),
    ("    atomic_json(out/'progress.json',state)\n",WORKER_COVERAGE+"    atomic_json(out/'progress.json',state)\n")])
eval_old=change('evaluation.py',[
    ('import json,math\n','import json,math\n'+IMPORT_EVAL),
    (OLD_SPEEDS,NEW_SPEEDS),
    ("  checks.update(direction_checks(speed,stats))\n",GOAL_CHECK+"  checks.update(direction_checks(speed,stats))\n")])

def line(data, text):
    found=[l for l in data.splitlines(keepends=True) if l.decode().rstrip('\r\n')==text.rstrip('\r\n')]
    assert len(found)==1,(text,found)
    return found[0]

env_new=(HERE/'environment_unitree.py').read_bytes()
env_override=env_new[env_new.index(b' # Training-only exact-command rehearsal;'):env_new.index(b' def configure_fixed_commands')]
worker_new=(HERE/'worker_unitree.py').read_bytes()
eval_new=(HERE/'evaluation.py').read_bytes()
constants={
 'ENV_BASE_SHA':hashlib.sha256(env_old).hexdigest(),
 'ENV_IMPORT':line(env_new,IMPORT_ENV),'ENV_OVERRIDE':env_override,
 'WORKER_IMPORT':b''.join(line(worker_new,l) for l in IMPORT_WORKER.splitlines()),
 'OLD_SIGNATURE':line(worker_old,OLD_SIGNATURE),'NEW_SIGNATURE':line(worker_new,NEW_SIGNATURE),
 'WORKER_METADATA':line(worker_new,WORKER_METADATA),'WORKER_COVERAGE':line(worker_new,WORKER_COVERAGE),
 'EVAL_IMPORT':line(eval_new,IMPORT_EVAL),'OLD_SPEEDS':line(eval_old,OLD_SPEEDS),
 'NEW_SPEEDS':line(eval_new,NEW_SPEEDS),'GOAL_CHECK':line(eval_new,GOAL_CHECK),
}
header='"""Pinned training-only curriculum migration; physics and old gates stay protected."""\n'
header+='import hashlib,json\nfrom pathlib import Path\n'
header+='EVALUATION_SPEEDS = (0.,.15,.35,.65,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.)\n'
header+='\n'.join(k+' = '+repr(v) for k,v in constants.items())+'\n'
body='''
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
'''
(HERE/'curriculum_provenance.py').write_text(header+body,encoding='utf-8')

supervisor=HERE/'supervisor_unitree.py';data=supervisor.read_text()
(BACKUP/'supervisor_unitree.py').write_bytes(supervisor.read_bytes())
def replace(a,b,count=1):
    global data
    assert data.count(a)==count,(a[:100],data.count(a))
    data=data.replace(a,b)
replace("('run_250',2.5)]","('run_250',2.5),('run_300',3.),('run_350',3.5),('run_400',4.),('run_450',4.5),('run_500',5.)]")
replace("def verify_resource_migration(prior,new_source,optimization_proof=''):","def verify_resource_migration(prior,new_source,optimization_proof='',curriculum_proof=''):")
replace(' import ast\n previous=json.loads'," if curriculum_proof:\n  assert not optimization_proof\n  from curriculum_provenance import verify_curriculum_migration\n  return verify_curriculum_migration(prior,new_source,curriculum_proof)\n import ast\n previous=json.loads")
replace("learning_rate=.001,optimization_proof=''):","learning_rate=.001,optimization_proof='',curriculum_proof=''):")
replace('  self.optimization_proof=optimization_proof\n',"  self.optimization_proof=optimization_proof\n  self.curriculum_proof=curriculum_proof\n  assert not curriculum_proof or (migrate_run and not optimization_proof),'Curriculum proof requires its own full checkpoint continuation'\n")
replace("'mass_estimate.json','status_io.py']:","'mass_estimate.json','status_io.py','command_curriculum.py','curriculum_provenance.py']:")
replace("'assets/motion_reference.npz']}","'assets/motion_reference.npz','command_curriculum.py','curriculum_provenance.py']}")
replace('self.optimization_proof)', 'self.optimization_proof,self.curriculum_proof)',count=3)
replace('   if self.optimization_proof:\n','   if self.optimization_proof or self.curriculum_proof:\n')
replace('optimization_proof=optimization_proof or None))','optimization_proof=optimization_proof or None,curriculum_proof=curriculum_proof or None))')
replace("  atomic(self.root/'resource_migration.json',dict(source=str(self.migrate_run),checkpoint=checkpoint,physics_reward_evaluator_and_PPO_unchanged=True))", "  atomic(self.root/'resource_migration.json',dict(source=str(self.migrate_run),checkpoint=checkpoint,physics_reward_and_PPO_unchanged=True,prior_speed_gates_unchanged=True,training_commands_changed=bool(self.curriculum_proof),curriculum_proof=self.curriculum_proof or None))")
replace("initial_checkpoint_scope='Resource-only migration; same actor/critic/Adam, physics/reward/PPO bytes verified; all current and retention gates re-run'", "initial_checkpoint_scope='Audited full actor/critic/Adam continuation; '+('training-only exact-command rehearsal, original reward/observation/action signature and prior gates preserved; ' if self.curriculum_proof else 'resource maintenance; ')+'all current and retention gates re-run'")
replace("p.add_argument('--optimization_proof',default='')", "p.add_argument('--optimization_proof',default='');p.add_argument('--curriculum_proof',default='')")
replace('args.learning_rate,args.optimization_proof)', 'args.learning_rate,args.optimization_proof,args.curriculum_proof)')
supervisor.write_text(data,encoding='utf-8')
print(json.dumps({'prior':str(prior),'canonical_repair_prepared':True,'live_frozen_source_changed':False},indent=2))
