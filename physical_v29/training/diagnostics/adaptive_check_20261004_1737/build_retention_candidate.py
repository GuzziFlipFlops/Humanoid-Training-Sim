"""Stage a training-only prior-command repair, without editing any live source."""
from pathlib import Path
import json,shutil
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
pause=json.loads((OUT/'diagnostic_pause.json').read_text())
OLD=Path(pause['run'])/'source/physical_v29'
NEW=OUT/'validation_source/physical_v29'
shutil.copytree(OLD,NEW,ignore=shutil.ignore_patterns('__pycache__','usd'),dirs_exist_ok=True)
supervisor=(HERE/'supervisor_unitree.py').read_text()
assert (HERE/'supervisor_unitree.py').read_bytes()==(Path(pause['run'])/'supervisor_unitree.py').read_bytes()
source=(OLD/'training/standing_retention.py').read_text()
source=source.replace("LEARNER_VERSION = 'standing_retention_history5_v2'", "LEARNER_VERSION = 'qualified_prior_retention_history5_v3'")
source=source.replace('command_index=None):', 'command_index=None, command_range=None):')
source=source.replace('self._observations=None;self._error=None;', 'self.command_range=command_range\n        if command_range is not None:assert .1<=command_range[0]<command_range[1]\n        self._observations=None;self._error=None;',1)
source=source.replace("ids=(raw[:,index:index+3].norm(dim=1)<=.1).nonzero(as_tuple=False).flatten()", "if self.command_range is None:\n            selected=raw[:,index:index+3].norm(dim=1)<=.1\n        else:\n            speed=raw[:,index:index+2].norm(dim=1)\n            selected=(speed>self.command_range[0])&(speed<=self.command_range[1]+1e-6)\n        ids=selected.nonzero(as_tuple=False).flatten()")
source=source.replace("learner_version=LEARNER_VERSION,scope=", "learner_version=LEARNER_VERSION,command_range=self.command_range,scope=")
source+='''

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
'''
(NEW/'training/standing_retention.py').write_text(source,encoding='utf-8',newline='\n')
worker=(OLD/'training/worker_unitree.py').read_text()
worker=worker.replace("parser.add_argument('--standing_prior_sha256',default='')", "parser.add_argument('--standing_prior_sha256',default='')\nparser.add_argument('--locomotion_prior_certificate',default='')")
worker=worker.replace('LEARNER_VERSION,validate_teacher','LEARNER_VERSION,validate_teacher,validate_qualification')
worker=worker.replace(" assert not args.standing_prior or", " assert not args.locomotion_prior_certificate or args.mode=='train','Locomotion prior is training-only'\n assert not args.standing_prior or",1)
worker=worker.replace("   assert actor_same,'Warm start changed actor tensors'", '''   assert actor_same,'Warm start changed actor tensors'
   if not args.actor_warm_start and same_reward:
    critic_same=all(torch.equal(v,runner.alg.critic.state_dict()[k]) for k,v in saved['critic_state_dict'].items())
    assert critic_same,'Full load changed critic tensors'
    optimizer_same=None
    if args.mode=='train':
     loaded=runner.alg.optimizer.state_dict();original=saved['optimizer_state_dict']
     optimizer_same=loaded['param_groups']==original['param_groups'] and loaded['state'].keys()==original['state'].keys()
     optimizer_same=optimizer_same and all(torch.equal(v.detach().cpu(),loaded['state'][k][field].detach().cpu()) if torch.is_tensor(v) else v==loaded['state'][k][field] for k,state in original['state'].items() for field,v in state.items())
     assert optimizer_same,'Full load changed Adam state'
    atomic_json(out/'full_load_proof.json',dict(actor_equal=True,critic_equal=critic_same,optimizer_equal=optimizer_same,
      source_sha256=sha(args.checkpoint),exploration_equal=True,learning_rate_restoration_follows=True))''')
worker=worker.replace('   retention=None','   retention=None;locomotion_retention=None',1)
token='   # Continuous heartbeat and GPU reserve checks at each PPO update.'
worker=worker.replace(token,'''   if args.locomotion_prior_certificate:
    assert args.speed>=1.5,'Prior locomotion regularization is reserved for running'
    certificate=json.loads(Path(args.locomotion_prior_certificate).read_text())
    maximum=validate_qualification(certificate,certificate['checkpoint'],reward_signature(),GATE_VERSION)
    assert .1<maximum<args.speed,'Teacher may cover only qualified prior commands'
    teacher=copy.deepcopy(runner.alg.actor)
    teacher_saved=torch.load(certificate['checkpoint'],weights_only=False,map_location='cpu')
    assert teacher_saved['infos']['reward_signature']==reward_signature(),'Teacher reward/runtime differs'
    validate_teacher(teacher_saved['infos'],CONTRACT,
     {key:sha(path) for key,path in [('asset_sha256',HERE/'assets/robot_training.urdf'),('motion_sha256',HERE/'assets/motion_reference.npz'),('actuator_sha256',HERE.parent/'assets/actuator_limits.json')]})
    teacher.load_state_dict(teacher_saved['actor_state_dict']);del teacher_saved
    locomotion_retention=StandingRetention(runner.alg.actor,teacher,command_range=(.1,maximum))
    atomic_json(out/'locomotion_retention.json',dict(certificate=certificate,weight=.25,max_samples=256,
     learner_version=LEARNER_VERSION,training_only=True,inference_teacher=False))
''' + token)
worker=worker.replace("    state['command_coverage']=", "    state['locomotion_retention']=locomotion_retention.read_audit() if locomotion_retention else None\n    state['command_coverage']=")
worker=worker.replace("   details['training_command_version']=", "   details['locomotion_prior_certificate']=args.locomotion_prior_certificate or None\n   details['training_command_version']=")
(NEW/'training/worker_unitree.py').write_text(worker,encoding='utf-8',newline='\n')
supervisor=supervisor.replace("def verify_resource_migration(prior,new_source,optimization_proof='',curriculum_proof=''):","def verify_resource_migration(prior,new_source,optimization_proof='',curriculum_proof='',retention_proof=''):")
supervisor=supervisor.replace(' if curriculum_proof:\n'," if retention_proof:\n  assert not optimization_proof and not curriculum_proof\n  from standing_retention import verify_retention_migration\n  return verify_retention_migration(prior,new_source,retention_proof)\n if curriculum_proof:\n",1)
supervisor=supervisor.replace("optimization_proof='',curriculum_proof=''):","optimization_proof='',curriculum_proof='',retention_proof=''):")
supervisor=supervisor.replace('  self.curriculum_proof=curriculum_proof',"  self.curriculum_proof=curriculum_proof\n  self.retention_proof=retention_proof\n  assert not retention_proof or (migrate_run and not curriculum_proof and not optimization_proof)")
supervisor=supervisor.replace('  self.retention_proof=retention_proof', "  self.retention_proof=retention_proof\n  if retention_proof:\n   proof=json.loads(Path(retention_proof).read_text())\n   assert hashlib.sha256((HERE/'supervisor_unitree.py').read_bytes()).hexdigest()==proof['supervisor_sha256']")
supervisor=supervisor.replace('self.optimization_proof,self.curriculum_proof)', 'self.optimization_proof,self.curriculum_proof,self.retention_proof)')
supervisor=supervisor.replace('if self.optimization_proof or self.curriculum_proof:', 'if self.optimization_proof or self.curriculum_proof or self.retention_proof:')
supervisor=supervisor.replace("curriculum_proof=curriculum_proof or None))", "curriculum_proof=curriculum_proof or None,retention_proof=retention_proof or None))")
token="  if mode=='eval' and label=='checkpoint_reload':"
supervisor=supervisor.replace(token,'''  if mode=='train' and speed>=1.5:
   qualified=[p for p in self.state['promotions'] if .1<p['speed']<speed]
   if qualified:
    from standing_retention import qualification_certificate
    from curriculum_provenance import training_reward_signature
    import ast
    GATE_VERSION=next(ast.literal_eval(n.value) for n in ast.parse((HERE/'evaluation.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GATE_VERSION' for t in n.targets))
    prior=max(qualified,key=lambda p:p['speed'])
    prior_run=Path(prior['checkpoint']).parent.parent
    receipt=qualification_certificate(prior_run,prior['checkpoint'],prior['speed'],training_reward_signature(HERE),GATE_VERSION)
    certificate=folder/'qualified_prior_certificate.json';atomic(certificate,receipt)
    command+=['--locomotion_prior_certificate',str(certificate)]
''' + token,1)
token="   self.state['promotions']=previous['promotions'][:]"
# Only the migration branch: a training-only repair rechecks its trusted baseline.
start=supervisor.index('  if self.migrate_run:',supervisor.index('  trusted=None;stage_start=0'))
pos=supervisor.index(token,start)
supervisor=supervisor[:pos]+'''   if self.retention_proof:
    baseline=previous['promotions'][-1]
    reports=self.gates(baseline['speed'],baseline['checkpoint'],'retention_teacher_requalification')
    assert reports and all(r['passed'] for r in reports),'Qualified teacher failed fresh retained skill verification'
    for p in previous['promotions']:p['requalification']='fresh_matching_prior_retention_teacher'
''' + supervisor[pos:]
supervisor=supervisor.replace("'resource maintenance; '","'qualified prior-command gradient repair; ' if self.retention_proof else 'resource maintenance; '")
supervisor=supervisor.replace('physics_reward_and_PPO_unchanged=True,prior_speed_gates_unchanged=True', 'physics_reward_and_PPO_algorithm_config_unchanged=True,training_gradient_repair=bool(self.retention_proof),retention_proof=self.retention_proof or None,prior_speed_gates_unchanged=True')
supervisor=supervisor.replace(" p=argparse.ArgumentParser();", " p=argparse.ArgumentParser();p.add_argument('--retention_proof',default='');",1)
supervisor=supervisor.replace('args.optimization_proof,args.curriculum_proof)', 'args.optimization_proof,args.curriculum_proof,args.retention_proof)')
(OUT/'candidate_supervisor_unitree.py').write_text(supervisor,encoding='utf-8',newline='\n')
for p in [NEW/'training/worker_unitree.py',NEW/'training/standing_retention.py',OUT/'candidate_supervisor_unitree.py']:
 compile(p.read_text(),str(p),'exec')
print('Staged two training files and the supervisor; rewards, inference, actions, physics and gates unchanged.')
