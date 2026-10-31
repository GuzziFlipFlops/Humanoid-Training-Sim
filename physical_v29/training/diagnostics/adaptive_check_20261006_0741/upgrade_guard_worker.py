"""Explicit signature migration for the narrowly proved termination correction."""
import hashlib,json,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=read(OUT/'guard_patch.json');source=Path(p['prior_run'])/'source/physical_v29';candidate=Path(p['source'])
old=(source/'training/worker_unitree.py').read_bytes();nl='\r\n' if b'\r\n' in old else '\n'
def b(x):return x.replace('\n',nl).encode()
function='''def validate_commanded_travel_checkpoint(checkpoint,signature,proof_path):
 """Explicit termination-only continuation; never a general signature bypass."""
 proof=json.loads(Path(proof_path).read_text())
 assert proof['schema']=='commanded_travel_guard_v1'
 assert signature==proof['old_reward_signature']
 assert sha(checkpoint) in proof['allowed_checkpoint_sha256']
 assert reward_signature()==proof['new_reward_signature']
 root=HERE.parent
 actual={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*')
  if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and 'usd' not in p.parts}
 assert actual==proof['source_hashes'],'Termination migration protected source mismatch'
 assert proof['cpu_tests_passed'] and not proof['gates_relaxed']
 return dict(schema=proof['schema'],proof_path=str(Path(proof_path).resolve()),
  proof_sha256=sha(proof_path),old_source_signature=signature,new_source_signature=reward_signature(),
  full_actor_critic_Adam_exploration_retained=True,gates_relaxed=False)

'''
changes=[("parser.add_argument('--actor_warm_start',action='store_true')\n", "parser.add_argument('--actor_warm_start',action='store_true')\nparser.add_argument('--termination_proof',default='')\n"),
 ('def agent_config():\n',function+'def agent_config():\n'),
 ("    assert saved['infos'].get('reward_signature')==reward_signature(),'Reward/observation/action source changed; use explicit audited actor transfer'\n",
  "    if saved['infos'].get('reward_signature')!=reward_signature():\n     assert args.termination_proof,'Reward/observation/action source changed; explicit termination proof required'\n     migration=validate_commanded_travel_checkpoint(args.checkpoint,saved['infos'].get('reward_signature'),args.termination_proof)\n     atomic_json(out/'termination_checkpoint_migration.json',migration)\n")]
changes.append(("    assert teacher_saved['infos']['reward_signature']==reward_signature(),'Teacher reward/runtime differs'\n",
 "    if teacher_saved['infos']['reward_signature']!=reward_signature():\n     assert args.termination_proof,'Teacher source changed; explicit termination proof required'\n     migration=validate_commanded_travel_checkpoint(certificate['checkpoint'],teacher_saved['infos']['reward_signature'],args.termination_proof)\n     atomic_json(out/'termination_teacher_migration.json',migration)\n"))
new=old
for before,after in changes:
 assert new.count(b(before))==1,before;new=new.replace(b(before),b(after))
(candidate/'training/worker_unitree.py').write_bytes(new);compile(new.decode(),str(candidate/'training/worker_unitree.py'),'exec')
sys.path.insert(0,str(source/'training'));from curriculum_provenance import training_reward_signature
old_sig=training_reward_signature(source/'training');new_sig=training_reward_signature(candidate/'training');assert old_sig!=new_sig
pause=read(OUT/'diagnostic_pause.json');state=read(OUT/'state_before_diagnosis.json');teacher=Path(state['promotions'][-1]['checkpoint'])
source_hashes={x.relative_to(candidate).as_posix():sha(x) for x in candidate.rglob('*') if x.is_file() and '__pycache__' not in x.parts and x.suffix!='.pyc' and 'usd' not in x.parts}
proof=dict(schema='commanded_travel_guard_v1',prior_run=p['prior_run'],checkpoint=pause['checkpoint'],checkpoint_sha256=pause['checkpoint_sha256'],
 old_reward_signature=old_sig,new_reward_signature=new_sig,allowed_checkpoint_sha256=[pause['checkpoint_sha256'],sha(teacher)],
 source_hashes=source_hashes,cpu_tests_passed=read(OUT/'travel_guard_cpu_tests.json')['passed'],gates_relaxed=False,
 environment_replacements=p['replacements'],worker_replacements=changes,
 implementation_hashes={k:source_hashes[k] for k in ['training/environment.py','training/worker_unitree.py']},
 reason='Correct artificial60m absolute travel failure; retain original60m margin beyond integrated action-command travel. Reward bodies/weights and verifier unchanged.',
 qualifies_locomotion=False)
(OUT/'termination_load_proof.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps({'signature_migration_explicit':True,'old':old_sig,'new':new_sig,'allowed_full_checkpoints':len(proof['allowed_checkpoint_sha256'])}))
