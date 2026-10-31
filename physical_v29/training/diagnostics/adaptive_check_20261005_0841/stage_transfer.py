"""Isolated training-only candidate. Protected source inventory stays byte-identical."""
import json,shutil,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
pause=json.loads((OUT/'diagnostic_pause.json').read_text());source=Path(pause['run'])/'source/physical_v29'
target=OUT/'validation_source/physical_v29';assert not target.exists()
shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc','usd'))
shutil.copy2(OUT/'phase_transfer.py',target/'training/phase_transfer.py')
shutil.copy2(OUT/'phase_context.json',target/'training/phase_context.json')
p=target/'training/worker_unitree.py';text=p.read_text()
text=text.replace("  if args.mode=='train':\n   retention=None;locomotion_retention=None",'''  if args.mode=='train':
   # Experimental labels only; the actual student inputs remain untouched.
   from phase_transfer import RunningRowTransfer
   transfer_teacher=copy.deepcopy(runner.alg.actor)
   transfer_path=Path(__file__).resolve().parents[5]/'runs/locomotion_20261004_r34_bilateral_swing_unloading/038_run_150_train_007/candidate.pt'
   transfer_saved=torch.load(transfer_path,map_location=envraw.device,weights_only=False)
   assert transfer_saved['infos']['reward_signature']==reward_signature()
   transfer_teacher.load_state_dict(transfer_saved['actor_state_dict']);del transfer_saved
   retention=None;locomotion_retention=None''')
text=text.replace("   upper_refinement=UpperBodyRefinement(runner.alg.actor,joint_names)","   upper_refinement=UpperBodyRefinement(runner.alg.actor,joint_names)\n   phase_transfer=RunningRowTransfer(runner.alg.actor,transfer_teacher,joint_names,json.loads((HERE/'phase_context.json').read_text()))")
text=text.replace("    state['command_coverage']=", "    state['phase_transfer']=phase_transfer.read_audit()\n    state['command_coverage']=")
assert 'phase_transfer=RunningRowTransfer' in text
# Explicit absolute source for this experiment only, pinned below.
flight=HERE/'runs/locomotion_20261004_r34_bilateral_swing_unloading/038_run_150_train_007/candidate.pt'
text=text.replace("Path(__file__).resolve().parents[5]/'runs/locomotion_20261004_r34_bilateral_swing_unloading/038_run_150_train_007/candidate.pt'",repr(str(flight)))
p.write_text(text)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def files(root):return {p.relative_to(root).as_posix():p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and 'usd' not in p.parts}
a,b=files(source),files(target)
changed={k for k in a.keys()|b.keys() if k not in a or k not in b or sha(a[k])!=sha(b[k])}
assert changed=={'training/worker_unitree.py','training/phase_transfer.py','training/phase_context.json'},changed
record=dict(diagnostic_only=True,production_changed=False,changed=sorted(changed),
  implementation_hashes={k:sha(b[k]) for k in changed},protected_files=len(set(a)-changed),
  teacher=str(flight),teacher_sha256=sha(flight),student_checkpoint=pause['checkpoint'],student_sha256=pause['checkpoint_sha256'])
(OUT/'staging.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
