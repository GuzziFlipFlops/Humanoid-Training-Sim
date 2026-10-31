"""Make the new dated handoff readable without changing scope or prior history."""
import json
from pathlib import Path
out=Path(__file__).resolve().parent;train=out.parents[1];root=train.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
write=lambda p,d:Path(p).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
d=read(out/'decision.json');old=list(d['next_tests']);old_diag=d['diagnostic_summary']
new=[
 'Preserve the current full learner, the separately passing 3.5 m/s pairs 908/917 and 605, actor 267 with all prior cases passing, and separately certified 349. Compare fresh 3.5 m/s direction, 0.15 m/s net travel, 1.5 m/s genuine/geometric flight and every earlier gate. Reviewed 920 also fails 2/2.5/3 m/s lateral direction and nominal 0.35 m/s bilateral stance. Two-seed component tests on exact 917 show that teacher legs with current arms overspeed at 0.15 and lose flight at 1.5; head changes also affect flight. This identifies closed-loop coordination, without isolating mechanical coupling. Next physically test the current body with qualified 349 arms only while retaining the CURRENT head, under each controller\'s truthful feedback. Resolve phase-specific effective leg targets, sole unloading and torso response on settled and transitioning histories. The inverse whole-upper test reached only about 0.9% flight; the arms-only inverse is untested. Never use 349 labels above its 3 m/s certificate. Select a command-local, state-conditioned arm-row or coupled transfer only after physical validation, exact current full-state 64/128-update controls, two learning seeds, all unchanged gates and independently seeded initial-state replication. Keep prior reweighting and rejected mean/gain/context/course candidates rejected without new evidence.',
 'Every check inspect actual head yaw/pitch, shoulder pitch/roll and elbow angle, physical target and velocity timelines at every speed, including bilateral amplitude, arm-leg phase and noncancelling slow bias. Reviewed 920 has 4.38-degree head RMS and 11.74/10.25-degree shoulder pitch ranges, with opposite held fast biases and slow 2.5/3 m/s reversals. The component split attributes most slow overspeed and the larger 1.5 m/s flight loss to the current arm controller, but current head alone yields 1.478/1.502% flight and is not qualified. Do not deploy diagnostic composites or infer a purely mechanical cause. Preserve upper refinement at 20 batches/update, forward head, 35/1 gains, legs, both teachers and full learned state. Collins 2009 Figure 6 provides human shoulder/elbow time plots, but numerical curves and exact speed, axes, rest poses and morphology mapping remain unverified. Mapped Unitree encoder curves and calibrated optical gaze remain unavailable. Do not fabricate curves, substitute production observations, raise caps or restart from scratch.',
 'Verify supervisor 39000, one actual worker, 55 frozen hashes, 53 protected authored files, local ground, exact actor/critic/Adam/exploration and both teachers. Count 27,507 inherited updates plus fresh completed and 50 hash-proven R40 recoveries; exclude 32 unsaved R39 updates and all smoke, capacity and diagnostic samples. Exclude the 126.955-second failed-metadata, 476.254-second compatibility and 371.378-second component-test pauses only through each timing endpoint. The canceled preboundary lookup never paused training. Resumed 920/923 load exact 917/920 respectively, with zero KL/command mismatch and all three auxiliaries at 20 batches/update. Keep target 8192, reserve 300 MiB, host guards/fallback and 180 continuous healthy TRAINING seconds; Fusion stays closed. Recorded 917 native 5 ms motor samples have zero driving clips and braking overspeed. The later 920 10 Hz ankle cap crossings cannot inherit that attribution. Only qualified 5 m/s, or qualified 4 m/s with causal native limits proving 5 blocked, ends training and automation; every endpoint needs all unchanged gates and 32 independently seeded initial-state trials per command.'
]
new_diag=('Exact 917 has 1152 robots over 20 seconds in the compatibility test and 704 in the head/arms split, each with two seeds and 32 individually seeded paired states per cohort; all survive. Qualified 349 gives about 1.81% genuine flight at 1.5 m/s. Teacher body with current whole upper gives about 0.94%, current arms only about 1.075%, and current head only 1.478/1.502%. At 0.15 m/s, current arms on teacher body/head reach about 0.285 m/s versus the teacher\'s 0.184; head only stays near 0.184. Context-conditioned whole-upper transfer is also unsafe. These are closed-loop component effects, not pure mechanical isolation. Every diagnostic composite remains outside production. Four actual PPO captures show mixed prior/upper alignment. No new 64/128-update candidate learning comparison or qualification is claimed.')
replacements=list(zip(old,new))+[(old_diag,new_diag)]
files=[out/'REPORT.md',out/'evidence.json',out/'automation_update_args.json',train/'automation/NEXT_CHECK.json',
 train/'ITERATION_HISTORY.md',root/'docs/robot/TRAINING_JOURNAL_20261001.md',
 Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md'),
 Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),
 Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]
def convert(x):
 if isinstance(x,str):
  for a,b in replacements:x=x.replace(a,b)
  return x
 if isinstance(x,list):return [convert(v) for v in x]
 if isinstance(x,dict):return {k:convert(v) for k,v in x.items()}
 return x
for p in files:
 if p.suffix=='.json':write(p,convert(read(p)))
 else:
  text=p.read_text(encoding='utf-8')
  assert text.count('<!-- adaptive_check_20261009_0547_record -->')==1 or p.name=='REPORT.md'
  for a,b in replacements:text=text.replace(a,b)
  p.write_text(text,encoding='utf-8')
d['next_tests']=new;d['diagnostic_summary']=new_diag;write(out/'decision.json',d)
print('New dated handoff made readable; older chronological entries preserved.')
