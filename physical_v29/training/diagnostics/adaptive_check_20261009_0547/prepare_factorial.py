"""Stage a bounded own-feedback head/arm factorial; no live source edits."""
from pathlib import Path
import shutil
out=Path(__file__).resolve().parent
dest=out/'head_arm_factorial';dest.mkdir(exist_ok=False)
for name in ['action_projection.py','prepare_and_pause.py','resume_after_diagnosis.py','teacher_body_probe.py']:
    shutil.copyfile(out/name,dest/name)
for name in ['prepare_and_pause.py']:
    p=dest/name;s=p.read_text();s=s.replace('HERE=OUT.parents[1]','HERE=OUT.parents[2]')
    s=s.replace('Bounded qualified-teacher body/current-upper compatibility on truthful settled/transition feedback; diagnostic hybrids never deployed',
        'Bounded own-feedback head versus arms factorial at the same saved actor; diagnostic compositions never deployed')
    p.write_text(s)
p=dest/'resume_after_diagnosis.py';s=p.read_text().replace('HERE=OUT.parents[1]','HERE=OUT.parents[2]');p.write_text(s)
p=dest/'teacher_body_probe.py';s=p.read_text()
s=s.replace('seeds=[4401,4411]','seeds=[4421,4431]')
s=s.replace("f'{n} robots,{steps*raw.step_dt}s,two seeds with{per} individually seeded paired initial states/cohort. Each policy reads its own cohort actual joint/history feedback. Qualified349/current body-upper compatibility and existing teacher-head-label context are diagnostic only.3.5 sentinel uses current actor only; no teacher extrapolation or production hybrid.'",
    "f'{n} robots,{steps*raw.step_dt}s,two seeds with{per} individually seeded paired initial states/cohort. Each policy reads its own actual feedback. Qualified349 legs plus current head-only,arm-only,and whole upper compositions are diagnostic only.3.5 sentinel is current-only. No mechanical-only isolation or production hybrid.'")
p.write_text(s)
controller='''"""Diagnostic own-feedback teacher head versus arm separation."""
import torch
MODES=('current','teacher','teacher_body_current_upper','teacher_current_head','teacher_current_arms')
def cohorts():
    rows=[]
    for seed in [4421,4431]:
        for speed in [.15,1.5]:
            for mode in MODES:
                rows.append(dict(name=f'{seed}_{speed}_{mode}',seed=seed,command=speed,initial_command=speed,mode=mode,transition=False))
        rows.append(dict(name=f'{seed}_3.5_current',seed=seed,command=3.5,initial_command=3.5,mode='current',transition=False))
    return rows
def combine_actions(current,teacher,context_teacher,group_modes,per,upper,qualified):
    assert current.shape==teacher.shape and len(group_modes)*per==len(current)
    result=current.clone()
    for group,mode in enumerate(group_modes):
        rows=slice(group*per,(group+1)*per)
        assert mode in MODES
        if mode=='current':continue
        assert bool(qualified[rows].all())
        result[rows]=teacher[rows]
        selected=upper if mode=='teacher_body_current_upper' else upper[:2] if mode=='teacher_current_head' else upper[2:] if mode=='teacher_current_arms' else []
        if selected:result[rows,selected]=current[rows,selected]
    return result
'''
(dest/'diagnostic_controllers.py').write_text(controller)
h=(out/'run_bounded_diagnosis.py').read_text()
start=h.index("    arguments = read(Path(cp).parent / 'run_config.json')['arguments']")
end=h.index('except BaseException:',start)
h=h[:start]+h[end:]
h=h.replace("launch([str(OUT / 'teacher_body_probe.py'), '--headless', '--device', 'cuda:0'], OUT / 'teacher_compatibility', 780)",
    "launch([str(OUT / 'teacher_body_probe.py'), '--headless', '--device', 'cuda:0'], OUT / 'teacher_compatibility', 780)\n    assert read(OUT / 'teacher_compatibility/result.json')['checkpoint_sha256'] == read(OUT.parent / 'teacher_compatibility/result.json')['checkpoint_sha256'], 'Factorial must use the same exact actor'\n    (ROOT / 'validation.json').write_text(json.dumps(dict(passed=True,smoke_and_full_complete=True,original_full_actor_preserved=True,diagnostic_PPO_updates=0,production_hybrid_deployed=False)))")
(dest/'run_bounded_diagnosis.py').write_text(h)
print(dest)
