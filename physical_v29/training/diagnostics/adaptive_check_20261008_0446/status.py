import json
from pathlib import Path
OUT=Path(__file__).resolve().parent;TRAIN=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
state=read(Path(read(TRAIN/'ACTIVE_RUN.json')['run'])/'state.json')
op=Path(state['operation_path']);result=dict(production_status=state['status'],operation=op.name)
if (op/'progress.json').exists():
    p=read(op/'progress.json');result['iteration']=p['iteration'];result['gpu']=p['gpu']
for f in ['diagnostic_pause.json','diagnostic_resume.json']:
    if (OUT/f).exists():result[f]=read(OUT/f)
root=OUT/'matched_shoulder'
if root.exists():
    result['diagnostics']={}
    for p in sorted(root.iterdir()):
        if not p.is_dir():continue
        r=read(p/'result.json') if (p/'result.json').exists() else None
        s=dict(completed=r.get('completed') if r else False)
        if (p/'progress.json').exists():
            v=read(p/'progress.json');s['iteration']=v['iteration'];s['new_gradient']=v['upper_refinement'].get('experimental_shoulder_mean');s['gpu']=v['gpu']
        elif (p/'stderr.log').exists() and (p/'stderr.log').stat().st_size:s['stderr_tail']=(p/'stderr.log').read_text()[-700:]
        result['diagnostics'][p.name]=s
print(json.dumps(result,indent=2))
