"""Reproduce the999/1000 audit boundary and verify actual gate provenance."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json
ROOT=Path(__file__).resolve().parent
TRAIN=ROOT.parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
run=Path(read(TRAIN/'ACTIVE_RUN.json')['run'])
checks={}
names=['1001_run_350_gate_332_nominal','999_run_350_gate_331_nominal','998_run_350_gate_330_nominal','1000_run_350_gate_331_push','099_run_150_train_020']
checks['old_lexical_order_selects_wrong_999']=sorted(names)[-1].startswith('999_')
ordered=sorted(names,key=lambda n:(int(n.split('_',1)[0]),n))
checks['numeric_order_selects_latest_1001']=ordered[-1].startswith('1001_')
checks['numeric_sequence_increases']=list(map(lambda n:int(n.split('_',1)[0]),ordered))==[99,998,999,1000,1001]
actual=[]
for nom in run.glob('*_gate_*_nominal'):
    if not (nom/'result.json').exists() or not read(nom/'result.json').get('completed'):continue
    suffix=nom.name.split('_',1)[1].replace('_nominal','_push')
    push=next(iter(run.glob('*_'+suffix)),None)
    if push and (push/'result.json').exists() and read(push/'result.json').get('completed'):
        current=Path(read(nom/'run_config.json')['arguments']['checkpoint'])
        assert current.resolve()==Path(read(push/'run_config.json')['arguments']['checkpoint']).resolve()
        actual.append((int(nom.name.split('_',1)[0]),str(current)))
latest=max(actual)
checks['actual_new_complete_pair_above999']=latest[0]>=1000
changes=[]
for name,path in [('record_owned_observation_handoff.py',TRAIN/'record_owned_observation_handoff.py'),
                  ('record_inventory.py',TRAIN/'history/record_inventory.py'),('review_gates.py',ROOT/'review_gates.py')]:
    source=path.read_text(encoding='utf-8')
    ast.parse(source)
    assert 'int(p.name.split(' in source
    changes.append(dict(path=str(path),before_sha256=sha(ROOT/'cpu_sort_before'/name),after_sha256=sha(path)))
checks['three_CPU_helpers_parse_and_use_numeric_prefix']=len(changes)==3
assert all(checks.values()),checks
result=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),passed=True,checks=checks,
    CPU_only_helpers=changes,actual_latest_complete_pair=latest,all_failed_before_sources_preserved=True,
    interpretation='Diagnostic latest-pair ordering only. The main supervisor obtains checkpoint from state/result, not this lexical gate sort. '
      'No live frozen worker,supervisor,reward,optimizer,physics or gate classifier was changed.')
(ROOT/'cpu_numeric_order_repair.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
