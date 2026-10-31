"""Pin the actual drained checkpoint and tested, narrowly scoped source changes."""
import datetime
import hashlib
import json
import sys
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parents[4]
HERE=ROOT/'physical_v29/training'
OUT=Path(__file__).parent
sys.path.insert(0,str(HERE))
from curriculum_provenance import verify_curriculum_migration
RUN=HERE/'runs/locomotion_20261004_r30_exact_graph_execution'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
state=read(RUN/'state.json')
drain=read(OUT/'r30_drain.json')
assert state['status']=='superseded_for_diagnostic_repair' and state['child_pid'] is None
assert drain['all_owned_workers_drained'] and drain['checkpoint']==state['checkpoint']
for process in psutil.process_iter(['name','cmdline']):
    args=process.info['cmdline'] or []
    if process.info['name']=='python.exe' and any(Path(a).name=='worker_unitree.py' for a in args):
        raise AssertionError(('Another Isaac worker still exists',process.pid,args))
names=['environment_unitree.py','worker_unitree.py','evaluation.py','command_curriculum.py','curriculum_provenance.py']
proof=dict(schema='exact_command_rehearsal_v1',qualifies_locomotion=False,
    recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),prior_run=str(RUN),
    checkpoint_sha256=sha(Path(state['checkpoint'])),
    implementation_hashes={'training/'+n:sha(HERE/n) for n in names},
    cpu_test=dict(path='cpu_test.json',sha256=sha(OUT/'cpu_test.json')),
    diagnosis='Repeated nominal/pushed retained 0.15 overspeed despite improved straight 0.65 alternating gait; old exact-speed exposure sparse',
    changes='Training-only exact-prior/frontier rehearsal and future running speeds through5; original rewards/physics/optimizer and prior classifiers preserved',
    supporting_tests=dict(continuation_cpu_tests='continuation_cpu_tests.log',running_extension='running_extension_tests.json',
                          upstream_counterfactuals='upstream_port_counterfactuals/counterfactual_rewards.json'),
    preserves=['full_actor','critic','Adam','exploration','adaptive_learning_rate','standing_teacher','iteration4_execution','physical_motor_limits'])
path=OUT/'continuation_proof.json';path.write_text(json.dumps(proof,indent=2))
verify_curriculum_migration(RUN,OUT/'cpu_freeze_fixture/source/physical_v29',path)
print(json.dumps(dict(proof=str(path),checkpoint=state['checkpoint'],checkpoint_sha256=proof['checkpoint_sha256'])))
