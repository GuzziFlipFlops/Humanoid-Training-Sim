"""Apply selected bitwise-equivalent execution after every required check passed."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile

here=Path(__file__).resolve().parent
workspace=here.parents[3]
training=here.parents[1]
experiment=json.loads((here/'experiment.json').read_text())
selected=here/'iteration_04/source/physical_v29'
implementation=['cad_differential.py','hardware_actuators.py','owned_cuda_graph.py']
tests={
 'math_cuda':'iteration_02/math_cuda.json',
 'actuator_parity':'iteration_04/benchmark_4096_retry/actuator_parity.json',
 'physical_parity_4096':'iteration_04/physical_parity_4096.json',
 'mixed_replay_parity':'iteration_04/mixed_replay_parity.json',
 'ppo_4096':'iteration_04/ppo_4096/result.json',
 'ppo_6144':'iteration_04/ppo_6144/result.json',
 'source_audit':'source_audit.json',
}
proof=dict(schema='exact_execution_optimization_v1',selected_iteration='04',
           prior_run=experiment['run'],checkpoint_sha256=experiment['checkpoint_sha256'],
           implementation_hashes={name:hashlib.sha256((selected/name).read_bytes()).hexdigest() for name in implementation},
           tests={name:dict(path=path,sha256=hashlib.sha256((here/path).read_bytes()).hexdigest()) for name,path in tests.items()},
           qualifies_locomotion=False,selection_reason='3.28x controlled inference and 3.15x actual 4096 PPO; iteration5 marginal gain within noise with additional environment changes',
           continuation='Preserve exact R29 actor, critic, exploration, Adam, adaptive LR and standing teacher; fresh current and retained-speed independent gates')
proof_path=here/'continuation_proof.json'
assert not proof_path.exists()
proof_path.write_text(json.dumps(proof,indent=2))
sys.path.insert(0,str(training))
import supervisor_unitree as supervisor
prior=Path(experiment['run'])
assert supervisor.verify_resource_migration(prior,selected,str(proof_path))['checkpoint']==experiment['checkpoint']

# Negative tests use disposable mutable copies, never any frozen experiment.
results=[]
with tempfile.TemporaryDirectory(prefix='migration_negative_',dir=here) as folder:
    assert Path(folder).resolve().is_relative_to(here)
    target=Path(folder)/'physical_v29'
    shutil.copytree(selected,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for relative in ['cad_differential.py','hardware_actuators.py','owned_cuda_graph.py',
                     'training/environment.py','training/environment_unitree.py','training/evaluation.py',
                     'assets/differential_geometry.json','assets/actuator_limits.json',
                     'training/assets/robot_training.urdf','training/mass_estimate.json']:
        path=target/relative;before=path.read_bytes();path.write_bytes(before+b'\n# rejected mutation\n')
        try:supervisor.verify_resource_migration(prior,target,str(proof_path))
        except AssertionError:results.append(dict(mutated=relative,rejected=True))
        else:raise AssertionError('Unauthorized source was accepted: '+relative)
        finally:path.write_bytes(before)
    bad=json.loads(proof_path.read_text());bad['qualifies_locomotion']=True
    bad_path=here/'negative_proof.json';bad_path.write_text(json.dumps(bad))
    try:supervisor.verify_resource_migration(prior,target,str(bad_path))
    except AssertionError:results.append(dict(mutated='qualification claim',rejected=True))
    else:raise AssertionError('Execution evidence was accepted as locomotion qualification')
    # Preserve this negative input as test evidence.
    assert supervisor.verify_resource_migration(prior,target,str(proof_path))

# Preserve originals and ensure no unrelated canonical edits have occurred.
for name in implementation[:2]:
    assert (training.parent/name).read_bytes()==(here/'canonical_before_apply'/name).read_bytes(),name
for name in implementation:shutil.copyfile(selected/name,training.parent/name)

# Test the real future freeze path without starting Isaac or touching ACTIVE_RUN.
freeze=here/'continuation_freeze_smoke'
chain=supervisor.Chain(str(freeze),300,18,64,initial_checkpoint=experiment['checkpoint'],
                       train_envs=8192,migrate_run=experiment['run'],learning_rate=.0001,
                       optimization_proof=str(proof_path))
assert supervisor.verify_resource_migration(prior,freeze/'source/physical_v29',str(proof_path))
config=json.loads((freeze/'chain_config.json').read_text())
assert 'physical_v29/owned_cuda_graph.py' in config['sources_sha256']
chain.state.update(status='freeze_smoke_complete_no_worker',child_pid=None)
chain.write()
report=dict(passed=True,positive_migration=True,negative_cases=results,
            future_frozen_helper_present=True,actor_only_transfer=False,canonical_applied=True,
            original_checkpoint_untouched=hashlib.sha256(Path(experiment['checkpoint']).read_bytes()).hexdigest()==experiment['checkpoint_sha256'])
(here/'continuation_integration.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
