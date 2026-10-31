"""Add verified local provenance to an RSL periodic save; preserve original."""
from pathlib import Path
import argparse
import hashlib
import json
import torch

HERE = Path(__file__).resolve().parent
OLD = HERE/'runs/locomotion_20261001_r2'
SOURCE = OLD/'005_stand_train_002/model_50.pt'
OUT = HERE/'checks/recovered_r2_model50'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(old=OLD, source=SOURCE, out=OUT):
    old, source, out = Path(old), Path(source), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads((source.parent/'run_config.json').read_text())
    assert config['contract'] == 'physical_v29_history4_ik_residual_v1'
    assert config['urdf_sha256'] == sha(HERE/'assets/robot_training.urdf')
    for name in ['environment.py', 'evaluation.py', 'mass_estimate.json', 'assets/motion_reference.npz']:
        assert sha(HERE/name) == sha(old/'source/physical_v29/training'/name), name
    for name in ['hardware_actuators.py', 'cad_differential.py', 'assets/actuator_limits.json', 'assets/differential_geometry.json']:
        assert sha(HERE.parent/name) == sha(old/'source/physical_v29'/name), name
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    assert isinstance(checkpoint['actor_state_dict'], dict)
    assert isinstance(checkpoint['critic_state_dict'], dict)
    assert checkpoint['optimizer_state_dict']
    checkpoint['infos'] = dict(contract=config['contract'], asset_sha256=config['urdf_sha256'],
                              speed=config['arguments']['speed'], source=str(source),
                              provenance='Periodic save from verified worker; metadata restored only; no behavior qualification')
    target = out/'resume.pt'
    torch.save(checkpoint, target)
    restored = torch.load(target, map_location='cpu', weights_only=False)
    for field in ['actor_state_dict', 'critic_state_dict']:
        for key, value in checkpoint[field].items():
            assert torch.equal(value, restored[field][key]), (field, key)
    (out/'manifest.json').write_text(json.dumps(dict(source=str(source), source_sha256=sha(source),
        recovered=str(target), recovered_sha256=sha(target), saved_iteration=checkpoint.get('iter'),
        modifications='infos metadata only; original untouched; actor/critic tensors unchanged',
        behavior_passed=False, run_config=config), indent=2))
    print(json.dumps(dict(checkpoint=str(target), iteration=checkpoint.get('iter'),
                          actor_and_critic_unchanged=True, qualification='unpassed standing candidate')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=str(OLD))
    parser.add_argument('--source', default=str(SOURCE))
    parser.add_argument('--output', default=str(OUT))
    args = parser.parse_args()
    main(args.run, args.source, args.output)
