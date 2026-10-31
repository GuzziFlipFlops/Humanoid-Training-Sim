"""V6 adds one heading term and fixes caching; other kernels/gates stay intact."""
import ast
import copy
import hashlib
from pathlib import Path
import unittest

HERE=Path(__file__).parent
FROZEN=HERE/'runs/locomotion_20261003_r27_heading_commands_checked/source/physical_v29'


def method(path,name):
    return next(n for n in ast.walk(ast.parse(path.read_text()))
                if isinstance(n,ast.FunctionDef) and n.name==name)


class Provenance(unittest.TestCase):
    def test_reward_kernels_and_weights_unchanged(self):
        now=copy.deepcopy(method(HERE/'stage_rewards.py','shaped_terms'))
        added=[n for n in now.body if isinstance(n,ast.Assign) and any(ast.unparse(t)=="terms['fixed_heading']" for t in n.targets)]
        self.assertEqual(len(added),1)
        self.assertEqual(ast.unparse(added[0].value),"torch.exp(-(metrics['heading_error_rad'] / 0.35).square()) * healthy * moving")
        now.body=[n for n in now.body if n not in added]
        self.assertEqual(ast.dump(now,include_attributes=False),
                         ast.dump(method(FROZEN/'training/stage_rewards.py','shaped_terms'),include_attributes=False))

    def test_all_preexisting_gate_checks_unchanged(self):
        self.assertEqual((HERE/'evaluation.py').read_bytes(),(FROZEN/'training/evaluation.py').read_bytes())
        self.assertEqual((HERE/'heading_control.py').read_bytes(),(FROZEN/'training/heading_control.py').read_bytes())

    def test_contact_mass_transmission_hardware_bytes_unchanged(self):
        for rel in ['training/environment.py','training/assets/robot_training.urdf','training/assets/motion_reference.npz',
                    'training/mass_estimate.json','training/support_geometry.py','training/evaluation_steps.py',
                    'hardware_actuators.py','assets/actuator_limits.json',
                    'assets/differential_geometry.json','assets/export_metadata.json']:
            self.assertEqual((HERE.parent/rel).read_bytes(),(FROZEN/rel).read_bytes(),rel)

    def test_cad_change_is_only_bypassing_tensor_cache(self):
        now=ast.parse((HERE.parent/'cad_differential.py').read_text())
        function=next(n for n in ast.walk(now) if isinstance(n,ast.FunctionDef) and n.name=='t')
        added=function.body.pop(0)
        expected=ast.parse('if isinstance(v, torch.Tensor):\n    return v.to(device=q.device, dtype=q.dtype)').body[0]
        self.assertEqual(ast.dump(added,include_attributes=False),ast.dump(expected,include_attributes=False))
        self.assertEqual(ast.dump(now,include_attributes=False),ast.dump(ast.parse((FROZEN/'cad_differential.py').read_text()),include_attributes=False))

    def test_observation_action_reset_and_command_functions_unchanged(self):
        for name in ['_get_observations','_pre_physics_step','_reset_idx','_update_observation_command','configure_fixed_commands']:
            self.assertEqual(ast.dump(method(HERE/'environment_unitree.py',name),include_attributes=False),
                ast.dump(method(FROZEN/'training/environment_unitree.py',name),include_attributes=False),name)

    def test_pinned_g1_h1_function_bodies_unchanged(self):
        for path in (FROZEN/'training/unitree_vendor').rglob('*'):
            if path.is_file() and path.suffix!='.pyc':
                rel=path.relative_to(FROZEN/'training')
                self.assertEqual((HERE/rel).read_bytes(),path.read_bytes(),str(rel))


if __name__=='__main__':unittest.main()
