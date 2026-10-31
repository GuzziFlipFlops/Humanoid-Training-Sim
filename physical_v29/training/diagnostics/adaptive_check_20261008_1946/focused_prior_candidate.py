"""Separate-process training candidate, not certified by the original signature."""
import json
import runpy
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
TRAIN = OUT / 'control_source/physical_v29/training'
sys.path[:0] = [str(OUT), str(TRAIN), str(TRAIN.parent)]
import command_curriculum
from focused_prior_sampler import sample_focused, VERSION, FOCUS

applied = '--focused-prior-rehearsal' in sys.argv
if applied:
    sys.argv.remove('--focused-prior-rehearsal')
    command_curriculum.sample_training_speeds = sample_focused
folder = Path(sys.argv[sys.argv.index('--output') + 1])
folder.mkdir(parents=True, exist_ok=True)
(folder / 'experimental_exposure.json').write_text(json.dumps({
    'version': VERSION, 'applied': applied, 'focus_commands': FOCUS,
    'scope': 'Only the existing 24% exact-prior bucket is reweighted at frontier 3.5. Standing 32% and frontier 44% mixture are unchanged. Expected focused command fractions increase from 3% to 4.8%; each other prior drops from 3% to 2.4%. Actual coverage must be measured.',
    'original_signature_does_not_certify_executed_training_sampler': applied,
    'actor_physics_rewards_motor_caps_teachers_upper_module_unchanged': True,
    'truthful_student_observations': True, 'no_inference_action_override': True,
    'diagnostic_actor_deployed': False
}, indent=2) + '\n', encoding='utf-8')
runpy.run_path(str(TRAIN / 'worker_unitree.py'), run_name='__main__')
