"""Actual upper objective distinguishability; fixtures are not gait references."""
import sys,json,hashlib
from pathlib import Path
import torch
OUT=Path(__file__).resolve().parent;TRAIN=OUT/'control_source/physical_v29/training'
sys.path[:0]=[str(TRAIN),str(TRAIN.parent)]
from upper_body_refinement import posture_loss
phase=torch.arange(128,dtype=torch.float64)*2*torch.pi/128
quiet=torch.zeros((128,8),dtype=torch.float64)
held=quiet.clone();held[:,2]=-1.5;held[:,3]=1.5
swing=quiet.clone();swing[:,2]=1.5*torch.sin(phase);swing[:,3]=-swing[:,2]
losses={label:float(posture_loss(x)[0]) for label,x in [('quiet',quiet),('held_opposite21_5deg',held),('zero_mean_opposite21_5deg_sine',swing)]}
assert losses=={k:0. for k in losses}
assert float(held[:,2:4].mean(0).abs().mean())==1.5
assert float(swing[:,2:4].mean(0).abs().max())<1e-15
result=dict(passed=True,fixtures=3,existing_objective_losses=losses,
    actual_frozen_module_sha256=hashlib.sha256((TRAIN/'upper_body_refinement.py').read_bytes()).hexdigest(),
    interpretation='The deployed pointwise comfortable-band and opposite-pitch symmetry terms intentionally give equal zero loss to quiet arms,held opposite21.5deg pitch and a zero-mean opposite sine. This is not a numerical defect or proof a sine is natural. The objective cannot by itself certify swing/low cycle bias. Fixture motion is synthetic,not human/Unitree data or deployed target.',
    next_hypothesis='If current actual target bias persists, evaluate phase-balanced per-command low-frequency shoulder-mean regularization confined to shoulder final-output rows. Derive bounded scale/eligibility using actual PPO/auxiliary gradients; full-state64/128 unchanged control and independent all-gate replication required before deployment.',
    source_or_training_changed=False,no_new_weight_or_reference_trajectory_deployed=True)
(OUT/'upper_objective_counterexamples.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
