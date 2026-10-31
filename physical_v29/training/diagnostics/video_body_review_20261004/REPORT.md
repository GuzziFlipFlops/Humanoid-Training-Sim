# Recorded body and gait review — October 4, 2026

The four recordings show the provisional training boxes, not the original CAD
body meshes. This was the wrong visual choice for a presentation of the user's
robot. The full v29 mesh export remains present. No CAD body disappeared during
walking, and the October 4 execution optimization did not remove the CAD.

## Frame evidence

All 2,000 H.264 frames decoded successfully: 500 frames at 50 FPS per clip,
exactly 10 seconds each. Five frames per speed were visually inspected at 0.02,
2.50, 5.00, 7.50 and 10.00 seconds. The same block geometry and gaps are visible
throughout these samples. See [all-speed close-ups](all_speeds_closeups.jpg), the
individual decoded source frames, and [machine-readable audit](audit.json).

The image shows empty-looking neck, waist, hip, knee, wrist and ankle regions
because the displayed shapes do not fill the actual link envelopes. Uniform
grey surfaces, low contrast against the grey scene, and a large black upper
band further reduce the presentation quality. The export issue is established
by source inspection; the precise cause of the black band was not diagnosed.

## Exact export cause

The recorder loaded its R32 frozen
`source/physical_v29/training/assets/robot_training.urdf`. That file contains
24 links, 23 joints, **24 box visuals and zero mesh visuals**. The full source
`physical_v29/assets/humanoid_physical_v29.urdf` contains the same link names,
joint frames and axes, with **140 visual meshes**. All referenced mesh files exist.

`training/build_asset.py:112` removes original visuals and collisions.
Lines 128–131 shrink selected envelopes; most selected casing boxes use factors
0.78, 0.78 and 0.87 along the three dimensions. Some intermediate hip links use
only 18 mm marker boxes. Lines 140–142 then use those same provisional boxes for
both visualization and collision. The video recorder at
`diagnostics/adaptive_check_20261004_1437/record_walking_gui.py:49` imports that
training asset without restoring the original visual meshes.

This also matters physically: missing surfaces are not solely invisible skins.
The training collision geometry underapproximates the real body, so these
recordings and proxy-based gates do not establish full-CAD clearance or contact
fidelity. The mass and inertia estimates include modeled parts and planning
allowances, but they do not restore missing collision surfaces. Execution parity
only preserved the existing proxy model. Current v31/v32 CAD changes remain
separate from the frozen v29 training and this review.

## The motion also needs separate review

The recordings use the preserved walking-baseline actor, SHA256
`51dbe9e21f69f63635399869250a38c8383392a82024293a16fbaa96d988dee2`, rather than the
current learning actor. All four ten-second traces have zero terminations.
Their pelvis heights remain approximately 0.898–0.909 m.

At the 1.0 m/s command, mean left/right knee angles are 43.87/45.14 degrees,
head pitch is -28.84 degrees and head yaw is +40.80 degrees in the saved native
joint frames. The head is visibly tilted/turned, and the knees remain deeply
flexed. These are recorded poses, not a human-motion score; restoring meshes
does not correct them. Survival and velocity qualification alone did not enforce
a natural head pose or certify the appearance of the gait.

The first CPU angle summary incorrectly labeled Isaac arrays with the CAD joint
declaration order. That diagnostic is preserved under
`attempt_01_wrong_joint_order` and is invalid. The corrected audit uses the actual
`ENVIRONMENT_READY` joint order from the same frozen asset's checkpoint worker
log. No policy, training source, CAD, physics or original video changed during
either audit. The recorder should store its actual runtime joint/body names
directly in future capture manifests to make this unambiguous.

## Prepared visual correction and its limits

[robot_training_full_visuals.urdf](playback_asset/robot_training_full_visuals.urdf)
is a separate playback asset: all 140 original v29 visuals are restored, with
resolved mesh paths. A saved/reloaded structural comparison verifies that every
joint, limit, collision and inertial element is exactly preserved from the
recorded training asset. Original source hashes and videos remain unchanged.

This asset is **prepared and structurally checked, not yet converted or rendered**.
No replacement video is claimed, and it is not installed into the active training
run. A corrected recording must use these CAD visuals, inspect their articulated
alignment and improve the camera/lighting. Full-body physical fidelity requires
a separate collision-clearance review; merely adding display meshes cannot
qualify that. Changing training collision geometry would require a fresh frozen
run and matching independent behavior tests.

The audit ran entirely on the CPU while existing training continued. It did not
launch another Isaac/CUDA worker or change rewards, motor limits or checkpoints.
