# Model optimization experiment, October 4, 2026

The five optimization attempts are complete, and iteration 4 was selected.
R30 now continues the original full R29 checkpoint in a fresh frozen run;
the recurring four-hour check remains paused. The user requested three to five
tested iterations before restarting training. These bounded experiments do not
qualify locomotion. The original R29 source and full checkpoint remain untouched.
See [final report](REPORT.md) and [continuation proof](continuation_proof.json).

## Scope and comparison

The existing training robot already has 24 rigid links and box collision shapes;
there are no detailed render meshes, stereo cameras, finger dynamics or passive
closed-loop constraints in this asset. The earlier profile attributed most time
to repeated exact rod differential calculations. Optimizing their execution is
preferable to deleting the mechanical constraints or changing the CAD layout.

Every iteration keeps joint frames/limits, link masses/inertias, collision boxes,
contact sensors, 5 ms physics steps, four substeps, self-collision, solver settings,
motor torque/speed envelopes, combined pitch/roll limits, policy/history ownership,
commands, rewards, and independent qualification gates. No Jacobian is reused
across a changed pose. CPU remains an eager fallback.

`benchmark.py` uses the actual saved RSL actor, critic and optimizer configuration,
with mean-action inference. It consumes the observation returned by the previous
step, as production PPO does. Its throughput measures environment transitions
including actor inference and telemetry, excluding PPO updates and initialization.
The old 24-tick profiler called `get_observations()` again on every tick, so its
number is not the controlled baseline here. Every variant runs in a fresh process
at 4,096 environments, seed 901, 24 warmup ticks and three 64-tick windows. This is
a short benchmark, not a long-duration memory or behavioral certificate.

One actual Isaac/CUDA worker runs at a time. GPU free memory must remain at least
300 MiB and host available memory at least 2,048 MiB. Outputs are preserved in
separate per-variant directories. `manifest.json` records frozen experiment files.

## Planned iterations

1. Prepare immutable geometry and share trigonometry within each differential
   state calculation, keeping the original operation ordering.
2. Replay differential tensor calculations using owned, bounded CUDA graphs.
3. Capture the complete hardware actuator calculation, including PD and all
   three coupled torque/speed envelopes.
4. Repair iteration 3's captured Python joint-index conversion by preparing
   CUDA index tensors outside capture; benchmark full actuation again.
5. Capture target feasibility projection and fresh current-pose transmission
   validity as two pure tensor operations; never substitute pre-physics state
   for post-physics telemetry.

Graph storage addresses must stay fixed; inputs are copied each call and results
are cloned so subsequent replay cannot overwrite saved rollouts or diagnostics.
This follows [PyTorch's CUDA graph guidance](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/).
Only pure tensor arithmetic is captured. Tests compare both numeric results and
tensor ownership against the frozen model, including invalid poses, different
input shapes, noncontiguous inputs, coupled torque requests and overspeed braking.

Final selection must consider measured gain, parity, physical replay, PPO/load
smoke, reserves and complexity. Speed alone does not justify accepting a variant.
