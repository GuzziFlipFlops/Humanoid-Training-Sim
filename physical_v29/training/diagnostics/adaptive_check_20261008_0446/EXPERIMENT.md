# Phase-balanced shoulder mean experiment

This is a local training hypothesis prompted by actual physical target and joint
timelines. The existing comfortable-band and pointwise symmetry loss intentionally
allows opposite held shoulder offsets. It is not a numerical defect.

The candidate retains the complete current checkpoint, existing upper refinement,
both qualified teachers, ordinary PPO and all physical/inference constraints. A
new derivative affects only the left and right shoulder-pitch rows of the final
actor layer. It has no direct shared-trunk, head, roll, elbow or lower-row derivative.
Physical coupling, ordinary PPO and global clipping can still affect locomotion.

Within each exact earlier command or one of two frontier speed bands, eight phase
bins receive equal weight. Only truthful, straight-command histories with stable
translation and at least two samples in every phase bin are eligible. The target
quantity is the average clipped shoulder offset, with a declared five-degree free
mean band and Huber slope, using the existing arm-loss scale0.025. These are local
experimental choices, not measured human/Unitree reference parameters. No swing
amplitude, timing curve or action override is imposed.

The extra final-layer gradient is capped at10% of the actual incoming PPO shoulder
gradient norm before the existing global clipping. Passive reproduction records
the unchanged upper auxiliary norm too. Both control and candidate compute these
diagnostics; only the candidate adds the new derivative. CPU fixtures independently
verify passive control gradient equality and exact direct output-row isolation.

Population bias cancellation remains a known limitation: opposite held biases on
different robots can yield a zero ensemble mean. This objective does not certify
individual cycle means or natural motion. Actual joint/target/velocity traces,
reset segmentation, population symmetry and gait gates remain required.

The sequential experiment includes unchanged-actor physical replay, four-update
PPO/full-load/reload smoke, exact full-state4096-environment64/128 control/candidate
continuations, matching320-robot20-second nominal/pushed gates and independently
seeded nominal-physics initial joint states. Every classifier and native limit is
unchanged. Original source signatures do not certify the experimental gradient;
no diagnostic learned actor may be silently inserted into production. All evidence
and failed attempts are retained. The original supervisor resumes in finally.
