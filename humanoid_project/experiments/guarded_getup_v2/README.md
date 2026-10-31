# Recovery revision v2 — September 3, 2026

Separate candidate; the approved walking/running assets and policies are unchanged.
See ../guarded_getup_v1/README.md for the unchanged collision, pose-bank, sensor,
actor/critic, actuator and strict evaluation contracts.

## Evidence and fixes

- v1 completed 200 PPO iterations at 8192 environments, then its perturbed
  evaluation crashed at 16 seconds: an Isaac cached inference tensor was updated
  outside torch.inference_mode. The push now uses a decorated inference helper;
  a unit test exercises an actual inference-created buffer. Preflight also runs
  a real perturbed rollout through the full 25 seconds.
- Its 128 nominal held-out supine trials had zero recoveries; mean maximum pelvis
  height was .0911 m. The height component gave no credit below .25 m. This was
  an observed reward blind zone, not proof that it was the only learning problem.
- Height progress now starts at .06 m. Upright progress uses (1+cos(tilt))/2,
  keeping a smooth signal beyond 90 degrees. Only improvements over the episode's
  previous best combined score are paid. Repeating the same bounce/height or
  remaining motionless pays no additional progress.
- Discovery progress budget is 4 instead of 1 per episode, still below the +20
  once-only seven-second completion reward. Reset initializes the score from the
  measured pose. This bounded shaping is NOT policy-invariant and partial-rise
  local optima remain possible. Independent full-recovery gates remain essential.
- Strict contacts, actual torso/pelvis tilt, COM support margin, anchor drift,
  two-second dwell plus five-second hold, post-success invalidation, self-collision
  and motor limits are unchanged. Standing rehearsal never counts as a lying
  recovery. No gait cadence, running reward or external assistance is added.

## Continuation and budget

The main run branches from v1's saved 200-iteration candidate, retaining actor
weights, actor normalizer and .7 action scale. Because rewards changed, migration
resets the critic and optimizer. Subsequent v2 blocks reload all three normally.
Smoke/benchmark policies are discarded, not used as the main starting policy.

Additional ceilings: 600 supine, 800 four-pose discovery, 400 refinement; 1800
new iterations, plus the inherited 200. Stages advance only on the original
per-pose nominal/perturbed gates. Three evaluated blocks without success or
height progress stop for diagnosis. Budget completion is not task success.

Preflight: 64 environments x 64 iterations, saved-policy perturbed reload,
unchanged stationary feasibility calibration, then 4096/6144/8192 x 4-iteration
scaling checks. Requested reserve 512 MiB, three consecutive low-free-memory
samples trigger memory-only fallback. Active state and GPU samples live in the
new logs/guarded_getup_v2_* directory. Monitor with monitor.py (or --run PATH).

The discovery/refinement structure was informed by the primary
[HumanUP project](https://humanoid-getup.github.io/) and
[paper](https://arxiv.org/abs/2502.12152); these reward changes are our engineering
candidate, not a reproduction or a guarantee of convergence.
