# Retention as a constraint — undeployed research lead

This check tests the earlier complete learned state first. It does not implement a new reward or optimizer. Replacing a checkpoint is insufficient if unchanged learning promptly sacrifices its independently passing commands.

The current promotion contract is a conjunction: current speed, stopped behavior and every earlier speed must pass. The training objective optimizes a mixed-command average with auxiliary target losses. A high average can coexist with a failed retained command. This describes a mismatch in what is measured; it does not prove which reward term or gradient causes the plateau.

[Achiam et al., Constrained Policy Optimization, ICML 2017](https://proceedings.mlr.press/v70/achiam17a.html) optimizes return subject to auxiliary expected-cost limits. Its practical update linearizes constraints, limits policy divergence and uses backtracking. Approximation and sampling can still produce infeasible steps; the paper includes a recovery direction. Its reported experiments use a single constraint. Those results do not certify this CAD robot, many command-specific constraints, deterministic mean-policy gates or every individual trial.

If the complete-state comparison remains unqualified, the next useful reward/optimization investigation is to preserve actual per-command behavior while improving the frontier, rather than retuning an arbitrary fixed penalty. Start with the reproduced 0.15 m/s path-efficiency and 1.5 m/s geometric-flight failures. Audit their complete returns, actual advantages and physical event histories. Derive prospective costs from unchanged classifier definitions and keep command identity separate so gains elsewhere cannot cancel a violation. Do not call an expected stochastic training cost an independently certified deterministic policy.

Before any versioned candidate:

- Reconstruct the intended trajectory costs from synchronized physical measurements, segment resets and command transitions, and compare them with the unchanged independent classifiers. Include stationary, diagonal/sliding, hopping, premature-exit and nominal-flight counterexamples.
- Establish sign, finite-gradient, clipping/importance-ratio and reset/bootstrap accounting. Measure the proposed update on the current truthful state distribution. Preserve compatible actor, critic, Adam and exploration; declare any new cost estimator separately.
- Keep the original complete reward as the comparison objective. A retained-action MSE, linearized bound or trust-region estimate is not a locomotion guarantee. No observation rewrite, teacher in inference, motor-cap change or relaxed gate is permitted.
- Require normal-limit physical checks, small real PPO/full-load/reload tests, exact 64/128-update controls, two learning seeds and every ordinary/diverse nominal/pushed/current/prior/head/direction/flight gate. Preserve both branches and all rejected evidence.

Do not label the existing PPO algorithm as defective from this research lead. Do not claim CPO was tested or deployed. A broad optimizer replacement is not yet selected; a narrow, measured retention constraint must justify its implementation and runtime cost first.
