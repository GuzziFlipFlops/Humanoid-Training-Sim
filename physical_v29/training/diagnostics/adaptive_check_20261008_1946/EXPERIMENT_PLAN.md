# Bounded failed-prior rehearsal comparison

The current 3.5 m/s learner repeatedly loses 0.15 m/s net travel and 1.5 m/s genuine flight. Exact rehearsal already exists: each of eight prior commands receives approximately 3% of all training robots. This experiment does not repair missing exact commands, alter running duty exposure or add a new teacher.

Only the existing 24% prior bucket at the 3.5 frontier is reallocated. Duplicating the two failed anchors raises their expected fractions to 4.8% each and reduces each other prior to 2.4%. Standing remains 32%; frontier remains 44%, with the same exact versus continuous mixture. Identical random draws outside the prior bucket produce bitwise identical commands. Every other frontier is unchanged. The tradeoff in other prior exposure must be tested.

Both branches start exact actor, critic, Adam and exploration from the current saved boundary, preserving the original upper refinement and both teachers. Two independent learning seeds (157 and167) use4,096 environments for64 then128 updates. Every64/128 checkpoint receives unchanged320-robot20-second nominal/pushed tests. At128, extra nominal-physics/pushed tests generate32 independent small joint initial states per command using4231/4251, paired between branches. Those additional seeds diversify initial conditions;157/167 separately replicate learning.

Before comparison: same-actor physical replay, passive real-PPO capture, current/prior normal-limit physics at two seeds with native5ms motor request/application recording,27 CPU sampler fixtures, eight actual64-environment PPO updates and four full-state reload updates. These checks do not themselves qualify locomotion or establish causality.

The hypothesis is a bounded allocation test motivated by persistent failures and smaller sampled prior contributions. Mixed PPO/teacher alignment and positive normalized advantages do not prove a sampling defect or globally opposed gradients. No arbitrary reward weight is changed.

Deployment requires independent current/prior retention, direction, flight and posture evidence at both learning seeds, with no new failures; a passing current speed alone is insufficient. Original frozen signatures do not certify the experimental sampler. Any successful candidate would require a narrowly versioned fresh freeze and strict compatible full-state migration. Diagnostic actors are never silently attached to production. Rejection preserves both successful and failed results and resumes the exact original learner.

The owned supervisor is suspended only after a saved boundary and its worker drains. Operations run sequentially with one actual CUDA worker and bounded budgets. Saved-boundary waiting/drain is separate from the6,000-second diagnostic work budget. Finally, all diagnostic workers exit and the same supervisor resumes. No endpoint is inferred from a budget, crash, cap spike or commanded speed.
