import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
g=read(OUT/'initial_gate_review.json');u=read(OUT/'upper_summary.json');a=read(OUT/'initial_learning_verification.json');b=read(OUT/'diagnostic_pause.json')
grad=read(OUT/'current_gradient_analysis.json');recent=[]
for pair in g['gate_pairs'][-10:]:
 recent.append(dict(checkpoint=pair['checkpoint'],current={label:{k:pair[label+'_cases'][-1][k] for k in ['signed_speed','lateral_speed','flight','failures']} for label in ['nominal','pushed']},
  prior_failures={label:[dict(command=c['command'],failures=c['failures']) for c in pair[label+'_cases'][:-1] if not c['passed']] for label in ['nominal','pushed']}))
(OUT/'recent_gate_trends.json').write_text(json.dumps(recent,indent=2))
data=[c['teacher_conditioning_lower_effective_RMS'] for batch in grad['batches'] for c in batch['per_command'] if c['command']<=1.]
summary=dict(actual_minibatches=4,teacher_lower_label_RMS_action_units_range=[min(data),max(data)],
 running_PPO_parameter_norm_range=[min(x['running_PPO_norm'] for x in grad['batches']),max(x['running_PPO_norm'] for x in grad['batches'])],
 teacher_parameter_norm_range=[min(x['prior_parameter_norm'] for x in grad['batches']),max(x['prior_parameter_norm'] for x in grad['batches'])],
 running_teacher_cosine_range=[min(x['running_teacher_cosine'] for x in grad['batches']),max(x['running_teacher_cosine'] for x in grad['batches'])],
 moving_score_reproduction_cosine_min=min(x['moving_surrogate_derivative_cosine'] for x in grad['batches']),
 inherited_tilted_context_matches_current_neutral_teacher=False,causal_fault_proven=False,
 stopped_hook_subtraction_remains_diagnostic_accounting_uncertainty=True)
(OUT/'gradient_summary.json').write_text(json.dumps(summary,indent=2))
report=f'''# Five-hour humanoid check — October 7, 08:44 UTC

Work in progress. No production source, reward, gain, cap, optimizer or actor change has been deployed.

R40 is healthy but the current3.5m/s stage remains unqualified. The exact initial133 actor travels3.420/3.429 signedm/s with0.599/0.577m/s lateral drift; all320 robots survive each20-second test. Retained1.5 geometric flight is1.181/1.118%, below the unchanged1.5% minimum.3.0 lateral fails both and pushed0.15 net travel fails. The last ten pairs are nonmonotonic, including an earlier0.211/0.188 lateral pair. Preserve useful actors and separately certified349/267; no current qualification inferred from reward or newer partial learning.

At the initial owned snapshot, all55 frozen hashes and one actual CUDA worker verify. R40 retains{a['actual_retained_main_updates']} actual current-run updates, including41 received full-state recovery updates, plus27507 inherited.32 interrupted unsaved R39 updates remain excluded. Rates through the last retained update before this pause are{a['active_PPO_samples_s']/1000:.2f}k active collection+PPO/{a['worker_effective_samples_s']/1000:.2f}k worker-effective/{a['whole_chain_samples_s']/1000:.2f}k whole-chain samples/s. Different denominators; capacity/smoke/diagnostic actors are excluded. Initial production is6144, target8192, reserve300MiB, host guards and180healthy-training-second ramps unchanged.

Actual upper angles/targets/velocities at exact133 were saved and both plots visually inspected. At1m shoulders range39.37/30.78deg and ipsilateral phase-159.8/-166.2deg. At3.5 shoulders range14.20/7.49deg, phases140.9/149.5deg, mean biases-13.63/+4.16deg and headRMS3.61deg. At2.5/3 shoulder bias drifts across seconds; range is not purely cycle swing.32 robots/command are sampled10Hz and16 representatives per gate50Hz, reset-segmented. Arm roll asymmetry/head oscillation persist; no natural-motion certification, calibrated world gaze or speed-matched Unitree encoder comparison.

The newer certified neutral-head teacher349 still receives head histories reconstructed from the older tilted-head walking teacher. Four genuine current-state PPO minibatches reproduce lower effective target differences{min(data):.3f}-{max(data):.3f} action units. Teacher/PPO gradients are not consistently opposed; this is a justified provenance hypothesis, not an established cause. Ten CPU command/gradient ownership fixtures pass. The bounded candidate removes legacy head conditioning only for the exact certified349 moving teacher. Student observations remain truthful. The original.25 MSE/23-output denominator, stopped teacher, upper-only refinement, PPO, full actor/critic/Adam/exploration, native physics/actions/caps and gates stay unchanged. Original reward signatures do not certify the diagnostic training override.

The owned supervisor39000 was suspended at exact139 SHA`{b['checkpoint_sha256']}`, its worker drained, and64/128-update4096env control/candidate tests run sequentially with full checkpoint load/reload and all320robot20s nominal/pushed prior/current checks. Diagnostic actors are excluded from main progress. Existing videos remain fulfilled and unsaved Fusion work is untouched. The same supervisor resumes after the decision unless a rigorously validated versioned continuation is justified.

The first pause bookkeeping attempt encountered a str/Path hashing error after its eval drained; its exception cleanup immediately resumed the same supervisor. The failed state/trace are preserved and the corrected helper repeated the saved-boundary pause. This is diagnostic bookkeeping, not a training failure or lost policy update.

Primary references reviewed: [Ross et al., state-distribution-aware imitation learning](https://proceedings.mlr.press/v15/ross11a.html), [Mirelman et al., arm swing amplitude/symmetry/smoothness](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136043), [official Unitree motion workflow](https://github.com/unitreerobotics/unitree_rl_mjlab). These motivate distribution and feature audits; they do not prove this local adaptation converges or supply a matching joint-angle reference.
'''
(OUT/'REPORT.md').write_text(report,encoding='utf-8');print(json.dumps(summary))
