"""Copy diagnostic harnesses only; never mutate the production frozen source."""
from pathlib import Path
OUT=Path(__file__).resolve().parent
OLD=OUT.parent/'adaptive_check_20261008_1946'
for name in ['diverse_eval.py','validate_physical_replay.py','motor_retention_probe.py']:
    text=(OLD/name).read_text()
    if name=='validate_physical_replay.py':text=text.replace('matched_focused_prior','matched_slow_reward')
    if name=='motor_retention_probe.py':
        text=text.replace('4291,4311','4491,4511')
        text=text.replace('[.15,1.5,3.,3.5] if actor==\'current\' else [.15,1.5,3.]',
            '[.15,.35,.65,1.,1.5,3.,3.5] if actor==\'current\' else [.15,.35,.65,1.,1.5,3.]')
        text=text.replace('448 physical robots','832 physical robots')
        text=text.replace("groups=np.array([g[0] for g in groups]),robots_per_group=per)",
            "groups=np.array([g[0] for g in groups]),robots_per_group=per)")
    (OUT/name).write_text(text)
text=(OUT/'prepare_and_pause.py').read_text().replace(
    'Bounded qualified-teacher body/current-upper compatibility on truthful settled/transition feedback; diagnostic hybrids never deployed',
    'Bounded slow-command reward precision experiment on the current exact full state versus unchanged control; all gates, weights and native limits retained')
(OUT/'prepare_and_pause.py').write_text(text)
text=(OUT/'resume_after_diagnosis.py').read_text().replace("'teacher_body_probe.py'}", "'teacher_body_probe.py','slow_reward_candidate.py'}")
(OUT/'resume_after_diagnosis.py').write_text(text)
text=(OLD/'run_focused_matched.py').read_text()
text=text.replace('matched_focused_prior','matched_slow_reward').replace('WORK_LIMIT=6000','WORK_LIMIT=6600')
text=text.replace("'focused_prior_candidate.py'", "'slow_reward_candidate.py'")
text=text.replace("wrapper=OUT/'focused_prior_candidate.py'", "wrapper=OUT/'slow_reward_candidate.py'")
text=text.replace('--focused-prior-rehearsal','--slow-tracking-tolerance')
text=text.replace('experimental_exposure.json','experimental_reward.json')
text=text.replace("experimental_source_sha256=sha(OUT/'focused_prior_sampler.py')", "experimental_source_sha256=sha(OUT/'slow_tracking_kernel.py')")
text=text.replace('original_signature_does_not_certify_changed_training_sampler=True,original_physics_rewards_caps_teachers_upper_unchanged=True,',
    'original_signature_does_not_certify_changed_training_reward=True,original_physics_caps_teachers_upper_unchanged=True,')
text=text.replace('tested_commands=[.15,1.5],expected_fractions_before=[.03,.03],expected_fractions_after=[.048,.048],\n        standing_fraction=.32,frontier_fraction=.44,prior_fraction=.24,other_prior_fraction_after=.024,',
    'changed_commands="0.1<speed<=1.0",old_tracking_std=.5,new_tracking_std="max(.10,.12*speed)",\n        unchanged_sampler=True,unchanged_all_weights=True,unchanged_faster_reward=True,')
text=text.replace('training_seeds=[157,167]','training_seeds=[179,191]').replace('independent_initial_state_seeds=[4231,4251]','independent_initial_state_seeds=[4531,4551]')
text=text.replace("hypothesis='Reallocate within existing prior bucket to reproduced persistent failures. Exact rehearsal already exists; no absence or causality is inferred.'",
    "hypothesis='Derive slow Gaussian error width from existing classifier tolerances. Test precise retained tracking without weakening faster gait. Instantaneous gait oscillations are a measured risk, not a proof of repair.'")
text=text.replace("'--seed','157'", "'--seed','179'").replace('for seed in [157,167]:','for seed in [179,191]:')
text=text.replace("('diverse_nominal',4231,False),('diverse_push',4251,True)", "('diverse_nominal',4531,False),('diverse_push',4551,True)")
# Leave physical evidence and the small real PPO/load checks before learning.
text=text.replace("launch([str(OUT/'motor_retention_probe.py'),'--headless','--device','cuda:0'],OUT/'motor_retention_response',480)",
    "launch([str(OUT/'motor_retention_probe.py'),'--headless','--device','cuda:0'],OUT/'motor_retention_response',600)\n    subprocess.run([sys.executable,'-X','utf8',str(OUT/'analyze_physical_reward.py')],check=True)\n    analysis=read(OUT/'physical_reward_counterfactual.json')\n    assert analysis['safe_to_test_learning'], 'Physical counterfactual rejected this width before learning; keep original learner'")
(OUT/'run_reward_matched.py').write_text(text)
print('Prepared isolated reward experiment and guaranteed same-supervisor resume.')
