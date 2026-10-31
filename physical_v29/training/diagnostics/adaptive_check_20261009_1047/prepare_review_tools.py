"""Adapt existing CPU-only evidence readers to the new declared reward test."""
from pathlib import Path
OUT=Path(__file__).resolve().parent;OLD=OUT.parent/'adaptive_check_20261008_1946'
for name,dest in [('review_focused_experiment.py','review_reward_experiment.py'),('monitor_diagnosis.py','monitor_diagnosis.py')]:
    text=(OLD/name).read_text().replace('matched_focused_prior','matched_slow_reward')
    text=text.replace('[157,167]','[179,191]').replace('matched_focused_review.json','matched_reward_review.json')
    text=text.replace('experimental_exposure.json','experimental_reward.json').replace('executed_sampler_exposure','executed_reward_override')
    text=text.replace('experimental_sampler_not_certified_by_original_signature','experimental_reward_not_certified_by_original_signature')
    text=text.replace('physics_rewards_caps_original_teachers_and_upper_unchanged','physics_caps_original_teachers_and_upper_unchanged')
    text=text.replace('focused_prior_candidate.py','slow_reward_candidate.py')
    (OUT/dest).write_text(text)
print('Prepared CPU comparison and liveness review.')
