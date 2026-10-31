from pathlib import Path
OUT=Path(__file__).resolve().parent;previous=OUT.parent/'adaptive_check_20261005_0841'
text=(previous/'pause_for_diagnosis.py').read_text().replace('locomotion_20261004_r35_upper_body_refinement','locomotion_20261005_r36_posture_conditioned_priors')
text=text.replace('Bounded normal-limit phase/objective/gradient and full-CAD frame diagnosis','Bounded normal-limit take-off timing and full-objective diagnosis')
(OUT/'pause_for_diagnosis.py').write_text(text)
text=(previous/'resume_after_diagnosis.py').read_text()
start=text.index("['physical_prior_causal.py'");end=text.index('] for x in a)',start)
text=text[:start]+"['physical_takeoff_probe.py','worker_unitree.py'"+text[end:]
(OUT/'resume_after_diagnosis.py').write_text(text)
print('Prepared owned saved-boundary pause/resume; production source unchanged.')
