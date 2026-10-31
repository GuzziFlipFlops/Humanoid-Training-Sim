from pathlib import Path
OUT=Path(__file__).resolve().parent;OLD=OUT.parent/'adaptive_check_20261008_1946'
text=(OUT/'analyze_current_gradients.py').read_text().replace('bounded_diagnosis/passive_PPO_capture','matched_slow_reward/passive_PPO_capture')
(OUT/'analyze_current_gradients.py').write_text(text)
(OUT/'attribute_motor_substeps.py').write_text((OLD/'attribute_motor_substeps.py').read_text())
print('Prepared current actual PPO and native CAD saved-data attribution.')
