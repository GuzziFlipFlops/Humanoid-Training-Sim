from pathlib import Path
OUT=Path(__file__).resolve().parent
s=(OUT/'physical_prior_causal.py').read_text().replace('speeds_to_test=[.35,.65,1.,1.5]','speeds_to_test=[.15]')
s=s.replace("out=OUT/'prior_causal_response'","out=OUT/'015_causal_response'")
s=s.replace('prior_causal_probe_failure.json','015_causal_probe_failure.json').replace('640 physical robots','160 physical robots')
(OUT/'physical_015_causal.py').write_text(s)
p=OUT/'resume_after_diagnosis.py';s=p.read_text().replace("'physical_phase_causal.py'","'physical_phase_causal.py','physical_015_causal.py','physical_015_phase_causal.py'")
p.write_text(s)
print('Prepared160-robot .15 controlled probe; no source/actor changes.')
