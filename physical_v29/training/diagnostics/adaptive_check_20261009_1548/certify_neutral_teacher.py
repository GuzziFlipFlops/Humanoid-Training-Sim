"""Strict CPU certificate construction without importing the Isaac runtime."""
import ast, hashlib, json, sys
from pathlib import Path
OUT=Path(__file__).resolve().parent
TRAIN=OUT/'control_source/physical_v29/training'
sys.path[:0]=[str(TRAIN),str(TRAIN.parent)]
from standing_retention import qualification_certificate,validate_qualification
from curriculum_provenance import training_reward_signature
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
module=ast.parse((TRAIN/'evaluation.py').read_text())
values=[ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='GATE_VERSION' for t in n.targets)]
assert len(values)==1 and isinstance(values[0],str)
gate_version=values[0]
ROOT=OUT/'neutral_requalification'
teacher=Path(read(OUT/'teacher_diagnosis_preparation.json')['teacher_checkpoint'])
cert=qualification_certificate(ROOT,str(teacher),3.,training_reward_signature(TRAIN),gate_version)
assert validate_qualification(cert,str(teacher),training_reward_signature(TRAIN),gate_version)==3.
(OUT/'neutral_teacher_certificate.json').write_text(json.dumps(cert,indent=2)+'\n')
pair=dict(checkpoint=str(teacher),sha256=sha(teacher),nominal=str(ROOT/'neutral_gate_nominal'),pushed=str(ROOT/'neutral_gate_push'))
for test in ['nominal','pushed']:
    ev=read(Path(pair[test])/'evaluation.json');pair[test+'_cases']=[]
    for c in ev['cases']:
        z=c['statistics'];pair[test+'_cases'].append(dict(command=z['command_m_s'],passed=c['passed'],failures=c['failures'],
            trials=z['trials'],survival=z['survival'],signed_speed=z['reset_frame_forward_speed_m_s'],lateral_speed=z['reset_frame_lateral_speed_m_s'],
            flight=z['geometric_flight_fraction'],statistics=z))
    pair[test+'_all_pass']=ev['passed']
(OUT/'neutral_teacher_gate_review.json').write_text(json.dumps(dict(latest=pair,promotions=[],run=read(OUT/'diagnostic_pause.json')['run']),indent=2)+'\n')
(OUT/'certificate_reader_repair.json').write_text(json.dumps(dict(passed=True,scope='CPU diagnostic import repair only',
    gate_version=gate_version,gate_version_source=str(TRAIN/'evaluation.py'),gate_version_source_sha256=sha(TRAIN/'evaluation.py'),
    original_frozen_validation_used=True,no_gate_or_signature_override=True,failed_attempt_preserved=True),indent=2)+'\n')
print(json.dumps(dict(all_prior_nominal=pair['nominal_all_pass'],all_prior_pushed=pair['pushed_all_pass'],
    cases={t:[{k:c[k] for k in ['command','passed','signed_speed','flight']} for c in pair[t+'_cases']] for t in ['nominal','pushed']})))
