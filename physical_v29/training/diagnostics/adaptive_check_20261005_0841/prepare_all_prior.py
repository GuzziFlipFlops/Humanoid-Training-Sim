import json,hashlib,shutil
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
context=json.loads((OUT/'phase_context.json').read_text());data=np.load(OUT/'015_causal_response/telemetry.npz')
obs=data['observations'][data['time']>=5.,32:64].reshape(-1,5,80)[:,-1]
phase=np.remainder(np.arctan2(obs[:,78],obs[:,79])/(2*np.pi),1.);index=np.floor(phase*64).astype(int)%64
table=np.full((64,6),np.nan,np.float32)
for b in range(64):
    rows=index==b
    if rows.any():table[b]=obs[rows][:,context['columns']].mean(0)
populated=np.flatnonzero(np.isfinite(table).all(1));assert len(populated)>=30
x=np.r_[populated-64,populated,populated+64]
for j in range(6):table[:,j]=np.interp(np.arange(64),x,np.tile(table[populated,j],3))
context['contexts']['0.15']=table.tolist();context['additional_015_source']=str(OUT/'015_causal_response/telemetry.npz')
context['additional_015_source_sha256']=hashlib.sha256((OUT/'015_causal_response/telemetry.npz').read_bytes()).hexdigest()
(OUT/'phase_context_all.json').write_text(json.dumps(context,indent=2)+'\n')
text=(OUT/'physical_phase_causal.py').read_text().replace('speeds_to_test=[.35,.65,1.,1.5]','speeds_to_test=[.15]')
text=text.replace("out=OUT/'phase_causal_response'","out=OUT/'015_phase_causal_response'")
text=text.replace("(OUT/'phase_context.json')","(OUT/'phase_context_all.json')")
text=text.replace('phase_causal_probe_failure.json','015_phase_causal_probe_failure.json').replace('640 physical robots','160 physical robots').replace('.35/.65/1./1.5 commands','.15m/s command')
(OUT/'physical_015_phase_causal.py').write_text(text)
source=Path(json.loads((OUT/'diagnostic_pause.json').read_text())['run'])/'source/physical_v29'
target=OUT/'all_prior_source/physical_v29';assert not target.exists()
shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc','usd'))
helper=(OUT/'phase_transfer.py').read_text().split('class RunningRowTransfer:')[0]
helper=helper[helper.index('import torch'):].replace("VERSION='diagnostic_neutral_head_running_rows_v1'","VERSION='teacher_head_phase_context_v1'")
(target/'training/teacher_head_context.py').write_text('"""Training-only teacher context; student/inference inputs are untouched."""\n'+helper)
shutil.copy2(OUT/'phase_context_all.json',target/'training/teacher_head_context.json')
text=(OUT/'prior_context_source/physical_v29/training/upper_body_refinement.py').read_text()
text=text.replace('from phase_transfer import context_inputs','from teacher_head_context import context_inputs').replace("'phase_context.json'","'teacher_head_context.json'")
text=text.replace("UPPER_REFINEMENT_VERSION='upper_output_posture_v1'","UPPER_REFINEMENT_VERSION='upper_output_posture_phase_prior_v2'")
text=text.replace('Original .25 derivative/denominator on lower outputs; zero upper copying.', 'Retain .25 MSE formula/denominator; declare phase-conditioned prior targets.')
text=text.replace("output_scope='Original direct torso/leg derivative, zero head/arm copying'","output_scope='Unchanged .25 lower MSE formula; declared conditioned prior targets; zero head/arm copying'")
text=text.replace("        result['teacher_phase_conditioned_samples']=", "        result['teacher_context_version']='teacher_head_phase_context_v1'\n        result['teacher_context_commands']=sorted(self.phase_tables)\n        result['teacher_phase_conditioned_samples']=")
(target/'training/upper_body_refinement.py').write_text(text)
cpu=(OUT/'cpu_prior_context_tests.py').read_text().replace("TRAIN=OUT/'prior_context_source/physical_v29/training'","TRAIN=OUT/'all_prior_source/physical_v29/training'")
cpu=cpu.replace('from phase_transfer import context_inputs','from teacher_head_context import context_inputs').replace("'phase_context.json'","'teacher_head_context.json'")
cpu=cpu.replace('eligible=command in [.35,.65,1.]','eligible=command in [.15,.35,.65,1.]')
cpu=cpu.replace('standing_and_015_bitwise_gradient_equal=True','standing_bitwise_gradient_equal=True')
cpu=cpu.replace('cpu_prior_context_tests.json','cpu_all_prior_tests.json');(OUT/'cpu_all_prior_tests.py').write_text(cpu)
script=(OUT/'run_matched_transfer.py').read_text().split('records={}')[0]
script=script.replace("'candidate':OUT/'validation_source/physical_v29'", "'candidate':OUT/'all_prior_source/physical_v29'")
script+=(OUT/'run_matched_prior_context.py').read_text().split('records={}')[1]
script=script.replace("folder,result=worker", "records={}\nfolder,result=worker",1)
script=script.replace('matched_prior_context','matched_all_prior').replace('prior_context_full_reload','all_prior_full_reload')
(OUT/'run_matched_all_prior.py').write_text(script)
resume=OUT/'resume_after_diagnosis.py';text=resume.read_text().replace("'run_matched_prior_context.py'", "'run_matched_prior_context.py','run_matched_all_prior.py'");resume.write_text(text)
print('All-prior candidate changes existing teacher targets only; no running-imitation hook.')
