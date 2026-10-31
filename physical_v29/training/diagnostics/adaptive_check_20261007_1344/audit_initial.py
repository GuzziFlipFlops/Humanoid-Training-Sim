import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
v=read(OUT/'initial_learning_verification.json');pause=read(OUT/'diagnostic_pause.json')
last=v['blocks'][-1]
print(json.dumps({'retained_current':v['actual_retained_main_updates'],'completed':v['completed_main_updates'],'recovered':v['hash_proven_recovered_main_updates'],
 'rates':[v[k] for k in ['active_PPO_samples_s','worker_effective_samples_s','whole_chain_samples_s']],
 'documented_pause_seconds_excluded':v['documented_diagnostic_pause_seconds_excluded'],
 'last_main':{k:last[k] for k in ['folder','updates','environments','minimum_free_gpu_MiB','minimum_host_available_MiB','private_memory_growth_MiB','torch_allocation_growth_MiB']},
 'paused_checkpoint':pause['checkpoint'],'sha256':pause['checkpoint_sha256']},indent=2))
qa={'recorded_utc':datetime.now(timezone.utc).isoformat(),'files':[],'human_Unitree_likeness_certified':False,'world_optical_gaze_calibrated':False,
 'findings':'Initial251 angle/velocity panels inspected:1m unequal arm swing and elbow flexion;2.5 large persistent opposite shoulder-pitch biases;3m bias slowly drifts;3.5 small right versus larger left fast pitch swing,roll asymmetry and gait-linked neck oscillations. Target excursions exceed measured fast arm motion. Population10Hz cannot certify high-frequency jitter.'}
for name in ['upper_joint_angles.png','upper_joint_velocities.png']:
 p=OUT/'initial_upper'/name;qa['files'].append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'visually_inspected':True})
(OUT/'plot_qa.json').write_text(json.dumps(qa,indent=2)+'\n')
(OUT/'heading_diagnosis_scope.json').write_text(json.dumps({'actual_scope':'Truthful heading gain/course/yaw physical diagnosis on exact254,not teacher-context learning.','preserved_pause_label':'The already-compiled copied pause helper retained an older descriptive label while its source label was updated. Functional pause/drain/checkpoint/source behavior is unchanged.','supervisor_same':True,'no_production_source_change':True},indent=2)+'\n')
