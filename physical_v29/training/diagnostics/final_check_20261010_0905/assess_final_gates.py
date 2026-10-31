import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;T=OUT.parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
write=lambda p,d:p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
now=datetime.now(timezone.utc).isoformat()
assessment={'recorded_utc':now,'stop_reason':'User ends this attempt after the single final scheduled check','qualified_speed_breakthrough':False,'qualified_5_or_4_endpoint':False,'causal_actuator_ceiling_established':False,'learning_after_stop':False,'comparisons':[]}
for scope in ['latest','selected']:
    review=read(OUT/(scope+'_gate_review.json'))['latest']
    for test in ['nominal','pushed']:
        path=Path(review[test])/'evaluation.json'
        ev=read(path)
        assessment['comparisons'].append(dict(scope=scope,test='ordinary_'+test,checkpoint=review['checkpoint'],sha256=review['sha256'],source=str(path),full_gate_passed=ev['passed'],cases=[dict(command=c['statistics']['command_m_s'],signed_speed=c['statistics']['reset_frame_forward_speed_m_s'],world_lateral=c['statistics']['reset_frame_lateral_speed_m_s'],flight=c['statistics']['geometric_flight_fraction'],survival=c['statistics']['survival'],passed=c['passed'],failures=c['failures']) for c in ev['cases']]))
    for test in ['nominal','pushed']:
        folder=OUT/(scope+'_diverse_'+test);ev=read(folder/'evaluation.json');receipt=read(folder/'completion_receipt.json');diversity=read(folder/'initial_diversity.json')
        assessment['comparisons'].append(dict(scope=scope,test='independently_seeded_'+test,checkpoint=receipt['checkpoint'],sha256=receipt['sha256'],seed=receipt['seed'],source=str(folder/'evaluation.json'),diversity=diversity,full_gate_passed=ev['passed'],cases=[dict(command=c['statistics']['command_m_s'],signed_speed=c['statistics']['reset_frame_forward_speed_m_s'],world_lateral=c['statistics']['reset_frame_lateral_speed_m_s'],flight=c['statistics']['geometric_flight_fraction'],survival=c['statistics']['survival'],passed=c['passed'],failures=c['failures']) for c in ev['cases']]))
assessment['preserved_iteration']=read(T/'diagnostics/substantial_iteration_20261010_0405/decision.json')
assessment['interpretation']='Current stage remains blocked by simultaneous direction and retention. Some exact actors pass 3.5 current cases, but no same actor passes all earlier cases. Recent higher signed speed without direction/retention is a smaller measurement improvement, not breakthrough. Earlier 4/4.5/5 extrapolation failed gait/direction, with no causal motor ceiling.'
write(OUT/'gate_assessment.json',assessment)
# Every listed PNG was actually visually inspected in this final turn.
paths=[]
for scope in ['latest','selected']:
 for test in ['nominal','pushed']:
  for band in ['slow','middle','fast']:
   paths.extend([OUT/(scope+'_upper')/test/band/f for f in ['upper_joint_angles.png','upper_joint_velocities.png']])
  paths.append(OUT/(scope+'_upper')/test/'bias/upper_bias_decomposition.png')
qa=dict(recorded_utc=now,visually_inspected=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],observations='Held unequal shoulder pitch, unequal roll/elbow motion, slow reversals at2.5/3m/s and gait-linked neck oscillation persist. Native head zero is torso-relative. Full population angle/cadence/phase statistics are separate from representative50Hz traces. No natural-motion, mapped Unitree-encoder or optical-gaze certification.',bias_scope='upper_bias_audit.py plots nominal data in both nominal/pushed folder locations. Pushed-folder bias duplicates are explicitly nominal; pushed angle/velocity and population statistics were independently inspected.',population_hz=10,representative_hz=50)
write(OUT/'upper_visual_QA.json',qa)
R41=T/'diagnostics/substantial_iteration_20261010_0405'
paths=[]
for seed in [347,359]:
 for branch in ['control','candidate']:
  folder=R41/f'upper_seed{seed}_{branch}_128'
  paths.extend(folder/p for p in ['upper_joint_angles.png','upper_joint_velocities.png','upper_bias_decomposition.png','pushed/upper_joint_angles.png','pushed/upper_joint_velocities.png'])
write(R41/'final_upper_visual_QA.json',dict(recorded_utc=now,visually_inspected=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],observations='Both seeds and routes retain held fast shoulder biases and unequal pitch/roll/elbow cycles. Candidate347 has larger negative head-yaw excursions in these representative traces; improvements elsewhere do not establish natural motion or full locomotion retention.',scope='20 actual angle/target/velocity/bias images; representative traces do not certify full-population fast jitter.'))
d=read(R41/'decision.json');d['upper_visual_QA_pending']=False;d['final_upper_visual_QA_receipt']=str(R41/'final_upper_visual_QA.json');write(R41/'decision.json',d)
print(json.dumps([dict(scope=x['scope'],test=x['test'],full_gate_passed=x['full_gate_passed'],fast=x['cases'][-1],failed=[{'command':c['command'],'failures':c['failures']} for c in x['cases'] if not c['passed']]) for x in assessment['comparisons']],indent=2))
