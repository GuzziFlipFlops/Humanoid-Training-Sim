import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;TRAINING=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
live=read(OUT/'liveness_cpu/live_handoff.json');learn=read(OUT/'liveness_learning_verification.json')
review=read(OUT/'liveness_gate_review.json');pair=review['latest'];qa=read(OUT/'plot_qa.json')
u=next(r for r in read(OUT/'closing_newer_upper/upper_joint_time_audit.json')['records'] if r['actor_tag']=='latest' and r['test']=='nominal' and r['command_m_s']==3.5)
assert live['state']['status']=='running' and len(live['actual_worker_pids'])==1 and all(live['frozen_source_integrity'].values())
assert learn['actual_retained_main_updates']==learn['completed_main_updates']+learn['hash_proven_recovered_main_updates']
assert u['checkpoint']==pair['checkpoint']
for name in ['upper_joint_angles.png','upper_joint_velocities.png','upper_bias_decomposition.png']:
    f=OUT/'closing_newer_upper'/name;qa['files'].append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),visually_inspected=True))
qa['closing_checkpoint']=pair['checkpoint'];qa['closing_observations']='344 timelines retain unequal fast pitch swing,held opposite biases at2.5m/s,slow3m/s bias changes,roll/elbow imbalance and gait-linked neck motion. All3 newer plots inspected;no natural/Unitree/optical-gaze certification.'
write(OUT/'plot_qa.json',qa)
c=[pair[k][-1] for k in ['nominal_cases','pushed_cases']]
text=f"Closing owned evidence {live['recorded_utc']}: the same R40 supervisor 39000 has one actual worker and all 55 hashes verified. It retains {learn['completed_main_updates']} completed + {learn['hash_proven_recovered_main_updates']} recovered = {learn['actual_retained_main_updates']} new main updates, plus 27,507 inherited. Exact actor {Path(pair['checkpoint']).parent.name} reaches {c[0]['signed_speed']:.3f}/{c[1]['signed_speed']:.3f} m/s signed at command3.5, but lateral {c[0]['lateral_speed']:.3f}/{c[1]['lateral_speed']:.3f} fails both direction tests. Both0.15 path-efficiency and1.5 geometric-flight tests fail; pushed3.0 lateral regresses. All survive. The earlier338 pair's3.0 recovery is preserved and is not attributed to344."
upper=f"The newer344 angle/target,velocity and slow-bias plots are also visually inspected. At3.5,head RMS{u['head_rms_deg']:.2f}deg and shoulder pitch ranges{u['joints']['left_shoulder_pitch']['median_per_robot_p95_p05_range_deg']:.2f}/{u['joints']['right_shoulder_pitch']['median_per_robot_p95_p05_range_deg']:.2f}deg. Fast asymmetry,held biases and gait-linked head oscillation persist;these diagnostics do not certify human/Unitree motion or optical gaze."
rates=[learn[k]/1000 for k in ['active_PPO_samples_s','worker_effective_samples_s','whole_chain_samples_s']]
timing=f"Fresh completed-main rates are {rates[0]:.2f}k active/{rates[1]:.2f}k worker/{rates[2]:.2f}k whole-chain samples/s, with {learn['documented_diagnostic_pause_seconds_excluded']:.1f}s documented suspensions excluded. Resumed341 loads exact338 actor/critic/Adam/exploration;344 also loads its predecessor exactly. All KL/command checks and three20-batch auxiliaries pass. Last6144 block minimumGPU{learn['blocks'][-1]['minimum_free_gpu_MiB']}MiB/host{learn['blocks'][-1]['minimum_host_available_MiB']}MiB,private+{learn['blocks'][-1]['private_memory_growth_MiB']}MiB,Torch allocation unchanged."
paragraph=text+' '+upper+' '+timing
write(OUT/'closing_liveness.json',dict(recorded_utc=datetime.now(timezone.utc).isoformat(),observation=live['recorded_utc'],summary=paragraph,
    checkpoint=pair['checkpoint'],sha256=pair['sha256'],current_stage_qualified=False,same_original_learner=True,no_source_or_actor_reset=True))
report=OUT/'REPORT.md';r=report.read_text(encoding='utf-8');r=r.replace('# Five-hour check — October7,18:45 UTC\n','# Five-hour check — October7,18:45 UTC\n\n'+paragraph+'\n\n[Newer exact gates](liveness_gate_review.json),[newer upper plots/data](closing_newer_upper/upper_joint_time_audit.json) and[resumed learning](liveness_learning_verification.json). The338 review below is the earlier saved diagnostic boundary.\n');report.write_text(r,encoding='utf-8')
marker='adaptive_check_20261007_1845_closing'
entry=f"\n<!-- {marker} -->\n### Closing exact evidence, {live['recorded_utc']}\n\n{paragraph}\n\nReport:{report}. Newer gates:{OUT/'liveness_gate_review.json'}. Newer upper:{OUT/'closing_newer_upper/upper_joint_time_audit.json'}. No production change; same ACTIVE5h scope/verified5 or conditional4 endpoint.\n<!-- /{marker} -->\n"
for f in [TRAINING/'ITERATION_HISTORY.md',Path('E:/Aditya/robot_sim/docs/robot/TRAINING_JOURNAL_20261001.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
    assert marker not in f.read_text(encoding='utf-8-sig')
    with f.open('a',encoding='utf-8') as stream:stream.write(entry)
for f in [Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]:
    old=f.read_text(encoding='utf-8-sig');assert marker not in old
    f.write_text(f"<!-- {marker} -->\n{paragraph} Read{report} and freshACTIVE_RUN/state. No production change;same ACTIVE5h endpoint. Earlier entries preserved below.\n<!-- /{marker} -->\n\n"+old,encoding='utf-8')
npath=TRAINING/'automation/NEXT_CHECK.json';n=read(npath)
n.update(recorded_utc=live['recorded_utc'],latest_observation=str(OUT/'liveness_cpu/live_handoff.json'),new_main_updates=learn['completed_main_updates'],
    actual_retained_current_run_main_updates=learn['actual_retained_main_updates'],selected_actor=dict(checkpoint=pair['checkpoint'],sha256=pair['sha256'],matching_gate_pair=str(OUT/'liveness_gate_review.json'),current_stage_qualified=False),
    throughput={k:learn[k] for k in ['active_PPO_samples_s','worker_effective_samples_s','whole_chain_samples_s']},
    mandatory_upper_audit=str(OUT/'closing_newer_upper/upper_joint_time_audit.json'),mandatory_upper_time_series=str(OUT/'closing_newer_upper/upper_joint_time_audit.json'),mandatory_upper_bias_audit=str(OUT/'closing_newer_upper/upper_bias_audit.json'),
    newer_matching_pair=str(OUT/'liveness_gate_review.json'),newer_gate_trend=text,
    latest_saved_actor_unqualified=dict(checkpoint=pair['checkpoint'],scope='Newest complete reviewed pair;newer partial actors unqualified'))
write(npath,n)
args=read(OUT/'automation_update_args.json');args['prompt']+='\n\n'+paragraph+' Read closing_liveness.json,liveness_gate_review.json,closing_newer_upper plots/data and liveness_learning_verification.json. The next check must use freshACTIVE_RUN/state;all priorities and durable scope remain.'
write(OUT/'automation_update_args.json',args)
print(paragraph)
