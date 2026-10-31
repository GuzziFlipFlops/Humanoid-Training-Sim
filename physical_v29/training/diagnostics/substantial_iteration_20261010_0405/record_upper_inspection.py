"""Explicitly inspected original upper plots, separately bound to actor1166."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;TRAIN=OUT.parents[1];FOLDER=OUT/'current_original_upper'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pair=read(OUT/'current_complete_gate_review.json')['latest']
plots=[FOLDER/'upper_joint_angles.png',FOLDER/'upper_joint_velocities.png',FOLDER/'pushed/upper_joint_angles.png',
 FOLDER/'pushed/upper_joint_velocities.png',FOLDER/'upper_bias_decomposition.png']
receipt=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),checkpoint=pair['checkpoint'],sha256=pair['sha256'],
 plots=[dict(path=str(p),sha256=sha(p),visually_inspected=True) for p in plots],
 observations='At1m/s shoulder pitch has large counter-swing and unequal roll/elbow motion. At2.5/3m/s shoulder means drift/reverse over several seconds while small faster motion remains. At3.5m/s opposite held pitch remains near-26/+8deg with much larger left swing; physical targets move farther than actual shoulders. Neck oscillation is gait-linked. Nominal/pushed qualitative patterns agree; no natural-motion or optical-gaze certification.',
 all_command_population_Hz=10,representative_trace_Hz=50,
 plotted_commands=[1.,2.5,3.,3.5],reset_segmentation_preserved=True,
 this_is_original_actor_1166_not_new_candidate_or_anchor_1169=True,
 candidate_128_timeline_inspection_pending_final_check=True)
(FOLDER/'visual_inspection.json').write_text(json.dumps(receipt,indent=2)+'\n')
data=read(FOLDER/'upper_joint_time_audit.json')
head=[dict(test=x['test'],head_rms_deg=x['head_rms_deg'],coordination=x['coordination']) for x in data['records'] if x['command_m_s']==3.5]
with (OUT/'REPORT.md').open('a',encoding='utf-8') as f:
 f.write('\n\n## Current original upper-motion review\n\nFresh fully matching actor1166, before anchor1169, has actual head/shoulder/elbow angle-target-velocity plots in current_original_upper. Five plots were visually inspected; population statistics cover every command. At3.5m/s neck RMS is3.190/3.197deg; opposite held shoulder pitch biases are about-26/+8deg, with unequal swing/roll/elbows. Slow2.5/3m reversals persist. These original-actor observations do not certify a newer candidate. The final check must inspect new128 actors; existing upper refinement remains20 batches/update.\n')
next=read(TRAIN/'automation/NEXT_CHECK.json')
next['substantial_iteration'].update(current_original_upper_QA=str(FOLDER/'visual_inspection.json'),
 current_original_upper_checkpoint=pair['checkpoint'],current_original_upper_sha256=pair['sha256'],
 candidate_upper_QA_pending=True)
temp=TRAIN/'automation/NEXT_CHECK.tmp';temp.write_text(json.dumps(next,indent=2)+'\n');temp.replace(TRAIN/'automation/NEXT_CHECK.json')
args=read(OUT/'automation_update_args.json')
args['prompt']+='\nLatest verified current-iteration setup: candidate completes8 genuine64-env smoke/reload PPO updates, exact actor/critic/Adam/exploration loads, zeroKL/mismatch and all three20-batch auxiliaries. Paired starting observations/actions/rewards/dones/returns/values/raw advantages are bitwise equal; only declared normalized advantages differ. Frozen baseline replay arrays/cases match exactly. Large4096-env two-seed64/128 comparison remains in progress; read actual completion/decision. Current original upper1166 angle/target/velocity/bias QA is separate from anchor1169 and new candidate actors. No diagnostic actor is deployed or speed breakthrough certified. Read the controlled-driver identity if it has not finished; prevent its auto-resume at the FINAL stop. Final check remains09:03UTC with exactlyone prompt, then stop and capture installed motor data/videos as requested.\n'
(OUT/'automation_update_args.json').write_text(json.dumps(args,indent=2)+'\n')
print(json.dumps(dict(inspected_plots=5,checkpoint=pair['checkpoint'],running_head=head),indent=2))
