"""Compare measured population motion, preserving exact actor and sampling scope."""
import json
from datetime import datetime, timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
records=[]
for seed in [229,241]:
    for branch in ['control','candidate']:
        folder=OUT/f'seed{seed}_{branch}_upper'
        if not (folder/'upper_target_bias_audit.json').exists():continue
        upper=read(folder/'upper_joint_time_audit.json');bias=read(folder/'upper_bias_audit.json');target=read(folder/'upper_target_bias_audit.json')
        for u in upper['records']:
            speed=u['command_m_s'];test=u['test']
            b=next(r for r in bias['records'] if r['test']==test and abs(r['speed_m_s']-speed)<1e-5)
            t=next((r for r in target['records'] if r['test']==test and abs(r['speed']-speed)<1e-5),None)
            records.append(dict(seed=seed,branch=branch,checkpoint=u['checkpoint'],sha256=u['sha256'],test=test,speed=speed,
                head_RMS_deg=u['head_rms_deg'],head_p95_deg=u['head_p95_deg'],
                actual_joints=u['joints'],coordination=u['coordination'],representative=u['representative_50hz'],
                actual_slow_bias=b['joints'],physical_target_response=t,
                population_sampling_Hz=10,representative_sampling_Hz=50,reset_segments_preserved=True))
complete=len(records)==80
result=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),all_four128_actors_complete=complete,
    records=records,scope='32 robots per command at10Hz;16 selected representatives per gate at50Hz. Focal plots are0.15/1.5/2.5/3.5; all10commands have nominal/pushed population statistics. 0.5Hz slow separation excludes3second edges. Phase must be assessed alongside amplitude. Different actor/seed results never qualify newer main learning.',
    human_Unitree_or_optical_gaze_certified=False,production_upper_gain_or_refinement_changed=False)
(OUT/'upper_matched_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(complete=complete,records=len(records),fast=[dict(seed=r['seed'],branch=r['branch'],test=r['test'],head_RMS=r['head_RMS_deg'],
    shoulder_range=[r['actual_joints'][n]['median_per_robot_p95_p05_range_deg'] for n in ['left_shoulder_pitch','right_shoulder_pitch']],
    slow_mean=[r['actual_slow_bias'][n]['slow_bias_mean_deg'] for n in ['left_shoulder_pitch','right_shoulder_pitch']]) for r in records if r['speed']==3.5]),indent=2))
