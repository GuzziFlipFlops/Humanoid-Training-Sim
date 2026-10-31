import argparse,json
from pathlib import Path
OUT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args()
u=json.loads((OUT/a.folder/'upper_joint_time_audit.json').read_text());b=json.loads((OUT/a.folder/'upper_bias_audit.json').read_text())
rows=[]
for r in u['records']:
 if r['actor_tag']!='latest' or r['test']!='nominal' or r['command_m_s'] not in [0.15,1.5,2.5,3.5]:continue
 bias=next(t for t in b['records'] if t['test']==r['test'] and t['speed_m_s']==r['command_m_s'])
 names=['left_shoulder_pitch','right_shoulder_pitch']
 rows.append(dict(command=r['command_m_s'],head_RMS=r['head_rms_deg'],range=[r['joints'][n]['median_per_robot_p95_p05_range_deg'] for n in names],
    slow_mean=[bias['joints'][n]['slow_bias_mean_deg'] for n in names],coordination=r['coordination']))
print(json.dumps(rows,indent=2))
