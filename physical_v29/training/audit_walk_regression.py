"""Read-only CPU audit of R23's lost walking/standing behavior."""
from pathlib import Path
import json, statistics, hashlib
import numpy as np
from PIL import Image, ImageDraw
from build_unitree_review import fk, pose
import xml.etree.ElementTree as ET

HERE = Path(__file__).parent
RUN = HERE / 'runs/locomotion_20261002_r23_standing_curriculum'
OUT = HERE / 'diagnostics/walk_regression_20261003'

def main():
    OUT.mkdir(exist_ok=True)
    gates=[];training=[];traces=[]
    for p in sorted(RUN.glob('*walk*gate*/evaluation.json')):
        g=json.loads(p.read_text())
        gates.append(dict(folder=p.parent.name,seed=g['seed'],perturb=g['perturb'],
            cases=[dict(command=c['statistics']['command_m_s'],passed=c['passed'],
                survival=c['statistics']['survival'],forward=c['statistics']['forward_speed_m_s'],
                steps_left=c['statistics']['steps_left'],steps_right=c['statistics']['steps_right'],
                failures=c['failures']) for c in g['cases']]))
    for p in sorted(RUN.glob('*train*/progress.jsonl')):
        rows=[json.loads(x) for x in p.read_text().splitlines()]
        if not rows:continue
        b=rows[-1]['behavior'];pol=rows[-1]['policy']
        training.append(dict(folder=p.parent.name,updates=len(rows),speed=rows[-1]['speed'],
            mean_fps=statistics.mean(x['num_envs']*24/(x['collect_seconds']+x['learn_seconds']) for x in rows),
            falls=b['falls_per_s'],standing_falls=b['stand_falls_per_s'],forward=b['moving_forward_speed'],
            healthy=b['healthy_fraction'],clearance_reward=b['reward_feet_clearance'],
            gait_reward=b['reward_gait'],raw_cost=b['shaping_raw_cost'],activity=b['shaping_activity'],
            learning_rate=pol['learning_rate'],mean_std=statistics.mean(pol['exploration_std']),
            action_p99=pol['normalized_observation_abs_p99']['action_history'],
            nominal_pose_drift=pol['nominal_pose_max_abs_drift_rad']))
    tree=ET.parse(RUN/'source/physical_v29/training/assets/robot_training.urdf').getroot()
    for folder in ['050_walk_035_gate_001_nominal','068_walk_035_gate_004_nominal','086_walk_035_gate_007_nominal']:
        with np.load(RUN/folder/'gait_diagnostics_50hz.npz') as z:
            d={k:z[k] for k in z.files}
        names=d['joint_names'].tolist()
        for robot in [0,2]:
            done=np.flatnonzero(d['done'][:,robot]);end=int(done[0]) if len(done) else len(d['time'])
            if not end:continue
            indices=np.unique(np.linspace(0,end-1,12).astype(int))
            sheet=Image.new('RGB',(1200,1000),'white');draw=ImageDraw.Draw(sheet)
            for k,i in enumerate(indices):
                x0=(k%3)*400;y0=(k//3)*250
                state=d['root_state'][i,robot];transforms,segments=fk(tree,dict(zip(names,d['q'][i,robot])),state)
                start=d['root_state'][0,robot,:3]
                def project(p):return (x0+200+int((p[0]-start[0])*150),y0+225-int(p[2]*128))
                draw.text((x0+5,y0+5),f't={d["time"][i]:.2f}s  vx={state[7]:.2f} h={state[2]:.2f}',fill='black')
                draw.text((x0+5,y0+21),'CPU FK, not a simulator screenshot',fill='black')
                draw.line([(x0+5,y0+225),(x0+395,y0+225)],fill='gray')
                for a,b,u,v in segments:
                    color='#176f9c' if b.startswith('left') else '#b05416' if b.startswith('right') else '#424242'
                    draw.line([project(u),project(v)],fill=color,width=4)
                for side in ['left','right']:
                    c=tree.find(f"link[@name='{side}_foot']/collision");size=np.fromstring(c.find('geometry/box').get('size'),sep=' ')
                    t=transforms[side+'_foot']@pose(c.find('origin'))
                    pts=[project((t@np.r_[np.array([x,y,-1])*size/2,1])[:3]) for x,y in [(-1,-1),(-1,1),(1,1),(1,-1)]]
                    draw.line(pts+[pts[0]],fill='black',width=2)
            image=f'{folder}_robot{robot}.png';sheet.save(OUT/image)
            q=d['q'][:end,robot];a=d['raw_action'][:end,robot]
            reason=d['failure_names'][d['failure_causes'][end,robot]].tolist() if len(done) else []
            traces.append(dict(folder=folder,robot=robot,command=float(d['commands'][robot]),
                first_failure_s=float(d['time'][end]) if len(done) else None,failure_causes=reason,
                first_height=float(d['root_state'][0,robot,2]),last_pre_failure_height=float(d['root_state'][end-1,robot,2]),
                raw_action_p99=float(np.quantile(np.abs(a),.99)),clipped_fraction=float((np.abs(a)>3).mean()),
                joint_ranges={n:[float(q[:,j].min()),float(q[:,j].max())] for j,n in enumerate(names)},frame_sheet=image))
    config=json.loads((RUN/'chain_config.json').read_text());hashes={}
    for name,want in config['sources_sha256'].items():
        if name.startswith('physical_assets/'):p=RUN/'source/physical_v29/assets'/name.split('/',1)[1]
        elif name.startswith('physical_v29/'):p=RUN/'source'/name
        elif name in ['vram_ramp.py','supervisor_unitree.py']:p=RUN/name
        else:p=RUN/'source/physical_v29/training'/name
        actual=hashlib.sha256(p.read_bytes()).hexdigest();assert actual==want,(name,actual,want);hashes[name]=actual
    report=dict(run=str(RUN),gates=gates,training=training,traces=traces,frozen_hashes_verified=len(hashes),
        scope='Read-only saved independent gates, all update audits, and first-16 synchronized traces. No qualification or policy change.')
    (OUT/'audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(gates=len(gates),training_updates=sum(x['updates'] for x in training),
        latest_training=training[-1],traces=[{k:v for k,v in x.items() if k!='joint_ranges'} for x in traces],
        frozen_hashes_verified=len(hashes),output=str(OUT)),indent=2))

if __name__=='__main__':main()
