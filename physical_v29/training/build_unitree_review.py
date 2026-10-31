"""Build a local frame/reward review and CPU reconstruction of saved rollouts.

Never launches Isaac, changes a policy, or treats reconstruction as a screenshot.
All geometry in reconstructed poses comes from the frozen evaluation URDF.
"""
from pathlib import Path
import argparse, json, math, xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).parent
DEFAULT = HERE / 'diagnostics/unitree_port_20261002'

def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= max(np.linalg.norm(axis), 1e-12)
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)

def pose(element):
    T = np.eye(4)
    if element is None:
        return T
    T[:3, 3] = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
    r, p, y = np.fromstring(element.get('rpy', '0 0 0'), sep=' ')
    T[:3, :3] = rotation([0,0,1], y) @ rotation([0,1,0], p) @ rotation([1,0,0], r)
    return T

def root_pose(state):
    T = np.eye(4)
    T[:3, 3] = state[:3]
    w, x, y, z = state[3:7]
    T[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                 [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                 [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]]
    return T

def fk(tree, q, state):
    transforms = {'pelvis': root_pose(state)}
    segments = []
    pending = list(tree.findall('joint'))
    while pending:
        progressed = False
        for j in pending[:]:
            parent, child = j.find('parent').get('link'), j.find('child').get('link')
            if parent not in transforms:
                continue
            T = transforms[parent] @ pose(j.find('origin'))
            axis = np.fromstring(j.find('axis').get('xyz', '1 0 0'), sep=' ') if j.find('axis') is not None else [1,0,0]
            R = np.eye(4)
            R[:3,:3] = rotation(axis, q.get(j.get('name'), 0.))
            transforms[child] = T @ R
            segments.append((parent, child, transforms[parent][:3,3], T[:3,3]))
            pending.remove(j)
            progressed = True
        assert progressed, 'Disconnected/cyclic URDF'
    return transforms, segments

def reconstruction(run, out, prefix='smoke', subdir='learned_reconstruction',
                   evaluation_folder='003_checkpoint_reload', cases=None,
                   policy_scope='actual 4-update PPO rollout'):
    trace_path = run / evaluation_folder / 'gait_diagnostics_50hz.npz'
    data = np.load(trace_path, allow_pickle=False)
    tree = ET.parse(run / 'source/physical_v29/training/assets/robot_training.urdf').getroot()
    frames_out = out / subdir
    frames_out.mkdir(exist_ok=True)
    names = data['joint_names'].tolist()
    summary, samples = [], []
    for label, robot in (cases if cases is not None else [(prefix+'_stand', 0), (prefix+'_walk015', 1)]):
        failures = np.flatnonzero(data['done'][:, robot])
        stop = int(failures[0]) if len(failures) else len(data['time'])
        # Exclude the done tick: its pose has already been automatically reset.
        frame_indices = list(range(0, stop, 5))
        if stop and (not frame_indices or frame_indices[-1] != stop-1):
            frame_indices.append(stop-1)
        first = data['root_state'][0, robot, :3]
        foot_error = []
        images = []
        for index in frame_indices:
            state = data['root_state'][index, robot]
            transforms, segments = fk(tree, dict(zip(names, data['q'][index, robot])), state)
            feet = [transforms[s+'_foot'][:3,3] for s in ['left','right']]
            foot_error.append(float(np.max(np.abs(np.array(feet)-data['foot_position'][index, robot]))))
            img = Image.new('RGB', (1000, 660), '#f5f5f2')
            draw = ImageDraw.Draw(img)
            draw.text((20, 12), f'{label} | t={data["time"][index]:.2f}s | command={data["commands"][robot]:.2f} m/s', fill='#111111')
            draw.text((20, 32), f'CPU reconstruction of {policy_scope}; not an Isaac screenshot or qualified gait', fill='#111111')
            draw.text((20, 52), f'vx={state[7]:.3f} m/s | height={state[2]:.3f} m | contacts={data["contact"][index,robot].tolist()}', fill='#111111')
            for panel, horizontal in enumerate([0,1]):
                cx = 250 + panel*500
                def projected(point):
                    p = np.asarray(point)
                    return (int(cx+(p[horizontal]-first[horizontal])*255), int(590-p[2]*285))
                draw.line([(panel*500+20,590),(panel*500+480,590)], fill='#999999', width=2)
                draw.text((panel*500+25,95), 'Side view: x/z' if horizontal==0 else 'Front view: y/z', fill='#111111')
                for parent, child, a, b in segments:
                    color = '#176f9c' if child.startswith('left') else '#b05416' if child.startswith('right') else '#424242'
                    draw.line([projected(a),projected(b)], fill=color, width=7)
                    px, py = projected(b)
                    draw.ellipse((px-3,py-3,px+3,py+3), fill=color)
                for side in ['left','right']:
                    link = tree.find(f"link[@name='{side}_foot']")
                    collision = link.find('collision')
                    box = collision.find('geometry/box')
                    size = np.fromstring(box.get('size'), sep=' ')
                    T = transforms[side+'_foot'] @ pose(collision.find('origin'))
                    corners = [np.array([sx,sy,-1])*size/2 for sx,sy in [(-1,-1),(-1,1),(1,1),(1,-1)]]
                    pts = [projected((T@np.r_[p,1])[:3]) for p in corners]
                    draw.line(pts+[pts[0]], fill='#176f9c' if side=='left' else '#b05416', width=4)
            filename = f'{label}_{index:04d}.png'
            img.save(frames_out/filename)
            images.append(img)
            samples.append(dict(case=label, t=float(data['time'][index]), frame=f'{subdir}/{filename}',
                                root=state.tolist(), q=data['q'][index,robot].tolist(),
                                target=data['target'][index,robot].tolist(),
                                contact=data['contact'][index,robot].tolist(), done=False,
                                terms_g1={},terms_h1={}, scope='CPU reconstruction; post-reset ticks excluded'))
        for start in range(0,len(images),12):
            sheet=Image.new('RGB',(1200,4*264),'white')
            for k,img in enumerate(images[start:start+12]):
                sheet.paste(img.resize((400,264)),((k%3)*400,(k//3)*264))
            sheet.save(frames_out/f'{label}_sheet_{start//12:02d}.jpg',quality=93)
        max_error=max(foot_error,default=0.)
        assert max_error < .001, f'FK/render mapping wrong: {max_error}'
        summary.append(dict(case=label, command_m_s=float(data['commands'][robot]),
                            first_failure_s=float(data['time'][stop]) if len(failures) else None,
                            frames=len(images), max_fk_vs_sim_foot_error_m=max_error,
                            scope=f'{policy_scope}; no qualification; CPU reconstruction, not screenshots'))
    (frames_out/'result.json').write_text(json.dumps(dict(source=str(trace_path), cases=summary),indent=2))
    return samples, summary

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--additional_run',type=Path,action='append',default=[])
    args=parser.parse_args()
    out=args.output
    live=json.loads((out/'scripted_visuals_r2/telemetry.json').read_text())
    frames=[dict(row,frame='scripted_visuals_r2/'+row['frame']) for row in live['rows'] if row.get('frame')]
    latest=out/'r20_policy_failure_visuals/telemetry.json'
    if latest.exists():
        replay=json.loads(latest.read_text())
        assert replay['joint_names']==live['joint_names']
        frames += [dict(row,frame='r20_policy_failure_visuals/'+row['frame']) for row in replay['rows'] if row.get('frame')]
    reconstructed, summary=reconstruction(args.run,out)
    frames+=reconstructed
    for additional_run in args.additional_run:
        revision=additional_run.name.split('_')[2]
        assert revision in ['r21','r22'],revision
        repaired, repaired_summary=reconstruction(additional_run,out,revision+'_smoke',revision+'_smoke_reconstruction')
        frames+=repaired
        summary+=repaired_summary
    payload=json.dumps(dict(frames=frames,joint_names=live['joint_names'],reconstruction=summary),separators=(',',':'))
    # Standalone local review, with only frame-event rows inlined (no network requests).
    template=(HERE/'unitree_review_template.html').read_text(encoding='utf-8')
    html=template.replace('<!-- DATA -->',payload)
    (out/'frame-review.html').write_text(html,encoding='utf-8')
    print(json.dumps(dict(review=str(out/'frame-review.html'),frame_count=len(frames),learned_reconstruction=summary),indent=2))

if __name__=='__main__':
    main()
