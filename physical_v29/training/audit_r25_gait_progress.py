"""Read-only CPU replay and kinematic inspection of saved independent gates.

No simulator, policy loading, CUDA allocation, reward edits, or qualification.
Replays the run's frozen landing classifier; rendered frames are CPU FK, not
Isaac screenshots. Loaded link/corner motion is not measured contact-patch slip.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from build_unitree_review import fk, pose


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(path):
    spec = importlib.util.spec_from_file_location('frozen_landing_replay', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def gate_trend(run):
    updates = 0
    rows = []
    for folder in sorted(run.iterdir()):
        if not folder.is_dir():
            continue
        result_path = folder/'result.json'
        if not result_path.exists():
            continue
        result = read(result_path)
        if '_train_' in folder.name:
            updates += result.get('iterations', result.get('updates', 0))
        if '_gate_' not in folder.name:
            continue
        for case in result.get('cases', []):
            rows.append(dict(folder=folder.name, full_walking_updates=updates,
                             pushed=bool(result['perturb']), passed=case['passed'],
                             failures=case['failures'], **case['statistics']))
    return rows


def unload_intervals(data):
    """Decompose physical unloaded bouts before settled landing classification."""
    dt = float(data['step_dt'])
    live = ~np.maximum.accumulate(data['done'].astype(bool),axis=0)
    bouts = []
    for robot, command in enumerate(data['commands']):
        if command <= .08:
            continue
        for side in range(2):
            contact = data['contact'][:,robot,side].astype(bool)
            stop = int(np.flatnonzero(~live[:,robot])[0]) if (~live[:,robot]).any() else len(contact)
            edges = np.diff(np.r_[False,~contact[:stop],False].astype(int))
            starts, ends = np.flatnonzero(edges==1),np.flatnonzero(edges==-1)
            previous_end = 0
            for j,(start,end) in enumerate(zip(starts,ends)):
                if end >= stop or data['time'][end] <= 2:
                    previous_end = end
                    continue
                next_start = starts[j+1] if j+1<len(starts) else stop
                duration = (end-start)*dt
                preceding = (start-previous_end)*dt
                following = (next_start-end)*dt
                maximum = float(data['clearance'][start:end,robot,side].max())
                reasons=[]
                if duration < .08-1e-5:
                    reasons.append('short_air')
                if preceding < .06-1e-5:
                    reasons.append('short_prior_load')
                if following < .10-1e-5:
                    reasons.append('short_settling_load')
                if maximum < (.02 if command>1.1 else .01):
                    reasons.append('low_clearance')
                bouts.append(dict(robot=robot,command_m_s=float(command),side=('left','right')[side],
                    takeoff_time_s=float(data['time'][start]),touchdown_time_s=float(data['time'][end]),
                    unloaded_duration_s=duration,preceding_load_duration_s=preceding,
                    following_load_duration_s=following,maximum_clearance_m=maximum,
                    physical_filter_reasons=reasons))
                previous_end=end
    groups=[]
    for command in np.unique(data['commands']):
        if command<=.08:
            continue
        for side in ('left','right'):
            rows=[r for r in bouts if np.isclose(r['command_m_s'],command) and r['side']==side]
            groups.append(dict(command_m_s=float(command),side=side,completed_unloaded_bouts=len(rows),
                mean_air_s=float(np.mean([r['unloaded_duration_s'] for r in rows])) if rows else None,
                mean_peak_clearance_m=float(np.mean([r['maximum_clearance_m'] for r in rows])) if rows else None,
                reason_counts={reason:sum(reason in r['physical_filter_reasons'] for r in rows)
                               for reason in ('short_air','short_prior_load','short_settling_load','low_clearance')}))
    return dict(groups=groups,bouts=bouts,scope='Contiguous unloaded intervals from the saved 50 Hz load flags, '
        'first 16 robots and first episode, touchdown after warmup. Reasons can overlap; not a replacement '
        'for the exact replay or proof of foot contact between samples.')


def replay(run, folder):
    path = run/folder/'gait_diagnostics_50hz.npz'
    with np.load(path, allow_pickle=False) as raw:
        data = {key:raw[key] for key in raw.files}
    count = data['q'].shape[1]
    tracker = module(run/'source/physical_v29/training/evaluation_steps.py').IndependentLandings(count, 'cpu')
    dt = float(data['step_dt'])
    alive = torch.ones(count, dtype=torch.bool)
    commands = torch.tensor(data['commands'], dtype=torch.float32)
    candidates, replayed = [], []
    with torch.inference_mode():
        for tick, time_s in enumerate(data['time']):
            alive &= ~torch.tensor(data['done'][tick], dtype=torch.bool)
            contact = torch.tensor(data['contact'][tick], dtype=torch.bool)
            state = torch.tensor(data['root_state'][tick], dtype=torch.float32)
            w, x, y, z = state[:, 3:7].unbind(1)
            forward = torch.stack([1-2*(y*y+z*z), 2*(x*y+w*z)], 1)
            forward /= forward.norm(dim=1, keepdim=True).clamp_min(.1)
            settling = tracker.armed & contact & (tracker.ground+dt >= .10-1e-5)
            settling &= alive[:, None] & (commands > .08)[:, None]
            landed, event_clearance, hops, touches = tracker.update(
                contact, torch.tensor(data['foot_position'][tick, :, :, :2]),
                torch.tensor(data['clearance'][tick]), state[:, :2], forward,
                commands, alive, dt)
            for key, value in [('landings', landed), ('hop_touches', hops),
                               ('same_foot_events', tracker.same_foot_events)]:
                assert np.array_equal(value.numpy(), data[key][tick]), f'{folder}: {key} mismatch at {tick}'
            replayed.append(landed.numpy().copy())
            for robot, side in torch.nonzero(settling).tolist():
                if time_s <= 2:
                    continue
                minimum = .02 if commands[robot] > 1.1 else .01
                clearance = float(tracker.maximum_clearance[robot, side])
                travel = float(tracker.impact_travel[robot, side])
                root_travel = float(tracker.impact_root_travel[robot, side])
                paired = bool(tracker.paired[robot, side])
                reasons = []
                if clearance < minimum:
                    reasons.append('insufficient_clearance')
                if travel < .020:
                    reasons.append('insufficient_forward_foot_placement')
                if root_travel < .003:
                    reasons.append('insufficient_forward_root_advance')
                if paired:
                    reasons.append('paired_hop')
                accepted = bool(landed[robot, side])
                assert accepted == (not reasons), 'Candidate reason decomposition differs from classifier'
                candidates.append(dict(robot=robot, side=('left','right')[side],
                    command_m_s=float(commands[robot]), time_s=float(time_s), accepted=accepted,
                    maximum_clearance_m=clearance, foot_forward_placement_m=travel,
                    root_forward_advance_m=root_travel, reasons=reasons))
    post = data['time'] > 2
    live = ~np.maximum.accumulate(data['done'].astype(bool), axis=0)
    delta = np.diff(data['foot_position'][..., :2], axis=0)/dt
    speed = np.linalg.norm(delta, axis=-1)
    loaded = data['contact'][1:].astype(bool) & data['contact'][:-1].astype(bool)
    loaded &= live[1:, :, None] & post[1:, None, None]
    names = data['joint_names'].tolist()
    tree = ET.parse(run/'source/physical_v29/training/assets/robot_training.urdf').getroot()
    limits = read(run/'source/physical_v29/assets/actuator_limits.json')
    per_robot = []
    for robot in range(count):
        valid = post & live[:, robot]
        indices = np.flatnonzero(valid)
        if not len(indices):
            per_robot.append(dict(robot=robot, command_m_s=float(data['commands'][robot]),
                survived=False, valid_postwarmup_ticks=0))
            continue
        root = data['root_state'][valid, robot]
        displacement = root[-1, :2]-root[0, :2]
        w, x, y, z = root[:, 3:7].T
        yaw = np.unwrap(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
        travel_heading = dict(
            world_xy_displacement_m=displacement.tolist(),
            net_displacement_m=float(np.linalg.norm(displacement)),
            path_distance_m=float(np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1).sum()),
            yaw_change_deg=float(np.rad2deg(yaw[-1]-yaw[0])),
            mean_absolute_yaw_rate_rad_s=float(np.abs(np.diff(yaw)/dt).mean()),
            mean_absolute_world_angular_z_rad_s=float(np.abs(root[:, 12]).mean()),
            scope='Valid first episode after warmup; global travel/heading diagnostics. '
                  'Existing gate does not check heading. Not a new promotion gate.')
        joint_rows = {}
        for name in ('left_hip_pitch', 'right_hip_pitch', 'left_knee_pitch', 'right_knee_pitch',
                     'left_ankle_pitch', 'right_ankle_pitch', 'left_hip_roll', 'right_hip_roll'):
            j = names.index(name)
            q = data['q'][valid, robot, j]
            tau = data['tau'][valid, robot, j]
            lim = tree.find(f"joint[@name='{name}']/limit")
            lo, hi = float(lim.get('lower')), float(lim.get('upper'))
            continuous = limits[name].get('continuous_Nm', limits[name]['effort_Nm'])
            joint_rows[name] = dict(min_q_deg=float(np.rad2deg(q.min())),
                max_q_deg=float(np.rad2deg(q.max())), excursion_deg=float(np.rad2deg(np.ptp(q))),
                limit_fraction=float(np.mean((q < lo+.01)|(q > hi-.01))),
                target_error_rms_rad=float(np.sqrt(np.mean((data['target'][valid, robot, j]-q)**2))),
                maximum_abs_torque_Nm=float(np.max(np.abs(tau))),
                continuous_torque_exceeded_fraction=float(np.mean(np.abs(tau)>continuous)))
        per_robot.append(dict(robot=robot, command_m_s=float(data['commands'][robot]),
            survived=bool(live[-1, robot]),
            root_x_advance_after_warmup_m=float(data['root_state'][indices[-1],robot,0]-data['root_state'][indices[0],robot,0]),
            accepted_left_steps=int(data['landings'][post, robot, 0].sum()),
            accepted_right_steps=int(data['landings'][post, robot, 1].sum()),
            same_foot_events=int(data['same_foot_events'][post, robot].sum()),
            maximum_clearance_m=data['clearance'][valid,robot].max(axis=0).tolist(),
            contact_fraction=data['contact'][valid,robot].mean(axis=0).tolist(),
            loaded_link_xy_motion_m_s=float(speed[:,robot][loaded[:,robot]].mean()) if loaded[:,robot].any() else None,
            travel_heading=travel_heading,
            joints=joint_rows))
    reasons_by_command = []
    for command in np.unique(data['commands']):
        for side in ('left', 'right'):
            events = [row for row in candidates if np.isclose(row['command_m_s'], command) and row['side']==side]
            reasons_by_command.append(dict(command_m_s=float(command), side=side, settled_air_qualified_candidates=len(events),
                accepted=sum(row['accepted'] for row in events),
                rejected_reasons={reason:sum(reason in row['reasons'] for row in events) for reason in
                    ('insufficient_clearance','insufficient_forward_foot_placement','insufficient_forward_root_advance','paired_hop')}))
    result = dict(evaluation=folder, trace_sha256=digest(path), classifier_sha256=digest(run/'source/physical_v29/training/evaluation_steps.py'),
        ticks=len(data['time']), robots=count, scalar_event_comparisons=len(data['time'])*count*4,
        replay_exactly_matches_saved_landings_hops_and_repeats=True,
        per_robot=per_robot, candidate_counts=reasons_by_command, candidates=candidates,
        unload_intervals=unload_intervals(data),
        scope='First 16 saved robots, fixed commands and first-episode alive mask. Counts after two seconds. '
              'Loaded link motion includes foot rolling/rotation; it is not measured contact-patch slip. '
              'Continuous torque exceedance is an instantaneous diagnostic, not a thermal rating or fault.')
    return data, result


def visuals(run, data, out, folder, robot=8):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tree = ET.parse(run/'source/physical_v29/training/assets/robot_training.urdf').getroot()
    names = data['joint_names'].tolist()
    time = data['time']
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(font_path), 14) if font_path.exists() else ImageFont.load_default()
    geom = []
    for side in ('left','right'):
        collision = tree.find(f"link[@name='{side}_foot']/collision")
        size = np.fromstring(collision.find('geometry/box').get('size'), sep=' ')
        corners = np.array([[sx*size[0]/2,sy*size[1]/2,-size[2]/2,1]
                             for sx,sy in [(-1,-1),(1,-1),(1,1),(-1,1)]])
        geom.append((pose(collision.find('origin')), corners))
    world_corners, errors = [], []
    for tick in range(len(time)):
        state = data['root_state'][tick,robot]
        transforms, _ = fk(tree, dict(zip(names,data['q'][tick,robot])), state)
        corners = []
        for j, side in enumerate(('left','right')):
            foot = transforms[side+'_foot']
            errors.append(float(np.max(np.abs(foot[:3,3]-data['foot_position'][tick,robot,j]))))
            offset, local = geom[j]
            corners.append((local@(foot@offset).T)[:, :3])
        world_corners.append(corners)
    assert max(errors) < .001, f'FK/trace disagreement {max(errors)}'
    world_corners = np.asarray(world_corners)
    corner_speed = np.linalg.norm(np.diff(world_corners[..., :2],axis=0)/float(data['step_dt']),axis=-1)
    contact = data['contact'][:,robot].astype(bool)
    valid = (time[1:] > 2) & ~np.maximum.accumulate(data['done'][:,robot].astype(bool))[1:]
    loaded = contact[1:] & contact[:-1] & valid[:,None]
    near = world_corners[1:,...,2] <= world_corners[1:,...,2].min(axis=2,keepdims=True)+.002
    nearest_corner_min_speed = np.where(near,corner_speed,np.inf).min(axis=2)
    corners_summary = dict(robot=robot, maximum_fk_error_m=max(errors),
        loaded_near_ground_corner_min_xy_motion_mean_m_s=float(nearest_corner_min_speed[loaded].mean()),
        fraction_loaded_with_all_near_ground_corners_faster_than_005_m_s=float((nearest_corner_min_speed[loaded]>.05).mean()),
        scope='Minimum horizontal speed among collision sole bottom corners within 2 mm of lowest corner. '
              'Tracks corresponding corner identities between ticks; excludes resets and warmup. '
              'An interior contact point is not sampled, and no pressure/impulse patch is measured.')
    fig, axes = plt.subplots(5,1,figsize=(12,12),sharex=True,layout='constrained')
    colors = ('#176f9c','#b05416')
    for side in range(2):
        axes[0].plot(time,data['foot_position'][:,robot,side,0]-data['root_state'][0,robot,0],label=('Left','Right')[side]+' foot link',color=colors[side])
        axes[1].plot(time,data['clearance'][:,robot,side]*1000,label=('Left','Right')[side],color=colors[side])
        axes[2].step(time,contact[:,side]+side*1.3,where='post',color=colors[side],label=('Left','Right')[side]+' loaded')
        idx=np.flatnonzero(data['landings'][:,robot,side])
        axes[2].scatter(time[idx],np.full(len(idx),.5+side*1.3),marker='v',color=colors[side],label=('Left','Right')[side]+' accepted')
        axes[4].plot(time[1:],np.where(loaded[:,side],nearest_corner_min_speed[:,side],np.nan),color=colors[side],alpha=.75,label=('Left','Right')[side]+' near-ground corner')
    axes[0].plot(time,data['root_state'][:,robot,0]-data['root_state'][0,robot,0],color='#222222',label='Pelvis')
    axes[1].axhline(10,color='#777777',linestyle='--',label='10 mm landing threshold')
    for name in ('left_hip_pitch','right_hip_pitch','left_knee_pitch','right_knee_pitch','left_ankle_pitch','right_ankle_pitch'):
        axes[3].plot(time,np.rad2deg(data['q'][:,robot,names.index(name)]),label=name)
    for ax,label in zip(axes,('World X travel (m)','Sole clearance (mm)','Contact / accepted landing','Joint angle (degrees)','Corner XY motion (m/s)')):
        ax.set_ylabel(label); ax.grid(alpha=.2); ax.axvline(2,color='#888888',linestyle=':')
        ax.legend(loc='upper left',fontsize=8,ncol=3)
    axes[-1].set_xlabel('Saved simulation time (s)')
    fig.suptitle(f'{folder}: saved independent rollout, robot {robot}, command {data["commands"][robot]:.2f} m/s\nCorner motion is kinematic diagnostic, not measured contact-patch slip')
    fig.savefig(out/f'{folder}_robot_{robot}_telemetry.png',dpi=135)
    plt.close(fig)
    accepted = np.flatnonzero(data['landings'][:,robot].any(axis=1)&(time>2))
    assert len(accepted), 'Select a robot with a saved qualifying landing for inspection'
    landing = int(accepted[0])
    sample_indices = np.linspace(max(0,landing-30),min(len(time)-1,landing+14),12).round().astype(int)
    sample_indices[np.argmin(np.abs(sample_indices-landing))] = landing
    sheet = Image.new('RGB',(4*512,3*396),'#ffffff')
    samples = []
    for k,index in enumerate(sample_indices):
        state=data['root_state'][index,robot]
        transforms,segments=fk(tree,dict(zip(names,data['q'][index,robot])),state)
        img=Image.new('RGB',(512,396),'#f7f7f3'); draw=ImageDraw.Draw(img)
        label=f't={time[index]:.2f}s; loaded L/R {int(contact[index,0])}/{int(contact[index,1])}'
        draw.text((10,7),label,font=font,fill='#111111')
        draw.text((10,26),f'clear L/R {data["clearance"][index,robot,0]*1000:.1f}/{data["clearance"][index,robot,1]*1000:.1f} mm',font=font,fill='#111111')
        draw.text((10,45),f'root travel {state[0]-data["root_state"][0,robot,0]:.2f} m; CPU FK',font=font,fill='#111111')
        for panel,axis in enumerate((0,1)):
            def project(point):
                return (int(128+panel*256+(point[axis]-state[axis])*158),int(372-point[2]*179))
            draw.text((12+panel*256,69),'Side X/Z' if panel==0 else 'Front Y/Z',font=font,fill='#333333')
            draw.line([(panel*256+8,372),(panel*256+248,372)],fill='#aaaaaa',width=2)
            for parent,child,a,b in segments:
                color=colors[0] if child.startswith('left') else colors[1] if child.startswith('right') else '#444444'
                draw.line([project(a),project(b)],fill=color,width=5)
                px,py=project(b); draw.ellipse((px-3,py-3,px+3,py+3),fill=color)
            for side in range(2):
                points=[project(point) for point in world_corners[index,side]]
                draw.line(points+[points[0]],fill=colors[side],width=3)
        if data['landings'][index,robot].any():
            draw.text((10,91),'QUALIFYING LANDING',font=font,fill='#224422')
        img.save(out/f'frame_{folder}_robot_{robot}_{index:04d}.png')
        sheet.paste(img,((k%4)*512,(k//4)*396))
        samples.append(dict(tick=int(index),time_s=float(time[index]),contacts=contact[index].tolist(),
            clearance_m=data['clearance'][index,robot].tolist(),accepted=data['landings'][index,robot].tolist()))
    sheet.save(out/f'{folder}_robot_{robot}_frame_sheet.png')
    return dict(**corners_summary, landing_tick=landing, samples=samples,
                frame_scope='CPU frozen-URDF FK of measured joint/root states, following root camera. No Isaac screenshot or gait certification.')


def group_support(path, run):
    audit = read(path)
    folder = Path(audit['source']).name
    with np.load(run/folder/'gait_diagnostics_50hz.npz',allow_pickle=False) as trace:
        commands = trace['commands']
        live = ~np.maximum.accumulate(trace['done'].astype(bool),axis=0)
        times = trace['time']
    keys = ('old_outside','shifted_rectangle_outside','full_sole_outside',
            'inset_sole_outside','near_ground_corner_outside','both_loaded')
    groups = []
    for command in np.unique(commands):
        rows = [row for row in audit['rows'] if np.isclose(commands[row['robot']],command)
                and live[np.argmin(np.abs(times-row['t'])),row['robot']]]
        groups.append(dict(command_m_s=float(command),first_episode_samples=len(rows),
            fractions={key:float(np.mean([row[key] for row in rows])) for key in keys}))
    return dict(evaluation=folder,source=str(path),source_sha256=digest(path),by_command=groups,
        scope='Same first-16/every-tenth-tick geometric audit, grouped by command and excluding '
              'post-failure reset poses. No measured contact pressure, gait or qualification.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--evaluation',action='append',required=True)
    parser.add_argument('--support-audit',type=Path,action='append',default=[])
    parser.add_argument('--visual-evaluation',action='append',default=[])
    parser.add_argument('--visual-robot',action='append',type=int,default=None)
    args=parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=True)
    reports=[]
    for folder in args.evaluation:
        data,report=replay(args.run,folder)
        if folder in args.visual_evaluation or (not args.visual_evaluation and '_006_nominal' in folder):
            report['visual_and_corner_audits']=[visuals(args.run,data,args.output,folder,robot)
                                               for robot in (args.visual_robot or [8])]
        (args.output/f'{folder}_replay.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        compact={key:value for key,value in report.items() if key not in ('candidates','per_robot','unload_intervals')}
        compact['unload_intervals']={key:value for key,value in report['unload_intervals'].items() if key!='bouts'}
        reports.append(compact)
    result=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),run=str(args.run),
                trend=gate_trend(args.run),replays=reports,
                grouped_support_audits=[group_support(path,args.run) for path in args.support_audit],
                scope='Read-only CPU audit of completed independent rollouts. Does not alter rewards, policy, live frozen files, or qualification.')
    (args.output/'progress_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(recorded_utc=result['recorded_utc'],replay_count=len(reports),
        exact_event_replays=all(row['replay_exactly_matches_saved_landings_hops_and_repeats'] for row in reports),
        trend_rows=len(result['trend']),output=str(args.output)),indent=2))


if __name__=='__main__':
    main()
