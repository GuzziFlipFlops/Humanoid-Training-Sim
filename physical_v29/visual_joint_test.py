"""Fast visible limb preview; invoked after AppLauncher by test_joints.py."""
import math
import time
import numpy as np


def run_visual_test(args,app,sim,robot,meta,report,save,step,measure,nominal,lookup,dt,motor_peaks):
    if args.seconds_per_joint<1:raise ValueError('Use at least one second per joint')
    state={'paused':False,'repeat':args.repeat,'stop':False}
    labels={}
    if not args.headless:
        import omni.ui as ui
        window=ui.Window('Humanoid joint preview',width=445,height=220)
        def pause():state['paused']=not state['paused']
        def repeat():state['repeat']=not state['repeat']
        def camera(which):
            views={'Front':((2.3,0.,1.1),(0.,0.,.8)),
                   'Side':((0.,-2.3,1.1),(0.,0.,.8)),
                   '3/4':((2.0,-2.0,1.35),(0.,0.,.8))}
            eye,target=views[which];sim.set_camera_view(eye=eye,target=target)
        with window.frame:
            with ui.VStack(spacing=7):
                labels['name']=ui.Label('Starting...',height=25)
                labels['angles']=ui.Label('',height=22)
                labels['cycle']=ui.Label('',height=22)
                with ui.HStack(height=28,spacing=5):
                    ui.Button('Pause / resume',clicked_fn=pause)
                    ui.Button('Toggle repeat',clicked_fn=repeat)
                    ui.Button('Stop',clicked_fn=lambda:state.update(stop=True))
                with ui.HStack(height=28,spacing=5):
                    for view in ('Front','Side','3/4'):
                        ui.Button(view,clicked_fn=lambda v=view:camera(v))
                ui.Label('Fixed base; gravity off. Wrists and fingers stay fixed.',height=25)
    rows=meta['joints'][:args.joint_count]
    if not rows:raise ValueError('No joints selected')
    def amplitude(name):
        if 'knee' in name:return 75
        if 'elbow' in name:return 90
        if 'ankle' in name:return 18 if 'pitch' in name else 12
        if name in ('waist_pitch','waist_roll'):return 12
        if 'head' in name:return 25
        return 35
    def wait_pause():
        while app.is_running() and state['paused'] and not state['stop']:
            sim.render();time.sleep(.02)
    report['mode']='fast_visual_preview'
    report['qualification']='Large smooth preview with measured motion/FK checks; strict settled validation is a separate run.'
    report['requested_seconds_per_joint']=args.seconds_per_joint
    report['cycles']=[]
    cycle=0;preview_started=time.perf_counter()
    while app.is_running() and not state['stop']:
        cycle+=1;records=[]
        for count,row in enumerate(rows,1):
            if not app.is_running() or state['stop']:break
            name=row['name'];idx=lookup[name];lo,hi=row['limits_rad']
            amp=math.radians(amplitude(name))
            points=[0.]
            for sign,room in ((-1,-lo),(1,hi)):
                if room>.025:points.append(sign*min(amp,room*.8))
            points.append(0.)
            speed=float(robot.actuators['cad_hardware'].records[name]['velocity_rad_s']) if args.hardware_actuators else 10.
            lengths=[]
            for a,b in zip(points,points[1:]):
                lengths.append(max(args.seconds_per_joint/(len(points)-1),math.pi*abs(b-a)/(2*speed*.75)))
            rec={'name':name,'targets_rad':points,'duration_s':sum(lengths),'samples':[]}
            max_tracking=0.;max_other=0.;max_fk=0.;max_angle=0.
            if labels:labels['name'].text=f'{count:02d}/{len(rows)}   {name.replace("_"," ")}'
            for start,end,duration in zip(points,points[1:],lengths):
                steps=max(1,math.ceil(duration/dt));segment_start=time.perf_counter()
                for k in range(steps):
                    wait_pause()
                    if not app.is_running() or state['stop']:break
                    u=(k+1)/steps
                    position=start+(end-start)*.5*(1-math.cos(math.pi*u))
                    velocity=(end-start)*.5*math.pi/(steps*dt)*math.sin(math.pi*u)
                    targets=nominal.clone();velocities=nominal.clone()
                    targets[0,idx]=position;velocities[0,idx]=velocity
                    tick_start=time.perf_counter();step(targets,velocity_targets=velocities)
                    if k%10==0 or k==steps-1:
                        q,errors=measure()
                        max_tracking=max(max_tracking,float(abs(q[idx]-position)))
                        max_other=max(max_other,float(np.max(np.abs(np.delete(q,idx)))))
                        max_fk=max(max_fk,max(e['position_error_m'] for e in errors))
                        max_angle=max(max_angle,max(e['rotation_error_rad'] for e in errors))
                        if labels:
                            labels['angles'].text=f'Target {math.degrees(position):+.1f} deg    Measured {math.degrees(q[idx]):+.1f} deg'
                            labels['cycle'].text=f'Cycle {cycle}    Repeat: {state["repeat"]}    ~{rec["duration_s"]:.1f} s / joint'
                    if not args.headless:time.sleep(max(0,dt-(time.perf_counter()-tick_start)))
                q,errors=measure()
                rec['samples'].append({'target_rad':end,'measured_rad':float(q[idx])})
            if state['stop'] or not app.is_running():break
            # A short settling interval prevents carry-over into the next axis.
            for _ in range(round(.25/dt)):
                start=time.perf_counter();step(nominal)
                if not args.headless:time.sleep(max(0,dt-(time.perf_counter()-start)))
            q,_=measure()
            rec.update(max_tracking_error_rad=max_tracking,max_other_joint_motion_rad=max_other,
                       fk_max_position_error_m=max_fk,fk_max_rotation_error_rad=max_angle,
                       reset_error_rad=float(np.abs(q).max()))
            moved=[s for s in rec['samples'] if abs(s['target_rad'])>.025]
            rec['pass']=all(s['target_rad']*s['measured_rad']>0 and abs(s['measured_rad'])>abs(s['target_rad'])*.65 for s in moved) and max_fk<.002 and max_angle<.01 and rec['reset_error_rad']<.06
            records.append(rec);report['joints']=records;save()
            print(f'VISUAL [{count:02d}/{len(rows)}] {name}: {"PASS" if rec["pass"] else "FAIL"} duration={rec["duration_s"]:.2f}s tracking_max={max_tracking:.4f}rad',flush=True)
        report['cycles'].append({'cycle':cycle,'joint_count':len(records),'all_pass':len(records)==len(rows) and all(r['pass'] for r in records)})
        report['all_pass']=report['cycles'][-1]['all_pass'];report['differential_applied_motor_peak_Nm']=motor_peaks
        report['finished']=time.time();save()
        report['preview_wall_seconds']=time.perf_counter()-preview_started
        report['preview_physics_dt_s']=dt
        report['invalid_transmission_steps_at_completion']=int(robot.actuators['cad_hardware'].invalid_transmission.any()) if args.hardware_actuators else 0
        save()
        print(f'VISUAL CYCLE {cycle}: {len(records)}/{len(rows)} joints, pass={report["all_pass"]}',flush=True)
        if args.headless or args.exit_after_cycle:break
        if not state['repeat']:
            if labels:labels['name'].text='Cycle finished. Toggle repeat to run again.'
            while app.is_running() and not state['stop'] and not state['repeat']:
                sim.render();time.sleep(.02)
    if args.headless and not report['all_pass']:raise AssertionError('Visual movement check failed; see JSON')
