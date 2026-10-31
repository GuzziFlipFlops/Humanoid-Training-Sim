from pathlib import Path
import py_compile
OUT=Path(__file__).resolve().parent;old=OUT.parent/'adaptive_check_20261004_1437/physical_takeoff_probe.py'
text=old.read_text().replace("(OUT/'drain.json')","(OUT/'diagnostic_pause.json')").replace("drain['all_owned_workers_drained']","drain['worker_drained']")
start=text.index("    groups=[('baseline'");end=text.index('    cfg=LocomotionCfg()',start)
text=text[:start]+'''    # name,command,retract_rad,retract_phase_start,push_extension_rad
    groups=[('baseline',1.5,0.,.40,0.),('retract_010',1.5,.10,.40,0.),
      ('retract_020',1.5,.20,.40,0.),('retract_030',1.5,.30,.40,0.),
      ('retract_early',1.5,.30,.35,0.),('retract_late',1.5,.30,.43,0.),
      ('push010_retract020',1.5,.20,.40,.10),('push020_retract030',1.5,.30,.40,.20),
      ('push030_retract030',1.5,.30,.40,.30),('push020_early_retract',1.5,.30,.35,.20),
      ('walk065_control',.65,0.,.40,0.),('stopped_control',0.,0.,.40,0.)]
    per=24;n=len(groups)*per
'''+text[end:]
text=text.replace("shift=torch.tensor([g[3] for g in groups],device=raw.device).repeat_interleave(per)",
 "retract_start=torch.tensor([g[3] for g in groups],device=raw.device).repeat_interleave(per)\n    extension=torch.tensor([g[4] for g in groups],device=raw.device).repeat_interleave(per)")
text=text.replace("raw.reset();observation=raw._get_observations();alive=", "raw.reset();reference_heading=raw.heading_target.clone();observation=raw._get_observations();alive=")
text=text.replace("    history={k:[] for k in", "    reward_terms={}\n    history={k:[] for k in")
text=text.replace("'failures']}","'failures','reward_phase','observed_command','action_command','target_clipped','target_slew','torque_saturation','healthy']}")
text=text.replace('range(500)','range(1000)')
start=text.index("        observed=observation['policy']");end=text.index('        for side in',start)
text=text[:start]+'''        observed_commands=observation['policy'].reshape(n,5,80)[:,-1,6:9].clone()
        with torch.inference_mode():action=policy(observation)
        action=action.clone()
        phase=raw.phase.clone();phasehalf=torch.remainder(phase,.5)
        # Smooth stance extension followed by shortening; all offsets go through
        # the ordinary action clamp, joint limits, CAD projection/slew and caps.
        retract_progress=((phasehalf-retract_start)/(.5-retract_start)).clamp(0.,1.)
        retract=torch.sin(retract_progress*torch.pi)*(phasehalf>=retract_start)
        push_progress=((phasehalf-.20)/.20).clamp(0.,1.)
        push=torch.sin(push_progress*torch.pi)*((phasehalf>=.20)&(phasehalf<.40))
        delta=(amplitude*retract-extension*push)*float(t>=3.)
'''+text[end:]
text=text.replace("unload_score=unload,failures=done)","""unload_score=unload,failures=done,reward_phase=raw.reward_proxy.cad_metrics['phase'],
            observed_command=observed_commands,action_command=raw.action_commands,
            target_clipped=raw.target_clipped,target_slew=raw.target_slew_limited,
            torque_saturation=(raw.actuator.computed_effort-raw.actuator.applied_effort).abs()>1e-3,
            healthy=raw.reward_port.last_diagnostics['healthy'])
        for k,v in raw.last_reward_terms.items():reward_terms.setdefault(k,[]).append(v.detach().cpu().numpy().copy())""")
text=text.replace("window=data['time']>=3.","window=data['time']>=5.")
text=text.replace("policy_phase_shift_cycles=g[3],","retract_phase_start=g[3],stance_extension_rad=g[4],")
text=text.replace("mean_forward_world_speed=float(data['root'][window,ids,7][good].mean()) if good.any() else None,", """mean_forward_world_speed=float((data['root'][window,ids,7]*np.cos(reference_heading[ids].cpu().numpy())[None,:]+data['root'][window,ids,8]*np.sin(reference_heading[ids].cpu().numpy())[None,:])[good].mean()) if good.any() else None,
            reward_terms={k:float(np.asarray(v)[window,ids][good].mean()) for k,v in reward_terms.items()},
            head_RMS_deg=float(np.rad2deg(np.sqrt((data['q'][window,ids][:,:,[raw.lookup['head_yaw'],raw.lookup['head_pitch']]]**2).sum(-1)[good].mean()))),
            target_clip_fraction=float(data['target_clipped'][window,ids].mean()),
            target_slew_fraction=float(data['target_slew'][window,ids].mean()),
            torque_saturation_fraction=float(data['torque_saturation'][window,ids].mean()),
            healthy_mean=float(data['healthy'][window,ids][good].mean()),""")
text=text.replace("**data,joint_names=", "**data,**{'term_'+k:np.asarray(v) for k,v in reward_terms.items()},reference_heading=reference_heading.cpu().numpy(),joint_names=")
text=text.replace("scope='96 physical robots, 10s; bounded diagnostic policy-phase or joint-action changes after3s. No external root support, no training source changes; automatic falls retained.'", "scope='288 physical robots,20s,24 per cohort; truthful policy observations. Bounded diagnostic action offsets after3s, statistics after5s. All normal CAD motor constraints, gravity and automatic falls retained. No support, phase rewrite or trained actor.'")
(OUT/'physical_takeoff_probe.py').write_text(text);py_compile.compile(str(OUT/'physical_takeoff_probe.py'),doraise=True)
print('Prepared truthful-observation, normal-limit controlled take-off grid.')
