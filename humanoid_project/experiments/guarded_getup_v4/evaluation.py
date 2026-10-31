"""Independent NumPy success audit of pre-reset measured traces, not core flags."""
import math
import numpy as np
from reward_core import Limits


def audit(trace, dt=.02):
    c = Limits()
    t, n = trace['height'].shape
    good = ((trace['height'] >= c.min_height) & (trace['height'] <= c.max_height)
            & (trace['pelvis_tilt'] <= c.max_pelvis_tilt) & (trace['torso_tilt'] <= c.max_torso_tilt)
            & (trace['planar_speed'] <= c.max_planar_speed)
            & (np.abs(trace['vertical_speed']) <= c.max_vertical_speed)
            & (trace['angular_speed'] <= c.max_angular_speed) & (trace['joint_speed_rms'] <= c.max_joint_speed_rms)
            & (trace['foot_load'].min(-1) >= c.min_each_foot_weight)
            & (trace['foot_load'].sum(-1) >= c.min_total_foot_weight)
            & (trace['foot_load'].sum(-1) <= c.max_total_foot_weight)
            & (trace['foot_tilt'].max(-1) <= c.max_foot_tilt)
            & (trace['foot_slip'].max(-1) <= c.max_slip)
            & (trace['nonfoot_force'] <= c.max_nonfoot_force)
            & (trace['support_margin'] >= c.min_support_margin)
            & trace['contacts_fresh'] & trace['safe'])
    for key, value in trace.items():
        if not np.isfinite(value).all(): raise ValueError('Invalid evaluation trace: ' + key)
    count = np.zeros(n, dtype=int)
    ever = np.zeros(n, dtype=bool)
    invalid = np.zeros(n, dtype=bool)
    failed = np.zeros(n, dtype=bool)
    xy = np.zeros((n, 2)); feet = np.zeros((n, 2, 2)); yaw = np.zeros(n)
    completion_time = np.full(n, np.nan)
    needed = math.ceil((c.dwell_s + c.post_hold_s) / dt)
    for i in range(t):
        failed |= trace['terminated'][i]
        start = good[i] & (count == 0) & ~ever & ~failed
        xy[start] = trace['xy'][i, start]
        feet[start] = trace['foot_xy'][i, start]
        yaw[start] = trace['yaw'][i, start]
        angle = trace['yaw'][i] - yaw
        angle = np.abs(np.arctan2(np.sin(angle), np.cos(angle)))
        valid = (good[i] & ~failed & (np.linalg.norm(trace['xy'][i] - xy, axis=-1) <= c.max_hold_drift)
                 & (np.linalg.norm(trace['foot_xy'][i] - feet, axis=-1).max(-1) <= c.max_hold_foot_drift)
                 & (angle <= c.max_hold_yaw))
        invalid |= ever & ~valid
        count = np.where(valid, count + 1, 0)
        new = (count >= needed) & ~ever
        completion_time[new] = (i + 1) * dt
        ever |= new
    passed = ever & ~invalid & ~failed
    rate = float(passed.mean())
    z = 1.96
    center = (rate + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(rate*(1-rate)/n + z*z/(4*n*n)) / (1+z*z/n)
    last = max(1, round(5. / dt))
    return dict(trials=n, strict_successes=int(passed.sum()), success_rate=rate,
                success_95ci=[max(0., center-half), min(1., center+half)],
                rise_then_lost_stand=int(invalid.sum()), unsafe_terminated=int(failed.sum()),
                any_completion=int(ever.sum()),
                max_height_mean=float(trace['height'].max(0).mean()),
                peak_penetration_m=float(trace['penetration'].max()),
                final5s_planar_speed=float(trace['planar_speed'][-last:].mean()),
                median_completion_s=float(np.median(completion_time[passed])) if passed.any() else None,
                peak_joint_speed=float(np.abs(trace['joint_vel']).max()),
                peak_estimated_torque=float(np.abs(trace['torque_estimate']).max()),
                warning='Torque is an implicit-drive estimate; this is simulation success, not hardware certification.')
