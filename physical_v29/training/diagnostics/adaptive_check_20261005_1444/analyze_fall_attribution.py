"""Coarse control-tick fall attribution for the passive substep trace."""
import json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent/'physical_speed_response'
d=np.load(P/'telemetry.npz');m=np.load(P/'substep_motor_200hz.npz')
# The saved alive flag is post-control-step. Do not claim exact fall onset
# inside the four actuator substeps of its terminal control tick.
alive=np.repeat(d['alive'],4,axis=0);clips=m['speed_clip'];ratios=m['cap_ratio'];ids=slice(160,192)
out=dict(command=2.5,all_motor_peak_cap_ratio_in_alive_control_ticks=float(ratios[:,ids][alive[:,ids]].max()),
 driving_clip_count_in_alive_control_ticks=int((clips[:,ids]&alive[:,ids,None]).sum()),
 driving_clip_count_in_failed_control_ticks=int((clips[:,ids]&~alive[:,ids,None]).sum()),
 cap_crossings_in_alive_control_ticks=int(((ratios[:,ids]>=1)&alive[:,ids,None]).sum()),
 cap_crossings_in_failed_control_ticks=int(((ratios[:,ids]>=1)&~alive[:,ids,None]).sum()),
 scope='Post-control-tick alive mask repeated four times; fall onset within the terminal20ms tick is not identified. Both pre-failure and failed-tick isolated driving clips occur. This untrained, eventually fallen cohort does not establish a sustained locomotion velocity ceiling.')
(P/'fall_attribution.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
