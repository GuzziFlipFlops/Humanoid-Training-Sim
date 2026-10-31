"""Resume the exact owned supervisor after every diagnostic worker exits."""
import json,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
import psutil
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
pause=read(OUT/'diagnostic_pause.json');run=Path(pause['run']);p=psutil.Process(pause['supervisor_pid'])
assert p.create_time()==pause['supervisor_create_time']
status_observations=[]
deadline=time.monotonic()+30
while True:
 status=p.status();status_observations.append({'time':time.time(),'status':status})
 if status==psutil.STATUS_STOPPED:break
 assert time.monotonic()<deadline,'Owned supervisor has not reported its suspended status; preserve pause and inspect identity.'
 time.sleep(1)
names={'worker_unitree.py','timing_probe.py','run_timing_probe.py','heading_candidate.py','matched_heading_test.py','diverse_heading_eval.py','capture_current_gradients.py','shoulder_candidate.py','diverse_eval.py','capture_policy_states.py','sequence_candidate.py','focused_prior_candidate.py','motor_retention_probe.py','arm_target_probe.py','teacher_body_probe.py','slow_reward_candidate.py','neutral_teacher_probe.py','neutral_teacher_candidate.py'}
for proc in psutil.process_iter(['cmdline']):
 assert not any(Path(x).name in names for x in proc.info['cmdline'] or []),'Diagnostic worker or harness still active'
s=read(OUT/'state_before_diagnosis.json');assert hashlib.sha256(Path(s['checkpoint']).read_bytes()).hexdigest()==pause['checkpoint_sha256']
now=time.time();s['updated_unix']=now;s['controlled_diagnosis_pause_seconds']=now-pause['suspended_unix'];write(run/'state.json',s)
a=read(HERE/'ACTIVE_RUN.json');assert Path(a['run'])==run;a['status']='running';write(HERE/'ACTIVE_RUN.json',a);p.resume()
r=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),same_supervisor=True,supervisor_pid=p.pid,checkpoint=s['checkpoint'],checkpoint_sha256=pause['checkpoint_sha256'],
 pause_seconds=s['controlled_diagnosis_pause_seconds'],suspended_unix=pause['suspended_unix'],resumed_unix=now,
 worker_drain_seconds=pause['paused_unix']-pause['suspended_unix'],training_source_changed=False,diagnostic_actors_deployed=False,
 strict_status_observations=status_observations)
write(OUT/'diagnostic_resume.json',r);print(json.dumps(r,indent=2))
