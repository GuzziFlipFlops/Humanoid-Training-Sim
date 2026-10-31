"""Prepare a same-heartbeat update and durable one-time video request."""
import json
import tomllib
from pathlib import Path
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
AUTOMATION = Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml')
current = tomllib.loads(AUTOMATION.read_text(encoding='utf-8-sig'))
assert current['kind']=='heartbeat' and current['status']=='ACTIVE'
assert current['rrule']=='FREQ=HOURLY;INTERVAL=3'
audit = HERE / 'recording_request_20261004'
audit.mkdir(exist_ok=True)
(audit/'automation_before.json').write_text(json.dumps(current,indent=2),encoding='utf-8')
priority = (
    'ONE-TIME PRIORITY FOR THE NEXT UPDATE/CHECK: The user explicitly requests a temporary '
    'training pause and real GUI videos with every original CAD part present, at different '
    'speeds including the newly demonstrated faster running. Before training repairs or '
    'reward redesign, read E:\\Aditya\\robot_sim\\physical_v29\\training\\automation\\NEXT_GUI_RECORDING.md '
    'and NEXT_CHECK.json recording_request. Follow that concrete pause/capture/frame-QA/resume '
    'workflow while the request is pending. Record one ten-second normal-real-time-playback '
    'MP4 per commanded0.15,0.35,0.65,1.0,1.5m/s and any higher actually demonstrated speeds. '
    'Ten simulated seconds must play in ten real seconds; slow rendering may take longer '
    'to capture. Use all original compatible CAD visual meshes, validate import and articulated '
    'attachment, improve camera/lighting, and inspect decoded frames so no body shells, '
    'limbs/hands or other mesh parts disappear. Use a separate visual-only playback asset with '
    'identical frozen training physics/joints/motor caps. Pause at an owned saved boundary, '
    'drain the worker before the GUI, preserve exact checkpoint/source lineage, deliver checked '
    'video links and resume/verify the same training supervisor after the GUI exits. Preserve '
    'failed attempts and resume safely on capture faults; no silent prolonged pause. Do not '
    'mark complete until videos, full-body frame QA and resume evidence are saved. After '
    'completion, mark recording_request complete and remove only this one-time priority from '
    'this same automation; keep all normal scope/cadence/endpoint instructions. Fresh status '
    'review diagnostics/status_review_20261004_2238/REPORT.md records R34checkpoint038 passing '
    'the1.5 running case nominal/pushed but failing0.15/0.35 speed retention. Preserve that '
    'actor and read newer gates; no whole-stage or5m/s qualification is inferred.\n\n'
)
assert 'ONE-TIME PRIORITY FOR THE NEXT UPDATE/CHECK' not in current['prompt']
args = {'id':current['id'],'mode':'update','kind':current['kind'],'name':current['name'],
    'prompt':priority+current['prompt'],'rrule':current['rrule'],'status':current['status'],
    'destination':'thread','targetThreadId':current['target_thread_id']}
if 'notification_policy' in current:
    args['notificationPolicy']=current['notification_policy']
elif 'notificationPolicy' in current:
    args['notificationPolicy']=current['notificationPolicy']
(audit/'automation_update_args.json').write_text(json.dumps(args,indent=2),encoding='utf-8')
next_path = HERE/'NEXT_CHECK.json'
handoff = json.loads(next_path.read_text(encoding='utf-8-sig'))
handoff['recording_request'] = {
    'id':'full-body-locomotion-next-check-20261004','status':'pending',
    'requested_utc':datetime.now(timezone.utc).isoformat(),'when':'next update/automated check',
    'workflow':str(HERE/'NEXT_GUI_RECORDING.md'),
    'pause_training':True,'resume_training_after_capture':True,'real_gui':True,
    'normal_playback_seconds_per_speed':10,'simulated_motion_seconds_per_speed':10,
    'commanded_speeds_m_s':[0.15,0.35,0.65,1.0,1.5],
    'also_include':'Higher speeds actually demonstrated by the next check',
    'visuals':'All compatible original CAD visual meshes; verify articulated attachment and full-body frame QA',
    'not_completed_by_scheduling':True,'deliverables':[]}
next_path.write_text(json.dumps(handoff,indent=2),encoding='utf-8')
durable = HERE/'THREE_HOUR_CHECK.md'
text = durable.read_text(encoding='utf-8-sig')
marker='<!-- next_gui_recording_20261004 -->'
assert marker not in text
block = f'''\n{marker}
## Next check: user-requested full-body video capture

The user requested a one-time temporary training pause and ten-second normal-time
GUI clips at multiple speeds including the newly demonstrated faster running.
Before ordinary repairs/redesign, read [NEXT_GUI_RECORDING.md](NEXT_GUI_RECORDING.md)
and `NEXT_CHECK.json` recording_request. Restore all compatible original CAD
visuals, validate articulated meshes and review frames; preserve frozen physics.
Pause/drain safely, capture longer wall time as needed, deliver verified MP4s,
then resume and verify the same training. Mark complete only after delivery,
frame QA and resume evidence, then remove only the one-time automation priority.
An already completed request must not run again. Preserve the global endpoint.
<!-- /next_gui_recording_20261004 -->
'''
durable.write_text(text.replace('## Identity and durable constraints',block+'\n## Identity and durable constraints',1),encoding='utf-8')
note = '''\n\n## October 4: next-check full-body GUI recording requested

Queued one-time priority in the same adaptive three-hour heartbeat: temporarily
pause/drain training, render all compatible original CAD visuals, record ten
seconds of normal-time motion per speed0.15/0.35/0.65/1.0/1.5 plus newly demonstrated
higher speeds, decode/review frames, deliver MP4s and resume the same owned run.
Prepared visual asset still requires actual conversion/rendered alignment QA.
No training pause, simulator launch or new video occurred while scheduling.
Workflow: physical_v29/training/automation/NEXT_GUI_RECORDING.md; pending state
in NEXT_CHECK.json. Remove the one-time prompt priority only after checked video
delivery and documented resume. Training sources and all gates remain unchanged.
'''
for path in [HERE.parent/'ITERATION_HISTORY.md',
             Path('E:/Aditya/robot_sim/docs/robot/TRAINING_JOURNAL_20261001.md')]:
    with path.open('a',encoding='utf-8') as f:f.write(note)
print(json.dumps({'arguments':str(audit/'automation_update_args.json'),
    'pending_request':handoff['recording_request'],'same_thread':args['targetThreadId'],
    'cadence_preserved':True}))
