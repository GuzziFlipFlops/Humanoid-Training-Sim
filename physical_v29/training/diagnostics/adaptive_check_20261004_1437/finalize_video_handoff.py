"""Append the user-requested capture/pause/resume iteration and current focus."""
import json,re
from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
resume=read(OUT/'video_resume.json');videos=read(OUT/'video_verification.json');snapshot=read(OUT/'replacement_status.json')
state=snapshot['state'];run=Path(snapshot['active_pointer']['run']);progress_path=Path(state['operation_path'])/'progress.jsonl'
rows=[json.loads(x) for x in progress_path.read_text().splitlines()] if progress_path.is_file() else []
assert state['status']=='running' and snapshot['actual_worker_count']==1
assert videos['passed'] and len(videos['clips'])==4 and all(c['falls']==0 for c in videos['clips'])
assert rows and all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
last=rows[-1];duration=resume['pause_seconds']
note=f'''## User-requested GUI recordings and exact training resume

The user requested a temporary training pause, actual GUI capture at several
walking speeds for ten seconds of real-time motion per clip, then resume.
The exact owned R32 supervisor PID {resume['supervisor_pid']} was suspended at
a saved evaluation boundary; its child drained before any GUI simulator started.
The same supervisor, in-memory curriculum and saved actor resumed after capture
and verification. Pause duration: {duration:.1f} wall seconds. No training restart,
policy change or critic/Adam reset occurred for these recordings.

Four actual Isaac Sim 5.1 GUI clips were captured from a tracking front/side
view at commanded0.15,0.35,0.65 and1.0m/s. Every clip is1280x720 H.264 MP4,
500 distinct control samples at50FPS, exactly10 simulated seconds played in10
real seconds. Each took about56 wall seconds to record. All decoded frames
were checked and representative start/middle/end frames visually inspected;
the whole robot and feet remain visible. All clips have zero falls. They depict
the frozen training geometry and deterministic actor, not a new CAD export or
independent qualification. Actual average signed speeds are0.181,0.404,0.704
and0.989m/s; target labels are explicit. Full synchronized telemetry is retained.

The first recording attempt produced a valid0.15 clip but failed while resetting
cached Isaac inference buffers before the next speed. The failed script, logs,
startup and exception are preserved in videos/walking_gui_20261004/attempt_01.
Reset now runs in the same inference context as rollout; the valid first clip
was preserved and remaining three completed. This changes only the recording
script, with no frozen training or physical changes.

Fresh qualification before the pause passed matching nominal/pushed standing,
0.35,0.65 and1.0m/s stages, with stopped and all earlier-speed retention. These
passes belong to the exact002_ppo_smoke/transferred_actor.pt checkpoint. The
1.5 baseline fails genuine-flight gates; no running,5m/s or velocity ceiling
qualification is claimed. The current learning actor needs new matching gates.

Resumed evidence: {snapshot['recorded_utc']}; stage{state['stage']},
{state['environment_count']} environments, one actual worker,46 hashes verified.
The current block has{len(rows)} observed PPO updates (through local iteration
{last['iteration']}), with every pre-update KL/action-command mismatch zero;
latest free VRAM{last['gpu']['free_mb']}MiB and host available
{last['gpu']['ram_available_mb']}MiB. Completed main blocks remain separately
counted at{state['total_training_iterations']} until this block finishes.

The suspended supervisor's recorded wrapper time for evaluation018 includes
the user video pause (about{duration/60:.1f} minutes), although its physical
rollout completed normally. Whole-chain throughput/budget clocks also include
this pause. Subtract the documented pause when comparing active training
throughput; do not diagnose a simulator stall from that wrapper duration.
The original timing and pause audit are retained rather than rewritten.

Video files and timing/provenance: `E:/Aditya/robot_sim/physical_v29/training/videos/walking_gui_20261004/README.md`.
Pause/resume: video_pause.json and video_resume.json. Decode/hash verification:
video_verification.json. Training remains active; the same three-hour adaptive
check continues toward the existing pause-and-brief endpoint. Computer left on.
'''
marker='walking_gui_capture_20261004';wrapped=f'<!-- {marker} -->\n'+note+f'<!-- end_{marker} -->\n\n'
files=[OUT/'REPORT.md',HERE/'ITERATION_HISTORY.md',HERE.parents[1]/'docs/robot/TRAINING_JOURNAL_20261001.md',
 Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md'),
 Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md'),
 Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md')]
for p in files:
    text=p.read_text(encoding='utf-8-sig');text=re.sub(r'<!-- '+marker+r' -->.*?<!-- end_'+marker+r' -->\s*','',text,flags=re.S)
    p.write_text(wrapped+text if p.name not in ('REPORT.md','ITERATION_HISTORY.md') else text.rstrip()+'\n\n'+wrapped,encoding='utf-8')
focus=f''' User-requested GUI capture is complete: four 10-second real-time playback MP4s at commanded 0.15, 0.35, 0.65 and 1.0 m/s are in training/videos/walking_gui_20261004, with decoded-frame checks, hashes and telemetry; zero falls. The same supervisor was temporarily suspended, worker drained, then resumed after the GUI exited. video_pause.json/video_resume.json document {duration:.1f} seconds of pause. Evaluation018 wrapper time and whole-chain clocks include that pause; exclude it when comparing active throughput. R32 freshly requalified standing and walking through 1.0 m/s on its exact transferred baseline before recording; current learning actor remains unqualified. Fresh resumed block {Path(state['operation_path']).name}: {len(rows)} observed updates, {state['environment_count']} environments, zero maximum pre-update KL/command mismatch, latest free GPU {last['gpu']['free_mb']} MiB and host {last['gpu']['ram_available_mb']} MiB. Preserve the improving run and inspect matching new geometric-flight/retention gates; recordings alone do not qualify running. The first recording reset-context failure is preserved and repaired only in diagnostic recording code. All training sources and motor limits remain unchanged since R32 freeze.'''
p=HERE/'automation/NEXT_CHECK.json';d=read(p);d.update(current_block_observed_updates=len(rows),current_block=state['operation_path'],
 current_progress=last,video_manifest=str(HERE/'videos/walking_gui_20261004/manifest.json'),user_capture_pause_seconds=duration,
 fresh_qualified_speeds=[q['speed'] for q in state['promotions']],qualification_checkpoint=state['promotions'][-1]['checkpoint'])
d['focus']+=focus;p.write_text(json.dumps(d,indent=2)+'\n')
p=OUT/'automation_update_args.json';args=read(p);args['prompt']+=focus;p.write_text(json.dumps(args,indent=2)+'\n')
print(json.dumps({'same_supervisor_resumed':True,'observed_updates':len(rows),'stage':state['stage'],'num_envs':state['environment_count'],'clips':4,'files_updated':len(files)},indent=2))
