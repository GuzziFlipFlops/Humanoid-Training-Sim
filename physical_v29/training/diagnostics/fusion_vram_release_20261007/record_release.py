"""Record normal Fusion shutdown and fresh owned training evidence; no simulator."""
import json,subprocess
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;TRAINING=OUT.parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
source=TRAINING/'diagnostics/adaptive_check_20261007_1344/fusion_close_owned/live_handoff.json'
live=read(source)
assert live['state']['status']=='running' and live['state']['supervisor_pid']==39000
assert len(live['actual_worker_pids'])==1 and len(live['frozen_source_integrity'])==55 and all(live['frozen_source_integrity'].values())
fusion=subprocess.run(['powershell','-NoProfile','-Command',"Get-Process -Name Fusion360,AdPreviewGenerator -ErrorAction SilentlyContinue | Select-Object Id,ProcessName | ConvertTo-Json -Compress"],capture_output=True,text=True)
assert not fusion.stdout.strip(),fusion.stdout
gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
total,used,free=[int(x.strip()) for x in gpu.split(',')]
run=Path(live['active_run']['run']);operation=Path(live['state']['operation_path'])
config=read(operation/'run_config.json') if (operation/'run_config.json').exists() else {}
receipt=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),fusion_process_absent=True,preview_process_absent=True,normal_UI_shutdown=True,forced_termination=False,unsaved_changes_discarded=False,unsaved_scope='Normal shutdown did not present an unsaved-change discard prompt. No Discard/Do not save was selected; no cloud save or design edit was initiated by this task.',queue_scope='Fusion reported one pending save/upload job. Its Pause unstarted jobs action was used; queued jobs are preserved for next launch. No active upload was force-killed.',update_scope='Update Later selected during shutdown; no update installation requested.',first_observed_gpu_MiB=dict(total=8188,used=7184,free=773),post_exit_gpu_MiB=dict(total=total,used=used,free=free),memory_attribution='Global snapshots across changing training phases; difference is not a matched measurement of Fusion dedicated memory or training throughput.',owned_observation_utc=live['recorded_utc'],run=str(run),supervisor_pid=39000,actual_worker_pids=live['actual_worker_pids'],operation=str(operation),operation_environment_count=live['state']['environment_count'],operation_arguments=config.get('arguments'),frozen_hashes_verified=55,training_paused=False,training_restarted=False,training_sources_changed=False,owned_snapshot=str(source))
(OUT/'resource_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
body=f"""# Fusion VRAM release — October 7

Fusion and its preview process are absent after normal UI shutdown. No process was force-killed. Normal close encountered one pending save/upload job; Fusion's **Pause unstarted jobs** action preserves queued jobs for the next launch. **Update Later** was then selected; no update was installed. No unsaved-change discard prompt was presented and no Discard/Do not save action was selected. Earlier Strap cam unsaved status was dated context, not assumed current state.

Global VRAM was 773 MiB free at the initial reading and {free} MiB free in the saved post-exit reading, on an 8188 MiB GPU. Training continues across these readings, so the difference is not an isolated Fusion memory measurement or a throughput benchmark.

Fresh owned observation {live['recorded_utc']} verifies the same R40 supervisor 39000, one actual worker and all 55 frozen hashes. Current operation `{operation.name}` has count {live['state']['environment_count']}; evaluation count is not PPO capacity. Training was neither paused nor restarted and no training source or policy parameter changed. Existing target 8192, 300 MiB reserve, host guard/fallback and 180 continuous healthy-training-second ramp remain. A startup or evaluation free-memory reading does not authorize an immediate capacity increase.

The user now explicitly authorized closing Fusion to release GPU memory. Do not reopen it merely for a routine training check. Preserve queued user work; future CAD/video tasks may require reopening only when necessary. [Resource receipt](resource_receipt.json) binds processes, memory and [fresh ownership evidence]({source.as_posix()}).
"""
(OUT/'REPORT.md').write_text(body,encoding='utf-8')
marker='fusion_vram_release_20261007_user_requested'
entry=f"\n<!-- {marker} -->\n## {receipt['recorded_utc']} — User-requested Fusion shutdown\n\nFusion and its preview process exited normally using Pause unstarted jobs and Update Later; no force-kill or unsaved-change discard. Global free VRAM initial773 MiB/post-exit{free} MiB; changing training phase prevents exact attribution. Same R40 supervisor39000/one worker/55 hashes continue. No training pause, restart, source or policy change. Existing target8192/reserve300/host guard/180healthyTRAINING ramp remains. Report: {OUT/'REPORT.md'}.\n<!-- /{marker} -->\n"
for p in [TRAINING/'ITERATION_HISTORY.md',Path('E:/Aditya/robot_sim/docs/robot/TRAINING_JOURNAL_20261001.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
    assert marker not in p.read_text(encoding='utf-8-sig')
    with p.open('a',encoding='utf-8') as stream:stream.write(entry)
npath=TRAINING/'automation/NEXT_CHECK.json';n=read(npath)
n['resource_release']=receipt
n['resource_release_report']=str(OUT/'REPORT.md')
n['fusion_user_preference']='User explicitly authorized closing Fusion to free VRAM on October7. It exited normally, queued saves preserved for next launch. Do not reopen during ordinary training checks; measure fresh live-training capacity and allow existing memory ramp.'
npath.write_text(json.dumps(n,indent=2)+'\n')
handoff=f"<!-- {marker} -->\nUser-requested resource change {receipt['recorded_utc']}: Fusion and AdPreviewGenerator exited normally; queued unstarted uploads are preserved for next launch. No force-kill/unsaved discard. Same R40/39000/one worker/55 hashes continue. Global post-exit free VRAM {free} MiB; phase-dependent, not a matched benchmark. Read {OUT/'REPORT.md'}, fresh ACTIVE_RUN and state. Do not reopen Fusion for routine training checks. Existing8192 target/reserve300/host fallback/180healthyTRAINING ramp remains; no policy/source/restart change.\n<!-- /{marker} -->\n\n"
for p in [Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]:
    old=p.read_text(encoding='utf-8-sig');assert marker not in old;p.write_text(handoff+old,encoding='utf-8')
print(json.dumps({k:receipt[k] for k in ['recorded_utc','post_exit_gpu_MiB','operation','operation_environment_count','actual_worker_pids','frozen_hashes_verified']},indent=2))
