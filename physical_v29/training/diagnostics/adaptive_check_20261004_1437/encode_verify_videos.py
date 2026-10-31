"""Encode H.264 without changing timing; verify all 500 frames and 10 seconds."""
import hashlib,json,subprocess
from pathlib import Path
import cv2,imageio_ffmpeg,numpy as np
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1];DEST=HERE/'videos/walking_gui_20261004'
manifest=json.loads((DEST/'manifest.json').read_text())
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();results=[]
for clip in manifest['clips']:
    target=Path(clip['video']);assert not target.exists(),'Preserve encoded output'
    command=[ffmpeg,'-hide_banner','-loglevel','warning','-i',clip['raw_video'],
             '-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
    completed=subprocess.run(command,capture_output=True,text=True)
    (DEST/(target.stem+'_encode.log')).write_text(completed.stderr)
    assert completed.returncode==0,completed.stderr
    cap=cv2.VideoCapture(str(target));assert cap.isOpened()
    fps=cap.get(cv2.CAP_PROP_FPS);declared=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));count=0;means=[];selected=[]
    while True:
        ok,frame=cap.read()
        if not ok:break
        assert frame.shape==(720,1280,3) and frame.std()>2.,'Invalid rendered frame'
        if count in (0,125,250,375,499):selected.append(frame)
        means.append(float(frame.mean()));count+=1
    cap.release()
    assert count==declared==500 and abs(fps-50)<1e-6 and abs(count/fps-10)<1e-6
    tiles=[cv2.resize(frame,(640,360)) for frame in selected]
    tiles.append(np.zeros_like(tiles[0]))
    sheet=np.concatenate([np.concatenate(tiles[:3],axis=1),np.concatenate(tiles[3:],axis=1)],axis=0)
    cv2.imwrite(str(DEST/(target.stem+'_inspection.jpg')),sheet)
    results.append({'video':str(target),'decoded_frames':count,'fps':fps,'duration_s':count/fps,
        'resolution':[1280,720],'all_frames_nonempty':True,'bytes':target.stat().st_size,
        'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'falls':clip['falls'],
        'command_m_s':clip['command_m_s'],'mean_signed_world_forward_speed_m_s':clip['mean_signed_world_forward_speed_m_s'],
        'record_wall_seconds':clip['record_wall_seconds']})
manifest['encoded_verified']=results
(DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(OUT/'video_verification.json').write_text(json.dumps({'passed':True,'clips':results},indent=2)+'\n')
lines=['# Walking GUI recordings — October 4, 2026','',
       'Actual Isaac Sim 5.1 GUI rendering of the frozen physical-v29 training robot and saved R32 actor. The training supervisor was suspended and its worker drained throughout capture. These are deterministic policy demonstrations, not independent qualification tests.','',
       'Each MP4 contains exactly 500 control frames at 50 FPS: 10 simulated seconds played in 10 real seconds. Recording took longer in wall time. The first two simulated seconds used to settle the gait are excluded. The camera follows from the front and side; floor guide lines are visual only and spaced one metre apart.','',
       'The visible robot is the simulation geometry used by this training run. No motor, action, joint, gravity or collision parameter was changed to make the clips.','',
       '| Target speed | Measured mean forward speed | Recording wall time | Video |','| --- | --- | --- | --- |']
for r in results:lines.append(f"| {r['command_m_s']:.2f} m/s | {r['mean_signed_world_forward_speed_m_s']:.3f} m/s | {r['record_wall_seconds']:.1f} s | [{Path(r['video']).name}]({Path(r['video']).name}) |")
lines+=['',f"Checkpoint: `{manifest['checkpoint']}`.",f"SHA256 `{manifest['checkpoint_sha256']}`.",'','Frame checks, per-clip measured timing and synchronized joint/root/contact/command data are preserved beside these files in manifest.json and *_telemetry.json. Original mp4v encodes are also preserved.']
(DEST/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(results,indent=2))
