"""Timing-preserving H.264 encoding, complete decode and visual QA sheets."""
import hashlib,json,subprocess
from pathlib import Path
import cv2,imageio_ffmpeg,numpy as np
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
DEST=Path(json.loads((OUT/'capture_config.json').read_text())['destination'])
manifest=json.loads((DEST/'manifest.json').read_text());ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();results=[]
for clip in manifest['clips']:
    target=Path(clip['video']);assert not target.exists(),'Preserve encoded output'
    command=[ffmpeg,'-hide_banner','-loglevel','warning','-i',clip['raw_video'],
             '-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
    completed=subprocess.run(command,capture_output=True,text=True)
    (DEST/(target.stem+'_encode.log')).write_text(completed.stderr)
    assert completed.returncode==0,completed.stderr
    cap=cv2.VideoCapture(str(target));assert cap.isOpened()
    fps=cap.get(cv2.CAP_PROP_FPS);declared=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));count=0;selected=[];hashes=[];motion=[];last=None
    while True:
        ok,frame=cap.read()
        if not ok:break
        assert frame.shape==(720,1280,3) and frame.std()>2.
        roi=frame[80:,300:980];hashes.append(hashlib.sha256(roi.tobytes()).hexdigest())
        if last is not None:motion.append(float(np.abs(roi.astype(np.float32)-last).mean()))
        last=roi.astype(np.float32)
        if count in (0,49,99,149,199):
            selected.append(frame);cv2.imwrite(str(DEST/f'{target.stem}_decoded_{count:03d}.png'),frame)
        count+=1
    cap.release()
    assert count==declared==200 and abs(fps-50)<1e-6 and abs(count/fps-4)<1e-6
    assert len(set(hashes))>=195,'Repeated/frozen rendered content; exclude top counter from check'
    rows=json.loads((DEST/(target.stem+'_telemetry.json')).read_text());assert len(rows)==200
    assert len({tuple(r['q']) for r in rows})==200,'Frozen physical motion'
    tiles=[cv2.resize(frame,(640,360)) for frame in selected]+[np.zeros((360,640,3),dtype=np.uint8)]
    cv2.imwrite(str(DEST/(target.stem+'_inspection.jpg')),np.concatenate([np.concatenate(tiles[:3],axis=1),np.concatenate(tiles[3:],axis=1)],axis=0))
    closeups=[cv2.resize(frame[55:675,390:885],(320,400)) for frame in selected]
    cv2.imwrite(str(DEST/(target.stem+'_closeups.jpg')),np.concatenate(closeups,axis=1))
    result={'video':str(target),'decoded_frames':count,'fps':fps,'duration_s':count/fps,'resolution':[1280,720],
        'all_frames_nonempty':True,'unique_image_roi_frames':len(set(hashes)),'unique_physical_q_frames':200,
        'mean_consecutive_roi_difference':float(np.mean(motion)),'bytes':target.stat().st_size,
        'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'falls':clip['falls'],
        'command_m_s':clip['command_m_s'],'mean_signed_world_forward_speed_m_s':clip['mean_signed_world_forward_speed_m_s'],
        'record_wall_seconds':clip['record_wall_seconds'],'geometric_flight_fraction':clip['geometric_flight_fraction'],
        'checkpoint':clip['checkpoint'],'checkpoint_sha256':clip['sha256'],
        'visual_inspection_frame_indices':[0,49,99,149,199]}
    results.append(result)
manifest['encoded_verified']=results;(DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(OUT/'decode_verification.json').write_text(json.dumps({'passed':True,'clips':results},indent=2)+'\n')
print(json.dumps(results,indent=2))
