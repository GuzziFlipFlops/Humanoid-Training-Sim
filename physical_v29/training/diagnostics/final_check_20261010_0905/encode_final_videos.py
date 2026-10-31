"""Normal-speed H264 export, complete decode and exact frame/native-sample audit."""
import hashlib,json,subprocess,time,sys
from pathlib import Path
import cv2,imageio_ffmpeg,numpy as np
OUT=Path(__file__).resolve().parent;DEST=Path(json.loads((OUT/'capture_config.json').read_text(encoding='utf-8-sig'))['destination'])
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
write=lambda p,d:p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
complete=(DEST/'manifest.json').exists()
assert complete or '--available' in sys.argv,'Full capture not complete'
manifest=read(DEST/('manifest.json' if complete else 'capture_progress.json'));ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();results=[]
previous=read(OUT/'decode_progress.json')['clips'] if (OUT/'decode_progress.json').exists() else []
gates=read(OUT/'selected_diverse_nominal/evaluation.json')['cases']
for clip in manifest['clips']:
 target=Path(clip['video']);case=next(c for c in gates if abs(c['statistics']['command_m_s']-clip['speed'])<1e-6)
 if '--available' in sys.argv:
  prior=next((r for r in previous if r['video']==str(target)),None)
  if prior and target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==prior['sha256']:
   results.append(prior);continue
 failures=', '.join(case['failures']) or 'this command passes; full-stage retention fails'
 # Full actor SHA and the exact per-command seeded gate scope are burned into pixels.
 safe_failure=failures.replace(':','-').replace("'",'')
 vf="drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':text='Actor SHA256 "+clip['sha256']+"':x=24:y=666:fontsize=13:fontcolor=white:box=1:boxcolor=black@0.65,drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':text='32-state nominal gate - "+safe_failure+"':x=24:y=688:fontsize=14:fontcolor=white:box=1:boxcolor=black@0.65"
 if not target.exists():
  cmd=[ffmpeg,'-hide_banner','-loglevel','warning','-i',clip['raw_video'],'-an','-vf',vf,'-c:v','libx264','-threads','2','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
  r=subprocess.run(cmd,capture_output=True,text=True);(DEST/(target.stem+'_encode.log')).write_text(r.stderr,encoding='utf-8');assert r.returncode==0,r.stderr
 cap=cv2.VideoCapture(str(target));assert cap.isOpened()
 fps=cap.get(cv2.CAP_PROP_FPS);declared=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));count=0;selected=[];hashes=[];motion=[];last=None;pts=[]
 while True:
  ok,frame=cap.read()
  if not ok:break
  assert frame.shape==(720,1280,3) and frame.std()>2
  roi=frame[80:660,300:980];hashes.append(hashlib.sha256(roi.tobytes()).hexdigest());pts.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
  if last is not None:motion.append(float(np.abs(roi.astype(np.float32)-last).mean()))
  last=roi.astype(np.float32)
  if count in (0,124,249,374,499):selected.append(frame);cv2.imwrite(str(DEST/f'{target.stem}_decoded_{count:03d}.png'),frame)
  count+=1
 cap.release();assert count==declared==500 and abs(fps-50)<1e-6 and abs(count/fps-10)<1e-6
 rows=read(Path(clip['alignment']));assert len(rows)==500
 np.testing.assert_allclose([r['video_time_s'] for r in rows],np.arange(500)*.02,atol=1e-8)
 np.testing.assert_allclose(pts,np.arange(500)*.02,atol=.001)
 with np.load(clip['motor_npz']) as z:
  assert z['position_rad'].shape==(2400,1,23) and len(z['interval_start_s'])==2400
  for i,row in enumerate(rows):
   a,b=row['motor_first_sample'],row['motor_last_sample'];assert a==400+i*4 and b==a+3
   assert abs(z['interval_end_s'][b]-row['post_step_sim_time_s'])<1e-7
  assert np.all(np.isfinite(z['applied_torque_Nm'])) and np.all(np.isfinite(z['velocity_rad_s']))
  np.testing.assert_allclose(z['signed_mechanical_power_W'],z['applied_torque_Nm']*z['velocity_rad_s'],rtol=2e-6,atol=.0003)
 unique_q=len({tuple(r['q']) for r in rows});unique_images=len(set(hashes))
 if clip['speed']>0:assert unique_q>=490 and unique_images>=490,'Frozen locomotion frames'
 tiles=[cv2.resize(f,(640,360)) for f in selected]+[np.zeros((360,640,3),dtype=np.uint8)]
 cv2.imwrite(str(DEST/(target.stem+'_inspection.jpg')),np.concatenate([np.concatenate(tiles[:3],axis=1),np.concatenate(tiles[3:],axis=1)],axis=0))
 closeups=[cv2.resize(f[70:660,400:860],(320,410)) for f in selected];cv2.imwrite(str(DEST/(target.stem+'_closeups.jpg')),np.concatenate(closeups,axis=1))
 result=dict(video=str(target),decoded_frames=count,fps=fps,duration_s=count/fps,resolution=[1280,720],all_frames_nonempty=True,unique_image_roi_frames=unique_images,unique_physical_q_frames=unique_q,mean_consecutive_roi_difference=float(np.mean(motion)),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),falls=clip['falls'],command_m_s=clip['speed'],episode_measured_signed_world_speed_m_s=clip['mean_signed_world_forward_speed_m_s'],record_wall_seconds=clip['record_wall_seconds'],checkpoint_sha256=clip['sha256'],frame_sample_alignment_passed=True,full_decoded_timestamps_match=True,visual_inspection_frame_indices=[0,124,249,374,499],seeded_population_case_passed=case['passed'],population_failures=case['failures'])
 results.append(result);write(OUT/'decode_progress.json',dict(clips=results));print(json.dumps(result),flush=True)
if complete:
 assert len(results)==10
 manifest['encoded_verified']=results;write(DEST/'manifest.json',manifest);write(OUT/'decode_verification.json',dict(passed=True,clips=results))
 print('All ten normal-speed videos completely decoded and motor/frame alignment verified.',flush=True)
else:
 print(f'{len(results)} available clips encoded and fully decoded; capture continues with learning stopped.',flush=True)
