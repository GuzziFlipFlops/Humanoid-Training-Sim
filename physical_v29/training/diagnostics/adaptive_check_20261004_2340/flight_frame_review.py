"""Extract decoded frames at telemetry-confirmed takeoff and neighboring ticks."""
import json
from pathlib import Path
import cv2,numpy as np
OUT=Path(__file__).resolve().parent;DEST=Path(json.loads((OUT/'capture_config.json').read_text())['destination'])
rows=json.loads((DEST/'locomotion_150_telemetry.json').read_text())
flight=[i for i,r in enumerate(rows) if min(r['clearance'])>.001 and max(r['force'])<1]
assert flight
centers=[flight[0],flight[len(flight)//2],flight[-1]]
indices=sorted(set(max(0,min(499,c+d)) for c in centers for d in (-1,0,1)))
cap=cv2.VideoCapture(str(DEST/'locomotion_150.mp4'));tiles=[];receipt=[]
for i in indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,frame=cap.read();assert ok
    tile=cv2.resize(frame[55:675,390:885],(320,400))
    cv2.putText(tile,f'frame{i} t={rows[i]["time_s"]:.2f}s',(3,20),cv2.FONT_HERSHEY_SIMPLEX,.43,(15,15,15),1,cv2.LINE_AA)
    tiles.append(tile);receipt.append({'frame':i,'time_s':rows[i]['time_s'],'sole_clearance_m':rows[i]['clearance'],'foot_force_N':rows[i]['force']})
cap.release()
while len(tiles)%3:tiles.append(np.zeros_like(tiles[0]))
sheet=np.concatenate([np.concatenate(tiles[i:i+3],axis=1) for i in range(0,len(tiles),3)],axis=0)
cv2.imwrite(str(DEST/'locomotion_150_flight_frames.jpg'),sheet)
(OUT/'flight_frame_samples.json').write_text(json.dumps({'flight_frames':flight,'review_samples':receipt,'scope':'Single-recorded robot; independent full-population gates remain separate.'},indent=2)+'\n')
print(json.dumps({'flight_frames':flight,'review_indices':indices}))
