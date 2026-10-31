import json, collections
from pathlib import Path
import cv2
from PIL import Image, ImageDraw

P=Path('E:/Aditya/robot_sim')
root=P/'physical_v29/training'
root.mkdir(exist_ok=True)
a=json.loads((P/'outputs/shin_cad_20260912/v29/mesh_manifest.json').read_text())['bodies']
groups=collections.defaultdict(lambda: [0,0.,set()])
for b in a:
    k=(b['link'],b['part'])
    groups[k][0]+=1; groups[k][1]+=b.get('volume_mm3',0.) or 0.; groups[k][2].add(b['appearance'])
rows=[dict(link=k[0],part=k[1],count=v[0],volume_cm3=v[1]/1000,appearances=sorted(v[2])) for k,v in groups.items()]
(root/'cad_part_volumes.json').write_text(json.dumps(rows,indent=2))
print('LARGEST CAD GROUPS',json.dumps(sorted(rows,key=lambda r:r['volume_cm3'],reverse=True)[:42],indent=1))
for speed in ['walk_1ms','run_4ms','run_5ms']:
    folder=P/'humanoid_project/reviews/working_locomotion_20260905_d'/speed
    files=list((folder/'video').glob('*.mp4'))
    if not files: files=list(folder.glob('*.mp4'))
    if not files: continue
    cap=cv2.VideoCapture(str(files[0])); fps=cap.get(cv2.CAP_PROP_FPS)
    sheet=Image.new('RGB',(1200,750),(20,20,20)); d=ImageDraw.Draw(sheet)
    for j,t in enumerate([3.,3.2,3.4,3.6,5.,7.,9.,11.,13.]):
        cap.set(cv2.CAP_PROP_POS_MSEC,t*1000); ok,im=cap.read()
        if ok:
            img=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)); img.thumbnail((400,225))
            x=(j%3)*400; y=(j//3)*250
            sheet.paste(img,(x,y)); d.text((x+8,y+225),f'{speed}, {t:.1f} sec',fill='white')
    dest=root/f'old_{speed}_frames.jpg'; sheet.save(dest); print('REVIEW',files[0],dest)
    cap.release()
