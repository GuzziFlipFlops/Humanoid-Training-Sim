"""Save official public gait examples and pinned secondary reward references."""
import hashlib,json,urllib.request
from pathlib import Path
from PIL import Image,ImageDraw,ImageOps
HERE=Path(__file__).parent
OUT=HERE/'diagnostics/unitree_port_20261002/reference';OUT.mkdir(parents=True,exist_ok=True)
manifest=[]
def get(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'robot-reward-validation'}),timeout=45).read()
for repo,paths in {
 'unitree_rl_mjlab':['src/tasks/velocity/config/h2/env_cfgs.py','src/tasks/velocity/velocity_env_cfg.py','src/tasks/velocity/mdp/rewards.py'],
 'unitree_rl_gym':['legged_gym/envs/h1/h1_config.py','legged_gym/envs/h1/h1_env.py','legged_gym/envs/g1/g1_config.py','legged_gym/envs/g1/g1_env.py']}.items():
    commit=json.loads(get('https://api.github.com/repos/unitreerobotics/'+repo+'/commits/main'))['sha']
    for path in paths:
        url=f'https://raw.githubusercontent.com/unitreerobotics/{repo}/{commit}/{path}'
        try:data=get(url)
        except Exception as error:manifest.append(dict(url=url,error=str(error)));continue
        file=repo+'_'+path.replace('/','_');(OUT/file).write_bytes(data)
        manifest.append(dict(file=file,url=url,commit=commit,sha256=hashlib.sha256(data).hexdigest()))
for name,url in {
 'g1_isaaclab':'https://oss-global-cdn.unitree.com/static/d879adac250648c587d3681e90658b49_480x397.gif',
 'g1_physical':'https://oss-global-cdn.unitree.com/static/6c17c6cf52ec4e26bbfab1fbf591adb2_480x270.gif'}.items():
    data=get(url);file=OUT/(name+'.gif');file.write_bytes(data);gif=Image.open(file)
    frames=[];time=0.
    for i in range(gif.n_frames):
        gif.seek(i);frame=gif.convert('RGB');dt=gif.info.get('duration',100)/1000
        frames.append((time,frame.copy()));time+=dt
    # Every original frame is included, in readable 4x3 sheets.
    for start in range(0,len(frames),12):
        sheet=Image.new('RGB',(1280,3*292),'white');draw=ImageDraw.Draw(sheet)
        for k,(t,frame) in enumerate(frames[start:start+12]):
            x=(k%4)*320;y=(k//4)*292
            img=ImageOps.contain(frame,(320,265));sheet.paste(img,(x+(320-img.width)//2,y+23))
            draw.text((x+6,y+4),f'{name} frame {start+k} t={t:.3f}s',fill='black')
        sheet.save(OUT/f'{name}_sheet_{start//12:02d}.jpg',quality=92)
    manifest.append(dict(file=file.name,url=url,sha256=hashlib.sha256(data).hexdigest(),frames=len(frames),seconds=time))
(OUT/'sources.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
