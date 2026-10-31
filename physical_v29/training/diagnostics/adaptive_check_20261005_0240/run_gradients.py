import json,subprocess,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent;d=json.loads((OUT/'diagnostic_pause.json').read_text());r=Path(d['run']);checkpoint=Path(d['checkpoint']);prior=json.loads((checkpoint.parent/'standing_retention.json').read_text());cert=checkpoint.parent/'qualified_prior_certificate.json'
assert cert.exists(),cert
cmd=[sys.executable,'-u','-X','utf8',str(OUT/'capture_gradients.py'),'--headless','--mode','train','--output',str(OUT/'gradient_ppo'),'--checkpoint',str(checkpoint),'--speed','1.5','--num_envs','64','--iterations','4','--learning_rate','0.0001','--standing_prior',prior['teacher'],'--standing_prior_sha256',prior['sha256'],'--locomotion_prior_certificate',str(cert)]
(OUT/'gradient_command.json').write_text(json.dumps(cmd,indent=2))
with (OUT/'gradient_stdout.log').open('w') as o,(OUT/'gradient_stderr.log').open('w') as e:
 result=subprocess.run(cmd,stdout=o,stderr=e)
print('gradient worker exit',result.returncode);sys.exit(result.returncode)
