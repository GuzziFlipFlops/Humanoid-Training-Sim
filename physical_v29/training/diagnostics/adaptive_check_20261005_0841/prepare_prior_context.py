"""Isolated added prior-target correction, compared against the running-only arm."""
import json,hashlib,shutil
from pathlib import Path
OUT=Path(__file__).resolve().parent
source=OUT/'validation_source/physical_v29';target=OUT/'prior_context_source/physical_v29'
assert not target.exists()
shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc','usd'))
p=target/'training/upper_body_refinement.py';text=p.read_text()
text=text.replace('import torch\n', 'import torch,json\nfrom pathlib import Path\nfrom phase_transfer import context_inputs\n')
text=text.replace('        super().__init__(*args,**kwargs)', '''        super().__init__(*args,**kwargs)
        self.phase_context=json.loads((Path(__file__).parent/'phase_context.json').read_text())
        self.phase_tables={float(k):torch.tensor(v,device=self.actor.mlp[-1].weight.device,
            dtype=self.actor.mlp[-1].weight.dtype) for k,v in self.phase_context['contexts'].items() if float(k)<=1.}
        self.conditioned_samples=0''')
text=text.replace("            target=self.teacher(observations[ids]).clamp(-3.,3.)",'''            teacher_input=observations[ids].clone()
            if self.command_range is not None:
                # Change only teacher head history on physically validated exact
                # prior-command cohorts. Student input and MSE derivative stay.
                teacher_raw=teacher_input['policy'];frames=teacher_raw.reshape(-1,5,80)
                for command,table in self.phase_tables.items():
                    eligible=((frames[:,:,6]-command).abs()<1e-6).all(1)
                    eligible&=(frames[:,:,78:80].square().sum(-1)>.5).all(1)
                    eligible&=(frames[:,-1,self.phase_context['columns'][:2]].abs()<.174533).all(1)
                    selected=eligible.nonzero().flatten()
                    if len(selected):
                        teacher_raw[selected]=context_inputs(teacher_raw[selected],table,self.phase_context['columns'])
                        self.conditioned_samples+=len(selected)
            target=self.teacher(teacher_input).clamp(-3.,3.)''')
text=text.replace("        return result\n\r", "        return result\n\r")
text=text.replace("        result.update(output_scope=", "        result['teacher_phase_conditioned_samples']=self.conditioned_samples;self.conditioned_samples=0\n        result.update(output_scope=")
assert 'teacher_input=observations[ids].clone()' in text
p.write_text(text)
prior=Path(json.loads((OUT/'diagnostic_pause.json').read_text())['run'])/'source/physical_v29'
def inventory(root):return {f.relative_to(root).as_posix():f for f in root.rglob('*') if f.is_file() and f.suffix!='.pyc' and '__pycache__' not in f.parts and 'usd' not in f.parts}
a,b=inventory(prior),inventory(target);sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
changed={k for k in a.keys()|b.keys() if k not in a or k not in b or sha(a[k])!=sha(b[k])}
assert changed=={'training/worker_unitree.py','training/upper_body_refinement.py','training/phase_transfer.py','training/phase_context.json'}
(OUT/'prior_context_staging.json').write_text(json.dumps(dict(diagnostic_only=True,changed=sorted(changed),implementation_hashes={k:sha(b[k]) for k in changed},protected_files=len(set(a)-changed)),indent=2)+'\n')
script=(OUT/'run_matched_transfer.py').read_text().split('records={}')[0]
script=script.replace("'candidate':OUT/'validation_source/physical_v29'", "'candidate':OUT/'prior_context_source/physical_v29'")
script+='''records={}
folder,result=worker(sources['candidate'],'matched_prior_context','train',checkpoint,num_envs=4096,iterations=64,learning_rate=.0001,seed=71)
records['candidate']={'train':str(folder),'result':result,'evaluations':[]}
for perturb,seed in [(False,901),(True,1301)]:
    ef,er=worker(sources['candidate'],'matched_prior_context'+('_push' if perturb else '_nominal'),'eval',folder/'candidate.pt',
        **dict(num_envs=192,duration=20,seed=seed,**({'perturb':True} if perturb else {})))
    records['candidate']['evaluations'].append(str(ef))
reload,reload_result=worker(sources['candidate'],'prior_context_full_reload','train',folder/'candidate.pt',num_envs=64,iterations=4,learning_rate=.0001,seed=71)
records['reload']=str(reload);records['qualifies_locomotion']=False;records['diagnostic_actors_deployed']=False
(OUT/'matched_prior_context_completed.json').write_text(json.dumps(records,indent=2)+'\\n')
'''
(OUT/'run_matched_prior_context.py').write_text(script)
resume=OUT/'resume_after_diagnosis.py';text=resume.read_text().replace("'run_matched_extension.py'", "'run_matched_extension.py','run_matched_prior_context.py'");resume.write_text(text)
print('Prepared validated prior-query conditioning on .35/.65/1 only, with original prior derivative/weight.')
