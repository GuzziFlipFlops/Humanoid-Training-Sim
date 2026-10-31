from pathlib import Path
OUT=Path(__file__).resolve().parent
text=(OUT/'run_matched_transfer.py').read_text().split('records={}')[0]
text+='''records={}
for name,source in sources.items():
    cp=OUT/('matched_'+name)/'candidate.pt'
    folder,result=worker(source,'continued_'+name,'train',cp,num_envs=4096,iterations=64,learning_rate=.0001,seed=71)
    records[name]={'train':str(folder),'result':result,'evaluations':[]}
    for perturb,seed in [(False,901),(True,1301)]:
        ef,er=worker(source,'continued_'+name+('_push' if perturb else '_nominal'),'eval',folder/'candidate.pt',
            **dict(num_envs=192,duration=20,seed=seed,**({'perturb':True} if perturb else {})))
        records[name]['evaluations'].append(str(ef))
    (OUT/'matched_extension_progress.json').write_text(json.dumps(records,indent=2)+'\\n')
records['diagnostic_actors_deployed']=False;records['qualifies_locomotion']=False
(OUT/'matched_extension_completed.json').write_text(json.dumps(records,indent=2)+'\\n')
'''
(OUT/'run_matched_extension.py').write_text(text)
text=(OUT/'analyze_matched_transfer.py').read_text()
text=text.replace("folder=OUT/('matched_'+name)","folder=OUT/('continued_'+name)")
text=text.replace("OUT/('matched_'+name+'_nominal')", "OUT/('continued_'+name+'_nominal')").replace("OUT/('matched_'+name+'_push')", "OUT/('continued_'+name+'_push')")
text=text.replace("matched_comparison.json", "matched_extension_comparison.json")
(OUT/'analyze_matched_extension.py').write_text(text)
text=(OUT/'resume_after_diagnosis.py').read_text().replace("'run_matched_transfer.py'", "'run_matched_transfer.py','run_matched_extension.py'")
(OUT/'resume_after_diagnosis.py').write_text(text)
print('Prepared second equal64-update full-state block; original checkpoint and production remain unchanged.')
