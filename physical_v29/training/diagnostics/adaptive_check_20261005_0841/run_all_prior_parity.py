from pathlib import Path
OUT=Path(__file__).resolve().parent
source=(OUT/'run_matched_all_prior.py').read_text().split('records={}')[0]
source+="\nfolder,result=worker(sources['candidate'],'all_prior_same_actor_replay','eval',checkpoint,num_envs=192,duration=20,seed=901)\n"
exec(compile(source,__file__,'exec'))
text=(OUT/'check_parity.py').read_text().replace("OUT/'candidate_same_actor_replay'","OUT/'all_prior_same_actor_replay'").replace("OUT/'physical_parity.json'","OUT/'all_prior_physical_parity.json'")
exec(compile(text,str(OUT/'check_parity.py'),'exec'))
