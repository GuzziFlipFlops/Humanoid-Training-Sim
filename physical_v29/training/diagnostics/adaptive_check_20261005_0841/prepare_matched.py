from pathlib import Path
OUT=Path(__file__).resolve().parent;OLD=OUT.parent/'adaptive_check_20261005_0540'
text=(OLD/'run_matched_exposure.py').read_text()
text=text.replace("'worker_unitree.py','physical_head_causal.py'", "'worker_unitree.py','physical_prior_causal.py','physical_phase_causal.py'")
text=text.replace("records={}\nparity_folder", "records={}\nsmoke_folder,smoke_result=worker(sources['candidate'],'candidate_smoke','train',checkpoint,num_envs=64,iterations=4,learning_rate=.0001,seed=71)\nrecords['smoke']=str(smoke_folder)\nparity_folder")
text=text.replace('matched_exposure_progress.json','matched_transfer_progress.json').replace('matched_exposure_completed.json','matched_transfer_completed.json')
(OUT/'run_matched_transfer.py').write_text(text)
worker=OUT/'validation_source/physical_v29/training/worker_unitree.py';text=worker.read_text()
text=text.replace("   transfer_saved=torch.load(transfer_path", "   assert sha(transfer_path)=='d41aeaaef963952487f01917f7defba618cbad4709f94b2e09bfd779e9b5e821'\n   transfer_saved=torch.load(transfer_path")
worker.write_text(text)
import json,hashlib
p=OUT/'staging.json';record=json.loads(p.read_text());record['implementation_hashes']['training/worker_unitree.py']=hashlib.sha256(worker.read_bytes()).hexdigest();p.write_text(json.dumps(record,indent=2)+'\n')
print('Prepared exact full-state sequential smoke, parity, matched PPO/gates and reload.')
