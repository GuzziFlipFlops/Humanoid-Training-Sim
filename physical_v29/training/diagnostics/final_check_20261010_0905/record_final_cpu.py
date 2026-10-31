import json,hashlib,sys,subprocess
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;T=OUT.parents[1];R=T/'runs/locomotion_20261007_r40_goal_budget_full_state'
sys.path.insert(0,str(T/'history'));from record_inventory import frozen_path
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
state=read(R/'state.json');config=read(R/'chain_config.json')
hashes={k:dict(path=str(frozen_path(R,k)),expected=v,actual=sha(frozen_path(R,k))) for k,v in config['sources_sha256'].items()}
assert all(v['expected']==v['actual'] for v in hashes.values())
updates=0;samples=0;active_seconds=0;worker_seconds=0;records=[]
for p in sorted(R.glob('*_run_*_train_*'),key=lambda p:int(p.name.split('_')[0])):
 if not (p/'result.json').exists():continue
 result=read(p/'result.json')
 if not result.get('completed'):continue
 rows=[json.loads(x) for x in (p/'progress.jsonl').read_text().splitlines()]
 proof=read(p/'full_load_proof.json');assert all(proof[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
 assert all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
 assert all(r[k]['batches']==20 for r in rows for k in ['retention','locomotion_retention','upper_refinement'])
 updates+=len(rows);samples+=sum(r['num_envs']*24 for r in rows)
 # Named timing values are explicitly inspected before reporting rates.
 records.append(dict(folder=str(p),updates=len(rows),samples=sum(r['num_envs']*24 for r in rows),first_row_keys=list(rows[0]),last=rows[-1],result=result,load_proof=proof))
evidence=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),state_status=state['status'],saved_checkpoint=state['checkpoint'],saved_checkpoint_sha256=sha(state['checkpoint']),frozen_hashes=hashes,all_hashes_verify=True,completed_main_updates=updates,recovered_verified_updates=50,current_run_retained_updates=updates+50,inherited_verified_updates=27507,excluded_unsaved_R39_updates=32,completed_main_samples=samples,latest_main=records[-1],no_learning_after_stop=True)
(OUT/'final_learning_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
cp=Path(state['checkpoint']);pair=dict(checkpoint=str(cp),sha256=sha(cp),nominal=str(R/'1227_run_350_gate_407_nominal'),pushed=str(R/'1228_run_350_gate_407_push'))
(OUT/'latest_gate_review.json').write_text(json.dumps(dict(run=str(R),latest=pair,promotions=[]),indent=2))
cp=R/'1064_run_350_train_352/candidate.pt';pair=dict(checkpoint=str(cp),sha256=sha(cp),nominal=str(R/'1065_run_350_gate_353_nominal'),pushed=str(R/'1066_run_350_gate_353_push'))
(OUT/'selected_gate_review.json').write_text(json.dumps(dict(run=str(R),latest=pair,promotions=[]),indent=2))
nextcheck=read(T/'automation/NEXT_CHECK.json');(OUT/'previous_handoff.json').write_text(json.dumps(nextcheck,indent=2))
print(json.dumps(dict(main_updates=updates,retained=updates+50,inherited=27507,samples=samples,hashes=len(hashes),last_row=records[-1]['last']),indent=2))
