"""Stage a narrow reward/flight-verification repair; does not deploy it."""
import hashlib,json,shutil,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
d=json.loads((OUT/'drain.json').read_text());prior=Path(d['run'])/'source/physical_v29/training'
sys.path.insert(0,str(prior))
from curriculum_provenance import training_reward_signature
candidate=OUT/'candidate_source/training';candidate.mkdir(parents=True,exist_ok=True)
before=OUT/'canonical_before';before.mkdir(exist_ok=True)
names=['stage_rewards.py','evaluation.py','reward_warm_start.py']
for name in names:
    assert (HERE/name).read_bytes()==(prior/name).read_bytes(),('Unexpected workspace divergence',name)
    if not (before/name).exists():shutil.copy2(HERE/name,before/name)
shutil.copy2(prior/'mass_estimate.json',candidate/'mass_estimate.json')
old=(prior/'stage_rewards.py').read_text()
new=old.replace('import torch\n','import torch\nimport json\nfrom pathlib import Path\nTOTAL_WEIGHT_N=json.loads((Path(__file__).parent/\'mass_estimate.json\').read_text())[\'total_mass_kg\']*9.81\n',1)
new=new.replace("REWARD_VERSION = 'cad_heading_alignment_retention_v6'","REWARD_VERSION = 'cad_running_phase_unload_v7'")
new=new.replace('def shaped_terms(','''def running_unload_quality(force, expected_swing, total_weight_N=TOTAL_WEIGHT_N):
    """Bounded positive swing-force feedback; quarter body-weight e-fold scale.

    This local adaptation follows periodic force composition. A partially
    unloaded foot receives continuous feedback before crossing contact gates.
    It is not a physical-flight classifier; the separate bonus requires both
    actual soles above1mm and both normal forces below1N.
    """
    score=torch.exp(-force.clamp_min(0.)/(total_weight_N*.25))
    return (score*expected_swing).sum(1)/expected_swing.sum(1).clamp_min(1)


def shaped_terms(''',1)
token="        running_flight=.25*(~metrics['contact'].any(1))*healthy*activity*running)"
assert token in new
new=new.replace(token,"        running_flight=.25*((metrics['clearance']>.001).all(1)&(metrics['foot_force'].abs()<1.).all(1))*healthy*activity*running,\n        running_unload=.5*running_unload_quality(metrics['foot_force'],expected_swing)*healthy*moving*running)")
new=new.replace('                gait_and_clearance_speed_multiplier=False,',"                running_unload_weight=.5,running_unload_force_scale_N=TOTAL_WEIGHT_N*.25,\n                running_unload_source='Local adaptation informed by Siekmann2011.01387 periodic force composition',\n                running_flight_requires='Both soles above1mm and both normal forces below1N',\n                gait_and_clearance_speed_multiplier=False,")
(candidate/'stage_rewards.py').write_text(new,encoding='utf-8',newline='\n')
old=(prior/'evaluation.py').read_text()
new=old.replace("GATE_VERSION='physical_landing_v5_fixed_direction'","GATE_VERSION='physical_landing_v6_geometric_flight'")
new=new.replace("  if speed>=1.5:checks['flight']=.015<stats['flight_fraction']<.45","  if speed>=1.5:\n   checks['flight']=.015<stats['flight_fraction']<.45\n   checks['geometric_flight']=.015<stats['geometric_flight_fraction']<.45")
new=new.replace("c.all(1),~c.any(1),m['drift'],outside],1)","c.all(1),~c.any(1),m['drift'],outside,((clearance>.001).all(1)&(m['foot_force'].abs()<1.).all(1))],1)")
new=new.replace('   two_feet_fraction=mean_col(12),flight_fraction=mean_col(13),', '   geometric_flight_fraction=mean_col(16),\n   two_feet_fraction=mean_col(12),flight_fraction=mean_col(13),')
assert new!=old and 'geometric_flight_fraction=mean_col(16)' in new
(candidate/'evaluation.py').write_text(new,encoding='utf-8',newline='\n')
old=(prior/'reward_warm_start.py').read_text()
sig=training_reward_signature(prior)
new=old.replace("INITIAL_EXPLORATION_STD = .35",f"RUNNING_SOURCE_REWARD = 'cad_heading_alignment_retention_v6'\nRUNNING_SOURCE_SIGNATURE = '{sig}'\nINITIAL_EXPLORATION_STD = .35")
new=new.replace('assert source in (SOURCE_REWARD,GAIT_SOURCE_REWARD),','assert source in (SOURCE_REWARD,GAIT_SOURCE_REWARD,RUNNING_SOURCE_REWARD),')
new=new.replace("    else:\n        assert float(info.get('speed',-1))", "    elif source==RUNNING_SOURCE_REWARD:\n        assert float(info.get('speed',-1))==1.5, 'Use the physically audited R31 running actor'\n        assert info.get('reward_signature')==RUNNING_SOURCE_SIGNATURE, 'v6 source differs from audited R31'\n    else:\n        assert float(info.get('speed',-1))",1)
new=new.replace('preserve_trained_exploration=source==GAIT_SOURCE_REWARD','preserve_trained_exploration=source in (GAIT_SOURCE_REWARD,RUNNING_SOURCE_REWARD)')
new=new.replace("'Only audited v29 v3-standing or v4-gait actor transfers are accepted'","'Only explicitly source-bound v29 actor transfers are accepted'")
(candidate/'reward_warm_start.py').write_text(new,encoding='utf-8',newline='\n')
proof=dict(prior_run=d['run'],source_checkpoint=d['checkpoint'],source_sha256=d['checkpoint_sha256'],
    source_reward_signature=sig,changed_files=names,changes='Added positive bounded running-only phase force feedback; require geometric flight in reward and new stricter gate; audited v6 actor-only transfer preserving exploration, fresh critic/Adam, fresh qualification',
    preserves='Walking/standing reward bytes within shaped_terms except zero running-only term; all inference/action/reset/phase/cadence/physics/motor/PPO sources unchanged',
    candidate_sha256={n:hashlib.sha256((candidate/n).read_bytes()).hexdigest() for n in names},deployed=False)
(OUT/'candidate_proof.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof,indent=2))
