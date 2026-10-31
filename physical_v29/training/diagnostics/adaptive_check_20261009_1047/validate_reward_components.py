"""Independent CPU validation of the completed experimental evidence."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;ROOT=OUT/'matched_slow_reward'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def check(name,value):
    checks.append(dict(name=name,passed=bool(value)));assert value,name

review=read(OUT/'matched_reward_review.json');pause=read(OUT/'diagnostic_pause.json')
check('Complete two-seed64/128 independent review',review['completed'] and len(review['rows'])==8)
complete=read(ROOT/'completion.json')
check('Harness completed before work-only limit',complete['completed'] and complete['work_seconds']<6600)
check('Exact current preserved checkpoint',sha(pause['checkpoint'])==pause['checkpoint_sha256'])
protected=read(OUT/'protected_source_copy.json')
check('Frozen authored and asset sources preserved',all(sha(Path(protected['source'])/n)==h and sha(OUT/'control_source/physical_v29'/n)==h for n,h in protected['protected_hashes'].items()))
cpu=read(OUT/'slow_reward_CPU_tests.json');parity=read(OUT/'physical_parity.json')
check('112 numeric/gradient fixtures and6 counterexample categories',cpu['passed'] and cpu['numerical_gradient_parity_cases']==112 and cpu['extra_categories']==6)
check('Every unchanged physical replay array and classifier case',parity['passed'] and parity['arrays_bitwise_equal']==51 and parity['all_cases_bitwise_equal'])
manifest=read(ROOT/'experiment_manifest.json')
check('Explicit changed-reward provenance',manifest['original_signature_does_not_certify_changed_training_reward'] and manifest['unchanged_all_weights'] and manifest['unchanged_faster_reward'])
check('Experimental kernel and wrapper recorded exactly',sha(OUT/'slow_tracking_kernel.py')==manifest['experimental_source_sha256'] and sha(OUT/'slow_reward_candidate.py')==manifest['wrapper_sha256'])

for name,count,source in [('passive_PPO_capture',4,pause['checkpoint']),('candidate_smoke',8,pause['checkpoint']),('candidate_full_reload',4,str(ROOT/'candidate_smoke/candidate.pt'))]:
    folder=ROOT/name;proof=read(folder/'full_load_proof.json');rows=[json.loads(r) for r in (folder/'progress.jsonl').read_text().splitlines()]
    check(name+' exact source full state',proof['source_sha256']==sha(source) and all(proof[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal']))
    check(name+' real PPO and unchanged three auxiliaries',len(rows)==count and all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 and all(r[k]['batches']==20 for k in ['retention','locomotion_retention','upper_refinement']) for r in rows))
for row in review['rows']:
    folder=Path(row['checkpoint']).parent
    source=pause['checkpoint'] if row['updates']==64 else ROOT/f'seed{row["seed"]}_{row["branch"]}_064_train/candidate.pt'
    check(folder.name+' load lineage',read(folder/'full_load_proof.json')['source_sha256']==sha(source))
    check(folder.name+' true64 updates and bounded memory',row['training_validation']['actual_updates']==64 and row['training_validation']['minimum_free_GPU_MiB']>=300 and row['training_validation']['post_warmup_torch_allocation_span_MiB']<64)
    check(folder.name+' original aux and explicit override',row['zero_KL_command_mismatch'] and row['original_auxiliaries_batches']==20 and row['executed_reward_override']['applied']==(row['branch']=='candidate'))
    for label in ['nominal','pushed']:
        check(folder.name+' '+label+' full trials',len(row[label+'_cases'])==10 and all(c['trials']==32 for c in row[label+'_cases']))
    if row['updates']==128:
        for label in ['diverse_nominal','diverse_push']:
            check(folder.name+' '+label+' independent32 states',len(row[label]['unique_initial_states_by_command'])==10 and all(n==32 for n in row[label]['unique_initial_states_by_command'].values()))
for seed in [179,191]:
    pair=[r for r in review['rows'] if r['seed']==seed and r['updates']==128]
    for label in ['diverse_nominal','diverse_push']:
        check('seed'+str(seed)+' paired'+label+' initial states',pair[0][label]['initial_state_sha256']==pair[1][label]['initial_state_sha256'])

physical=read(OUT/'motor_retention_response/result.json');motor=read(OUT/'motor_substep_attribution.json');objective=read(OUT/'physical_reward_counterfactual.json')
check('Current and qualified-prior832 robot native physical comparison',physical['completed'] and physical['actor_loaded_exact'] and physical['maximum_command_difference']==0 and len(physical['groups'])==26 and all(g['robots']==32 and g['survival']==1 for g in physical['groups']))
check('Full physical reward sum reconstructs',physical['maximum_reward_term_sum_error']<1e-3)
check('Independent5ms requested/applied driving attribution',motor['passed'] and motor['independent_saved_numpy_driving_clip_mask_matches'] and motor['checkpoint_sha256']==pause['checkpoint_sha256'])
check('Original learner safely resumed',read(OUT/'diagnostic_resume.json')['same_supervisor'] and not read(OUT/'diagnostic_resume.json')['diagnostic_actors_deployed'])
result=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),completed=True,passed=True,count=len(checks),checks=checks,
    diagnostic_learning_updates=4+8+4+8*64,diagnostic_learning_excluded_from_main_progress=True,
    original_signature_does_not_certify_experimental_reward=True,
    candidate_acceptance_not_implied_by_execution_checks=True)
(OUT/'experiment_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
