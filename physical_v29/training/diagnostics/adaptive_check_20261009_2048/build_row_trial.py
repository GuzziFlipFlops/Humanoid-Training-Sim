"""Stage a bounded, sequential three-route comparison with explicit decision gate."""
from pathlib import Path
import hashlib,json
OUT=Path(__file__).resolve().parent
OLD=OUT.parent/'adaptive_check_20261009_1548/coupled_training_trial'
text=(OLD/'run_coupled_trial.py').read_text()
text=text.replace("ROOT=OUT/'matched_coupled_teacher'","ROOT=OUT/'matched_row_route'")
text=text.replace('WORK_LIMIT=7200','WORK_LIMIT=10800')
text=text.replace('training_seeds=[229,241]','training_seeds=[251,263]')
text=text.replace('independent_initial_state_seeds=[4671,4691]','independent_initial_state_seeds=[4751,4771]')
text=text.replace("hypothesis='The fresh neutral-head267 teacher retains prior skills under current physics. Its body+arms with current head passed two-seed compatibility, whereas its body with current whole upper lost flight/return. Test coupled21-row truthful labels locally at the failing0.15/1.5 commands while preserving head gradients and all other objectives.'",
 "hypothesis='The rejected coupled target altered shared features despite zero direct head rows. Compare only its selected derivative restricted to final21 output rows against the same shared target and unchanged full-state control. Original PPO, other teachers, upper, weights, inputs and physics remain. Final rows do not guarantee off-command or physical retention.'")
text=text.replace("original_MSE_denominator_outputs=23,","original_MSE_denominator_outputs=23,branches=['control','shared','rows'],route_helper_sha256=sha(OUT/'final_row_route.py'),CPU_fixtures=read(OUT/'final_row_CPU_checks.json'),")
# Capture actual original learner gradients before the coupled route is applied.
insertion="""    f=ROOT/'passive_PPO_capture'
    launch([str(OUT/'capture_current_gradients.py'),'--mode','train','--output',str(f),'--checkpoint',cp,
        '--num_envs','4096','--iterations','4','--learning_rate','.0001','--seed','251']+teacher+common,f,260)
    validate_train(f,4)
    with (OUT/'row_response_analysis.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-X','utf8',str(OUT/'analyze_row_response.py')],check=True,stdout=log,stderr=subprocess.STDOUT)
"""
text=text.replace('    # Recheck compatibility on this newer, exact trial checkpoint.',insertion+'    # Recheck compatibility on this newer, exact trial checkpoint.')
text=text.replace("[str(wrapper),'--coupled-neutral-teacher','--mode','train'","[str(wrapper),'--coupled-neutral-teacher','--final-row-route','--mode','train'")
text=text.replace("'--seed','229'","'--seed','251'")
text=text.replace("    for seed in [229,241]:", """    (ROOT/'stage_ready.json').write_text(json.dumps(dict(physical_support=True,CPU_response=read(OUT/'current_row_response.json'),smoke_and_reload=True)))
    deadline=time.monotonic()+900
    while not (ROOT/'learning_decision.json').exists():
        assert time.monotonic()<deadline,'No diagnostic learning decision; safely resume original learner'
        time.sleep(2)
    assert read(ROOT/'learning_decision.json')['proceed'],'Diagnostic learning declined; safely resume original learner'
    for seed in [251,263]:""")
text=text.replace("for branch in ['control','candidate']:","for branch in ['control','shared','rows']:")
text=text.replace("(['--coupled-neutral-teacher'] if branch=='candidate' else [])","(['--coupled-neutral-teacher']+(['--final-row-route'] if branch=='rows' else []) if branch!='control' else [])")
text=text.replace("validate_train(f,64,branch=='candidate')","validate_train(f,64,branch!='control')")
text=text.replace("if branch=='candidate':","if branch!='control':")
text=text.replace("('diverse_nominal',4671,False),('diverse_push',4691,True)","('diverse_nominal',4751,False),('diverse_push',4771,True)")
text=text.replace("    if applied is not None:assert read(folder/'experimental_teacher_target.json')['applied']==applied", """    if applied is not None:
        annotation=read(folder/'experimental_teacher_target.json');assert annotation['applied']==applied
        if annotation['final_output_rows_only']:
            assert all(r['locomotion_retention']['final_row_route']['batches']==20 and
                r['locomotion_retention']['final_row_route']['parameter_contributions']=={'weight':20,'bias':20} for r in rows)
            assert all(r['locomotion_retention']['final_row_route']['head_rows']==[11,16] for r in rows)
""")
(OUT/'run_row_trial.py').write_text(text)
compile(text,str(OUT/'run_row_trial.py'),'exec')
for name in ['neutral_teacher_candidate.py','final_row_route.py','analyze_row_response.py']:
 compile((OUT/name).read_text(),str(OUT/name),'exec')
(OUT/'row_trial_staging.json').write_text(json.dumps(dict(compiled=True,files={n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in
 ['run_row_trial.py','neutral_teacher_candidate.py','final_row_route.py','command_local_teacher_target.py','analyze_row_response.py']}),indent=2)+'\n')
print('Staged three-route trial; no live training mutation.')
