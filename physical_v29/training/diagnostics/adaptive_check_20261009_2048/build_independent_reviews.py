"""Adapt CPU-only reviewers to the new explicitly bound three-route design."""
from pathlib import Path
OUT=Path(__file__).resolve().parent;OLD=OUT.parent/'adaptive_check_20261009_1548'
for source,destination in [('validate_coupled_trial.py','validate_row_trial.py'),('review_coupled_trial.py','review_row_trial.py')]:
 text=(OLD/source).read_text().replace("TRIAL = OUT / 'coupled_training_trial'","TRIAL = OUT").replace("ROOT = TRIAL / 'matched_coupled_teacher'","ROOT = TRIAL / 'matched_row_route'")
 text=text.replace("TRIAL=OUT/'coupled_training_trial';ROOT=TRIAL/'matched_coupled_teacher'","TRIAL=OUT;ROOT=TRIAL/'matched_row_route'")
 text=text.replace('[229, 241]','[251, 263]').replace('[229,241]','[251,263]')
 text=text.replace("['control', 'candidate']","['control', 'shared', 'rows']").replace("['control','candidate']","['control','shared','rows']")
 text=text.replace("branch == 'candidate'","branch != 'control'").replace("branch=='candidate'","branch!='control'")
 text=text.replace("TRIAL / 'remainder_pause/diagnostic_resume.json'","TRIAL / 'diagnostic_resume.json'")
 text=text.replace('actual_updates == 512 and samples == 50331648 and len(operations) == 32','actual_updates == 768 and samples == 75497472 and len(operations) == 48')
 text=text.replace('all_32_comparison_operations_completed_and_owned_workers_cleaned','all_48_comparison_operations_completed_and_owned_workers_cleaned')
 text=text.replace('false_positive_URDF_order_abort_preserved=True, corrected_review_uses_observed_Isaac_order=True','review_uses_observed_Isaac_order=True, previous_reviewer_error_not_repeated=True')
 text=text.replace('len(records)==8','len(records)==12')
 text=text.replace("candidates=[r for r in records if r['branch']=='candidate']","candidates=[r for r in records if r['branch']=='rows']")
 text=text.replace("matched_coupled_teacher_review.json","matched_row_route_review.json")
 if source.startswith('validate'):
  text=text.replace("            actual_updates += len(rows);", """            assert executed['final_output_rows_only']==(branch=='rows')
            if branch=='rows':
                assert executed['route_helper_sha256']==sha(TRIAL/'final_row_route.py')
                for row in rows:
                    route=row['locomotion_retention']['final_row_route']
                    assert route['head_rows']==head and route['batches']==20
                    assert route['parameter_contributions']=={'weight':20,'bias':20}
                    assert not route['selected_teacher_trunk_derivative']
                    assert route['PPO_and_other_auxiliary_paths_unchanged_before_global_clipping']
            actual_updates += len(rows);""")
  text=text.replace("assert '--coupled-neutral-teacher' not in argv","assert '--coupled-neutral-teacher' not in argv and '--final-row-route' not in argv")
 else:
  text=text.replace('    records=records,all_expected_components_complete=all_expected,',"    branch_all_gates_pass={b:all_expected and all(t['passed'] for r in records if r['branch']==b for t in r['tests'].values()) for b in ['control','shared','rows']},records=records,all_expected_components_complete=all_expected,")
 compile(text,str(OUT/destination),'exec');(OUT/destination).write_text(text)
monitor=(OUT/'monitor_check.py').read_text().replace("ROOT=OUT/'coupled_training_trial/matched_coupled_teacher'","ROOT=OUT/'matched_row_route'").replace("OUT/'coupled_training_trial/teacher_compatibility_response'","OUT/'teacher_compatibility_response'")
(OUT/'monitor_check.py').write_text(monitor)
print('Staged CPU validation, complete case review and compact owned progress.')
