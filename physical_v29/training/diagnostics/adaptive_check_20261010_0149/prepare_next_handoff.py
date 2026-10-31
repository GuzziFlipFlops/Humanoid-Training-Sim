"""Synchronize this check's canonical fields and explicitly scope older work."""
import hashlib,json
from pathlib import Path
def refresh_canonical(d,out,train,v,pair,summary,priorities,stamp):
    owned=json.loads((out/'final_cpu/live_handoff.json').read_text());state=owned['state']
    current=Path(state['checkpoint']);selected=Path(pair['checkpoint']);previous=json.loads((out/'previous_NEXT_CHECK.json').read_text())
    historical={k:previous.get(k) for k in ['current_gradient_analysis','shared_target_head_effect','neutral_teacher_physical_response','matched_coupled_teacher_review','fixed_state_weight_response']}
    for key in historical:d.pop(key,None)
    d.update(recorded_utc=stamp,report=str(out/'REPORT.md'),decision=str(out/'decision.json'),active_run=str(train/'ACTIVE_RUN.json'),summary=summary,closing_summary=summary,
        latest_observation=v['owned_snapshot_utc'],latest_liveness_utc=v['owned_snapshot_utc'],latest_liveness=str(out/'final_cpu/live_handoff.json'),
        current_stage=v['stage'],current_stage_qualified=pair['nominal_all_pass'] and pair['pushed_all_pass'],running_qualified=False,velocity_ceiling_proven=False,
        computer_on=True,production_change=False,completed_main_updates=v['completed_main_updates'],new_main_updates=v['actual_retained_main_updates'],
        recovered_main_updates=v['hash_proven_recovered_main_updates'],actual_retained_current_run_main_updates=v['actual_retained_main_updates'],
        partial_observed_main_updates=v['observed_partial_main_updates'],current_owned_operation=v['operation_path'],current_checkpoint=str(current),
        current_checkpoint_sha256=hashlib.sha256(current.read_bytes()).hexdigest(),selected_actor=str(selected),selected_actor_sha256=pair['sha256'],latest_matching_pair=pair,
        reviewed_speed_case_passes_both=all(next(c['passed'] for c in pair[t+'_cases'] if c['command']==3.5) for t in ['nominal','pushed']),
        focused_tests=priorities,next_tests=priorities,focus=priorities,resumed_learning_evidence=str(out/'final_learning_verification.json'),final_verification=str(out/'final_learning_verification.json'),
        retained_main_totals=dict(updates=v['actual_retained_main_updates'],samples=sum(b['samples'] for b in v['blocks']),active_seconds=sum(b['active_seconds'] for b in v['blocks']),worker_seconds=sum(b['worker_seconds'] for b in v['blocks'])),
        retained_main_throughput={k:v[k] for k in ['active_PPO_samples_s','worker_effective_samples_s','whole_chain_samples_s','documented_diagnostic_pause_seconds_excluded']},
        resources=dict(last_completed_main=v['blocks'][-1],target_environments=8192,reserve_MiB=300,healthy_training_ramp_seconds=180,evaluation_or_diagnostic_memory_is_not_capacity=True),
        mandatory_upper_audit=str(out/'final_upper/upper_joint_time_audit.json'),mandatory_upper_bias_audit=str(out/'final_upper/upper_bias_audit.json'),upper_target_bias_audit=str(out/'final_upper/upper_target_bias_audit.json'),
        mandatory_all_speed_trace_checkpoint=str(selected),mandatory_all_speed_trace_sha256=pair['sha256'],upper_matched_comparison=str(out/'upper_matched_comparison.json'),
        synchronized_com_contact_audit=str(out/'closing_COM_contact_audit.json'),motor_speed_audit=str(out/'closing_motor_nominal.json'),pushed_motor_speed_audit=str(out/'closing_motor_pushed.json'),motor_audit_checkpoint=str(selected),
        neutral_teacher_certificate=str(out/'neutral_teacher_certificate.json'),fresh_earlier_complete_state_certificate=str(out/'matched_full_state/earlier_baseline_review.json'),
        earlier_complete_state_CPU_preparation=str(out/'older_branch_CPU_preflight.json'),matched_complete_state_review=str(out/'matched_full_state_review.json'),starting_state_motion_comparison=str(out/'starting_motion_comparison.json'),
        full_baseline_replay_parity=str(out/'full_baseline_parity.json'),mixed_training_metric_comparison=str(out/'training_metric_comparison.json'),
        measured_flight_retention_comparison=str(out/'flight_retention_analysis.json'),undeployed_reward_constraint_research=str(out/'constraint_reward_review.md'),
        primary_upper_reference_review=str(out/'reference_review.json'),experiment_validation=str(out/'experiment_validation.json'),diagnostic_resumes=[str(out/'diagnostic_resume.json')],
        previous_full_handoff_snapshot=str(out/'previous_NEXT_CHECK.json'),historical_diagnostic_paths=dict(previous_full_handoff=str(out/'previous_NEXT_CHECK.json'),previous_check=str(out.parent/'adaptive_check_20261009_2048/REPORT.md'),earlier_actor_specific_gradient_and_target_evidence=historical),
        history_inventory=str(train/'history/RUN_INVENTORY.json'),automation_verified_active=False,latest_matching_pair_file=str(out/'closing_gate_review.json'),newer_posture_certified=False,
        automation_verification=str(out/'automation_inventory_verification.json'),inventory_verification=str(out/'automation_inventory_verification.json'),final_owned_receipt=str(out/'final_cpu/live_handoff.json'),
        final_owned_observation_utc=v['owned_snapshot_utc'],latest_owned_saved_main_updates=v['actual_retained_main_updates'],latest_owned_completed_main_updates=v['completed_main_updates'],latest_owned_main_receipt=v['blocks'][-1])
