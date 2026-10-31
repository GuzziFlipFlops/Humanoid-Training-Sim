"""Refresh exact live evidence and the same heartbeat's final dated focus."""
import hashlib,json,subprocess,sys,tomllib
from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
proc=subprocess.run([sys.executable,'-u','-X','utf8',str(HERE/'record_owned_observation_handoff.py')],capture_output=True,text=True)
(OUT/'final_status_recorder_stdout.log').write_text(proc.stdout)
(OUT/'final_status_recorder_stderr.log').write_text(proc.stderr)
assert proc.returncode==0,proc.stderr
live=read(HERE/'diagnostics/walk_regression_20261003/live_handoff.json')
(OUT/'live_handoff.json').write_text(json.dumps(live,indent=2)+'\n')
run=Path(live['active_run']['run']);state=live['state'];progress=live.get('worker_progress')
gates=[]
for p in sorted(run.glob('*_gate_*/result.json')):
    result=read(p);cfg=read(p.parent/'run_config.json')
    gates.append({'folder':p.parent.name,'completed':result['completed'],'passed':result['passed'],
       'checkpoint':cfg['arguments']['checkpoint'],'duration_s':result['duration_s'],'perturb':result['perturb'],
       'cases':[{'command_m_s':c['statistics']['command_m_s'],'passed':c['passed'],
           'trials':c['statistics']['trials'],'signed_speed_m_s':c['statistics']['reset_frame_forward_speed_m_s'],
           'genuine_flight_fraction':c['statistics'].get('geometric_flight_fraction'),
           'failed_checks':[k for k,v in c['checks'].items() if not v]} for c in result['cases']]})
current=Path(state['operation_path']);load=read(current/'full_load_proof.json') if (current/'full_load_proof.json').is_file() else None
summary={'recorded_utc':live['recorded_utc'],'run':str(run),'status':state['status'],'stage':state['stage'],
    'current_operation':current.name,'completed_main_updates':state['total_training_iterations'],
    'checkpoint':state.get('checkpoint'),'promotions':[{'stage':p['stage'],'speed':p['speed'],'checkpoint':p['checkpoint']} for p in state['promotions']],
    'current_environments':state['environment_count'],'selected_next_count':state.get('next_training_environment_count'),
    'actual_worker_pids':live['actual_worker_pids'],'frozen_source_integrity':live['frozen_source_integrity'],
    'gpu':live['gpu'],'system_memory':live['system_memory'],'benchmarks':live['benchmarks'],
    'latest_independent_gates':gates[-4:],'full_load_proof':load,
    'current_progress':None if not progress else {'iteration':progress['iteration'],'gpu':progress['gpu'],'ppo':progress['ppo'],
        'action_command_difference':progress['behavior']['action_command_difference'],
        'locomotion_retention':progress.get('locomotion_retention')},
    'qualification_complete':state['status']=='all_requested_stages_passed','scope':'Dated verified snapshot; read fresh state at next check'}
(OUT/'replacement_status.json').write_text(json.dumps(summary,indent=2)+'\n')
next_check=read(HERE/'automation/NEXT_CHECK.json');next_check.update(recorded_utc=summary['recorded_utc'],latest_verified_state=summary)
(HERE/'automation/NEXT_CHECK.json').write_text(json.dumps(next_check,indent=2)+'\n')
notes='\n\n## Latest verified replacement observation\n\n'+f"Observed {summary['recorded_utc']}: {summary['status']}, stage {summary['stage']}, operation {current.name}, current count {state['environment_count']}, selected next count {state.get('next_training_environment_count')}, {state['total_training_iterations']} completed new main updates. Fresh promoted speeds: {[p['speed'] for p in state['promotions']]}. Frozen hashes and owned processes verify in live_handoff.json. Startup/capacity/smokes are excluded from main learning progress. Read replacement_status.json for matching gate cases, memory and exact checkpoint provenance; this snapshot is not a final running/5m/s qualification.\n"
report=OUT/'REPORT.md';report.write_text(report.read_text()+notes)
config=tomllib.loads(Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml').read_text())
args=read(OUT/'automation_update_args.json')
extra=f" Fresh replacement observation{summary['recorded_utc']}: status{summary['status']},stage{summary['stage']},operation{current.name},current count{state['environment_count']},selected next{state.get('next_training_environment_count')},new completed main{state['total_training_iterations']},fresh promoted speeds{[p['speed'] for p in state['promotions']]}. Read replacement_status.json and live_handoff.json; startup/capacity/smoke is not main learning or running qualification."
args['prompt']=config['prompt'].split(' Fresh replacement observation')[0]+extra
(OUT/'automation_update_args.json').write_text(json.dumps(args,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['latest_independent_gates','frozen_source_integrity','benchmarks','current_progress','promotions']},indent=2))
