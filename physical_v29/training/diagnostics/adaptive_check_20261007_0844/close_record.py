import json
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;TRAINING=OUT.parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
receipt=read(OUT/'recording_receipt.json');a=read(OUT/'automation_verification.json');assert a['status']=='ACTIVE' and a['five_hour_cadence']
inventory=read(TRAINING/'history/RUN_INVENTORY.json')
physical=inventory['physical_runs'];verification=[p for r in physical for p in r['frozen_source_verification']]
result={'recorded_utc':datetime.now(timezone.utc).isoformat(),'physical_runs':len(physical),'physical_operations':sum(len(r['operations']) for r in physical),
    'missing_manifest_artifacts':sum(p['actual_sha256'] is None for p in verification),
    'differing_manifest_hashes':sum(p['actual_sha256'] is not None and not p['matches'] for p in verification),
    'read_warnings':inventory['read_warnings'],'inventory_generated_utc':inventory['generated_utc']}
assert result['differing_manifest_hashes']==0 and not result['read_warnings']
(OUT/'inventory_verification.json').write_text(json.dumps(result,indent=2)+'\n')
receipt.update(native_update_pending=False,native_update_confirmed=True,same_active_five_hour_heartbeat_verified=True,inventory_verified=True,completed_utc=result['recorded_utc'])
(OUT/'recording_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
fault={'scope':'Preserved summary of first diagnostic bookkeeping failure,not a reconstructed raw traceback','error':"AttributeError: 'str' object has no attribute 'read_bytes'",'action':'Exception cleanup resumed same supervisor; helper then corrected Path conversion and repeated owned saved-boundary suspension.','lost_main_updates':False,'exact_short_pause_duration_available':False,'failed_state':str(OUT/'failed_first_pause_state.json')}
(OUT/'first_pause_bookkeeping_failure.json').write_text(json.dumps(fault,indent=2)+'\n')
marker='adaptive_check_20261007_0844_completion'
closing=f'''\n<!-- {marker} -->
Closing confirmation {result['recorded_utc']}: native app confirms same ACTIVEfive-hour heartbeat/thread/notification preference; durable scope/5-or-qualified4 endpoint prefix is byte-identical. No duplicate created. CPU history inventory has{result['physical_runs']} physical runs,{result['physical_operations']} operations,zero differing hashes/read warnings;{result['missing_manifest_artifacts']} historical missing manifests remain documented. Eight angle/velocity images have visual QA. Exact resumed142 main64 updates and matching143/144 are verified; later145 partial learner remains separately unqualified. No production source/gain/reward/cap/state reset or diagnostic actor deployed. Reports and earlier history preserved. See {OUT/'automation_verification.json'} and {OUT/'inventory_verification.json'}.
<!-- /{marker} -->\n'''
for p in [OUT/'REPORT.md',TRAINING/'ITERATION_HISTORY.md',Path('E:/Aditya/robot_sim/docs/robot/TRAINING_JOURNAL_20261001.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
    assert marker not in p.read_text(encoding='utf-8-sig')
    with p.open('a',encoding='utf-8') as f:f.write(closing)
next_path=TRAINING/'automation/NEXT_CHECK.json';n=read(next_path)
n['automation_verification']=str(OUT/'automation_verification.json');n['inventory_verification']=str(OUT/'inventory_verification.json')
next_path.write_text(json.dumps(n,indent=2)+'\n')
print(json.dumps(result,indent=2))
