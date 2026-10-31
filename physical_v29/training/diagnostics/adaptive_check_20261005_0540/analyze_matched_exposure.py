"""Compare independent full-population cases; never select from reward alone."""
import json,hashlib
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def describe(folder):
    gate=read(folder/'evaluation.json');data=np.load(folder/'telemetry.npz')
    names=list(data['joint_names']);ids=[names.index('head_yaw'),names.index('head_pitch')]
    rows=[]
    for index,case in enumerate(gate['cases']):
        s=case['statistics'];head=data['q'][:,data['case_labels']==index][:,:,ids]
        rows.append({'command':s['command_m_s'],'passed':case['passed'],'failures':case['failures'],
            'survival':s['survival'],'signed_speed':s['reset_frame_forward_speed_m_s'],
            'flight_fraction':s['geometric_flight_fraction'],'heading_deg':s['final_heading_abs_mean_deg'],
            'lateral_speed':s['reset_frame_lateral_speed_m_s'],
            'head_RMS_deg':float(np.rad2deg(np.sqrt((head**2).sum(-1).mean())))})
    return rows
result={'qualifies_locomotion':False,'diagnostic_actors_deployed':False,'arms':{}}
for name in ['control','candidate']:
    folder=OUT/('matched_'+name);result['arms'][name]={'nominal':describe(OUT/('matched_'+name+'_nominal')),
        'pushed':describe(OUT/('matched_'+name+'_push'))}
    rows=[json.loads(x) for x in (folder/'progress.jsonl').read_text().splitlines()]
    proof=read(folder/'full_load_proof.json')
    assert proof['actor_equal'] and proof['critic_equal'] and proof['optimizer_equal'] and proof['exploration_equal']
    assert len(rows)==64
    assert all(x['ppo']['pre_update_kl']==0 and x['behavior']['action_command_difference']==0 for x in rows)
    result['arms'][name]['training']={'updates':len(rows),'environments':4096,'full_load':proof,
        'max_pre_update_KL':max(x['ppo']['pre_update_kl'] for x in rows),
        'max_command_difference':max(x['behavior']['action_command_difference'] for x in rows),
        'mean_running_fraction':sum(x['command_coverage']['exact_command_fractions']['1.5'] for x in rows)/len(rows),
        'active_samples_per_second':len(rows)*4096*24/sum(x['collect_seconds']+x['learn_seconds'] for x in rows),
        'worker_effective_samples_per_second':len(rows)*4096*24/read(folder/'result.json')['seconds'],
        'minimum_free_GPU_MiB':min(x['gpu']['free_mb'] for x in rows),
        'minimum_host_available_MiB':min(x['gpu']['ram_available_mb'] for x in rows)}
reload=OUT/'candidate_full_reload';rows=[json.loads(x) for x in (reload/'progress.jsonl').read_text().splitlines()]
proof=read(reload/'full_load_proof.json');assert all(proof[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
assert len(rows)==4 and all(x['ppo']['pre_update_kl']==0 and x['behavior']['action_command_difference']==0 for x in rows)
result['full_reload']={'passed':True,'proof':proof,'updates':4,'pre_update_KL':0,'command_difference':0}
(OUT/'matched_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:{p:[{key:r[key] for key in ['command','passed','signed_speed','flight_fraction','head_RMS_deg']} for r in v[p]] for p in ['nominal','pushed']} for k,v in result['arms'].items()},indent=2))
