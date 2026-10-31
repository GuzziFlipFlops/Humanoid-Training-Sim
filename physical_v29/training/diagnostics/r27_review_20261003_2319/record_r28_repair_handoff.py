"""CPU-only smoke/memory/command evidence; never starts a simulator."""
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

import numpy as np

OUT = Path(__file__).resolve().parent
TRAINING = OUT.parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    active=read(TRAINING/'ACTIVE_RUN.json');run=Path(active['run'])
    assert run.name=='locomotion_20261004_r28_heading_alignment_cache_fixed'
    record=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),active=active,state=read(run/'state.json'),operations=[],completed=False)
    for name in ['001_physics_preflight','002_ppo_smoke','003_checkpoint_reload','004_phase_retention_smoke','005_phase_retention_reload','006_benchmark_4096','007_benchmark_6144','008_benchmark_8192']:
        folder=run/name
        if not folder.exists():continue
        item=dict(name=name)
        if (folder/'result.json').exists():item['result']=read(folder/'result.json')
        if (folder/'warm_start.json').exists():item['transfer']=read(folder/'warm_start.json')
        progress=folder/'progress.jsonl'
        if progress.exists():
            rows=[]
            for line in progress.read_text(encoding='utf-8').splitlines():
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:continue
            if rows:
                warm=rows[1:] or rows
                item['progress']=dict(updates=len(rows),pre_update_kl_max=max(r['ppo']['pre_update_kl'] for r in rows),
                    command_difference_max=max(r['behavior']['action_command_difference'] for r in rows),
                    private_mb=[r['gpu']['process_private_mb'] for r in rows],
                    cuda_mb=[r['gpu']['torch_allocated_mb'] for r in rows],
                    min_free_vram_mb=min(r['gpu']['free_mb'] for r in rows),
                    latest_heading_error_abs_rad=rows[-1]['behavior'].get('heading_error_abs_rad'),
                    raw_environment_steps_s=rows[-1]['num_envs']*24/statistics.mean(r['collect_seconds']+r['learn_seconds'] for r in warm))
        trace=folder/'gait_diagnostics_50hz.npz'
        if trace.exists():
            with np.load(trace) as data:
                item['trace_command_difference_max']=float(np.abs(data['observed_command']-data['action_command']).max())
        record['operations'].append(item)
    record['scope']='Fresh execution/memory diagnostics. Smokes and benchmarks are not independent gait qualification.'
    record['completed']=True
    (OUT/'r28_execution_handoff.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    compact=[dict(name=item['name'],completed=item.get('result',{}).get('completed'),progress=item.get('progress'),
        command_difference=item.get('trace_command_difference_max')) for item in record['operations']]
    print(json.dumps(dict(recorded_utc=record['recorded_utc'],operations=compact,scope=record['scope']),indent=2))


if __name__=='__main__':main()
