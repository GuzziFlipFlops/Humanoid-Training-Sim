"""Resolve v6's last checkpoint that passed every applicable hard gate."""
import hashlib
import json
from pathlib import Path


def accepted_source(state_path, original_hashes):
    state_path=Path(state_path)
    state=json.loads(state_path.read_text())
    if (state['status']!='stopped_gate' or state.get('stage')!='speed'
            or float(state.get('speed_ceiling',0.))!=4.5):
        raise RuntimeError('Expected the stopped guarded-v6 4.5 m/s speed stage')
    if state['original_hashes']!=original_hashes:
        raise RuntimeError('Original assets/policies changed since v3')
    gate=state['last_gate']
    if not gate['walking_pass'] or gate['stage_pass']:
        raise RuntimeError('Expected walking retention plus a narrowly failed v6 task gate')
    failures=[]
    for batch in gate['checks']:
        failures.extend(name for name,value in batch['stage'].items() if not value)
    if set(failures)!={'torso_lean_occupancy'}:
        raise RuntimeError(f'V7 repair is allowed only for the measured occupancy miss, got {failures}')
    # The 4.5 m/s candidates never passed all task gates.  Resume the preceding
    # 4.0 m/s traction checkpoint whose paired gate passed in full.
    source_gate_path=state_path.parent/'00_traction_4_0200_gate.json'
    source_gate=json.loads(source_gate_path.read_text())
    if not source_gate['walking_pass'] or not source_gate['stage_pass']:
        raise RuntimeError('The v6 4.0 m/s source gate is not fully passing')
    checkpoint=Path(source_gate['checkpoint'])
    prefix=source_gate['evaluated_as']
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for seed in (47,83):
        result=json.loads((state_path.parent/f'{prefix}_eval_{seed}'/'result.json').read_text())
        if not result.get('completed') or result['source_sha256']!=digest:
            raise RuntimeError('Checkpoint differs from evaluated artifact')
    scale=next((h['result']['action_scale'] for h in state['history']
                if h.get('result',{}).get('checkpoint')==str(checkpoint)),None)
    if scale != .35:
        raise RuntimeError(f'Unexpected native action scale: {scale}')
    return checkpoint,scale,source_gate
