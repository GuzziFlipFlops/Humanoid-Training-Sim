"""Resolve v7's recorded last checkpoint that passed every hard gate."""
import hashlib
import json
from pathlib import Path


def accepted_source(state_path, original_hashes):
    state_path=Path(state_path)
    state=json.loads(state_path.read_text())
    if (state['status']!='manually_stopped_after_gate' or state.get('stage')!='speed'
            or float(state.get('speed_ceiling',0.))!=4.5):
        raise RuntimeError('Expected the evidence-preserving guarded-v7 stop')
    if state['original_hashes']!=original_hashes:
        raise RuntimeError('Original assets/policies changed since v3')
    gate=state['last_gate']
    if not gate['walking_pass'] or gate['stage_pass']:
        raise RuntimeError('Expected walking retention plus a narrowly failed v7 task gate')
    failures=[]
    for batch in gate['checks']:
        failures.extend(name for name,value in batch['stage'].items() if not value)
    if set(failures)!={'torso_lean_occupancy'}:
        raise RuntimeError(f'V8 repair is allowed only for the measured occupancy miss, got {failures}')
    # The 4.5 m/s candidates never passed all task gates.  Resume the preceding
    # 4.0 m/s traction checkpoint whose paired gate passed in full.
    source_gate=state['resume_gate']
    if not source_gate['walking_pass'] or not source_gate['stage_pass']:
        raise RuntimeError('The v6 4.0 m/s source gate is not fully passing')
    checkpoint=Path(state['manual_stop']['last_fully_passing_checkpoint'])
    if source_gate['checkpoint'] != str(checkpoint):
        raise RuntimeError('Recorded passing checkpoint differs from the preserved source gate')
    prefix=source_gate['evaluated_as']
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    source_run=Path(state['resume_source']).parent
    for seed in (47,83):
        result=json.loads((source_run/f'{prefix}_eval_{seed}'/'result.json').read_text())
        if not result.get('completed') or result['source_sha256']!=digest:
            raise RuntimeError('Checkpoint differs from evaluated artifact')
    scale=state.get('action_scale')
    if scale != .35:
        raise RuntimeError(f'Unexpected native action scale: {scale}')
    return checkpoint,scale,source_gate
