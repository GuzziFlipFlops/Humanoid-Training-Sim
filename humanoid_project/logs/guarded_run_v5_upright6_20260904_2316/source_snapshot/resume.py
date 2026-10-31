"""Resolve only the completed, measured v4 checkpoint; never overwrite it."""
import hashlib
import json
from pathlib import Path


def accepted_source(state_path, original_hashes):
    state_path=Path(state_path)
    state=json.loads(state_path.read_text())
    if state['status']!='completed':
        raise RuntimeError('Expected the completed guarded v4 chain')
    if state['original_hashes']!=original_hashes:
        raise RuntimeError('Original assets/policies changed since v3')
    checkpoint=Path(state['final_checkpoint'])
    gate=state['last_gate']
    prefix=gate['evaluated_as']
    if gate['checkpoint']!=str(checkpoint) or not gate['walking_pass'] or not gate['stage_pass']:
        raise RuntimeError('Resume checkpoint was not approved by both gates')
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for seed in (47,83):
        result=json.loads((state_path.parent/f'{prefix}_eval_{seed}'/'result.json').read_text())
        if not result.get('completed') or result['source_sha256']!=digest:
            raise RuntimeError('Checkpoint differs from evaluated artifact')
    return checkpoint,state['action_scale'],gate
