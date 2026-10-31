"""Concise current upper population/phase report from existing CPU evidence."""
import argparse, json
from pathlib import Path
OUT = Path(__file__).resolve().parent
p = argparse.ArgumentParser(); p.add_argument('--folder', default='initial_upper'); a = p.parse_args()
u = json.loads((OUT / a.folder / 'upper_joint_time_audit.json').read_text())
b = json.loads((OUT / a.folder / 'upper_bias_audit.json').read_text())
print(json.dumps(dict(record_count=len(u['records']), example_keys=list(u['records'][0]),
    bias_keys=list(b['records'][0]), nominal_fast=next(x for x in u['records'] if x['actor_tag']=='latest'
        and x['test']=='nominal' and x['command_m_s']==3.5)), indent=2))
