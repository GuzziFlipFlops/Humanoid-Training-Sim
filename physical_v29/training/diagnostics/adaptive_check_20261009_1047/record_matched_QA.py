"""Hash-backed four-plot QA receipt for a matched experimental actor."""
import argparse,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();folder=OUT/a.folder
assert folder.resolve().is_relative_to(OUT.resolve())
notes=json.loads((folder/'QA_notes.json').read_text(encoding='utf-8'))
names=['upper_joint_angles.png','upper_joint_velocities.png','upper_bias_decomposition.png','upper_target_bias.png']
assert notes['inspected_files']==names
audit=json.loads((folder/'audit_set.json').read_text())
assert not audit['all_commands_explicit_nominal_and_pushed_traces']
receipt=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),visually_inspected=True,
    plots={n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in names},notes=notes['notes'],
    all_command_population_statistics=True,representative_trace_QA=True,
    explicit_all_command_nominal_pushed_trace_QA=False,natural_motion_certified=False)
(folder/'visual_QA.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
audit.update(all_plots_visually_inspected=True,plots_need_visual_QA=False,visual_QA=str(folder/'visual_QA.json'))
(folder/'audit_set.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,indent=2))
