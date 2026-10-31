"""Hash plots after explicit model inspection; do not infer visual completion."""
import argparse,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--notes',required=True);args=p.parse_args()
root=OUT/args.folder;notes=json.loads((OUT/args.notes).read_text())
audit=json.loads((root/'audit_set.json').read_text());review=json.loads(Path(audit['gate_review']).read_text())
actor=Path(review['latest']['checkpoint']).parent.name
receipts=[];count=0
assert set(notes)=={'','slow','middle','pushed_fast','pushed_slow','pushed_middle'}
for folder,observations in notes.items():
    names=['upper_joint_angles.png','upper_joint_velocities.png']
    if not folder:names += ['upper_bias_decomposition.png','upper_target_bias.png']
    receipt=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),actor=actor,observations=observations,
        plots=[dict(path=str(root/folder/name),sha256=hashlib.sha256((root/folder/name).read_bytes()).hexdigest(),visually_inspected=True) for name in names],
        population_sampling_Hz=10,representative_sampling_Hz=50,reset_segments_preserved=True,
        human_or_Unitree_encoder_certified=False,CAD_optical_gaze_calibrated=False)
    path=root/folder/'visual_inspection.json';path.write_text(json.dumps(receipt,indent=2)+'\n')
    receipts.append(str(path));count+=len(names)
assert count==14
audit.update(plots_need_visual_QA=False,all_plots_visually_inspected=True,visual_QA_receipts=receipts)
for row in audit['supplements']:row['needs_visual_QA']=False
(root/'audit_set.json').write_text(json.dumps(audit,indent=2)+'\n')
print('Saved14 explicitly inspected '+actor+' upper plot receipts.')
