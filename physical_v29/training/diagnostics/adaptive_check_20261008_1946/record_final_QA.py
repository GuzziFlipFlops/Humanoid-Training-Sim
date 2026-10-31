"""Save receipts only for the fourteen final plots actually inspected."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent
folder=OUT/'final_upper'
pair=json.loads((OUT/'final_gate_review.json').read_text())['latest']
notes={
 '.':'Exact794 nominal/pushed population and representative data. Held opposite shoulder pitch and unequal fast amplitudes persist. At2.5 bias evolves without reversing sign in the displayed nominal robot; at3m it crosses slowly. Targets carry these slow changes. At3.5, pitch is smaller than target swing and elbow/roll amplitudes differ; neck motion is gait-linked. Native zero is torso-relative, not calibrated optical gaze.',
 'slow':'Nominal stopped angle variation is small. The saved stopped qd offsets are descriptive and un-attributed, not proof of a runtime fault. At0.15/0.35/0.65, large opposed shoulder offsets, unequal elbow amplitudes and gait-linked neck/roll cycles persist.',
 'middle':'Nominal1/1.5/2/2.5: opposed shoulder pitch means, unequal elbow swing, attenuated pitch at faster speeds, gait-linked head movement. Target and measured amplitudes/phases differ. No reset spans are joined.',
 'pushed_fast':'Pushed1/2.5/3/3.5: actual angles and targets carry slow shoulder changes; fast shoulder pitch remains attenuated, with unequal elbow/roll amplitudes. Small push transients do not remove persistent held bias.',
 'pushed_slow':'Pushed stopped response near5s settles without a reset. Moving slow commands retain unequal arm and head cycles; left shoulder and elbow response differs from right. All slow actual velocity timelines inspected.',
 'pushed_middle':'Pushed1/1.5/2/2.5 actual targets, angles and velocities inspected. Slow bias evolves at2/2.5, while higher-frequency elbow/roll/neck motion and shoulder asymmetry persist. Bounds or low head RMS do not certify natural motion.'}
for label,note in notes.items():
 f=folder/label
 names=['upper_joint_angles.png','upper_joint_velocities.png']+(['upper_bias_decomposition.png','upper_target_bias.png'] if label=='.' else [])
 record=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),checkpoint=pair['checkpoint'],sha256=pair['sha256'],
  plots=[dict(path=str(f/n),sha256=hashlib.sha256((f/n).read_bytes()).hexdigest(),visually_inspected=True) for n in names],
  observations=note,population_sampling='32 robots per command at10Hz',representative_sampling='16 total selected robots per gate at50Hz, reset-segmented',
  natural_human_Unitree_certified=False,mapped_speed_matched_Unitree_encoder_curves_available=False,CAD_optical_gaze_calibrated=False)
 (f/'visual_inspection.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
audit=json.loads((folder/'audit_set.json').read_text())
audit.update(plots_need_visual_QA=False,all_plots_visually_inspected=True,inspected_plot_count=14)
for r in audit['supplements']:r['needs_visual_QA']=False
(folder/'audit_set.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
assert all((OUT/f'seed{seed}_{branch}_upper/visual_inspection.json').exists() for seed in [157,167] for branch in ['control','candidate'])
print('Exact794 fourteen plots inspected; four128-update diagnostic actors also have receipts.')
