"""Record model visual QA actually performed on all fourteen initial plots."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent;folder=OUT/'initial_upper'
notes={
 '.':'767 exact nominal/pushed. Gait-linked head oscillation; opposite held shoulder pitch at every moving speed. Shoulder roll and elbow amplitudes unequal. At 2.5 and 3 the plotted representative has held bias rather than the slow reversals seen in other earlier actors. Physical targets carry these biases; left target/actual bias mismatch is larger than right. Native-zero head is torso-relative. Actual motion remains uncertified.',
 'slow':'Nominal stopped motion small; 0.15/0.35/0.65 shoulder swing has large opposed offsets and unequal elbow amplitude. Gait-linked yaw dips and asymmetric roll persist. Measured stopped qd residual is descriptive and is not used to infer a production defect.',
 'middle':'Nominal 1/1.5/2/2.5: shoulder pitch alternation about opposed held means, fast pitch amplitude attenuated. Elbow coordination and left/right velocities differ. Head oscillation increases with cadence.',
 'pushed_fast':'Pushed 1/2.5/3/3.5: small transient changes about the same held pitch biases, fast shoulder target motion is larger than actual motion. Unequal elbow amplitude and gait-linked head yaw persist.',
 'pushed_slow':'Pushed stopped transient near 5s is visible, settles without a reset; slow walking head/shoulder/elbow cycles persist with bilateral asymmetry.',
 'pushed_middle':'Pushed 1/1.5/2/2.5 target-versus-angle phase and amplitude differences are visible, with persistent large opposed shoulder means and elbow asymmetry.'}
for label,note in notes.items():
    f=folder/label
    names=['upper_joint_angles.png','upper_joint_velocities.png']+(['upper_bias_decomposition.png','upper_target_bias.png'] if label=='.' else [])
    record=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),exact_actor='767_run_350_train_253',
        plots=[dict(path=str(f/n),sha256=hashlib.sha256((f/n).read_bytes()).hexdigest(),visually_inspected=True) for n in names],observations=note,
        population_sampling='32 robots per command at10Hz',representative_sampling='16 total robots per gate at50Hz, reset-segmented',
        natural_human_Unitree_certified=False,mapped_speed_matched_Unitree_encoder_curves_available=False,CAD_optical_gaze_calibrated=False)
    (f/'visual_inspection.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
record=json.loads((folder/'audit_set.json').read_text());record['plots_need_visual_QA']=False
for r in record['supplements']:r['needs_visual_QA']=False
(folder/'audit_set.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print('Fourteen actual plots inspected; exact767 scope saved.')
