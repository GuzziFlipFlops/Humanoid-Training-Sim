"""Maintain local actuator calculations and the current robot report."""
import json
import math
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
OUT=HERE/'analysis/20260930_actuators'
a=json.loads((OUT/'calculation.json').read_text())
rows=[]
for name,r in a['joint_types'].items():
    cont='Unknown' if r['continuous_Nm'] is None else f"{r['continuous_Nm']:.2f}"
    note='Neutral, pure-axis; coupled/pose-dependent' if 'differential' in r else 'Ideal/user assumption'
    if name=='elbow_pitch':note='24 V stall / no-load; continuous unknown'
    if name.startswith('head_'):note='12 V rated / stall / no-load'
    if name=='shoulder_pitch':note='Physical 4:1; motor torque capped to 62.5%'
    rows.append(f"| {name.replace('_',' ')} | {cont} | {r['effort_Nm']:.2f} | {r['velocity_rad_s']*60/(2*math.pi):.1f} | {note} |")
text='''# User actuator assumptions and CAD transmission — September 30, 2026

This replaces the previous pending hardware selection for the **physical v29** simulation. It does not revise CAD or the separate old 22-axis training robot. User assumptions are accepted as planning inputs, not manufacturer-qualified thermal ratings. Source masses/inertias remain test proxies. Documentation is local only.

7010: **18 N·m sustained / 45 N·m peak / 400 RPM at factory 8:1 output**. 6010: **15 / 30 / 400** at factory output. The internal eight reduction is already included. Waist yaw has an optional **50 N·m** user-proposed boost; the default remains45 because boost duration/temperature have not been supplied.

DJI10018 77KV: **8 N·m sustained / 18 N·m peak at the bare shaft**. Hip pitch and roll now both use **15:1 cycloidal**, superseding the earlier17:1 hip-pitch plan. Knee uses **8:1 planetary ×2:1 chain =16:1**. Ideal no-load speed uses43.2 V nominal and50.4 V full charge, not the assumption that peak torque persists to no-load speed.

| Joint, same on both sides | Sustained/rated N·m | Peak/stall N·m | Speed ceiling RPM | Meaning |
|---|---:|---:|---:|---|
'''+ '\n'.join(rows)+'''

## Shoulder input limiting

The additional shoulder-pitch planetary is physically4:1. To achieve a2.5× equivalent torque cap, its geared motor output is limited to **9.375 N·m sustained /18.75 N·m peak**:4×9.375=37.5 and4×18.75=75 N·m ideal joint output. Speed is still400/4=100 RPM. If only peak input is limited, full sustained motor torque could instead yield60 N·m ideal; the present configuration also derates sustained torque to match the user's intended2.5 multiplier. Neither reduction is newly modeled in CAD.

## Exact saved CAD differential geometry

Geometry is reconstructed from the independently reopened native v29 shaft axes and upper/lower rod-eye centres in `native_joint_map_saved.json`, hash recorded in `assets/differential_geometry.json`. Current motor-output angles are incremental from the baked saved pose; mirrored native axes and small neutral ankle offsets are retained.

Let q=[pitch,roll], θ=[motor A,motor B]. The closure solver preserves each measured rod length and the 23 mm ankle /26 mm waist crank radius. It calculates **J(q)=∂θ/∂q**, then uses **θ̇=Jq̇** and **τ_joint=Jᵀτ_motor**. Torque commands are transformed to both motor shafts, limited there, then transformed back. Independent URDF limits cannot implement this coupling.

Near neutral, approximate envelopes are:

- Ankle peak: **|τ_pitch|/156.2 + |τ_roll|/89.8 ≤1**. Sustained: denominators62.5 and35.9. Small saved left/right linkage asymmetries are included in the actual matrix rather than this rounded symmetric formula.
- Waist peak: **|τ_pitch|/111.9 + |τ_roll|/75.4 ≤1**. Sustained: denominators55.9 and37.7.

Both axes can produce torque simultaneously. Both cannot generally reach their individual pure-axis maxima simultaneously. For example, ankle115 N·m pitch plus65 N·m roll exceeds a45 N·m motor limit near neutral, even though each component separately fits. Ankle100 pitch plus20 roll fits at zero speed. Waist60 pitch plus20 roll fits at zero speed.

The fixed1.15 ankle ratio belongs to an older simulation and is not used. The neutral per-motor pitch ratio is about1.736 and roll about1.000; waist ratios are1.864 and1.257. All change with posture. The sampled current anatomical waist range includes positions without feasible rod closure or ±60° crank travel; the actuator flags these, and a future task must terminate/reset invalid poses. Native angular ranges have not been silently widened or replaced by a rectangular workspace claim.

`cad_differential.py` reproduces transmission kinematics, motor positions, speed mapping and virtual work. `hardware_actuators.py` enforces the coupled limits in Isaac Lab. Passive rods/cranks remain attached packaging visuals in the anatomical URDF; this is a mathematical transmission, not a full passive closed-loop PhysX mechanism or animated rod mesh. Rod-end contacts/35° tilt, spring compliance, strength and losses are not qualified by this model.

## Loss sensitivity

All headline torque products above are ideal. At85% total transmission efficiency, hip pitch/roll become **102 sustained /229.5 peak**, and knee **108.8 /244.8**. Speed reductions remain15 and16. At90% efficiency the selected shoulder-pitch caps give33.75 /67.5 N·m. Rod-linkage friction needs measurement separately. This sensitivity is not an assumed measured efficiency.

## Supplier torque/current and speed/voltage constants

The attached Excel tables were read without substituting their rated/maximum torque or speed rows for the user's assumptions. The **2026 exoskeleton Sheet1** has WGG7010-8 F14=0.65 N·m/A and F13=9.25 outputRPM/V; WGG6010-8 C14=0.39 and C13=21.6667. The other table's first sheet has different GIM6010-8 constants M16=0.47 for24 V and N16=0.65 for48 V. These variants must not be treated as identical purchased hardware. Full cells and SHA256 source hashes are in `assets/motor_constants_20260930.json`.

| Reference | Approx. current for sustained assumption | Approx. current for peak assumption |
|---|---:|---:|
| WGG7010-8,0.65 N·m/A | 27.7 A at18 N·m | 69.2 A at45;76.9 A at50 |
| WGG6010-8,0.39 N·m/A | 38.5 A at15 N·m | 76.9 A at30 |
| GIM6010-8 24 V,0.47 | 31.9 A at15 | 63.8 A at30 |
| GIM6010-8 48 V,0.65 | 23.1 A at15 | 46.2 A at30 |
| DJI77KV,ideal Kt=9.5493/77≈0.1240 | 64.5 A at8 shaft N·m | 145.1 A at18 shaft N·m |

These are **T/Kt using each source's current convention**. Sheets do not identify phase RMS/peak, Iq or bus current, and the supplier's constants and operating data are not perfectly self-consistent. They cannot be summed to obtain battery current or used uncalibrated as controller phase limits. DJI conversion is an ideal SI estimate; actual winding/controller convention is unresolved.

WGG7010 speed from Kv is **399.6 RPM nominal /466.2 full**; it does not establish >500 RPM for this8:1 variant. WGG6010 gives936 /1092 before the user's400 ceiling. GIM6010-8 instead gives498.6 /581.7. A50 N·m 7010 boost costs approximately23.5% more I²R heat than45 N·m at the same resistance, but does not establish a continuous50 rating.

## Servo sources and custom control

[Feetech STS3215-C018](https://www.feetechrc.com/525603.html) lists10 kg·cm rated,30 kg·cm stall and0.222 s/60° no-load at12 V:0.981 /2.942 N·m and45.05 RPM. The [user elbow screenshot](C:/Users/rshan/AppData/Local/Temp/codex-clipboard-c3ac298b-fce9-4b1e-9d43-fee7849c38f7.png) lists170 kg·cm stall and0.21 s/60° at24 V:16.67 N·m and47.62 RPM. Replacing the elbow driver/encoder does not establish continuous torque or enlarge its mechanical torque-speed envelope. The custom driver remains proposed hardware.

The Isaac brushed-servo envelope is a documented straight-line no-load-to-stall approximation, not a measured curve. At9.4 N·m elbow effort it predicts about20.8 RPM, rather than47.6. BLDCs retain the requested current/torque caps below user speed ceilings; real voltage sag and loaded curves are pending.

## 12S3P battery comparison

36 cells; nominal43.2 V and full50.4 V. Series count raises voltage, three parallel raises current/capacity.

- **Ampace JP50:** strong choice for explicitly specified short bursts. [Manufacturer product page](https://en.ampace.com/products/power-tools-product-matrix/jp/69) lists5 Ah,50 A high-power continuous and180 A for5 s; the detailed parameter table additionally distinguishes60 A maximum continuous from40 A full-discharge continuous. These are different duty qualifications. 12S3P is15 Ah, about644–648 Wh, approximately2.59 kg bare cells. Full-discharge cell arithmetic is120 A; headline50 A gives150 A;180 A/5 s gives540 A for5 s at cell level, not an assembled-pack guarantee.
- **Molicel P60C:** [manufacturer page](https://www.molicel.com/product/inr-21700-p60c/) lists6 Ah and100 A discharge. 12S3P gives18 Ah /777.6 Wh /2.70 kg bare cells, with300 A listed-current arithmetic. The public page does not establish unrestricted continuous duty or a120 A pulse duration; get the application datasheet before selecting a thermal budget. This is the highest listed discharge current among these reviewed manufacturer references, not a proof of the best robot pack.

No reviewed reference establishes the user's75 A continuous plus>120 A burst as an unrestricted general21700 rating. A practical first full-discharge budget for the JP50 example is **120 A pack**, about5.18 kW nominal before sag;150 A is6.48 kW under its separate higher-current duty. Actual pack limits are the minimum of cells, temperature, BMS, interconnects, wiring, connectors and fuse. Pulse duration and repetition matter.12S voltage does not add torque beyond the driver current cap.24 V elbows and12 V neck require appropriate regulated rails. Exoskeleton sheet voltage range stops at48 V, so exact electronics must be verified for50.4 V full12S.

Battery sufficiency cannot be proved from the sum of phase-current estimates. A gait power trace needs simultaneous shaft speed/torque, calibrated copper losses and converter efficiency. No verified battery-current/thermal/regeneration model has been added to Isaac.

## Isaac integration and verification

Use `make_actuators()` from `hardware_actuators.py`; each of the23 axes is assigned user-derived limits. `assets/actuator_limits.json` is current. `assets/actuator_limits_pending.json` is retained as historical pre-configuration evidence, not an active configuration. URDF differential limits are sampled bounding boxes only; the custom actuator provides actual shared constraints. Hardware boost is optional and off by default.

`test_transmissions.py` checks both ankles and waist against independent finite differences, rod closure, virtual-work power, simultaneous-axis motor clipping and overspeed motoring. Results are in `validation/actuators_20260930/transmission_validation.json`. The Isaac movement check uses `test_joints.py --headless --device cpu --hardware_actuators --validation_dir validation/actuators_20260930` and keeps the original native-axis evidence separate. This check remains fixed-base, gravity-free, no contacts and proxy masses with a test-only0.05 kg·m² armature; PD gains are fixture choices, not identified hardware gains. Initial unstable explicit-PD trial evidence is retained. Passing movement/FK checks do not establish walking capability or thermally sustained torque.
'''
# Keep readable prose around numeric values and model names.
for before,after in {
    'remains45':'remains 45', 'DJI10018 77KV':'DJI 10018 77 KV',
    'earlier17:1':'earlier 17:1', 'uses43.2':'uses 43.2', 'and50.4':'and 50.4',
    'physically4:1':'physically 4:1', 'a2.5':'a 2.5', '/18.75':' / 18.75',
    ':4×9.375=37.5 and4×18.75=75':': 4 × 9.375 = 37.5 and 4 × 18.75 = 75',
    'still400/4=100':'still 400 / 4 = 100', 'yield60':'yield 60',
    "intended2.5":"intended 2.5", 'q=[pitch,roll]':'q = [pitch, roll]',
    'θ=[motor A,motor B]':'θ = [motor A, motor B]', '/26 mm':' / 26 mm',
    '≤1':'≤ 1', 'denominators62.5 and35.9':'denominators 62.5 and 35.9',
    'denominators55.9 and37.7':'denominators 55.9 and 37.7',
    'ankle115':'ankle 115', 'plus65':'plus 65', 'a45':'a 45',
    'Ankle100':'Ankle 100', 'Waist60':'Waist 60', 'fixed1.15':'fixed 1.15',
    'about1.736 and roll about1.000':'about 1.736 and roll about 1.000',
    'are1.864 and1.257':'are 1.864 and 1.257', 'At85%':'At 85%',
    '/229.5':' / 229.5', '/244.8':' / 244.8', 'remain15 and16':'remain 15 and 16',
    'At90%':'At 90%', 'give33.75 /67.5':'give 33.75 / 67.5',
    'outputRPM/V':'output RPM/V', 'for24 V and N16=0.65 for48 V':'for 24 V and N16=0.65 for 48 V',
    'WGG7010-8,0.65':'WGG7010-8, 0.65', 'WGG6010-8,0.39':'WGG6010-8, 0.39',
    'at18':'at 18', 'at45;76.9':'at 45; 76.9', 'at50':'at 50',
    'at15':'at 15', 'at30':'at 30', 'DJI77KV,ideal':'DJI 77 KV, ideal',
    'at8 shaft':'at 8 shaft', 'this8:1':'this 8:1', "user's400":"user's 400",
    'A50 N·m':'A 50 N·m', 'approximately23.5%':'approximately 23.5%',
    'than45':'than 45', 'continuous50':'continuous 50', 'lists10':'lists 10',
    'rated,30':'rated, 30', 'and0.222':'and 0.222', 'at12':'at 12',
    '/2.942':' / 2.942', 'and45.05':'and 45.05', 'lists170':'lists 170',
    'and0.21':'and 0.21', 'at24':'at 24', 'and47.62':'and 47.62',
    'At9.4':'At 9.4', 'about20.8':'about 20.8', 'than47.6':'than 47.6',
    'nominal43.2':'nominal 43.2', 'full50.4':'full 50.4', 'lists5':'lists 5',
    'Ah,50':'Ah, 50', 'and180':'and 180', 'for5 s':'for 5 s',
    'distinguishes60':'distinguishes 60', 'from40':'from 40', 'is15':'is 15',
    'is120':'is 120', 'headline50':'headline 50', 'gives150':'gives 150',
    ';180 A/5 s gives540':'; 180 A / 5 s gives 540', 'lists6':'lists 6',
    'and100':'and 100', 'gives18':'gives 18', '/777.6':' / 777.6', '/2.70':' / 2.70',
    'with300':'with 300', 'a120':'a 120', "user's75":"user's 75", 'general21700':'general 21700',
    'about5.18':'about 5.18', 'sag;150':'sag; 150', 'is6.48':'is 6.48',
    'matter.12S':'matter. 12S', 'cap.24 V':'cap. 24 V', 'and12 V':'and 12 V',
    'at48 V':'at 48 V', 'for50.4':'for 50.4', 'the23':'the 23',
    'test-only0.05':'test-only 0.05',
}.items():
    text=text.replace(before,after)
validation_path=HERE/'validation/actuators_20260930/joint_validation.json'
if validation_path.exists():
    result=json.loads(validation_path.read_text())
    if result.get('all_pass') and len(result.get('joints',[]))==23:
        text+='\n\n**Completed actuator movement check:** all 23 joints passed tracking, unchanged-axis, frame-derived FK and reset checks. No invalid differential poses were encountered. Recorded motor-shaft peaks stayed within the 30 N·m waist / 45 N·m ankle caps. The exact report and final stdout are `validation/actuators_20260930/joint_validation.json` and `isaac_final_stdout.log`. Continuous-torque duty is recorded but not time/thermal enforced; no locomotion training was started.\n'
path=OUT/'ACTUATOR_CALCULATIONS.md';path.write_text(text,encoding='utf-8')
repo=Path(r'C:\Users\rshan\Documents\GitHub\Humanoid-Robot\docs\robot')
mirror=Path(r'E:\Aditya\robot_sim\docs\robot')
for dest in (repo,mirror):
    shutil.copy2(path,dest/'ACTUATOR_CALCULATIONS.md')
    p=dest/'TORQUE_NOTES.md';old=p.read_text(encoding='utf-8')
    if '## Current user configuration' not in old:
        old=old.replace('# Torque evidence and remaining qualification','# Torque evidence and remaining qualification\n\n## Current user configuration — September 30, 2026\n\n[Current actuator calculations](ACTUATOR_CALCULATIONS.md) supersede the earlier selection assumptions below:7010 18/45 N·m,6010 15/30, both400 RPM at factory output; DJI8/18 shaft N·m with15:1 hips and16:1 knees; additional4:1 shoulder pitch with2.5-equivalent torque cap; CAD-dependent coupled ankle/waist mapping. Actual mass/contact/thermal qualification and elbow continuous torque remain pending. Historical evidence below is retained.\n')
        p.write_text(old,encoding='utf-8')
    p=dest/'BUILD_REPORT.md';old=p.read_text(encoding='utf-8')
    old=old.replace('**Stopped for user torque/velocity configuration.** Each entry in `physical_v29/assets/actuator_limits_pending.json` is null. URDF-required effort/velocity numbers and the test\'s PD gains are explicit fixture placeholders, not selected motor limits. No training was started, and the old 1.524 m / 22-axis training model remains separate.','**Actuator assumptions updated September 30.** [Current calculations](ACTUATOR_CALCULATIONS.md) record the user-selected torque/RPM assumptions and the saved-CAD differential Jacobians. `physical_v29/assets/actuator_limits.json` and the explicit `hardware_actuators.py` are current; the null pending file is historical. Current masses/contact/thermal behavior remain unqualified. No training started; the old22-axis model is separate.')
    if '[Current actuator calculations]' not in old:old=old.replace('## Files and report map','Current user actuator configuration: [Current actuator calculations](ACTUATOR_CALCULATIONS.md).\n\n## Files and report map')
    p.write_text(old,encoding='utf-8')
    p=dest/'SIMULATION.md';old=p.read_text(encoding='utf-8')
    marker='## September 30 actuator configuration'
    if marker not in old:old+= '\n\n'+marker+'\n\nThe user now supplied motor torque/RPM assumptions. See [Actuator calculations](ACTUATOR_CALCULATIONS.md) and the active `physical_v29/assets/actuator_limits.json`. The previous null pending file and original fixture report remain historical. `hardware_actuators.make_actuators()` applies saved-CAD pose-dependent ankle/waist Jacobians and clips shared motor torque/speed. A new movement fixture uses `--hardware_actuators --validation_dir validation/actuators_20260930`; it remains no-gravity/proxy-dynamics and does not certify locomotion. No training started.\n'
    p.write_text(old,encoding='utf-8')
p=HERE/'README.md';old=p.read_text(encoding='utf-8')
marker='## September 30 user actuator configuration'
if marker not in old:old+='\n\n'+marker+'\n\nCurrent values are `assets/actuator_limits.json`; the old null pending file above is historical. Use `hardware_actuators.make_actuators()` in task configurations for CAD-based coupled ankle/waist torque and speed, rather than independent URDF effort boxes. Run `test_transmissions.py` and `test_joints.py --headless --device cpu --hardware_actuators --validation_dir validation/actuators_20260930`. Calculations, source-cell constants, assumptions and remaining qualification are in [Actuator calculations](analysis/20260930_actuators/ACTUATOR_CALCULATIONS.md). No training started; masses remain proxies.\n'
p.write_text(old,encoding='utf-8')
print(path)
