"""Maintain current local documentation from measured/configured evidence."""
import csv,json,math,shutil
from pathlib import Path

HERE=Path(__file__).resolve().parent
DOCS=Path(r'C:\Users\rshan\Documents\GitHub\Humanoid-Robot\docs\robot')
MIRROR=HERE.parent/'docs/robot'
ANALYSIS=HERE/'analysis/20260930_actuators'
META=json.loads((HERE/'assets/export_metadata.json').read_text())
PROFILE=META['actuator_profile'];VALID=HERE/META['hardware_validation_dir']
a=json.loads((ANALYSIS/'calculation.json').read_text())
movement=json.loads((VALID/'joint_validation.json').read_text())
preview=json.loads((HERE/(META['hardware_validation_dir']+'_visual')/'joint_validation.json').read_text())
assert movement['all_pass'] and len(movement['joints'])==23
assert preview['all_pass'] and len(preview['joints'])==23
FIX=HERE.parent/'outputs/shin_cad_20260912/v29/wrist_review_20260930'
wrist=json.loads((FIX/'verification.json').read_text())
reopen=json.loads((FIX/'native_reopen.json').read_text())
assert wrist['all_pass'] and reopen['all_pass']
backup=ANALYSIS/'revisions/local_docs_before_eff95';backup.mkdir(parents=True,exist_ok=True)
for name in ('ACTUATOR_CALCULATIONS.md','BUILD_REPORT.md','SIMULATION.md','BOM.csv','BOM.md','OPEN_ITEMS.md','EXPORTS.md','EVIDENCE.md','TORQUE_NOTES.md'):
    if not (backup/name).exists():shutil.copy2(DOCS/name,backup/name)
old_report=(backup/'ACTUATOR_CALCULATIONS.md').read_text(encoding='utf-8')
electrical=old_report.split('## Supplier torque/current and speed/voltage constants',1)[1].split('## Isaac integration and verification',1)[0]
old_servo=electrical.split('## Servo sources and custom control',1)[1].split('## 12S3P battery comparison',1)[0]
new_servo='''

[Feetech STS3215-C018](https://www.feetechrc.com/525603.html) specifies 10 kg·cm rated / 30 kg·cm stall and 0.222 s/60° no-load at 12 V: 0.981 / 2.942 N·m and 45.05 RPM. The user elbow screenshot gives 170 kg·cm stall and 0.21 s/60° at 24 V: 16.67 N·m and 47.62 RPM no-load.

The user now selects **7 N·m sustained at 45 RPM** for the elbow. Both sustained torque and the operating command cap are 7 N·m; the advertised stall/no-load values remain metadata. This is an assumed operating box, not a measured loaded curve. It supersedes the earlier linear elbow approximation. The neck retains its documented linear stall-to-no-load planning curve. Custom driver/encoder feedback remains planned hardware; mechanical and thermal qualification is pending.

'''
electrical=electrical.replace(old_servo,new_servo)
rows=[]
for name,r in a['joint_types'].items():
    note='Neutral pure-axis; shared motor limits' if 'differential' in r else 'User planning assumption'
    if name=='elbow_pitch':note='User 7 N·m / 45 RPM operating box'
    if name.startswith('head_'):note='12 V rated / stall / no-load'
    if name=='shoulder_pitch':note='Physical 4:1; capped input torque'
    rows.append(f"| {name.replace('_',' ')} | {r['continuous_Nm']:.2f} | {r['effort_Nm']:.2f} | {r['velocity_rad_s']*60/(2*math.pi):.1f} | {note} |")
report='''# Current actuator calculations — September 30, 2026

Current profile: `eff95_elbow7_20260930`. Values are user-selected planning assumptions for physical v29. Actual link mass/inertia, thermal duty and contact dynamics remain unqualified. Factory-geared motor outputs and servo outputs already include their internal reduction; they are not derated twice.

**Efficiency is 95% overall for each added transmission or rod linkage.** Knee 8×2 uses one overall 95% assumption, rather than 95% twice. Direct 7010/6010 and neck/elbow outputs retain their supplied output values. The shoulder-pitch speed target is 120 RPM, requiring 480 RPM at its geared motor output, below the adopted 500 RPM cap.

| Joint, same on both sides | Sustained/rated N·m | Peak/operating cap N·m | Speed ceiling RPM | Meaning |
|---|---:|---:|---:|---|
'''+ '\n'.join(rows)+'''

## Transmission and torque caps

DJI 10018 77 KV uses 8 N·m sustained / 18 burst at the bare shaft. Both hip pitch and roll use 15:1: 8×15×0.95=114 sustained and 18×15×0.95=256.5 peak. Knee uses 8:1 planetary ×2:1 chain: 121.6 / 273.6 N·m. Nominal no-load estimates at 43.2 V are 221.76 RPM hips and 207.9 RPM knees; full-charge estimates are 258.72 / 242.55. These are speed ceilings, not claims that peak torque persists to no-load speed.

Shoulder pitch physically adds 4:1. Input caps remain 9.375 N·m sustained / 18.75 peak, equivalent to a 2.5 multiplier of the 6010 torque ratings before losses. Joint output is **35.625 / 71.25 N·m** after 95% efficiency. Its selected 120 RPM needs 480 motor-output RPM. The extra planetary is still planned hardware, not a newly modeled gearbox. Shoulder roll remains direct 15 / 30 N·m at 400 RPM.

## CAD-dependent ankle and waist differentials

`assets/differential_geometry.json` derives rod-eye centres, shaft axes and rod lengths from the saved native v29 joint frames. For q=[pitch,roll] and shaft angles θ, the solver computes J(q)=∂θ/∂q. Velocity mapping is θ̇=Jq̇; the selected torque model is **τ_joint=0.95 Jᵀ τ_motor**. Requested effort is transformed to both motor shafts, clipped there and transformed back. This scalar loss model is a planning approximation, not a measured bidirectional friction/regeneration model.

Near neutral, approximate peak envelopes are:

- Ankles: |τ_pitch|/148.42 + |τ_roll|/85.34 ≤1. Sustained denominators are 59.37 and 34.14 N·m.
- Waist: |τ_pitch|/106.27 + |τ_roll|/71.67 ≤1. Sustained denominators are 53.14 and 35.83 N·m.

Actual saved linkage asymmetry and posture dependence are included in the runtime matrix. Pitch and roll can act simultaneously, but cannot both reach their separate pure-axis maxima simultaneously. Motor-output speed caps remain 400 RPM; efficiency does not change geometric speed reduction. Ankle neutral pitch/roll ratios are about 1.736 / 1.000 and waist 1.864 / 1.257. The older fixed 1.15 ankle factor is unused.

Some corners of the anatomical waist limit rectangle have no feasible rod closure or allowed crank travel. Invalid poses are flagged; these tests use valid pure-axis samples. Passive rod/crank visuals are attached packaging meshes, not an animated passive PhysX closed loop. Rod-end tilt/contact, compliance and whole-workspace strength remain unqualified.

## Supplier torque/current and speed/voltage constants
'''+electrical+'''## Isaac checks and GUI preview

Use `hardware_actuators.make_actuators()` with `assets/actuator_limits.json`. URDF differential limits are bounding boxes; the custom actuator supplies actual shared constraints. Continuous torque is recorded but no time/thermal model is enforced. The optional 50 N·m waist-yaw boost stays off.

`validation/eff95_elbow7_20260930` contains passing finite-difference/closure/95%-power/shared-limit tests and all **23** strict Isaac movement/FK/reset checks. `validation/eff95_elbow7_20260930_visual` records the larger, smooth fast preview. Both use fixed base, disabled gravity/contacts, proxy masses/inertia and test-only 0.05 kg·m² armature. Strict gains are 120/15 for waist/hip/knee and 35/4 otherwise; fast preview uses 120/8 on heavy joints at 10 ms steps. Neither fixture qualifies locomotion or real load capability.

```powershell
& 'E:\\Aditya\\robot_sim\\env_isaaclab\\Scripts\\python.exe' -X utf8 `
  'E:\\Aditya\\robot_sim\\physical_v29\\test_joints.py' `
  --device cpu --hardware_actuators --visual_fast --seconds_per_joint 2.5 --repeat
```

The GUI labels each joint, shows target/measured angles, and provides front/side/three-quarter views, pause, repeat and stop. Each axis takes about 2.5 simulated seconds plus a brief reset; wall time depends on rendering. Wrists and fingers stay static. Close the window to end the test. `--exit_after_cycle` closes after one pass.

## Wrist correction

The v29 hand mesh copy omitted the 6° resting shoulder rotation that the forearms and adapters already carried. Native mesh coordinates were assumed to be root-context coordinates. All eight complete hand category meshes are now rotated into the adapter frames, preserving original meshes hidden. Six M2.5 axes are coaxial with the adapters within 0.003 mm. All 1537 BReps and 47 joints are unchanged. The corrected full native archive independently reopens; affected simulation visuals are synchronized. Evidence: `outputs/shin_cad_20260912/v29/wrist_review_20260930`.

All documentation remains local. No Git commit, push or publication was performed.
'''
(ANALYSIS/'ACTUATOR_CALCULATIONS.md').write_text(report,encoding='utf-8')
(DOCS/'ACTUATOR_CALCULATIONS.md').write_text(report,encoding='utf-8')
with (DOCS/'BOM.csv').open(newline='',encoding='utf-8-sig') as f:
    reader=csv.DictReader(f);fields=reader.fieldnames;bom=list(reader)
updates={
 'ACT-02':'Factory output 15 N.m sustained / 30 peak / 400 RPM; shoulder pitch adopts 500 RPM motor-output cap and 120 RPM joint target through planned 4:1. Two waist motors use 95% rod-linkage efficiency. Direct shoulder roll output unchanged. Exact variant/current convention to confirm.',
 'ACT-05':'User assumes 7 N.m sustained at 45 RPM with a 7 N.m operating command cap. Advertised 24 V stall 16.67 N.m / no-load 47.62 RPM retained as metadata. RDS51150 65x30x48 case is authorized proxy; exact RDS51160 dimensions and real thermal/loaded behavior unverified.',
 'TRN-01':'8:1 planetary x2:1 chain =16:1. At one overall assumed 95% efficiency: 121.6 sustained /273.6 peak N.m. Actual gearbox, chain length/tensioning and strength remain to design.',
 'TRN-03':'Both hips use 15:1 cycloidal. Assumed 95% efficiency gives114 sustained /256.5 peak N.m. Actual gears, support, friction and strength remain unqualified.',
 'TRN-04':'15:1 cycloidal, replacing previous17:1 pitch plan. Assumed95% gives114 sustained /256.5 peak N.m. Not a finished gearset.',
 'TRN-05':'Physical4:1 external planetary. Motor input cap9.375 sustained /18.75 peak gives35.625 /71.25 joint N.m at95%;120 joint RPM requires480 geared-motor RPM, adopted cap500. Planned, not modeled.',
}
for row in bom:
    if row['id'] in updates:row['notes']=updates[row['id']]
with (DOCS/'BOM.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(bom)
def edit(name,pairs):
    path=DOCS/name;text=path.read_text(encoding='utf-8')
    for old,new in pairs:text=text.replace(old,new)
    path.write_text(text,encoding='utf-8')
edit('BOM.md',[('elbow continuous torque remain to qualify','measured elbow continuous duty remain to qualify')])
edit('BUILD_REPORT.md',[
 ('Saved Fusion source **Humanoid_complete_v29_revised**, version 1,','Saved Fusion source **Humanoid_complete_v29_revised**, version '+str(reopen['source_version'])+', wrist alignment corrected,'),
 ('Actual masses, thermal duty, elbow continuous torque and contact dynamics remain unqualified.','Current profile uses95% added-transmission efficiency,7 N·m sustained elbow at45 RPM and120 RPM shoulder pitch. Actual masses, measured thermal/loaded duty and contact dynamics remain unqualified.'),
 ('| September 30 actuator test | PASS: all 23 axes tracked/reset with user-assumed motor caps and CAD-dependent ankle/waist transmission; no invalid transmission samples. Fixed base, no gravity/contacts, proxy inertia. |','| Current95% actuator test | PASS: all23 axes tracked/reset, FK matched, no invalid transmission samples;7 N·m/45 RPM elbow and120 RPM shoulder pitch. Fixed base, no gravity/contacts, proxy inertia. |\n| Wrist alignment correction | PASS: both hand assemblies aligned; six adapter hole axes match,1537 solid parts and47 joints unchanged; corrected full F3D independently reopens. |'),
])
simdoc='''# Physical v29 Isaac Lab handoff

Current package: `E:/Aditya/robot_sim/physical_v29`. Physical robot:1.6002 m,24 rigid links,23 controlled body axes,140 colored visual meshes. The older22-axis training asset remains separate.

## Run the fast GUI preview

```powershell
& 'E:\\Aditya\\robot_sim\\env_isaaclab\\Scripts\\python.exe' -X utf8 `
  'E:\\Aditya\\robot_sim\\physical_v29\\test_joints.py' `
  --device cpu --hardware_actuators --visual_fast --seconds_per_joint 2.5 --repeat
```

Each joint makes a smooth visible sweep, then resets. Joint names and target/measured angles appear in the control panel. Front/side/three-quarter cameras, pause, repeat and stop are available. About2.5 simulated seconds per axis plus0.25 reset; rendering can extend wall time. Wrists and fingers remain fixed. Closing the window ends the preview; use `--exit_after_cycle` for one pass that closes automatically. No walking/training starts.

## Current configuration and evidence

[Actuator calculations](ACTUATOR_CALCULATIONS.md) describes `eff95_elbow7_20260930`:95% overall added transmissions/rod-linkages, elbow7 N·m sustained and operating cap at45 RPM, shoulder pitch120 RPM with480/500 RPM motor output. Factory8:1 outputs are not derated twice. `hardware_actuators.make_actuators()` applies the saved-CAD pose-dependent ankle/waist Jacobians and shared motor constraints. Independent URDF boxes alone cannot implement those constraints.

All23 axes passed strict tracking, companion-axis,24-link FK and reset checks in `validation/eff95_elbow7_20260930/joint_validation.json`. Three differential analytical/float32/shared-limit checks passed in `transmission_validation.json`. Large smooth preview evidence is separate in `validation/eff95_elbow7_20260930_visual`. Earlier ideal-efficiency actuator evidence remains in `validation/actuators_20260930`; original fixture evidence is `validation/joint_validation.json`.

Strict check:

```powershell
& 'E:\\Aditya\\robot_sim\\env_isaaclab\\Scripts\\python.exe' -X utf8 `
  'E:\\Aditya\\robot_sim\\physical_v29\\test_joints.py' --headless --device cpu --hardware_actuators
```

## Scope and file ownership

Base fixed; gravity and contacts disabled. Masses are1 kg/link proxies; inertia,0.05 kg·m² armature and PD gains are test choices. Strict fixture is5 ms,120/15 on waist/hip/knee and35/4 elsewhere. Fast preview is10 ms and120/8 on heavy joints. This checks motion/control plumbing; it does not qualify load capability, collision-free travel, standing, thermal duty or battery current. Continuous torque is documented without a time/thermal model.

CAD front−Y maps to Isaac+X. Shoulder6° rest abduction is baked into geometry, q=0 is saved rest, knee flexion is positive/backward and elbow flexion negative. Hands are static official reference assemblies. Wrist hand meshes now include the missing rest rotation; eight affected visual STLs are synchronized. New full F3D independently reopens. Passive differential rods remain fixed packaging visuals; runtime mathematical closure supplies their varying ratios. Some combined waist corners are infeasible and are flagged.

`assets/actuator_limits.json` is current; `actuator_limits_pending.json` is historical. Preserve all140 `assets/meshes` files and the entire `assets/usd/configuration` directory. Use `--force_convert` after editing visuals/URDF. Exact source hashes and wrist provenance are in `assets/export_metadata.json`. Toolkit cleanup can hang after a completed report; stop only the owned test processes after confirming completion. No Git commit/push or locomotion training was performed.
'''
(DOCS/'SIMULATION.md').write_text(simdoc,encoding='utf-8')
(HERE/'README.md').write_text(simdoc.replace('[Actuator calculations](ACTUATOR_CALCULATIONS.md)','[Actuator calculations](analysis/20260930_actuators/ACTUATOR_CALCULATIONS.md)'),encoding='utf-8')
edit('OPEN_ITEMS.md',[('Elbow continuous torque is unknown.','Elbow7 N·m sustained at45 RPM is a user assumption; qualify it on the actual actuator.'),('Decide whether the shoulder-pitch continuous cap should also be reduced to the 2.5-equivalent torque level.','Current shoulder pitch uses35.625 /71.25 N·m at95% and120 RPM; verify the physical4:1 stage and motor speed margin.')])
edit('TORQUE_NOTES.md',[('Actual mass/contact/thermal qualification and elbow continuous torque remain pending.','Current95% profile,7 N·m/45 RPM elbow and120 RPM shoulder pitch supersede the earlier assumptions below; real loaded/thermal duty remains pending.')])
for name in ('EXPORTS.md','EVIDENCE.md'):
    p=DOCS/name;text=p.read_text(encoding='utf-8');marker='## Wrist alignment and current actuator update'
    if marker in text:text=text.split(marker)[0].rstrip()
    text+='\n\n'+marker+'\n\nCorrected native archive: `'+wrist['archive']+'`; SHA256 `'+wrist['sha256']+'`. Native reopen passed1545 visible shapes,47 joint frames/states/ranges and all parent grounds. Originals remain hidden and the before archive is preserved. Live source version'+str(reopen['source_version'])+'. Evidence and reviewed image: `'+str(FIX)+'`.\n\nActive95% actuator/strict movement evidence: `'+str(VALID)+'`. Fast GUI preview evidence: `'+str(VALID)+'_visual`. The simulation visual meshes match the corrected hands; all joint frames and actuator settings are preserved. The older downloaded handoff remains historical unless refreshed.\n'
    p.write_text(text,encoding='utf-8')
for name in ('ACTUATOR_CALCULATIONS.md','BUILD_REPORT.md','SIMULATION.md','BOM.csv','BOM.md','OPEN_ITEMS.md','EXPORTS.md','EVIDENCE.md','TORQUE_NOTES.md'):
    text=(DOCS/name).read_text(encoding='utf-8')
    for a,b in [('current95%','current 95%'),('Current95%','Current 95%'),('all23','all 23'),('All23','All 23'),('all140','all 140'),('passed1545','passed 1545'),('shapes,47','shapes, 47'),('version2','version 2'),('robot:1','robot: 1'),('older22','older 22'),('about2','about 2'),('About2','About 2'),('plus0','plus 0'),('seconds per','seconds per'),('elbow7','elbow 7'),('at45','at 45'),('pitch120','pitch 120'),('torque;7','torque; 7'),('profile uses95','profile uses 95'),('efficiency,7','efficiency, 7'),('and120','and 120'),('match,1537','match, 1537'),('and47','and 47'),('at95','at 95'),('uses35','uses 35'),('pitch6','pitch 6')]:text=text.replace(a,b)
    # Preserve literal profile/path tokens after prose spacing replacements.
    text=text.replace('eff95_elbow 7_20260930','eff95_elbow7_20260930')
    (DOCS/name).write_text(text,encoding='utf-8');shutil.copy2(DOCS/name,MIRROR/name)
evidence=DOCS/'evidence/v29/wrist_review_20260930';evidence.mkdir(parents=True,exist_ok=True)
for name in ('verification.json','native_reopen.json','fix_record.json','wrists_aligned.png','simulation_visual_sync.json'):
    shutil.copy2(FIX/name,evidence/name)
print(json.dumps({'local_docs':str(DOCS),'profile':PROFILE,'strict_joints':23,'preview_joints':23,'native_wrist_reopen_pass':True,'BOM_rows':len(bom)},indent=2))
