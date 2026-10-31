"""Reconcile the local v29 report/BOM with the September 30 actuator inputs."""
import csv
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
REPO = Path(r'C:\Users\rshan\Documents\GitHub\Humanoid-Robot')
DOCS = REPO / 'docs/robot'
MIRROR = HERE.parent / 'docs/robot'
validation = json.loads((HERE / 'validation/actuators_20260930/joint_validation.json').read_text())
passed = validation.get('all_pass') and len(validation.get('joints', [])) == 23

with (DOCS / 'BOM.csv').open(newline='', encoding='utf-8-sig') as stream:
    reader = csv.DictReader(stream)
    fields = reader.fieldnames
    rows = list(reader)
by_id = {row['id']: row for row in rows}
updates = {
    'ACT-01': {'notes': 'September 30 user assumption at factory 8:1 output: 18 N.m sustained, 45 peak, 400 RPM. Optional waist-yaw 50 N.m boost remains off. WGG7010 sheet Kt 0.65 N.m/A; exact GWK/GIM/WGG purchased variant/current convention to confirm.'},
    'ACT-02': {'notes': 'September 30 user assumption at factory 8:1 output: 15 N.m sustained, 30 peak, 400 RPM. Two waist differential motors; two shoulder rolls; two shoulder pitches with additional physical 4:1 and capped input torque. WGG/GIM variants have different Kt. Each complete motor is one purchase.'},
    'ACT-03': {'notes': 'User assumes 8 N.m sustained / 18 burst at bare 77 KV shaft. Hip pitch AND roll each 15:1 cycloidal; knee 8:1 planetary x 2:1 chain. Estimated motor CAD and gearsets still require measurement/design.'},
    'ACT-04': {'notes': '12 V STS3215-C018 source: 10 kg.cm rated / 30 stall, 0.222 s/60 degrees no-load. Configured 0.981 / 2.942 N.m and 45.05 RPM; thermal duty not qualified. Older 7.4 V variant differs.', 'source_url': 'https://www.feetechrc.com/525603.html'},
    'ACT-05': {'notes': 'Case fixes to upper arm; both coaxial outputs carry forearm fork. Authorized RDS51150 65 x 30 x 48 mm packaging proxy, not verified RDS51160 dimensions. User screenshot at 24 V: 170 kg.cm stall = 16.67 N.m, 0.21 s/60 degrees = 47.62 RPM no-load. Continuous torque and actual custom driver remain unqualified.'},
    'TRN-01': {'notes': 'User 8:1 internal planetary target. With knee chain stage the total is 16:1, giving ideal 128 sustained / 288 peak N.m from shaft assumptions. Available housing is not a finished gearset.'},
    'TRN-03': {'notes': 'September 30 user target 15:1 cycloidal: ideal 120 sustained / 270 peak N.m. Actual gearset, losses, support and strength not completed.'},
    'TRN-04': {'status': 'Undesigned; 15:1 user target', 'notes': 'September 30 both hip pitch and roll use 15:1, superseding previous 17:1 hip-pitch plan. Ideal 120 sustained / 270 peak N.m; actual gearset, backlash, losses and strength require qualification.'},
    'PWR-01': {'status': '12S3P plan; JP50/P60C candidates, not selected', 'notes': '36 cells. JP50 example: 15 Ah, 43.2 V nominal / 50.4 V full; 120 A full-discharge cell arithmetic or separate 180 A/cell 5 s pulse duty. P60C: 18 Ah and manufacturer-listed 100 A/cell discharge; detailed duty not obtained. See ACTUATOR_CALCULATIONS.md. Actual pack limits, thermal qualification and sourcing remain open.'},
    'ELE-09': {'status': 'Custom current-sense/encoder control planned', 'notes': 'User intends custom elbow driver with encoder feedback and force control. Serial/PWM interface, current calibration and sustained load capability require design; selected servo mechanics remain unchanged.'},
}
for row_id, values in updates.items():
    by_id[row_id].update(values)
if 'TRN-05' not in by_id:
    row = dict.fromkeys(fields, '')
    row.update(id='TRN-05', category='Transmissions', item='Additional shoulder-pitch planetary', quantity='2', unit='gearbox', status='4:1 user plan; not modeled or designed', evidence='September 30 user actuator selection', notes='Physical 4:1 adds to the factory 8:1. Input capped to 9.375 sustained / 18.75 peak N.m for ideal joint 37.5 / 75 N.m; speed remains 100 RPM. Continuous input cap is an explicit interpretation; if only peak is capped, sustained ideal output could be 60 N.m.')
    rows.append(row)
with (DOCS / 'BOM.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
intro = '''# Current physical robot BOM — v29

[Build report](BUILD_REPORT.md) · [CSV](BOM.csv) · [CAD inventory](CAD_PARTS.csv) · [Actuator calculations](ACTUATOR_CALCULATIONS.md).

Quantities describe current CAD or explicitly planned hardware. Component references must not be counted twice. Prices remain blank except the dated September 29 camera quote. This is not a current purchasing quotation.

September 30 motor assumptions are configured in the isolated physical simulation. Gearset designs, thermal ratings, exact motor variants and elbow continuous torque remain to qualify. The additional shoulder-pitch 4:1 stage is planned hardware, not new CAD. The active casing set has 51 cosmetic pieces and two structural waist halves; old generic elbow bearings and guards are excluded.
'''
groups = {}
for row in rows:
    groups.setdefault(row['category'], []).append(row)
for category, items in groups.items():
    intro += f'\n## {category}\n\n| ID | Item | Quantity | Status |\n|---|---|---:|---|\n'
    for row in items:
        intro += f"| {row['id']} | {row['item']} | {row['quantity']} | {row['status']} |\n"
intro += '\nExact interfaces, sources and qualifications are in the CSV. Actual masses, fastener quantities, printed load capacity and assembled battery limits remain open. Documentation stays local; no commit or push.\n'
(DOCS / 'BOM.md').write_text(intro, encoding='utf-8')

def edit(name, replacements):
    path = DOCS / name
    text = path.read_text(encoding='utf-8')
    for old, new in replacements:
        text = text.replace(old, new)
    path.write_text(text, encoding='utf-8')

edit('BUILD_REPORT.md', [
    ('**Hardware torque and velocity limits are pending your configuration.** Each entry in `physical_v29/assets/actuator_limits_pending.json` is null. URDF-required effort/velocity numbers and the test\'s PD gains are explicit fixture placeholders, not selected motor limits. No training was started, and the old 1.524 m / 22-axis training model remains separate.', '**September 30 actuator assumptions are configured.** [Current calculations](ACTUATOR_CALCULATIONS.md) record the selected torque/RPM inputs, source constants and saved-CAD ankle/waist Jacobians. Use `physical_v29/assets/actuator_limits.json` with `hardware_actuators.make_actuators()`; the null pending file is historical. Actual masses, thermal duty, elbow continuous torque and contact dynamics remain unqualified. No training started; the older 22-axis model stays separate.'),
    ('Exact validation command, fixture and pending limit configuration', 'Current actuator configuration and original movement-fixture evidence'),
])
if passed:
    edit('BUILD_REPORT.md', [
        ('| Isaac Lab joint test | PASS: every body joint tracked legal sample commands, other axes stayed near neutral, all 24 link poses matched frame-derived FK, then returned to neutral |', '| Original Isaac Lab joint test | PASS: every body joint tracked legal sample commands, other axes stayed near neutral, all 24 link poses matched frame-derived FK, then returned to neutral |\n| September 30 actuator test | PASS: all 23 axes tracked/reset with user-assumed motor caps and CAD-dependent ankle/waist transmission; no invalid transmission samples. Fixed base, no gravity/contacts, proxy inertia. |')
    ] if '| September 30 actuator test |' not in (DOCS / 'BUILD_REPORT.md').read_text(encoding='utf-8') else [])
edit('OPEN_ITEMS.md', [
    ('Hardware limits are deliberately waiting for the user; no questions were sent during the work.', 'September 30 motor torque/RPM assumptions are now configured. Physical thermal, electrical and load qualification remains open.'),
    ('1. Configure continuous/peak torque, speed and control behavior for all 23 body axes. `physical_v29/assets/actuator_limits_pending.json` has null entries; fixture limits are not hardware settings.', '1. Qualify the configured user assumptions against exact hardware variants, loaded torque-speed curves and thermal duty. Current values are `physical_v29/assets/actuator_limits.json`; the former null file is historical. Elbow continuous torque is unknown. Decide whether the shoulder-pitch continuous cap should also be reduced to the 2.5-equivalent torque level.'),
    ('8. Select host, cells, BMS, converters, control interfaces and cooling/wiring.', '8. Select host, exact JP50/P60C or alternative cells, BMS, converters, control interfaces and cooling/wiring. Confirm 50.4 V full-charge compatibility and pulse repetition; cell current arithmetic does not qualify the pack.'),
])
edit('EXPORTS.md', [
    ('Mass/inertia and actuator drive values are test proxies; hardware limits remain pending.', 'Mass/inertia remain test proxies. September 30 user actuator assumptions are configured in the workspace package; use the active `actuator_limits.json` and custom coupled actuator from [Actuator calculations](ACTUATOR_CALCULATIONS.md). Earlier downloaded packages retain original fixture settings unless updated.'),
])
edit('TORQUE_NOTES.md', [
    ('below:7010 18/45 N·m,6010 15/30, both400 RPM at factory output; DJI8/18 shaft N·m with15:1 hips and16:1 knees; additional4:1 shoulder pitch with2.5-equivalent torque cap;', 'below: 7010 18/45 N·m, 6010 15/30, both 400 RPM at factory output; DJI 8/18 shaft N·m with 15:1 hips and 16:1 knees; additional 4:1 shoulder pitch with a 2.5-equivalent torque cap;'),
    ('No hardware torque/velocity limits were configured in the final Isaac fixture.', 'The original axis fixture used test drives. The September 30 custom actuator now uses user-supplied limits; physical qualification remains pending.'),
])

current = '''## Current actuator model — September 30

[Actuator calculations](ACTUATOR_CALCULATIONS.md) and `assets/actuator_limits.json` are current. `hardware_actuators.make_actuators()` implements pose-dependent ankle/waist rod-closure Jacobians and shared motor limits. Factory 8:1 torque is already included. Both hips use 15:1; knees use 16:1 total; shoulder pitch has an additional physical 4:1 with capped torque.

Run `test_joints.py --headless --device cpu --hardware_actuators --validation_dir validation/actuators_20260930`. Reports are separate from the original native-axis fixture below. Both fixtures use proxy mass/inertia, fixed base, no gravity and no contacts. Passive rod meshes are not animated as a closed-loop physics mechanism. Continuous duty, battery current, loaded BLDC curves and thermal behavior are not modeled. No training started.
'''
if passed:
    current += '\nThe new actuator fixture passed all 23 body axes, including FK/reset checks, with no invalid differential poses and motor-shaft torques inside their caps. See `validation/actuators_20260930/joint_validation.json` and `isaac_final_stdout.log`. Fixture armature/gains are test choices.\n'
for path in (DOCS / 'SIMULATION.md', HERE / 'README.md'):
    text = path.read_text(encoding='utf-8')
    for marker in ('## September 30 actuator configuration', '## September 30 user actuator configuration'):
        text = text.split(marker)[0].rstrip()
    # Replace an earlier generated current section on a repeated invocation.
    if '## Current actuator model — September 30' in text:
        before, remainder = text.split('## Current actuator model — September 30', 1)
        text = before.rstrip() + '\n\n' + remainder.split('## Original native-axis movement fixture\n', 1)[1].lstrip()
    first_break = text.find('\n\n', text.find('Package:')) if 'Package:' in text else text.find('\n\n')
    text = text[:first_break] + '\n\n' + current + '\n## Original native-axis movement fixture\n' + text[first_break:]
    text = text.replace('Configure those from the real hardware after this handoff.', 'This file is preserved as pre-configuration history; current user assumptions are in `assets/actuator_limits.json`.')
    text = text.replace('A hardware dynamics/actuator configuration must be added as the next step after the values are chosen.', 'The new explicit actuator uses the active configuration; future tasks still need measured dynamics and thermal/electrical limits.')
    text = text.replace('hardware settings awaiting user input.', 'historical null settings before the September 30 inputs.')
    if path == HERE / 'README.md':
        text = text.replace('[Actuator calculations](ACTUATOR_CALCULATIONS.md)', '[Actuator calculations](analysis/20260930_actuators/ACTUATOR_CALCULATIONS.md)')
    path.write_text(text + '\n', encoding='utf-8')

file_locations = REPO / 'Learning/HumanoidTraining/FILE_LOCATIONS.md'
text = file_locations.read_text(encoding='utf-8')
text = text.replace('pending hardware values `assets/actuator_limits_pending.json`', 'current user assumptions `assets/actuator_limits.json` and `hardware_actuators.py`; former null values `assets/actuator_limits_pending.json` are historical')
text = text.replace('Hardware effort/velocity, actual mass/inertia and contact dynamics remain pending.', 'September 30 effort/velocity assumptions and CAD-dependent differential mapping are configured. Actual mass/inertia, loaded curves, thermal duty and contact dynamics remain unqualified. See `docs/robot/ACTUATOR_CALCULATIONS.md` and `physical_v29/validation/actuators_20260930`.')
file_locations.write_text(text, encoding='utf-8')

for name in ('ACTUATOR_CALCULATIONS.md', 'BOM.csv', 'BOM.md', 'BUILD_REPORT.md', 'OPEN_ITEMS.md', 'EXPORTS.md', 'TORQUE_NOTES.md', 'SIMULATION.md'):
    shutil.copy2(DOCS / name, MIRROR / name)
print(json.dumps({'local_docs': str(DOCS), 'BOM_rows': len(rows), 'new_actuator_check_all_pass': bool(passed)}, indent=2))
