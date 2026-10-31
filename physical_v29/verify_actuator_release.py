"""Verify preserved CAD geometry, input sheets and completed actuator evidence."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
meta=json.loads((HERE/'assets/export_metadata.json').read_text())
OUT = HERE / meta['hardware_validation_dir']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def geometry(path):
    root = ET.parse(path).getroot()
    for joint in root.findall('joint'):
        limit = joint.find('limit')
        if limit is not None:
            limit.attrib.pop('effort', None)
            limit.attrib.pop('velocity', None)
    return ET.tostring(root)

source_urdf = HERE / 'analysis/20260930_actuators/humanoid_physical_v29_before_actuators.urdf'
current_urdf = HERE / 'assets/humanoid_physical_v29.urdf'
assert geometry(source_urdf) == geometry(current_urdf), 'Geometry, joint frames or motion ranges changed'
constants = json.loads((HERE / 'assets/motor_constants_20260930.json').read_text())
for source in constants['sources']:
    assert sha(source['path']) == source['sha256'], source['path']
movement = json.loads((OUT / 'joint_validation.json').read_text())
assert movement['all_pass'] and len(movement['joints']) == 23
assert movement['invalid_transmission_steps'] == 0
transmission = json.loads((OUT / 'transmission_validation.json').read_text())
assert transmission['all_pass'] and len(transmission['differentials']) == 3
limits = json.loads((HERE / 'assets/actuator_limits.json').read_text())
assert len(limits) == 23
for joint in ET.parse(current_urdf).getroot().findall('joint'):
    if joint.get('name') in limits:
        row = limits[joint.get('name')]
        assert float(joint.find('limit').get('effort')) == row.get('URDF_bounding_effort_Nm', row['effort_Nm'])
        assert float(joint.find('limit').get('velocity')) == row.get('URDF_bounding_velocity_rad_s', row['velocity_rad_s'])

names = ['cad_differential.py', 'hardware_actuators.py', 'test_joints.py',
         'visual_joint_test.py',
         'assets/humanoid_physical_v29.urdf', 'assets/usd/humanoid_physical_v29.usd',
         'assets/actuator_limits.json', 'assets/differential_geometry.json',
         'assets/motor_constants_20260930.json', 'analysis/20260930_actuators/calculation.json']
result = {
    'all_pass': True,
    'scope': 'Actuator release coherence; unchanged URDF geometry/axes/ranges; original XLSX inputs unchanged; completed transmission and Isaac movement evidence. Not a mass/contact/thermal/pack qualification.',
    'joint_count': len(limits),
    'URDF_geometry_frames_and_ranges_unchanged': True,
    'input_XLSX_hashes_unchanged': True,
    'movement_all_pass': True,
    'transmission_all_pass': True,
    'source_snapshot_sha256': {name: sha(HERE / name) for name in names},
    'fixture_PD': {'waist_hip_knee_stiffness': 120, 'waist_hip_knee_damping': 15,
                   'other_stiffness': 35, 'other_damping': 4,
                   'test_only_armature_kg_m2': .05},
    'maximum_tracking_error_rad': max(sample['tracking_error_rad'] for j in movement['joints'] for sample in j['samples']),
    'maximum_fk_position_error_m': max(sample['fk_max_position_error_m'] for j in movement['joints'] for sample in j['samples']),
    'motor_peak_Nm': movement['differential_applied_motor_peak_Nm'],
}
(OUT / 'release_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({key: value for key, value in result.items() if key != 'source_snapshot_sha256'}, indent=2))
