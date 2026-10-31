"""Reproduce the user's September 30 motor assumptions and CAD transmission."""
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import torch
from cad_differential import CadDifferential

HERE=Path(__file__).resolve().parent
ASSETS=HERE/'assets'
OUT=HERE/'analysis/20260930_actuators'
OUT.mkdir(parents=True,exist_ok=True)
EFFICIENCY=.95
PROFILE='eff95_elbow7_20260930'
native_path=Path(r'E:\Aditya\robot_sim\outputs\shin_cad_20260912\v29\native_joint_map_saved.json')
rows=json.loads(native_path.read_text()); native={j['name']:j for j in rows}
meta=json.loads((ASSETS/'export_metadata.json').read_text())
rad_s=lambda rpm:rpm*2*math.pi/60
specs={}
for key,prefix,waist in [('left_ankle','Left_',False),('right_ankle','Right_',False),('waist','WJ',True)]:
    pitch=native['WJ01_lower_waist_pitch' if waist else prefix+'Ankle_pitch']
    roll=native['WJ02_upper_waist_roll' if waist else prefix+'Ankle_roll']
    spec={'pitch_origin_mm':pitch['origin_mm'],'pitch_axis':pitch['axis'],'roll_origin_mm':roll['origin_mm'],'roll_axis':roll['axis'],'joint_names':['waist_pitch','waist_roll'] if waist else [key+'_pitch',key+'_roll'],'motor_on_moving_link':waist,'motors':[],'source':str(native_path),'source_sha256':hashlib.sha256(native_path.read_bytes()).hexdigest(),'scope':'Exact saved shaft and rod-eye closure geometry. Motor angular limits checked. Rod-end contact angles, elastic deformation, friction and passive mesh animation not simulated.'}
    for name in ('L','R') if waist else ('A','B'):
        drive=native['WJ_drive_'+name if waist else prefix+'Motor_'+name+'_output']
        eye=native['WJ_rod_'+name+'_pin' if waist else prefix+'Rod_'+name+'_pin_swivel']
        bottom=native['WJ_rod_'+name+'_base' if waist else prefix+'Rod_'+name+'_lower_ball']
        v=np.array(eye['origin_mm'])-drive['origin_mm']
        spec['motors'].append({'name':drive['name'],'center_mm':drive['origin_mm'],'axis':drive['axis'],'crank_vector_mm':v.tolist(),'bottom_eye_mm':bottom['origin_mm'],'rod_length_mm':float(np.linalg.norm(np.array(eye['origin_mm'])-bottom['origin_mm'])),'motor_limits_rad':drive['limits_rad']})
    specs[key]=spec
(ASSETS/'differential_geometry.json').write_text(json.dumps(specs,indent=2),encoding='utf-8')

analysis={'profile':PROFILE,'validation_dir':'validation/'+PROFILE,'assumptions':{'geared_7010':[18,45,400],'geared_6010':[15,30,400],'7010_6010_torque_speed_domain':'Factory 8:1 gearbox output; internal losses already included. User-selected planning limits; loaded envelopes pending.','DJI10018_shaft':[8,18,77],'voltage_nominal_V':43.2,'voltage_full_V':50.4,'transmission_efficiency':EFFICIENCY,'efficiency_status':'User planning assumption: 95% overall for added transmissions/rod linkages. Knee 8x2 is one overall 95% stage, not 95% twice. Factory-geared motor and servo output ratings are not derated a second time.','shoulder_pitch_physical_external_reduction':4,'shoulder_pitch_equivalent_torque_multiplier':2.5,'shoulder_pitch_motor_output_speed_cap_RPM':500,'shoulder_pitch_joint_speed_cap_RPM':120,'elbow_user_sustained_and_operating_cap_Nm':7,'elbow_user_loaded_RPM':45,'waist_yaw_boost_Nm':50,'boost_status':'Optional user-proposed current increase, not a validated continuous/thermal rating.'},'differentials':{},'joint_types':{},'sources':json.loads((ASSETS/'motor_constants_20260930.json').read_text())}
limits={}
for key,spec in specs.items():
    model=CadDifferential(spec)
    neutral=torch.zeros((1,2),dtype=torch.float64)
    theta,J,valid,error=model.state(neutral)
    assert valid.all() and error.max()<1e-6
    torque=30 if key=='waist' else 45
    continuous=15 if key=='waist' else 18
    invt=torch.linalg.inv(J[0].T)
    pure_peak=(EFFICIENCY*torque/invt.abs().amax(0)).tolist()
    pure_cont=(EFFICIENCY*continuous/invt.abs().amax(0)).tolist()
    pure_speed=(400/J[0].abs().amax(0)).tolist()
    jmeta=[next(j for j in meta['joints'] if j['name']==n) for n in spec['joint_names']]
    grids=[np.linspace(*j['limits_rad'],31) for j in jmeta]
    q=torch.tensor(np.array(np.meshgrid(*grids,indexing='ij')).reshape(2,-1).T,dtype=torch.float64)
    angles,jacs,ok,err=model.state(q)
    good=jacs[ok]
    assert len(good)>0
    pure=EFFICIENCY*torque/torch.linalg.inv(good.transpose(1,2)).abs().amax(1)
    bbox_effort=EFFICIENCY*torque*good.abs().sum(1).amax(0)
    bbox_speed=400*torch.linalg.inv(good).abs().sum(2).amax(0)
    item={'neutral_J_motor_rad_per_joint_rad':J[0].tolist(),'neutral_motor_angle_rad':theta[0].tolist(),'neutral_pure_axis_continuous_Nm':pure_cont,'neutral_pure_axis_peak_Nm':pure_peak,'neutral_pure_axis_speed_cap_RPM':pure_speed,'sampled_grid_count':len(q),'closure_valid_grid_count':int(ok.sum()),'sampled_pure_peak_range_Nm':torch.stack((pure.amin(0),pure.amax(0)),1).tolist(),'grid_max_closure_error_mm':float(err[ok].max()),'geometry':spec}
    analysis['differentials'][key]=item
    for i,name in enumerate(spec['joint_names']):
        limits[name]={'continuous_Nm':pure_cont[i],'effort_Nm':pure_peak[i],'velocity_rad_s':rad_s(pure_speed[i]),'URDF_bounding_effort_Nm':float(bbox_effort[i]),'URDF_bounding_velocity_rad_s':rad_s(float(bbox_speed[i])),'differential':key,'transmission_efficiency':EFFICIENCY,'motor_continuous_Nm':continuous,'motor_peak_Nm':torque,'motor_speed_cap_RPM':400,'independent_limits_valid':False,'status':'POSE_DEPENDENT_COUPLED_MOTOR_LIMITS'}

def direct(name,cont,peak,rpm,**extra):
    item={'continuous_Nm':cont,'effort_Nm':peak,'velocity_rad_s':rad_s(rpm),'max_joint_RPM':rpm,**extra}
    limits[name]=item
for name in [j['name'] for j in meta['joints']]:
    if name in limits:continue
    if name=='waist_yaw' or name.endswith('hip_yaw'):
        direct(name,18,45,400,motor='7010-8',optional_boost_peak_Nm=50 if name=='waist_yaw' else None)
    elif name.endswith('hip_pitch') or name.endswith('hip_roll'):
        direct(name,8*15*EFFICIENCY,18*15*EFFICIENCY,77*43.2/15,motor='DJI10018 77KV',reduction=15,transmission_efficiency=EFFICIENCY,full_charge_no_load_RPM=77*50.4/15)
    elif name.endswith('knee_pitch'):
        direct(name,8*8*2*EFFICIENCY,18*8*2*EFFICIENCY,77*43.2/16,motor='DJI10018 77KV',reduction=16,planetary=8,chain=2,transmission_efficiency=EFFICIENCY,full_charge_no_load_RPM=77*50.4/16)
    elif name.endswith('shoulder_pitch'):
        direct(name,15*2.5*EFFICIENCY,30*2.5*EFFICIENCY,120,motor='6010-8',physical_external_reduction=4,transmission_efficiency=EFFICIENCY,motor_continuous_cap_Nm=15*2.5/4,motor_peak_cap_Nm=30*2.5/4,motor_speed_cap_RPM=500,motor_RPM_at_joint_cap=480)
    elif name.endswith('shoulder_roll'):
        direct(name,15,30,400,motor='6010-8')
    elif name.endswith('elbow_pitch'):
        direct(name,7,7,45,motor='RDS51160',supply_V=24,status='USER_ASSUMED_7NM_AT_45RPM_OPERATING_CAP',speed_model='user_operating_torque_speed_box',advertised_stall_Nm=170*.0980665,advertised_no_load_RPM=10/.21,qualification='User assumes 7 N.m sustained at 45 RPM. Operating command cap is 7; advertised stall remains metadata, not available operating torque. Not a measured curve.')
    elif name.startswith('head_'):
        direct(name,10*.0980665,30*.0980665,10/.222,motor='STS3215-C018',supply_V=12,continuous_status='Manufacturer rated torque; thermal duty conditions need confirmation',speed_model='linear_no_load_to_stall_planning_approximation')
    else:raise ValueError(name)
assert len(limits)==23
for name,item in limits.items():
    group=name.replace('left_','').replace('right_','')
    analysis['joint_types'].setdefault(group,item)
const=analysis['sources']['variants']
analysis['currents_A_supplier_convention']={
 'WGG7010-8':{'Kt_output_Nm_A':const['WGG7010-8']['output_Kt_Nm_A'],'continuous':18/.65,'peak':45/.65,'optional_50Nm':50/.65,'Kv_output_rpm_V':9.25,'Kv_nominal_output_RPM':9.25*43.2,'Kv_full_output_RPM':9.25*50.4},
 'WGG6010-8':{'Kt_output_Nm_A':.39,'continuous':15/.39,'peak':30/.39,'shoulder_pitch_cont_cap':9.375/.39,'shoulder_pitch_peak_cap':18.75/.39,'Kv_output_rpm_V':21.6666666666667,'Kv_nominal_output_RPM':21.6666666666667*43.2,'Kv_full_output_RPM':21.6666666666667*50.4},
 'GIM6010-8_24V':{'Kt_output_Nm_A':.47,'continuous':15/.47,'peak':30/.47},
 'GIM6010-8_48V':{'Kt_output_Nm_A':.65,'continuous':15/.65,'peak':30/.65},
 'DJI10018_77KV_ideal_SI':{'Kt_shaft_Nm_A':60/(2*math.pi*77),'continuous':8/(60/(2*math.pi*77)),'peak':18/(60/(2*math.pi*77)),'status':'Ideal Kv-to-Kt conversion; controller current convention/winding resistance unknown'}
}
analysis['battery_candidates']={
 'Ampace_JP50':{'cell_Ah':5,'cell_full_discharge_continuous_A':40,'cell_high_power_continuous_headline_A':50,'cell_maximum_continuous_A':60,'cell_pulse_A':180,'pulse_seconds':5,'pack_nominal_V':43.2,'pack_full_V':50.4,'pack_Ah':15,'pack_Wh':648,'three_parallel_cell_arithmetic_full_discharge_cont_A':120,'three_parallel_cell_arithmetic_high_power_headline_A':150,'three_parallel_cell_arithmetic_pulse_A':540,'source':'https://en.ampace.com/products/power-tools-product-matrix/jp/69','qualification':'Cell arithmetic only. Detailed source distinguishes full-discharge continuous, maximum continuous and headline high-power duty. Pack BMS/interconnects/fuses/temperature and pulse repetition not established.'},
 'Molicel_P60C':{'cell_Ah':6,'manufacturer_listed_discharge_A':100,'pack_Ah':18,'pack_Wh':777.6,'three_parallel_listed_discharge_A':300,'source':'https://www.molicel.com/product/inr-21700-p60c/','qualification':'Manufacturer public page does not establish unrestricted continuous duty or pulse duration; application datasheet needed.'}
}
analysis['sim_limit_model']='Pose-dependent 95%-efficient transmission and shared per-motor current/speed limits. Neck linear torque-speed approximation; elbow user-assumed 7 Nm/45 RPM operating box. No verified voltage sag, thermal duration, regeneration or whole-pack current model.'
(ASSETS/'actuator_limits.json').write_text(json.dumps(limits,indent=2),encoding='utf-8')
(OUT/'calculation.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
with (OUT/'joint_capabilities.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f);w.writerow(['joint','continuous_Nm','peak_Nm','maximum_joint_RPM','note'])
    for name,item in analysis['joint_types'].items():w.writerow([name,item['continuous_Nm'],item['effort_Nm'],item['velocity_rad_s']*60/(2*math.pi),item.get('status','User assumptions; ideal transmission; torque and max speed not simultaneous')])

urdf=ASSETS/'humanoid_physical_v29.urdf'
backup=OUT/'humanoid_physical_v29_before_actuators.urdf'
if not backup.exists():shutil.copy2(urdf,backup)
tree=ET.parse(urdf)
for joint in tree.getroot().findall('joint'):
    if joint.get('name') not in limits:continue
    item=limits[joint.get('name')];lim=joint.find('limit')
    lim.set('effort',str(item.get('URDF_bounding_effort_Nm',item['effort_Nm'])))
    lim.set('velocity',str(item.get('URDF_bounding_velocity_rad_s',item['velocity_rad_s'])))
tree.write(urdf,encoding='utf-8',xml_declaration=True)
meta['hardware_limits_status']='USER_ASSUMPTIONS_'+PROFILE+'; voltage/thermal/pack qualification pending'
meta['actuator_profile']=PROFILE
meta['hardware_validation_dir']='validation/'+PROFILE
meta['actuator_limits_file']='assets/actuator_limits.json'
meta['differential_geometry_file']='assets/differential_geometry.json'
meta['limit_runtime']='Use hardware_actuators.make_actuators(); independent URDF differential limits are only bounding values.'
(ASSETS/'export_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
print(json.dumps({k:{n:v for n,v in item.items() if n.startswith('neutral_') or n.endswith('_count')} for k,item in analysis['differentials'].items()},indent=2))
print(json.dumps(analysis['joint_types'],indent=2))
