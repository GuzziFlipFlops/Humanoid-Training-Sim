"""Execute pinned upstream reward function bodies with explicit v29 mappings.

No discovery bonus, synthetic-IK tracking or standing imitation is added.
Unitree Apache-2.0 source and local Isaac Lab source are kept in unitree_vendor.
AST loading avoids importing simulator plugins for CPU adversarial tests.
"""
from __future__ import annotations
import ast, math
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
import torch

REWARD_VERSION = 'unitree_g1_lifetime_v2'
VENDOR = Path(__file__).parent / 'unitree_vendor'

def gait_parameters(commands, name='g1'):
    """Published walk parameters; explicit cadence/duty adaptation above 1 m/s.

    Public G1's 0.55 duty has no flight and was published for <=1 m/s.
    Preserve it for walking. Running uses the same original function with
    parameter tensors: 0.43 duty and the independently evaluated cadence.
    """
    speed=torch.linalg.vector_norm(commands[:,:2],dim=1)
    if name=='h1':return torch.full_like(speed,.6),torch.full_like(speed,.55)
    running=speed>=1.5
    return torch.where(running,1/(.9+.5*speed),.8),torch.where(running,.43,.55)

@dataclass
class Entity:
    name: str
    joint_ids: object = field(default_factory=lambda:slice(None))
    body_ids: object = field(default_factory=lambda:slice(None))
    joint_names: object = None
    body_names: object = None

def quat_apply_inverse(q, v):
    xyz=q[...,1:]; t=2*torch.cross(xyz,v,dim=-1)
    return v-q[...,:1]*t+torch.cross(xyz,t,dim=-1)

def yaw_quat(q):
    w,x,y,z=q.unbind(-1)
    angle=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))/2
    zeros=torch.zeros_like(angle)
    return torch.stack([torch.cos(angle),zeros,zeros,torch.sin(angle)],-1)

def definitions():
    """Compile original bodies, in the same override order as upstream mdp."""
    scope=dict(torch=torch,SceneEntityCfg=Entity,quat_apply_inverse=quat_apply_inverse,yaw_quat=yaw_quat)
    required={'energy','foot_clearance_reward','feet_gait','is_alive','is_terminated','lin_vel_z_l2',
              'ang_vel_xy_l2','joint_vel_l2','joint_acc_l2','action_rate_l2',
              'joint_pos_limits','joint_deviation_l1','flat_orientation_l2',
              'base_height_l2','contact_forces','undesired_contacts',
              'track_lin_vel_xy_yaw_frame_exp','track_ang_vel_z_exp','feet_slide'}
    for file in ['isaac_rewards.py','velocity_rewards.py','rewards.py']:
        tree=ast.parse((VENDOR/file).read_text(encoding='utf-8'))
        funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in required]
        module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+funcs,type_ignores=[])
        ast.fix_missing_locations(module);exec(compile(module,str(VENDOR/file),'exec'),scope)
    assert required<=scope.keys(),required-scope.keys()
    return {k:scope[k] for k in required}

FUNCTIONS=definitions()

def profile(name='g1'):
    """Read weights/functions/parameters directly from pinned RewardsCfg AST."""
    cfg=ast.parse((VENDOR/f'{name}_cfg.py').read_text(encoding='utf-8'))
    cls=next(n for n in cfg.body if isinstance(n,ast.ClassDef) and n.name=='RewardsCfg')
    terms={}
    for node in cls.body:
        if not isinstance(node,ast.Assign):continue
        args={k.arg:k.value for k in node.value.keywords}
        fname=args['func'].attr
        expression=ast.fix_missing_locations(ast.Expression(args.get('params',ast.Dict(keys=[],values=[]))))
        params=eval(compile(expression,'<upstream parameters>','eval'),
                    {'SceneEntityCfg':Entity,'math':math})
        terms[node.targets[0].id]=dict(function=fname,weight=ast.literal_eval(args['weight']),params=params)
    return terms

class Scene(dict):
    def __init__(self,robot,contacts):
        super().__init__(robot=robot,contact_forces=contacts);self.sensors={'contact_forces':contacts}

class UnitreeRewards:
    def __init__(self,joint_names,body_names,sensor_names,root_height,foot_origin_height,name='g1',termination_weight=0.,positive_living=False):
        self.name=name;self.terms=profile(name)
        self.termination_weight=float(termination_weight);self.positive_living=bool(positive_living)
        if self.termination_weight:
            assert name=='g1' and self.termination_weight==-200.,'Explicit pinned H2 terminal-fault cost'
            self.terms['physical_termination']=dict(function='is_terminated',weight=self.termination_weight,params={})
        assert not self.positive_living or self.termination_weight,'Clip living reward before a separate fault cost'
        self.mapping={}
        groups={'arms':[n for n in joint_names if 'shoulder' in n or 'elbow' in n],
                'waists':[n for n in joint_names if 'waist' in n],
                'torso':[n for n in joint_names if 'waist' in n],
                'legs':[n for n in joint_names if 'hip_roll' in n or 'hip_yaw' in n],
                'hips':[n for n in joint_names if 'hip_roll' in n or 'hip_yaw' in n]}
        feet=[body_names.index(s+'_foot') for s in ['left','right']]
        sensorfeet=[sensor_names.index(s+'_foot') for s in ['left','right']]
        for key,term in self.terms.items():
            for cfg in term['params'].values():
                if not isinstance(cfg,Entity):continue
                if cfg.joint_names is not None:
                    group=key.removeprefix('joint_deviation_');mapped=groups[group]
                    assert mapped,(name,key)
                    cfg.joint_ids=[joint_names.index(n) for n in mapped]
                    self.mapping[key]=mapped
                if cfg.body_names is not None:
                    if key=='undesired_contacts':cfg.body_ids=[i for i in range(len(sensor_names)) if i not in sensorfeet]
                    else:cfg.body_ids=sensorfeet if cfg.name=='contact_forces' else feet
                    self.mapping[key+':'+cfg.name]=[sensor_names[i] if cfg.name=='contact_forces' else body_names[i] for i in cfg.body_ids]
        # Robot-specific dimensions, retaining the original formulas and weights.
        self.terms['base_height']['params']['target_height']=float(root_height)
        # Upstream ankle-body height means link-origin height, not sole clearance.
        self.terms['feet_clearance']['params']['target_height']=float(foot_origin_height)+(.1 if name=='g1' else .15)
        self.adaptations={'base_height_m':float(root_height),'foot_link_clearance_target_m':self.terms['feet_clearance']['params']['target_height'],
                          'foot_origin_standing_height_m':float(foot_origin_height),'joint_mapping':self.mapping}

    def evaluate(self,env):
        raw={}
        for key,term in self.terms.items():
            params=term['params']
            if key=='gait' and self.name=='g1':
                period,threshold=gait_parameters(env.command_manager.get_command('base_velocity'),self.name)
                params=dict(params,period=period,threshold=threshold)
            raw[key]=FUNCTIONS[term['function']](env,**params)
        # ManagerBased RewardManager accumulates into a floating point buffer.
        # Contact-count terms return integer tensors even for integer weights.
        weighted={key:value*float(self.terms[key]['weight']) for key,value in raw.items()}
        if self.positive_living:
            living=sum(value for key,value in weighted.items() if key!='physical_termination')
            # Published Unitree RL Gym aggregation: clip regular rewards first,
            # then add terminal cost. Never clip away the physical-fault cost.
            raw['living_clip_adjustment']=living.clamp_min(0.)-living
            weighted['living_clip_adjustment']=raw['living_clip_adjustment']
        total=sum(weighted.values())*env.step_dt
        return total,raw,weighted

    def manifest(self):
        return dict(profile=self.name,reward_version=REWARD_VERSION,adaptations=self.adaptations,
                    gait_parameters=dict(walk_period_s=.8,walk_duty=.55,run_min_speed_m_s=1.5,run_frequency_hz='0.9 + 0.5 * command_speed',run_duty=.43) if self.name=='g1' else dict(period_s=.6,duty=.55),
                    terms={k:dict(function=v['function'],weight=v['weight']) for k,v in self.terms.items()},
                    lifetime_adaptation=dict(termination_weight=self.termination_weight,positive_living=self.positive_living,
                        termination_source='Unitree RL MJLab H2 velocity task at 1425b15f73bd4095f0df53709d7c389c3eb9e790',
                        clipping_source='Unitree RL Gym LeggedRobot.compute_reward at 276801e46c5d433564f24658bac64f254b7d2d4b',timeout_penalized=False),
                    source='unitree_vendor/sources.json',scope='Original core bodies/weights; declared anatomy and optional Unitree lifetime adaptations')
