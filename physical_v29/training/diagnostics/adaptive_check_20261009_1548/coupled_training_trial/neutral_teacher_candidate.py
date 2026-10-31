"""Undeployed, explicit training-only coupled neutral-teacher experiment."""
import copy,hashlib,json,runpy,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent;TRAIN=OUT/'control_source/physical_v29/training'
sys.path[:0]=[str(OUT),str(TRAIN),str(TRAIN.parent)]
from command_local_teacher_target import eligible,target_derivative,VERSION,COMMANDS
applied='--coupled-neutral-teacher' in sys.argv
if applied:sys.argv.remove('--coupled-neutral-teacher')
folder=Path(sys.argv[sys.argv.index('--output')+1]);folder.mkdir(parents=True,exist_ok=True)
mode=sys.argv[sys.argv.index('--mode')+1];assert not applied or mode=='train'
cert=json.loads((OUT/'neutral_teacher_certificate.json').read_text())
receipt=dict(version=VERSION,applied=applied,training_only=True,diagnostic_actor_deployed=False,
    original_signature_does_not_certify_executed_training_target_override=applied,
    commands=list(COMMANDS),neutral_teacher_checkpoint=cert['checkpoint'],neutral_teacher_sha256=cert['checkpoint_sha256'],
    unchanged_weight=.25,unchanged_denominator_outputs=23,unchanged_sample_bound=256,
    current_head_direct_gradient_zero=True,other_command_teacher_targets_and_rows_unchanged=True,
    physics_caps_rewards_commands_PPO_critic_exploration_upper_posture_unchanged=True,
    truthful_student_observations=True,no_inference_teacher=True,
    helper_sha256=hashlib.sha256((OUT/'command_local_teacher_target.py').read_bytes()).hexdigest())
(folder/'experimental_teacher_target.json').write_text(json.dumps(receipt,indent=2)+'\n')
if applied:
    import torch
    from upper_body_refinement import LocomotionOnlyRetention
    from teacher_head_context import context_inputs
    from standing_retention import validate_qualification,validate_teacher
    from curriculum_provenance import training_reward_signature
    original_init=LocomotionOnlyRetention.__init__;original_regularize=LocomotionOnlyRetention._regularize;original_audit=LocomotionOnlyRetention.read_audit
    def initialize(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        self.neutral_counts={str(command):0 for command in COMMANDS}
        if self.command_range is None:return
        from evaluation import GATE_VERSION
        assert validate_qualification(cert,cert['checkpoint'],training_reward_signature(TRAIN),GATE_VERSION)==3.
        saved=torch.load(cert['checkpoint'],weights_only=False,map_location='cpu')
        assert saved['infos']['reward_signature']==training_reward_signature(TRAIN)
        validate_teacher(saved['infos'],'physical_v29_history5_unitree_defaultoffset_v2',
            {key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in [('asset_sha256',TRAIN/'assets/robot_training.urdf'),
             ('motion_sha256',TRAIN/'assets/motion_reference.npz'),('actuator_sha256',TRAIN.parent/'assets/actuator_limits.json')]})
        self.neutral_teacher=copy.deepcopy(self.teacher)
        self.neutral_teacher.load_state_dict(saved['actor_state_dict']);self.neutral_teacher.eval().requires_grad_(False)
        assert all(torch.equal(value.to(self.neutral_teacher.state_dict()[key].device),self.neutral_teacher.state_dict()[key]) for key,value in saved['actor_state_dict'].items())
    def regularize(self,module,inputs,mean):
        if self.command_range is None:return original_regularize(self,module,inputs,mean)
        observations=self._observations;self._observations=None
        if observations is None or not mean.requires_grad:return
        raw=observations['policy'];index=self.command_index if self.command_index is not None else raw.shape[-1]-80+6
        speed=raw[:,index:index+2].norm(dim=1)
        selected=(speed>self.command_range[0])&(speed<=self.command_range[1]+1e-6)
        ids=selected.nonzero(as_tuple=False).flatten()
        if not ids.numel():return
        stride=max(1,(ids.numel()+self.max_samples-1)//self.max_samples);ids=ids[::stride][:self.max_samples]
        with torch.no_grad():
            chosen=eligible(raw[ids],self.phase_context['columns'][:2])
            teacher_input=observations[ids].clone();teacher_raw=teacher_input['policy'];frames=teacher_raw.reshape(-1,5,80)
            for command,table in self.phase_tables.items():
                conditioned=((frames[:,:,6]-command).abs()<1e-6).all(1)
                conditioned&=(frames[:,:,78:80].square().sum(-1)>.5).all(1)
                conditioned&=(frames[:,-1,self.phase_context['columns'][:2]].abs()<.174533).all(1)
                conditioned&=~chosen
                rows=conditioned.nonzero().flatten()
                if len(rows):
                    teacher_raw[rows]=context_inputs(teacher_raw[rows],table,self.phase_context['columns']);self.conditioned_samples+=len(rows)
            original_target=self.teacher(teacher_input).clamp(-3.,3.)
            neutral_target=original_target.clone()
            if chosen.any():neutral_target[chosen]=self.neutral_teacher(observations[ids][chosen]).clamp(-3.,3.)
            target,difference,derivative,included=target_derivative(mean[ids],original_target,neutral_target,chosen,self.upper,self.upper[:2],self.weight)
            for command in COMMANDS:self.neutral_counts[str(command)]+=int((chosen&(raw[ids,index]-command).abs().lt(1e-6)).sum())
            error=difference.square().sum()
        self._error=error if self._error is None else self._error+error;self._samples+=difference.numel();self._batches+=1
        mean.register_hook(lambda gradient:gradient.index_add(0,ids,derivative))
    def audit(self):
        result=original_audit(self)
        if self.command_range is not None:
            result.update(experimental_training_target_version=VERSION,neutral_teacher_sha256=cert['checkpoint_sha256'],
                command_local_coupled_samples=dict(self.neutral_counts),head_excluded=self.upper[:2],
                output_scope='Same .25/23 MSE; at eligible settled0.15/1.5 truthful states,neutral267 body+six arm derivatives and zero head derivative; other commands original conditioned349 body-only targets.',
                original_signature_does_not_certify_override=True)
        self.neutral_counts={str(command):0 for command in COMMANDS}
        return result
    LocomotionOnlyRetention.__init__=initialize;LocomotionOnlyRetention._regularize=regularize;LocomotionOnlyRetention.read_audit=audit
runpy.run_path(str(TRAIN/'worker_unitree.py'),run_name='__main__')
