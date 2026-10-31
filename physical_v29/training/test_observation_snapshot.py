"""Exercise actual observation/reset statements and installed PPO on CPU."""
import argparse,ast,copy,json
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
from unitree_reward_port import gait_parameters

HERE=Path(__file__).parent
OLD=HERE/'runs/locomotion_20261002_r23_standing_curriculum/source/physical_v29/training'

def read_method(path,name):
    return next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name==name)

def compiled(method):
    scope=dict(torch=torch,gait_parameters=gait_parameters,FRAMES=80)
    mod=ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))
    exec(compile(mod,'<actual environment method>','exec'),scope);return scope[method.name]

def reset_function():
    assignments=[]
    for path in [HERE/'environment.py',HERE/'environment_unitree.py']:
        method=read_method(path,'_reset_idx')
        for n in ast.walk(method):
            if isinstance(n,ast.Assign) and any('self.history[' in ast.unparse(t) or 'self.critic_history[' in ast.unparse(t) for t in n.targets):assignments.append(copy.deepcopy(n))
    assert len(assignments)==2
    method=ast.parse('def clear_histories(self,env_ids):\n ids=env_ids').body[0];method.body+=assignments
    return compiled(method)

def fake():
    n=8
    d=NS(root_ang_vel_b=torch.randn(n,3)*.1,projected_gravity_b=torch.tensor([0.,0.,-1.]).expand(n,3),
        joint_pos=torch.randn(n,23)*.1,joint_vel=torch.randn(n,23)*.2,root_lin_vel_b=torch.randn(n,3)*.1)
    return NS(robot=NS(data=d),num_envs=n,device='cpu',cfg=NS(evaluation=True),commands=torch.zeros(n,3),
        phase=torch.zeros(n),episode_length_buf=torch.ones(n),step_dt=.02,stand=torch.zeros(23),
        actions=torch.randn(n,23)*.1,history=torch.randn(n,5,80)*.1,critic_history=torch.randn(n,5,83)*.1,
        _last_history_tick=-1,common_step_counter=7,_update_observation_command=lambda:None)

def run(out):
    torch.set_num_threads(1);torch.manual_seed(94)
    rows=torch.tensor([0,2,6]);clear=reset_function();results=[]
    for label,path in [('frozen_r23',OLD/'environment_unitree.py'),('owned_snapshot',HERE/'environment_unitree.py')]:
        get=compiled(read_method(path,'_get_observations'));env=fake();views=get(env)
        snapshots={k:v.clone() for k,v in views.items()};obs=TensorDict(views,batch_size=[8])
        actor=MLPModel(obs,{'actor':['policy']},'actor',23,hidden_dims=[512,256,128],activation='elu',obs_normalization=False,
            distribution_cfg=dict(class_name='GaussianDistribution',init_std=.35,std_type='log'))
        ck=HERE/'runs/locomotion_20261002_r23_standing_curriculum/047_stand_train_008_part_04/candidate.pt'
        actor.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['actor_state_dict'])
        critic=MLPModel(obs,{'critic':['critic']},'critic',1,hidden_dims=[16],activation='elu',obs_normalization=False)
        storage=RolloutStorage('rl',8,1,obs,[23]);alg=PPO(actor,critic,storage)
        with torch.no_grad():
            alg.act(obs)
            clear(env,rows)
            assert (env.history[rows]==0).all() and (env.critic_history[rows]==0).all()
            env.common_step_counter+=1;next_obs=TensorDict(get(env),batch_size=[8])
            dones=torch.zeros(8);dones[rows]=1.
            alg.process_env_step(next_obs,torch.ones(8)*.2,dones,{})
            stored=storage.observations[0]
            old_mean,old_std=[p[0] for p in storage.distribution_params]
            now=actor(stored);std=actor.distribution.log_std_param.exp().expand_as(now)
            kl=torch.distributions.kl_divergence(torch.distributions.Normal(old_mean,old_std),torch.distributions.Normal(now,std)).sum(-1)
        record=dict(case=label,policy_snapshot_corruption=float((views['policy']-snapshots['policy']).abs().max()),
            critic_snapshot_corruption=float((views['critic']-snapshots['critic']).abs().max()),
            preupdate_kl_reset_rows=float(kl[rows].mean()),preupdate_kl_other_rows=float(kl[[1,3,4,5,7]].mean()))
        if label=='frozen_r23':
            assert record['policy_snapshot_corruption']>0. and record['critic_snapshot_corruption']>0. and record['preupdate_kl_reset_rows']>1e-4
        else:
            assert record['policy_snapshot_corruption']==record['critic_snapshot_corruption']==0.
            assert record['preupdate_kl_reset_rows']<1e-8 and record['preupdate_kl_other_rows']<1e-8
        results.append(record)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    report=dict(passed=True,cases=results,scope='Actual AST observation/reset statements, qualified actor, installed CPU PPO. A controlled ownership regression, not a learned gait.')
    (out/'observation_snapshot_tests.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
