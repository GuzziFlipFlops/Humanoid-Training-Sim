"""V29 curriculum rewards informed by the verified five-foot walker and G1.

    Keep pinned Unitree function bodies; declare every local scale/replacement.
    No moving IK action reference, success reset or camera. Training-only
    standing retention is separate from this reward and absent in inference.
"""
import torch
from unitree_reward_port import UnitreeRewards, gait_parameters
from support_geometry import horizontal_support

REWARD_VERSION = 'cad_heading_alignment_retention_v6'
COST_BUDGET = 2.0
SURVIVAL_RATE = 2.0
TERMINATION_RATE = -200.0


def shaped_terms(metrics, commands, root_quat, root_pos, core_raw, core_costs, terminated):
    """Reward rates. Smooth bounded costs preserve nonnegative lifetime returns.

    A +2 survival rate offsets costs bounded below -2. There is no aggregate
    positive floor. Terminal ticks receive -200 only, independent of posture.
    This removes the incentive to exit merely to avoid negative future costs.
    """
    stopped = commands[:,:2].norm(dim=1)<=.1
    moving = ~stopped
    running = commands[:,:2].norm(dim=1)>=1.5
    alive = (~terminated).float()
    upright = torch.exp(-torch.sin(metrics['tilt']).square()/.25**2)*(metrics['tilt']<torch.pi/2)
    torso = torch.exp(-torch.sin(metrics['torso_tilt']).square()/.25**2)*(metrics['torso_tilt']<torch.pi/2)
    height = torch.exp(-((metrics['height']-metrics['reference_height'])/.06).square())
    healthy = upright*torso*height
    quiet = torch.exp(-metrics['velocity'][:,:2].square().sum(1)/.25**2
                     -metrics['angular'].square().sum(1)/.5**2)
    both = metrics['contact'].all(1).float()
    _, support_excess = horizontal_support(root_quat,root_pos,metrics['foot_position'],metrics['com'],metrics['contact'])
    support = both*torch.exp(-(support_excess/.04).square())
    # G1/H1 phase feedback must exist before the first translational step.
    # The former speed-norm multiplier erased it at rest and gave backward or
    # lateral movement the same activation as forward travel.
    activity = (metrics['velocity'][:,0]/commands[:,:2].norm(dim=1).clamp_min(.1)).clamp(0.,1.)
    _, duty = gait_parameters(commands)
    leg_phase = torch.remainder(metrics['phase'][:,None]+torch.tensor([0.,.5],device=commands.device),1.)
    expected_swing = leg_phase>=duty[:,None]
    # Use actual sole clearance and unloading. Stationary loaded feet get zero.
    clearance_target = torch.where(running,.065,.040)
    swing_quality = torch.exp(-((metrics['clearance']-clearance_target[:,None])/.025).square())
    swing_quality *= ((~metrics['contact']) & expected_swing).float()
    terms = dict(survival=SURVIVAL_RATE*alive,
        upright=3.5*upright,torso_upright=.75*torso,height=2.5*height,
        stationary=quiet*healthy*stopped,
        stand_two_feet=1.5*both*healthy*stopped,
        stand_anchor=torch.exp(-(metrics['drift']/.15).square())*healthy*stopped,
        stand_com_support=.5*support*healthy*stopped,
        track_lin_vel_xy=4.0*core_raw['track_lin_vel_xy']*healthy*moving,
        track_ang_vel_z=.5*core_raw['track_ang_vel_z']*healthy,
        gait=.5*core_raw['gait']*healthy*moving,
        feet_clearance=.35*swing_quality.amax(1)*healthy*moving,
        running_flight=.25*(~metrics['contact'].any(1))*healthy*activity*running)
    # A body-frame velocity can match while walking in a circle. The small,
    # oscillatory instantaneous yaw-rate kernel did not teach correction in
    # measured R27 command interventions. Reward the episode's fixed heading
    # directly; this positive rate is bounded and absent at stopped commands.
    terms['fixed_heading'] = torch.exp(-(metrics['heading_error_rad']/.35).square())*healthy*moving
    core_costs = dict(core_costs)
    core_costs['walking_flight_cost'] = -.5*(~metrics['contact'].any(1)).float()*moving*(~running)
    raw_cost = sum(-value for value in core_costs.values())
    if raw_cost.device.type=='cpu':
        assert torch.all(raw_cost>=0.), 'Cost terms must be nonpositive'
    bounded_cost = COST_BUDGET*torch.tanh(raw_cost/COST_BUDGET)
    factor = torch.where(raw_cost>0.,bounded_cost/raw_cost.clamp_min(1e-8),torch.ones_like(raw_cost))
    terms.update({key:value*factor for key,value in core_costs.items()})
    terms = {key:value*alive for key,value in terms.items()}
    terms['physical_termination'] = TERMINATION_RATE*terminated.float()
    if raw_cost.device.type=='cpu':
        assert all(torch.isfinite(value).all() for value in terms.values()), 'Nonfinite reward telemetry'
    return terms, dict(raw_cost=raw_cost,bounded_cost=bounded_cost,healthy=healthy,
        support_excess_m=support_excess,activity=activity,
        expected_swing_fraction=expected_swing.float().mean(1))


class CurriculumRewards(UnitreeRewards):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,termination_weight=TERMINATION_RATE,positive_living=False,**kwargs)
        # Same exact action-rate function, coefficient used by approved walker.
        self.terms['action_rate']['weight'] = -.01

    def evaluate(self,env):
        _, raw, upstream_weighted = super().evaluate(env)
        # Height/orientation are now informative positive kernels. Costs retain
        # the other pinned G1 functions, including slide, limits and contacts.
        costs = {key:value for key,value in upstream_weighted.items()
                 if self.terms[key]['weight']<0 and key not in ['physical_termination','base_height','flat_orientation_l2']}
        data = env.scene['robot'].data
        terms, diagnostics = shaped_terms(env.cad_metrics,env.command_manager.get_command('base_velocity'),
            data.root_quat_w,data.root_pos_w,raw,costs,env.termination_manager.terminated)
        self.last_diagnostics = diagnostics
        return sum(terms.values())*env.step_dt,raw,terms

    def manifest(self):
        result = super().manifest()
        result.update(reward_version=REWARD_VERSION,
            heading_command=dict(version='fixed_reset_heading_yaw_rate_v1',gain=.5,
                maximum_rad_s=.2,reference='Reset heading held for the complete episode',
                stopped_command_zero=True,action_time_command_snapshot=True,
                source='Isaac Lab UniformVelocityCommand heading mode; local task adaptation'),
            scope='Declared V29 curriculum adaptation; pinned G1 cost/phase bodies and verified old standing shaping',
            standing=dict(upright_weight=3.5,height_weight=2.5,torso_weight=.75,
                bilateral_weight=1.5,quiet_weight=1.,anchor_weight=1.,support_weight=.5,
                gravity_projected_support=True,stopped_threshold_m_s=.1),
            locomotion=dict(velocity_weight=4.,healthy_gating=True,gait_weight=.5,
                fixed_heading_weight=1.,fixed_heading_width_rad=.35,
                fixed_heading_reference='Episode reset heading; wrapped error, no moving target',
                fixed_heading_source='Local adaptation informed by Isaac Lab direct humanoid heading alignment',
                swing_clearance_weight=.35,walk_sole_target_m=.04,run_sole_target_m=.065,
                loaded_or_stationary_clearance_reward=0.,running_flight_weight=.25,
                gait_and_clearance_speed_multiplier=False,clearance_only_expected_swing=True,
                running_flight_forward_activity=True,walking_flight_cost_weight=-.5),
            lifetime_adaptation=dict(termination_weight=TERMINATION_RATE,positive_living=False,
                survival_rate=SURVIVAL_RATE,smooth_cost_budget=COST_BUDGET,
                regular_reward_nonnegative=True,terminal_regular_reward=0.,timeout_penalized=False,
                rationale='Smooth bounded costs plus survival offset; no aggregate reward flooring'),
            action_rate_weight=-.01)
        return result
