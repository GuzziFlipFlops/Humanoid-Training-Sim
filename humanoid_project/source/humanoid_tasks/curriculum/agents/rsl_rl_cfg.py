"""PPO settings and the reviewed overnight iteration budget."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class HumanoidBasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 600
    save_interval = 50
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.45,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=8.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


def _runner(name: str, iterations: int):
    @configclass
    class StageRunner(HumanoidBasePPORunnerCfg):
        def __post_init__(self):
            self.max_iterations = iterations
            self.experiment_name = name

    return StageRunner


HumanoidStandPPORunnerCfg = _runner("humanoid5ft_stand", 500)
HumanoidTurnStandingPPORunnerCfg = _runner("humanoid5ft_turn_standing", 700)
HumanoidGetUpDiscoveryPPORunnerCfg = _runner("humanoid5ft_getup_discovery", 1800)
HumanoidGetUpRefinePPORunnerCfg = _runner("humanoid5ft_getup_refine", 1000)
HumanoidWalkPPORunnerCfg = _runner("humanoid5ft_walk", 1600)
HumanoidRunFastPPORunnerCfg = _runner("humanoid5ft_run_fast", 2500)
HumanoidSprint4PPORunnerCfg = _runner("humanoid5ft_sprint4", 3200)
