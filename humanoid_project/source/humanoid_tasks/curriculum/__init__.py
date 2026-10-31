"""Gym registrations for the staged humanoid curriculum."""

import gymnasium as gym

from . import agents


STAGES = {
    "Stand": ("HumanoidStandEnvCfg", "HumanoidStandPPORunnerCfg"),
    "TurnStanding": ("HumanoidTurnStandingEnvCfg", "HumanoidTurnStandingPPORunnerCfg"),
    "GetUpDiscovery": ("HumanoidGetUpDiscoveryEnvCfg", "HumanoidGetUpDiscoveryPPORunnerCfg"),
    "GetUpRefine": ("HumanoidGetUpRefineEnvCfg", "HumanoidGetUpRefinePPORunnerCfg"),
    "Walk": ("HumanoidWalkEnvCfg", "HumanoidWalkPPORunnerCfg"),
    "RunFast": ("HumanoidRunFastEnvCfg", "HumanoidRunFastPPORunnerCfg"),
    "Sprint4": ("HumanoidSprint4EnvCfg", "HumanoidSprint4PPORunnerCfg"),
    "NaturalStand": ("HumanoidNaturalStandEnvCfg", "HumanoidStandPPORunnerCfg"),
    "NaturalWalk": ("HumanoidNaturalWalkEnvCfg", "HumanoidWalkPPORunnerCfg"),
    "NaturalWalkFixed1": ("HumanoidNaturalWalkFixed1EnvCfg", "HumanoidWalkPPORunnerCfg"),
    "GaitWalk": ("HumanoidGaitWalkEnvCfg", "HumanoidWalkPPORunnerCfg"),
    "GaitSprint4": ("HumanoidGaitSprint4EnvCfg", "HumanoidSprint4PPORunnerCfg"),
    "NaturalRun": ("HumanoidNaturalRunEnvCfg", "HumanoidRunFastPPORunnerCfg"),
    "NaturalSprint4": ("HumanoidNaturalSprint4EnvCfg", "HumanoidSprint4PPORunnerCfg"),
}


for stage_name, (env_cfg, agent_cfg) in STAGES.items():
    task_id = f"Isaac-Humanoid5ft-{stage_name}-v0"
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{env_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:{agent_cfg}",
        },
    )
    gym.register(
        id=task_id.replace("-v0", "-Play-v0"),
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{env_cfg}_PLAY",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:{agent_cfg}",
        },
    )
