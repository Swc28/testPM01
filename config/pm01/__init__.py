import gymnasium as gym

from . import agents

# 速度行走任务（训练 / 演示）
gym.register(
    id="Isaac-Velocity-MyTask-PM01-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:PM01MyTaskEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PM01MyTaskPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-MyTask-PM01-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:PM01MyTaskEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PM01MyTaskPPORunnerCfg",
    },
)


# 蹲起任务（训练 / 演示）
gym.register(
    id="Isaac-Squat-MyTask-PM01-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.squat_env_cfg:PM01MyTaskSquatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PM01SquatPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Squat-MyTask-PM01-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.squat_env_cfg:PM01MyTaskSquatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PM01SquatPPORunnerCfg",
    },
)
