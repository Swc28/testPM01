"""离屏相机录制蹲起策略视频（headless 无显示器可用）。

原理：官方 play.py 的 --video 走 RecordVideo 视口截图，无显示环境下拿到的是黑帧；
本脚本使用 Play 场景中的 TiledCamera 传感器（RTX 离屏渲染）逐帧取 RGB 写 mp4。

用法（仓库根目录）：
    python -m engineai_rl_lab.tasks.mytask.record_video \\
        --task Isaac-Squat-MyTask-PM01-Play-v0 --num_envs 1 \\
        --load_run 2026-10-06_19-37-55 --checkpoint model_4999.pt \\
        --video_length 500
输出：./videos/<任务>_<run>_<checkpoint>.mp4（可用 --output 指定路径）

注意：相机传感器只配置在 *_PLAY 环境里，task 必须用 Play 任务名。
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record squat policy video via offscreen tiled camera.")
parser.add_argument("--task", type=str, default="Isaac-Squat-MyTask-PM01-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--load_run", type=str, default=None, help="训练 run 时间戳目录名")
parser.add_argument("--checkpoint", type=str, default="model_4999.pt")
parser.add_argument("--video_length", type=int, default=500, help="录制步数（每步 0.02s，500 步 = 10s）")
parser.add_argument("--output", type=str, default=None, help="输出 mp4 路径，默认 ./videos/<task>_<run>_<ckpt>.mp4")
parser.add_argument("--fps", type=int, default=50, help="视频帧率（环境 50Hz，50 即实时回放）")
parser.add_argument("--env_id", type=int, default=0, help="录制哪个环境的相机")
parser.add_argument("--seed", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 强制 headless + 开启相机渲染（场景中有 TiledCamera，不开会报
# "A camera was spawned without the --enable_cameras flag"）
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- 运行时导入（必须在 AppLauncher 之后）----
import gymnasium as gym
import imageio.v2 as imageio
import torch

from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import engineai_rl_lab.tasks  # noqa: F401  # 注册所有任务

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.seed = agent_cfg.seed

    # checkpoint 路径解析（与 play.py 相同逻辑）
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, args_cli.load_run, args_cli.checkpoint)
    run_dir = os.path.dirname(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # 离屏相机（PLAY 场景中注册的 view_cam）
    camera = env.unwrapped.scene.sensors["view_cam"]

    obs = env.get_observations()
    frames = []
    for step in range(args_cli.video_length):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)
        rgb = camera.data.output["rgb"][args_cli.env_id, :, :, :3].cpu().numpy()
        frames.append(rgb)
        if step % 100 == 0:
            print(f"[INFO]: recorded {step}/{args_cli.video_length}")

    env.close()

    if args_cli.output is None:
        run_name = os.path.basename(run_dir)
        ckpt_name = os.path.splitext(os.path.basename(resume_path))[0]
        out_dir = os.path.abspath("videos")
        os.makedirs(out_dir, exist_ok=True)
        args_cli.output = os.path.join(out_dir, f"{args_cli.task}_{run_name}_{ckpt_name}.mp4")

    imageio.mimsave(args_cli.output, frames, fps=args_cli.fps)
    print(f"[INFO]: video saved to: {args_cli.output}")


def _patch_actor_critic():
    """与 play.py 相同的 checkpoint 兼容补丁（std/log_std 与分布裁剪）。"""
    from rsl_rl.modules import ActorCritic as _AC
    from rsl_rl.modules import ActorCriticRecurrent as _ACR

    _orig_update_dist = _AC._update_distribution
    _orig_load_state_dict = _AC.load_state_dict

    def _safe_update_distribution(self, obs):
        _orig_update_dist(self, obs)
        if self.distribution is not None:
            self.distribution = torch.distributions.Normal(
                self.distribution.mean,
                torch.clamp(self.distribution.scale, min=1e-6),
            )

    def _compatible_load_state_dict(self, state_dict, strict=True):
        has_std = "std" in state_dict
        has_log_std = "log_std" in state_dict
        if has_std and not has_log_std and self.noise_std_type == "log":
            state_dict["log_std"] = torch.log(torch.clamp(state_dict.pop("std"), min=1e-7))
            self.noise_std_type = "log"
        elif has_log_std and not has_std and self.noise_std_type == "scalar":
            state_dict["std"] = torch.exp(state_dict.pop("log_std"))
            self.noise_std_type = "scalar"
        elif has_std and has_log_std:
            pass
        return _orig_load_state_dict(self, state_dict, strict=strict)

    _AC._update_distribution = _safe_update_distribution
    _ACR._update_distribution = _safe_update_distribution
    _AC.load_state_dict = _compatible_load_state_dict
    _ACR.load_state_dict = _compatible_load_state_dict


if __name__ == "__main__":
    _patch_actor_critic()
    main()
    simulation_app.close()
