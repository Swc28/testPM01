from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


def both_feet_airborne_penalty(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1) > 1.0
    both_airborne = torch.sum(contacts.int(), dim=1) == 0
    return both_airborne.float()


def base_height_l2(env, target_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2]
    return torch.square(height - target_height)


def knee_bend_penalty(env, threshold: float = 1.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    knee_joint_ids = asset.find_joints(".*KNEE_PITCH.*")[0]
    if len(knee_joint_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    knee_pos = joint_pos[:, knee_joint_ids]
    penalty = torch.sum(torch.clamp(knee_pos - threshold, min=0.0), dim=1)
    return penalty


def arm_swing_reward(env, max_amplitude: float = 0.35, pos_weight: float = 0.6, vel_weight: float = 0.4, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    joint_vel = asset.data.joint_vel
    shoulder_joint_ids = asset.find_joints(".*SHOULDER_PITCH.*")[0]
    if len(shoulder_joint_ids) < 2:
        return torch.zeros(env.num_envs, device=env.device)
    left_pos = joint_pos[:, shoulder_joint_ids[0]]
    right_pos = joint_pos[:, shoulder_joint_ids[1]]
    left_vel = joint_vel[:, shoulder_joint_ids[0]]
    right_vel = joint_vel[:, shoulder_joint_ids[1]]
    # 1. 幅度限制：两边都不能超出范围（防止一手举头顶）
    amp_mask = (torch.abs(left_pos) < max_amplitude) & (torch.abs(right_pos) < max_amplitude)
    # 2. 位置反向：左右肩位置异号（一前一后）
    pos_reward = -left_pos * right_pos
    pos_reward = torch.clamp(pos_reward, min=0.0)
    # 3. 速度反向：左右肩速度异号（真正在摆动，不是摆好不动）
    vel_reward = -left_vel * right_vel
    vel_reward = torch.clamp(vel_reward, min=0.0)
    # 组合：必须满足幅度限制，同时鼓励位置+速度都反向
    reward = amp_mask.float() * (pos_weight * pos_reward + vel_weight * vel_reward)
    return reward


def hip_extension_penalty(env, threshold: float = 0.5, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    hip_joint_ids = asset.find_joints(".*HIP_PITCH.*")[0]
    if len(hip_joint_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    hip_pos = joint_pos[:, hip_joint_ids]
    penalty = torch.sum(torch.clamp(hip_pos - threshold, min=0.0), dim=1)
    return penalty


def shoulder_flexion_penalty(env, threshold: float = 0.3, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    shoulder_joint_ids = asset.find_joints(".*SHOULDER_PITCH.*")[0]
    if len(shoulder_joint_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    shoulder_pos = joint_pos[:, shoulder_joint_ids]
    penalty = torch.sum(torch.clamp(shoulder_pos - threshold, min=0.0), dim=1)
    return penalty


def hip_flexion_reward(env, min_angle: float = 0.1, max_angle: float = 0.6, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    hip_joint_ids = asset.find_joints(".*HIP_PITCH.*")[0]
    if len(hip_joint_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    hip_pos = joint_pos[:, hip_joint_ids]
    flex = -hip_pos
    reward = torch.sum(torch.clamp(flex - min_angle, min=0.0) * torch.clamp(max_angle - flex, min=0.0), dim=1)
    return reward


def hip_symmetry_penalty(env, threshold: float = 0.3, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    hip_joint_ids = asset.find_joints(".*HIP_PITCH.*")[0]
    if len(hip_joint_ids) < 2:
        return torch.zeros(env.num_envs, device=env.device)
    left_hip = joint_pos[:, hip_joint_ids[0]]
    right_hip = joint_pos[:, hip_joint_ids[1]]
    diff = torch.abs(left_hip - right_hip)
    penalty = torch.clamp(diff - threshold, min=0.0)
    return penalty
