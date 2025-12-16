#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DDPG训练脚本 - 速度闭环控制版
目标：控制车辆在保持最佳滑移率（不打滑）的前提下，尽快加速至目标速度并保持稳定。
"""
import numpy as np
import torch
import os
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# 请确保这两个模块路径正确，或者根据你的项目结构调整import
from pycarsimlib.rl.ddpg_agent import DDPGAgent
from pycarsimlib.rl.env_carsim_simulink import CarsimSimulinkEnv


def train_ddpg_simulink(
    max_episodes: int = 500,
    max_torque: float = 1000.0,      # 提升最大扭矩，确保动力充足
    target_slip_ratio: float = 0.1, # 冰雪低附着路面最佳滑移率
    target_speed: float = 30.0,     # [新增] 目标巡航速度 (km/h)
    log_dir: str = "logs"
):
    # --- 1. 定义奖励函数权重 (Reward Weights) ---
    # 这里的权重决定了智能体的学习方向
    reward_weights = {
            # [正向激励] 
            'w_tracking': 0,     # 高斯满分 +40 分。起步误差大时接近0，但不是负数。
            'w_accel': 150,         
            'w_energy': -10,
            # [约束]
            'w_consistency': -20.0, 
            'w_yaw': -2.0,
            'w_slip': -10.0,        
            'w_smooth': -30.0       
        }
    
    # 记录超参数用于 TensorBoard 展示
    hyperparams = {
        'Target Speed (km/h)': target_speed,
        'Max Torque (Nm)': max_torque,
        'Target Slip': target_slip_ratio,
        'Max Episodes': max_episodes,
        'Batch Size': 128,
        'Actor LR': 1e-4,
        'Critic LR': 1e-3,
        'Gamma': 0.99
    }

    # --- 2. 初始化日志 ---
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"speed_tracking_{current_time}")
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=log_path)
    
    # 将参数表写入 TensorBoard 的 Text 面板
    md_table = "### Reward Coefficients\n| Key | Value |\n|---|---|\n"
    for k, v in reward_weights.items():
        md_table += f"| {k} | {v} |\n"
    
    md_table += "\n### Hyperparameters\n| Key | Value |\n|---|---|\n"
    for k, v in hyperparams.items():
        md_table += f"| {k} | {v} |\n"
        
    writer.add_text("Configuration/Parameters", md_table, 0)
    
    print(f"训练日志将保存至: {log_path}")
    print(f"请在终端运行: tensorboard --logdir={log_dir} 查看曲线")

    # --- 3. 初始化环境 ---
    env = CarsimSimulinkEnv(
        sim_time_s=20.0,        # [修改] 增加到20秒，给足时间进入巡航状态
        delta_time_s=0.01,
        max_torque=max_torque,
        target_slip_ratio=target_slip_ratio,
        target_speed=target_speed,
        reward_weights=reward_weights, # 传入权重字典
        send_port=9202,
        recv_port=8087
    )
    
    # 获取维度
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    
    # --- 4. 初始化 Agent ---
    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bound=1.0,
        hidden_dim=256,
        batch_size=128,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        buffer_capacity=100000,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 训练变量
    best_reward = -float('inf')
    noise_scale = 0.5       # 初始探索噪声
    min_noise = 0.05        # 最小噪声
    noise_decay = 0.95     # 噪声衰减率
    
    print("\n========== 开始训练 (Speed Tracking Task) ==========")
    print(f"目标速度: {target_speed} km/h | 最大扭矩: {max_torque} Nm")
    
    for episode in range(max_episodes):
        # 重置环境
        state, info = env.reset()
        agent.reset_noise()
        episode_reward = 0
        step_count = 0
        
        # 统计数据容器
        slip_errors = []
        speed_errors = []
        torque_smoothness = []
        
        start_time = time.time()
        
        while True:
            # 1. 选择动作
            action = agent.select_action(state, noise_scale=noise_scale)
            
            # 2. 环境交互
            next_state, reward, done, info = env.step(action)
            
            # 3. 存储经验
            agent.push(state, action, reward, next_state, done)
            
            # 4. 模型训练
            c_loss, a_loss = agent.train_step()
            
            # 更新状态
            state = next_state
            episode_reward += reward
            step_count += 1
            
            # 收集统计信息
            if "slip_error" in info: slip_errors.append(info["slip_error"])
            if "speed_error" in info: speed_errors.append(abs(info["speed_error"]))
            if "torque_smoothness" in info: torque_smoothness.append(info["torque_smoothness"])
            
            if done:
                break
        
        # --- Episode 结束处理 ---
        duration = time.time() - start_time
        
        # 计算平均指标
        avg_slip_error = np.mean(slip_errors) if slip_errors else 0.0
        avg_speed_error = np.mean(speed_errors) if speed_errors else 0.0
        avg_smoothness = np.mean(torque_smoothness) if torque_smoothness else 0.0
        final_speed = info.get('vx', 0)
        
        # 1. 记录到 TensorBoard
        writer.add_scalar('Train/Reward', episode_reward, episode)
        writer.add_scalar('Train/Avg_Speed_Error_kmh', avg_speed_error, episode) # 关注这个曲线是否下降
        writer.add_scalar('Train/Final_Speed_kmh', final_speed, episode)         # 关注这个是否趋向100
        writer.add_scalar('Train/Avg_Slip_Error', avg_slip_error, episode)
        writer.add_scalar('Train/Torque_Smoothness', avg_smoothness, episode)
        writer.add_scalar('Train/Noise', noise_scale, episode)
        
        if c_loss is not None:
            writer.add_scalar('Loss/Critic', c_loss, episode)
            writer.add_scalar('Loss/Actor', a_loss, episode)
            
        # 2. 保存最佳模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_model(os.path.join(log_path, "best_model.pt"))
            print(f"🚀 新纪录! Ep {episode+1} Reward: {episode_reward:.1f} (已保存)")
            
        # 3. 定期 Checkpoint
        if (episode + 1) % 50 == 0:
            agent.save_model(os.path.join(log_path, f"checkpoint_ep{episode+1}.pt"))
            
        # 4. 噪声衰减
        noise_scale = max(min_noise, noise_scale * noise_decay)
        
        # 打印进度
        print(f"Ep {episode+1}/{max_episodes} | "
              f"Reward: {episode_reward:.1f} | "
              f"EndSpeed: {final_speed:.1f} km/h | "
              f"Err: {avg_speed_error:.1f} | "
              f"Time: {duration:.1f}s")
              
    # 训练结束
    agent.save_model(os.path.join(log_path, "final_model.pt"))
    writer.close()
    env.close()
    print("训练完成!")


if __name__ == "__main__":
    # 确保 logs 目录存在
    os.makedirs("logs", exist_ok=True)
    
    train_ddpg_simulink()