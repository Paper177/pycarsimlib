#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
速度追踪版 (直接CarsimManager通信版)
更新点：
1. 移除UDP通信，直接使用CarsimManager与CarSim通信。
2. 索引映射更新。
"""
import numpy as np
from datetime import timedelta
from typing import Dict, Tuple, Optional, Any
from pycarsimlib.core import CarsimManager

class CarsimSimulinkEnv:
    # 车辆控制信号名称常量
    STEERING_SIGNAL = "IMP_STEER_SW"  # 方向盘转角信号
    BRAKE_SIGNAL = "IMP_PCON_BK"     # 制动主缸油压信号
    THROTTLE_SIGNAL = "IMP_THROTTLE_ENGINE"  # 油门信号
    TORQUE_SIGNAL_FL = "IMP_TORQUE_FL"  # 左前轮扭矩信号
    TORQUE_SIGNAL_FR = "IMP_TORQUE_FR"  # 右前轮扭矩信号
    TORQUE_SIGNAL_RL = "IMP_TORQUE_RL"  # 左后轮扭矩信号
    TORQUE_SIGNAL_RR = "IMP_TORQUE_RR"  # 右后轮扭矩信号
    
    # 车辆状态信号名称常量
    LONGITUDINAL_SPEED = "Vx"  # 纵向速度信号
    LONGITUDINAL_ACCEL = "Ax"  # 纵向加速度信号
    SLIP_FL = "SLIP_FL"  # 左前轮滑移率
    SLIP_FR = "SLIP_FR"  # 右前轮滑移率
    SLIP_RL = "SLIP_RL"  # 左后轮滑移率
    SLIP_RR = "SLIP_RR"  # 右后轮滑移率
    YAW_RATE = "Yaw"  # 横摆角速度信号
    
    def __init__(
        self,
        carsim_db_dir: str,
        sim_time_s: float = 20.0,
        delta_time_s: float = 0.01,
        max_torque: float = 1500.0,
        target_slip_ratio: float = 0.1, 
        target_speed: float = 100.0,
        vehicle_type: str = "normal_vehicle",
        reward_weights: dict = None
    ):
        self.carsim_db_dir = carsim_db_dir
        self.sim_time_s = sim_time_s
        self.delta_time = timedelta(seconds=delta_time_s)
        self.max_steps = int(sim_time_s / delta_time_s)
        self.max_torque = max_torque
        self.target_slip_ratio = target_slip_ratio
        self.target_speed = target_speed
        self.vehicle_type = vehicle_type
        
        # 默认权重配置
        default_weights = {
            'w_speed': 0.15,      # 速度奖励
            'w_accel': 2.0,       # 加速度奖励
            'w_energy': -0.0,     # 能耗
            'w_consistency': -0.0,# 一致性
            'w_yaw': -0.0,        # 横摆
            'w_slip': -0.1,       # 滑移率
            'w_smooth': -0.0      # 平滑
        }
        self.weights = default_weights.copy()
        if reward_weights:
            self.weights.update(reward_weights)
            
        # 运行时变量
        self.cm: Optional[CarsimManager] = None  # CarSim管理器实例
        self.runtime: float = 0.0  # 当前回合已运行时间
        self.last_torque = np.zeros(4)
        self.filter_alpha = 0.6 
        
        # 状态维度现在是 7
        # 顺序: [Vx, Ax, S_FL, S_FR, S_RL, S_RR, Yaw]
        self.state_dim = 7 
        self.action_dim = 4
        self.current_step = 0

    def reset(self) -> np.ndarray:
        """重置环境并返回初始状态"""
        # 如果存在CarSim管理器实例，先关闭它
        if self.cm:
            self.cm.close()
            self.cm = None
        
        # 初始化CarSim管理器并启动仿真
        self.cm = CarsimManager(
            database_dir=self.carsim_db_dir,
            vehicle_type=self.vehicle_type,
            output_file_name="sim_output",
            delta_time=self.delta_time
        )
        
        # 初始化回合变量
        self.current_step = 0
        self.runtime = 0.0
        self.last_torque = np.zeros(4)
        
        # 获取初始状态
        state_raw = self._receive_state_raw()
        return np.array(state_raw)

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """执行动作并返回新状态、奖励、是否结束和额外信息"""
        # 将动作映射到实际扭矩值
        target_torque = action * self.max_torque
        real_torque = target_torque
        
        # 创建动作字典，格式与CarsimManager兼容
        action_dict = {
            self.TORQUE_SIGNAL_FL: real_torque[0],
            self.TORQUE_SIGNAL_FR: real_torque[1],
            self.TORQUE_SIGNAL_RL: real_torque[2],
            self.TORQUE_SIGNAL_RR: real_torque[3],
            self.STEERING_SIGNAL: 0.0,  # 默认方向盘转角为0
            self.BRAKE_SIGNAL: 0.0,     # 默认制动压力为0
            self.THROTTLE_SIGNAL: 1.0   # 默认油门为1.0
        }
        
        # 执行仿真步
        observations = self.cm.step(action_dict, self.delta_time)
        
        # 获取原始状态
        raw_state = self._receive_state_raw()
        next_state = self._normalize_state(raw_state)
        
        # 计算奖励
        reward, reward_details = self._calculate_reward(raw_state, real_torque, self.last_torque)
        
        # 更新回合变量
        self.last_torque = real_torque
        self.current_step += 1
        self.runtime += self.delta_time.total_seconds()

        # 打印进度信息
        details_str = " | ".join([f"{k}: {v:.3f}" for k, v in reward_details.items()])
        if self.current_step % 100 == 0:
            print(f"step: {self.current_step}, reward: {reward:.4f}, vx: {raw_state[0]:.2f} km/h || {details_str}")

        # 检查是否结束
        done = self.current_step >= self.max_steps
        
        # 构造额外信息字典
        info = {
            "slip_error": np.mean(np.abs(raw_state[2:6] - self.target_slip_ratio)), 
            "vx": raw_state[0], 
            "ax": raw_state[1], 
            **reward_details
        }
        
        return next_state, reward, done, info

    def _receive_state_raw(self) -> np.ndarray:
        """
        直接从CarSimManager获取状态数据
        顺序: Vx, Ax, S_FL, S_FR, S_RL, S_RR, Yaw
        """
        # 从最新的观测值字典中提取状态
        if self.cm.observations:
            obs = self.cm.observations
            return np.array([
                obs.get(self.LONGITUDINAL_SPEED, 0.0),  # Vx (km/h)
                obs.get(self.LONGITUDINAL_ACCEL, 0.0),  # Ax (m/s^2)
                obs.get(self.SLIP_FL, 0.0) * 100,        # S_FL (%，转换为百分比)
                obs.get(self.SLIP_FR, 0.0) * 100,        # S_FR (%，转换为百分比)
                obs.get(self.SLIP_RL, 0.0) * 100,        # S_RL (%，转换为百分比)
                obs.get(self.SLIP_RR, 0.0) * 100,        # S_RR (%，转换为百分比)
                obs.get(self.YAW_RATE, 0.0)              # Yaw (deg)
            ], dtype=np.float32)
        else:
            return np.zeros(self.state_dim, dtype=np.float32)

    def _normalize_state(self, raw_state):
        """
        [修改] 针对 7维数据进行归一化
        raw_state: [Vx, Ax, S1, S2, S3, S4, Yaw]
        """
        norm_state = raw_state.copy()
        
        # 1. 速度 Vx (索引0) -> /100
        norm_state[0] = raw_state[0] / 100.0 
        
        # 2. 加速度 Ax (索引1) -> /5.0 (归一化到 -1~1 范围)
        norm_state[1] = raw_state[1] / 1.0    
        
        # 3. 滑移率 Slips (索引2-5) ->
        norm_state[2:6] = raw_state[2:6]/100
        
        # 4. 横摆角速度 Yaw (索引6) -> /50.0
        norm_state[6] = raw_state[6] / 10.0   
        
        return norm_state

    def _calculate_reward(self, state, current_torque, last_torque):
        if np.all(state == 0): return 0.0, {}
        
        # [修改] 基于新索引提取物理量
        vx = state[0]           # km/h
        ax = state[1]           # m/s^2
        slip = state[2:6] # 假设 Simulink 发来的是 %, 否则去掉 *0.01
        beta = state[6]     # deg偏航角
        
        w = self.weights 
        
        # 1. Speed
        r_speed = w['w_speed'] * (vx) 
        
        # 2. Accel
        if ax > 0.0:
            r_accel = w['w_accel'] * (ax)
        else:
            r_accel = 0.0
        
        # 3. Energy
        torque_norm = current_torque / self.max_torque
        r_energy = w['w_energy'] * np.sum(np.square(torque_norm))
        
        # 4. Slip (前后轮分离)
        # 提取前后轮滑移率
        slip_FL = slip[0] # FL
        slip_FR = slip[1] # FR
        slip_RL = slip[2] # RL
        slip_RR = slip[3] # RR
        
        # 设定阈值
        threshold_front = 0.10  # 前轮 10%
        threshold_rear  = 0.15  # 后轮 15%
        
        # 计算超出的部分 (小于阈值则为0)
        excess_FL = np.maximum(0.0, slip_FL - threshold_front)
        excess_FR = np.maximum(0.0, slip_FR - threshold_front)
        excess_RL = np.maximum(0.0, slip_RL - threshold_rear)
        excess_RR = np.maximum(0.0, slip_RR - threshold_rear)
        
        if vx > 3: # 只有车动起来才算滑移惩罚
            r_slip_FL = w['w_slip'] * excess_FL
            r_slip_FR = w['w_slip'] * excess_FR
            r_slip_RL = w['w_slip'] * excess_RL
            r_slip_RR = w['w_slip'] * excess_RR
            
            # 总滑移奖励
            r_slip = r_slip_FL + r_slip_FR + r_slip_RL + r_slip_RR
        else:
            r_slip = 0.0

        # 5. Consistency
        diff_front = abs(current_torque[0] - current_torque[1])
        diff_rear = abs(current_torque[2] - current_torque[3])
        r_consistency = w['w_consistency'] * ((diff_front + diff_rear) / self.max_torque)
        
        # 6. Smooth
        delta_torque = (current_torque - last_torque) / self.max_torque
        smooth_l2 = np.mean(np.square(delta_torque))
        r_smooth = w['w_smooth'] * smooth_l2 * 10.0

        # 7. Yaw
        r_beta = w['w_beta'] * abs(beta)

        total_reward = r_speed + r_accel + r_energy + r_consistency + r_slip + r_smooth + r_beta

        reward_details = {
            "R_Speed": r_speed,
            "R_Accel": r_accel,
            "R_Energy": r_energy,
            "R_Consis": r_consistency,
            "R_Slip": r_slip,
            "R_Smooth": r_smooth,
            "R_Beta": r_beta
        }

        return total_reward, reward_details
    
    # [修改] 维度更新
    def get_state_dim(self): return 7 
    def get_action_dim(self): return 4
    def close(self) -> None:
        """关闭环境资源"""
        if self.cm:
            self.cm.close()
            self.cm = None