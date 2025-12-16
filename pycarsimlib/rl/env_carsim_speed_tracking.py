from datetime import timedelta
from typing import Dict, Tuple, Optional, List, Any
import os
import shutil

import numpy as np

from pycarsimlib.core import CarsimManager


class CarsimSpeedTrackingEnv:
    """
    基于CarSim的车辆速度跟踪强化学习环境

    - 动作空间: 2维连续空间 [-1, 1],映射到 [油门%, 制动%]
    - 状态空间: [当前速度(km/h), 速度误差(km/h)]
    - 奖励函数: -|速度误差|*2 - （油门% + 制动%)*1
    """
    
    # 车辆控制信号名称常量
    STEERING_SIGNAL = "IMP_STEER_SW"  # 方向盘转角信号
    BRAKE_SIGNAL = "IMP_PCON_BK"     # 制动主缸油压信号
    THROTTLE_SIGNAL = "IMP_THROTTLE_ENGINE"  # 油门信号
    
    # 车辆状态信号名称常量
    LONGITUDINAL_SPEED = "Vx"  # 纵向速度信号
    
    def __init__(
        self,
        carsim_db_dir: str,
        delta_time_s: float = 0.1,
        episode_time_s: float = 10.0,
        target_speed: float = 30.0,
        vehicle_type: str = "normal_vehicle",
        max_throttle: float = 1.0,  # 1.0代表100%油门
        max_brake: float = 8.0      # 最大制动主缸油压(MPa)
    ) -> None:
        """
        初始化CarSim速度跟踪环境
        
        参数:
            carsim_db_dir: CarSim数据库目录路径
            delta_time_s: 仿真步长（秒）
            episode_time_s: 单回合最大时间（秒）
            target_speed: 目标速度（km/h）- CarSim读取的Vx单位为km/h
            vehicle_type: 车辆类型
            max_throttle: 最大油门值（1.0表示100%）
            max_brake: 最大制动主缸油压(MPa),默认8.0MPa
        """
        # 配置参数
        self.carsim_db_dir = carsim_db_dir
        self.delta_time = timedelta(seconds=delta_time_s)
        self.episode_time_s = episode_time_s
        self.max_steps = int(episode_time_s / delta_time_s)
        self.vehicle_type = vehicle_type
        self.max_throttle = max_throttle
        self.max_brake = max_brake
        self.target_speed = target_speed
        
        # 运行时变量
        self.cm: Optional[CarsimManager] = None  # CarSim管理器实例
        self.runtime: float = 0.0  # 当前回合已运行时间
        
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置环境
        返回:state: 初始状态 [当前速度, 速度误差], info: 附加信息
        """
        # 关闭会话
        self.close()
        
        # 创建新的CarSim实例
        self.cm = CarsimManager(
            carsim_db_dir=self.carsim_db_dir,
            vehicle_type=self.vehicle_type,
        )
        
        # 重置时间
        self.runtime = 0.0
        
        # 0时刻动作获取初始状态
        zero_action = {
            self.STEERING_SIGNAL: 0.0,
            self.BRAKE_SIGNAL: 0.0,
            self.THROTTLE_SIGNAL: 0.0
        }
        
        obs_dict, _, _ = self.cm.step(
            action=zero_action,
            delta_time=self.delta_time,
        )
        
        # 获取初始速度和状态
        current_speed = self._get_vehicle_speed(obs_dict)   #从Carsim读取
        speed_error = self.target_speed - current_speed
        
        # 状态
        state = np.array([current_speed, speed_error], dtype=np.float32)
        info = {
            "speed": current_speed,
            "v_ref": self.target_speed,
            "run_time": self.runtime
        }
        
        return state, info
    
    def step(self, action_np: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        执行一步环境交互
        
        参数:
            action_np: 2维动作数组 [throttle_action, brake_action],范围[-1, 1]
            
        返回:
            next_state: 下一个状态
            reward: 即时奖励
            done: 是否终止
            info: 附加信息
        """
        # 将RL动作映射到CarSim控制信号
        control_signals = self._map_action_to_control(action_np)
        
        # 执行CarSim仿真步
        obs_dict, terminated_flag, _ = self.cm.step(
            action=control_signals,
            delta_time=self.delta_time
        )
        
        # 更新时间
        self.runtime += self.delta_time.total_seconds()
        
        # 获取当前速度和计算奖励
        current_speed = self._get_vehicle_speed(obs_dict)
        reward = self._calculate_reward(current_speed, control_signals)
        
        # 检查是否终止
        done = bool(terminated_flag) or (self.runtime >= self.episode_time_s)
        
        # 构建下一个状态
        speed_error = self.target_speed - current_speed
        next_state = np.array([current_speed, speed_error], dtype=np.float32)
        
        # 构建信息字典
        info = {
            "speed": current_speed,
            "v_ref": self.target_speed,
            "throttle": control_signals[self.THROTTLE_SIGNAL],
            "brake": control_signals[self.BRAKE_SIGNAL],
            "run_time": self.runtime
        }
        
        return next_state, reward, done, info
    
    def _map_action_to_control(self, action: np.ndarray) -> Dict[str, float]:
        """
        将RL动作映射到CarSim控制信号
        
        参数:
            action: RL动作数组 [throttle_action, brake_action],范围[-1, 1]
            
        返回:
            控制信号字典,包含方向盘、油门和制动值
        """
        # 确保动作在有效范围内
        a = np.clip(action, -1.0, 1.0)
        
        # 将[-1, 1]映射到[0, max_throttle]和[0, max_brake]
        # 油门映射到百分比,制动映射到主缸油压(MPa)
        throttle = float((a[0] + 1.0) * 0.5 * self.max_throttle)
        brake = float((a[1] + 1.0) * 0.5 * self.max_brake)
        
        return {
            self.STEERING_SIGNAL: 0.0,
            self.BRAKE_SIGNAL: brake,
            self.THROTTLE_SIGNAL: throttle
        }
    
    def _get_vehicle_speed(self, obs_dict: Dict[str, Any]) -> float:
        """
        从观察字典中获取车辆速度
        
        参数:
            obs_dict: CarSim输出的观察字典
            
        返回:
            车辆纵向速度（km/h）- CarSim输出的Vx单位为km/h
        """
        # 直接从CarSim获取Vx信号作为纵向速度
        return float(obs_dict.get(self.LONGITUDINAL_SPEED, 0.0))
    
    def _calculate_reward(self, current_speed: float, control_signals: Dict[str, float]) -> float:
        """
        计算奖励函数

        参数:
            current_speed: 当前车辆速度
            control_signals: 当前控制信号
            
        返回:
            奖励值
        """
        # 计算速度误差
        speed_error = self.target_speed - current_speed
        
        # 计算控制努力（归一化到最大控制量）
        normalized_throttle = control_signals[self.THROTTLE_SIGNAL] / self.max_throttle
        normalized_brake = control_signals[self.BRAKE_SIGNAL] / self.max_brake
        
        # 根据速度误差调整控制努力惩罚策略
        # 当速度低于目标速度时,减少对油门的惩罚,鼓励加速
        # 当速度高于目标速度时,减少对刹车的惩罚,鼓励减速
        if speed_error < 0:  # 当前速度低于目标速度
            # 速度低时,对油门惩罚减轻,主要惩罚刹车
            control_effort = 0.01 * normalized_throttle + normalized_brake
        else:  # 当前速度高于目标速度
            # 速度高时,对刹车惩罚减轻,主要惩罚油门
            control_effort = 1*normalized_throttle + 0.01 * normalized_brake
        
        # 奖励 = -|速度误差| - 控制努力惩罚 - 互斥行为惩罚
        reward = -abs(speed_error) * 2 - 1 * control_effort
        
        return reward
    
    def close(self) -> None:
        """
        关闭环境并清理资源
        """
        if self.cm is not None:
            try:
                self.cm.close()
            except Exception:
                # 静默处理异常,确保资源被释放
                pass
            finally:
                self.cm = None
    
    def save_results_into_carsimdb(self, results_source_dir: str = "", results_target_dir: str = "") -> None:
        """
        将仿真结果保存到CarSim数据库目录
        
        参数:
            results_source_dir: 结果源目录路径（默认使用当前工作目录下的Results文件夹）
            results_target_dir: 结果目标目录路径（默认使用CarSim数据库下的Results文件夹）
        """
        # 设置默认路径
        if not results_source_dir:
            results_source_dir = os.path.join(os.getcwd(), "Results")
        
        if not results_target_dir:
            results_target_dir = os.path.join(self.carsim_db_dir, "Results")
        
        # 复制结果目录
        print(f"正在保存结果到: {results_target_dir}")
        shutil.copytree(results_source_dir, results_target_dir, dirs_exist_ok=True)
        print("结果保存成功。")