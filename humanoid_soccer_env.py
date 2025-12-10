"""
HumanoidSoccerEnv - 人形機器人足球環境

這個環境讓 humanoid 機器人學習走向足球並踢向球門。
基於 Gymnasium 的 MujocoEnv 建立。

使用方式：
    env = HumanoidSoccerEnv(render_mode="human")
    obs, info = env.reset()
    
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
"""

import numpy as np
import os
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


class HumanoidSoccerEnv(MujocoEnv, utils.EzPickle):
    """
    人形機器人足球環境
    
    觀察空間 (348 維)：
        - humanoid 的 qpos (24維): 位置
        - humanoid 的 qvel (23維): 速度
        - cinert (140維): 慣性
        - cvel (84維): 質心速度
        - qfrc_actuator (23維): 作用力
        - cfrc_ext (84維): 外部接觸力
        - 球的位置 (3維): ball_pos
        - 球的速度 (3維): ball_vel
        - 球到球門的向量 (3維): ball_to_goal
        - 機器人到球的向量 (3維): robot_to_ball
    
    動作空間 (21 維)：
        - 21 個關節的力矩
    
    獎勵函數：
        - 接近球的獎勵
        - 球移動向球門的獎勵
        - 進球大獎
        - 跌倒懲罰
        - 存活獎勵
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 67,  # 基於 timestep=0.003 和 frame_skip=5
    }
    
    def __init__(
        self,
        xml_file: str = None,
        frame_skip: int = 5,
        default_camera_config: dict = None,
        forward_reward_weight: float = 1.25,
        ctrl_cost_weight: float = 0.1,
        healthy_reward: float = 5.0,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: tuple = (0.8, 2.1),
        reset_noise_scale: float = 1e-2,
        # 足球相關參數
        ball_reward_weight: float = 2.0,       # 接近球的獎勵權重
        kick_reward_weight: float = 5.0,       # 踢球向球門的獎勵權重
        goal_reward: float = 100.0,            # 進球獎勵
        ball_initial_distance: float = 1.5,    # 球的初始距離
        goal_position: tuple = (5.0, 0.0, 0.0), # 球門位置
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            ball_reward_weight,
            kick_reward_weight,
            goal_reward,
            ball_initial_distance,
            goal_position,
            **kwargs,
        )
        
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        
        # 足球相關
        self._ball_reward_weight = ball_reward_weight
        self._kick_reward_weight = kick_reward_weight
        self._goal_reward = goal_reward
        self._ball_initial_distance = ball_initial_distance
        self._goal_position = np.array(goal_position)
        
        # 追蹤上一步的球位置（用於計算球的移動）
        self._prev_ball_pos = None
        
        # 如果沒指定 xml_file，使用預設路徑
        if xml_file is None:
            xml_file = os.path.join(os.path.dirname(__file__), "humanoid_soccer.xml")
        
        # 設定觀察空間
        # 先暫時設一個較大的值，之後會在 _get_obs 中自動調整
        # humanoid qpos(24) + qvel(23) + 球相關(12) = 基礎 59
        # 但實際會包含更多資訊（cinert, cvel, etc.）
        # 我們先設為 None，讓 MujocoEnv 自動推斷
        observation_space = None  # 會在初始化後重新設定
        
        # 預設相機設定
        if default_camera_config is None:
            default_camera_config = {
                "trackbodyid": 1,
                "distance": 4.0,
                "lookat": np.array([0.0, 0.0, 1.0]),
                "elevation": -20.0,
            }
        
        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config=default_camera_config,
            observation_space=observation_space,
            **kwargs,
        )
        
        # 找到球的 body id
        self._ball_body_id = self.model.body("ball").id
        self._ball_joint_id = self.model.joint("ball_joint").id
        
        # 初始化後，根據實際觀察重新設定觀察空間
        sample_obs = self._get_obs()
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=sample_obs.shape, dtype=np.float64
        )
        
    @property
    def healthy_reward(self):
        return float(self.is_healthy or self._terminate_when_unhealthy) * self._healthy_reward
    
    def control_cost(self, action):
        return self._ctrl_cost_weight * np.sum(np.square(action))
    
    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.qpos[2] < max_z
        return is_healthy
    
    @property
    def terminated(self):
        terminated = (not self.is_healthy) if self._terminate_when_unhealthy else False
        return terminated
    
    def _get_ball_position(self):
        """獲取球的位置"""
        # 球的 qpos 在 humanoid qpos 之後
        # humanoid 有 24 個 qpos (7 for root + 17 joints)
        # 球的 freejoint 有 7 個 qpos (3 pos + 4 quat)
        ball_qpos_start = 24  # humanoid 的 qpos 數量
        ball_pos = self.data.qpos[ball_qpos_start:ball_qpos_start + 3].copy()
        return ball_pos
    
    def _get_ball_velocity(self):
        """獲取球的速度"""
        # 球的 qvel 在 humanoid qvel 之後
        # humanoid 有 23 個 qvel
        # 球的 freejoint 有 6 個 qvel (3 linear + 3 angular)
        ball_qvel_start = 23  # humanoid 的 qvel 數量
        ball_vel = self.data.qvel[ball_qvel_start:ball_qvel_start + 3].copy()
        return ball_vel
    
    def _get_robot_position(self):
        """獲取機器人（軀幹）的位置"""
        return self.data.qpos[:3].copy()
    
    def _get_obs(self):
        """組合觀察空間"""
        # 獲取 humanoid 的 qpos 和 qvel（排除球的部分）
        # humanoid 有 24 個 qpos，23 個 qvel
        position = self.data.qpos[:24].flat.copy()
        velocity = self.data.qvel[:23].flat.copy()
        
        # cinert, cvel, qfrc_actuator, cfrc_ext 的大小取決於模型
        # 我們取所有可用的資料
        com_inertia = self.data.cinert.flat.copy()
        com_velocity = self.data.cvel.flat.copy()
        actuator_forces = self.data.qfrc_actuator[:23].flat.copy()  # 只取 humanoid 的
        external_contact_forces = self.data.cfrc_ext.flat.copy()
        
        # 球的資訊
        ball_pos = self._get_ball_position()
        ball_vel = self._get_ball_velocity()
        robot_pos = self._get_robot_position()
        
        # 計算相對向量
        ball_to_goal = self._goal_position - ball_pos
        robot_to_ball = ball_pos - robot_pos
        
        return np.concatenate([
            position,
            velocity,
            com_inertia,
            com_velocity,
            actuator_forces,
            external_contact_forces,
            ball_pos,
            ball_vel,
            ball_to_goal,
            robot_to_ball,
        ])
    
    def step(self, action):
        # 記錄動作前的狀態
        robot_pos_before = self._get_robot_position()
        ball_pos_before = self._get_ball_position()
        
        # 執行動作
        self.do_simulation(action, self.frame_skip)
        
        # 獲取動作後的狀態
        robot_pos_after = self._get_robot_position()
        ball_pos_after = self._get_ball_position()
        
        # ==================== 計算獎勵 ====================
        
        # 1. 存活獎勵
        healthy_reward = self.healthy_reward
        
        # 2. 控制成本
        ctrl_cost = self.control_cost(action)
        
        # 3. 接近球的獎勵
        dist_to_ball_before = np.linalg.norm(robot_pos_before[:2] - ball_pos_before[:2])
        dist_to_ball_after = np.linalg.norm(robot_pos_after[:2] - ball_pos_after[:2])
        approach_ball_reward = self._ball_reward_weight * (dist_to_ball_before - dist_to_ball_after)
        
        # 4. 踢球向球門的獎勵
        ball_to_goal_before = np.linalg.norm(self._goal_position[:2] - ball_pos_before[:2])
        ball_to_goal_after = np.linalg.norm(self._goal_position[:2] - ball_pos_after[:2])
        kick_reward = self._kick_reward_weight * (ball_to_goal_before - ball_to_goal_after)
        
        # 5. 進球獎勵
        goal_reward = 0.0
        if ball_pos_after[0] > 4.9 and abs(ball_pos_after[1]) < 1.0 and ball_pos_after[2] < 1.0:
            goal_reward = self._goal_reward
        
        # 總獎勵
        reward = (
            healthy_reward
            - ctrl_cost
            + approach_ball_reward
            + kick_reward
            + goal_reward
        )
        
        # ==================== 終止條件 ====================
        terminated = self.terminated
        
        # 如果進球也終止
        if goal_reward > 0:
            terminated = True
        
        # 獲取觀察
        observation = self._get_obs()
        
        # 資訊
        info = {
            "reward_survive": healthy_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_approach_ball": approach_ball_reward,
            "reward_kick": kick_reward,
            "reward_goal": goal_reward,
            "robot_position": robot_pos_after,
            "ball_position": ball_pos_after,
            "distance_to_ball": dist_to_ball_after,
            "ball_to_goal_distance": ball_to_goal_after,
            "is_healthy": self.is_healthy,
            "goal_scored": goal_reward > 0,
        }
        
        # truncated 由 TimeLimit wrapper 處理
        truncated = False
        
        return observation, reward, terminated, truncated, info
    
    def reset_model(self):
        """重置環境"""
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        
        # 重置 humanoid 位置
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        
        # 加入隨機噪音到 humanoid
        qpos[:24] = self.init_qpos[:24] + self.np_random.uniform(
            low=noise_low, high=noise_high, size=24
        )
        qvel[:23] = self.init_qvel[:23] + self.np_random.uniform(
            low=noise_low, high=noise_high, size=23
        )
        
        # 重置球的位置（在機器人前方，加一點隨機性）
        ball_x = self._ball_initial_distance + self.np_random.uniform(-0.3, 0.3)
        ball_y = self.np_random.uniform(-0.5, 0.5)
        ball_z = 0.11  # 球的半徑
        
        # 設定球的 qpos (位置 + 四元數)
        qpos[24:27] = [ball_x, ball_y, ball_z]  # 位置
        qpos[27:31] = [1, 0, 0, 0]  # 四元數（無旋轉）
        
        # 球的速度設為 0
        qvel[23:29] = 0
        
        self.set_state(qpos, qvel)
        
        return self._get_obs()
    
    def reset(self, *, seed=None, options=None):
        """重置環境並回傳觀察和資訊"""
        obs, info = super().reset(seed=seed, options=options)
        
        # 加入額外資訊
        ball_pos = self._get_ball_position()
        robot_pos = self._get_robot_position()
        
        info.update({
            "ball_position": ball_pos,
            "robot_position": robot_pos,
            "distance_to_ball": np.linalg.norm(robot_pos[:2] - ball_pos[:2]),
            "ball_to_goal_distance": np.linalg.norm(self._goal_position[:2] - ball_pos[:2]),
        })
        
        return obs, info
    

def make_soccer_env(render_mode=None, **kwargs):
    """
    建立足球環境的輔助函數
    
    使用方式：
        from humanoid_soccer_env import make_soccer_env
        env = make_soccer_env(render_mode="human")
    """
    return HumanoidSoccerEnv(render_mode=render_mode, **kwargs)


# ==================== 測試程式碼 ====================
if __name__ == "__main__":
    print("測試 HumanoidSoccerEnv...")
    
    # 建立環境
    env = HumanoidSoccerEnv(render_mode="human")
    
    print(f"觀察空間: {env.observation_space.shape}")
    print(f"動作空間: {env.action_space.shape}")
    
    # 重置環境
    obs, info = env.reset()
    print(f"初始觀察維度: {obs.shape}")
    print(f"初始球位置: {info['ball_position']}")
    print(f"初始機器人位置: {info['robot_position']}")
    
    # 跑幾步測試
    for i in range(500000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 100 == 0:
            print(f"Step {i}: reward={reward:.3f}, dist_to_ball={info['distance_to_ball']:.3f}")
        
        if terminated or truncated:
            print(f"Episode 結束於 step {i}")
            if info.get('goal_scored'):
                print("🎉 進球了！")
            obs, info = env.reset()
    
    env.close()
    print("測試完成！")