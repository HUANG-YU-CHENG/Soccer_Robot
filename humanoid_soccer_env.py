"""
HumanoidSoccerEnv - 人形機器人足球環境 (修正版)

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
import mujoco


class HumanoidSoccerEnv(MujocoEnv, utils.EzPickle):
    """
    人形機器人足球環境
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 67,
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
        ball_reward_weight: float = 2.0,
        kick_reward_weight: float = 5.0,
        goal_reward: float = 100.0,
        ball_initial_distance: float = 1.5,
        goal_position: tuple = (5.0, 0.0, 0.0),
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
        
        # 如果沒指定 xml_file，使用預設路徑
        if xml_file is None:
            xml_file = os.path.join(os.path.dirname(__file__), "humanoid_soccer.xml")
        
        # 預設相機設定
        if default_camera_config is None:
            default_camera_config = {
                "trackbodyid": 1,
                "distance": 5.0,
                "lookat": np.array([0.0, 0.0, 1.0]),
                "elevation": -20.0,
            }
        
        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,  # 先設為 None，稍後會重新設定
            default_camera_config=default_camera_config,
            **kwargs,
        )
        
        # ========== 關鍵：正確找到球的 qpos/qvel 位置 ==========
        self._ball_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
        self._ball_qpos_adr = self.model.jnt_qposadr[self._ball_joint_id]
        self._ball_qvel_adr = self.model.jnt_dofadr[self._ball_joint_id]
        
        # humanoid root joint
        self._root_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        self._root_qpos_adr = self.model.jnt_qposadr[self._root_joint_id]
        
        # humanoid 的 qpos 數量（root 之後到 ball 之前）
        self._humanoid_qpos_end = self._ball_qpos_adr
        self._humanoid_qvel_end = self._ball_qvel_adr
        
        print(f"[DEBUG] ball qpos 起始: {self._ball_qpos_adr}, qvel 起始: {self._ball_qvel_adr}")
        print(f"[DEBUG] humanoid qpos 範圍: 0-{self._humanoid_qpos_end}, qvel 範圍: 0-{self._humanoid_qvel_end}")
        print(f"[DEBUG] init_qpos[:7] (humanoid 位置+姿態): {self.init_qpos[:7]}")
        print(f"[DEBUG] init_qpos 球位置: {self.init_qpos[self._ball_qpos_adr:self._ball_qpos_adr+3]}")
        
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
        # humanoid 的 z 位置在 qpos[2]
        is_healthy = min_z < self.data.qpos[2] < max_z
        return is_healthy
    
    @property
    def terminated(self):
        terminated = (not self.is_healthy) if self._terminate_when_unhealthy else False
        return terminated
    
    def _get_ball_position(self):
        """獲取球的位置"""
        return self.data.qpos[self._ball_qpos_adr:self._ball_qpos_adr + 3].copy()
    
    def _get_ball_velocity(self):
        """獲取球的速度"""
        return self.data.qvel[self._ball_qvel_adr:self._ball_qvel_adr + 3].copy()
    
    def _get_robot_position(self):
        """獲取機器人（軀幹）的位置"""
        return self.data.qpos[:3].copy()
    
    def _get_obs(self):
        """組合觀察空間"""
        # humanoid 的 qpos 和 qvel（不包含球）
        position = self.data.qpos[:self._humanoid_qpos_end].flat.copy()
        velocity = self.data.qvel[:self._humanoid_qvel_end].flat.copy()
        
        # 球的資訊
        ball_pos = self._get_ball_position()
        ball_vel = self._get_ball_velocity()
        robot_pos = self._get_robot_position()
        
        # 計算相對向量
        ball_to_goal = self._goal_position - ball_pos
        robot_to_ball = ball_pos - robot_pos
        
        # 簡化的觀察空間（不包含 cinert, cvel 等複雜資訊）
        return np.concatenate([
            position,
            velocity,
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
        
        if goal_reward > 0:
            terminated = True
        
        observation = self._get_obs()
        
        info = {
            "reward_survive": healthy_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_approach_ball": approach_ball_reward,
            "reward_kick": kick_reward,
            "reward_goal": goal_reward,
            "robot_position": robot_pos_after.copy(),
            "ball_position": ball_pos_after.copy(),
            "distance_to_ball": dist_to_ball_after,
            "ball_to_goal_distance": ball_to_goal_after,
            "is_healthy": self.is_healthy,
            "goal_scored": goal_reward > 0,
        }
        
        truncated = False
        
        return observation, reward, terminated, truncated, info
    
    def reset_model(self):
        """重置環境"""
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        
        # 複製初始狀態（這包含 XML 中定義的初始位置）
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        
        # ========== 重要：保留 humanoid 的初始高度 ==========
        # init_qpos 已經包含 XML 中定義的正確位置 (0, 0, 1.4)
        # 我們只對關節角度加噪音，不動位置和姿態
        
        # humanoid 的 qpos 結構：
        # [0:3] = x, y, z 位置
        # [3:7] = 四元數 (w, x, y, z) 姿態
        # [7:] = 各關節角度
        
        # 只對關節角度加小噪音（從索引 7 開始到球之前）
        joint_start = 7
        joint_end = self._humanoid_qpos_end
        num_joints = joint_end - joint_start
        
        if num_joints > 0:
            qpos[joint_start:joint_end] += self.np_random.uniform(
                low=noise_low, high=noise_high, size=num_joints
            )
        
        # 對 humanoid 的速度加小噪音
        qvel[:self._humanoid_qvel_end] += self.np_random.uniform(
            low=noise_low, high=noise_high, size=self._humanoid_qvel_end
        )
        
        # ========== 重置球的位置 ==========
        # 球放在機器人前方
        ball_x = self._ball_initial_distance + self.np_random.uniform(-0.3, 0.3)
        ball_y = self.np_random.uniform(-0.3, 0.3)
        ball_z = 0.15  # 稍微高一點，讓球自然落到地面
        
        # 設定球的 qpos (位置 xyz + 四元數 wxyz)
        qpos[self._ball_qpos_adr:self._ball_qpos_adr + 3] = [ball_x, ball_y, ball_z]
        qpos[self._ball_qpos_adr + 3:self._ball_qpos_adr + 7] = [1, 0, 0, 0]
        
        # 球的速度設為 0
        qvel[self._ball_qvel_adr:self._ball_qvel_adr + 6] = 0
        
        self.set_state(qpos, qvel)
        
        # ========== 關鍵修正：讓機器人自然下落並穩定接地 ==========
        # 執行幾步零動作（不輸入任何力），讓重力作用
        # 這確保機器人腳部接觸地面，而不是依賴模型學會對抗重力
        zero_action = np.zeros(self.action_space.shape[0])
        for _ in range(10):
            self.do_simulation(zero_action, self.frame_skip)
        
        return self._get_obs()
    
    def reset(self, *, seed=None, options=None):
        """重置環境並回傳觀察和資訊"""
        obs, info = super().reset(seed=seed, options=options)
        
        # 加入額外資訊
        ball_pos = self._get_ball_position()
        robot_pos = self._get_robot_position()
        
        info.update({
            "ball_position": ball_pos.copy(),
            "robot_position": robot_pos.copy(),
            "distance_to_ball": np.linalg.norm(robot_pos[:2] - ball_pos[:2]),
            "ball_to_goal_distance": np.linalg.norm(self._goal_position[:2] - ball_pos[:2]),
        })
        
        return obs, info


def make_soccer_env(render_mode=None, **kwargs):
    """建立足球環境的輔助函數"""
    return HumanoidSoccerEnv(render_mode=render_mode, **kwargs)


# ==================== 測試程式碼 ====================
if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("測試 HumanoidSoccerEnv")
    print("=" * 60)
    
    # 建立環境
    env = HumanoidSoccerEnv(render_mode="human")
    
    print(f"\n📊 環境資訊:")
    print(f"   觀察空間: {env.observation_space.shape}")
    print(f"   動作空間: {env.action_space.shape}")
    
    # 重置環境
    obs, info = env.reset()
    print(f"\n📍 初始狀態:")
    print(f"   觀察維度: {obs.shape}")
    print(f"   機器人位置: {info['robot_position']}")
    print(f"   球位置: {info['ball_position']}")
    print(f"   到球距離: {info['distance_to_ball']:.3f}")
    print(f"   球到球門距離: {info['ball_to_goal_distance']:.3f}")
    
    # 跑幾步測試
    print(f"\n🎮 開始測試（觀察視窗中的機器人）...")
    print(f"   提示：機器人會先從空中落下，然後開始隨機動作")
    print(f"   因為是隨機動作，機器人會很快倒下，這是正常的！\n")
    
    episode_reward = 0
    episode_count = 0
    
    for i in range(2000):
        # 隨機動作
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        
        # 強制渲染（確保視窗更新）
        env.render()
        
        # 稍微減慢速度，讓人眼能看清楚
        time.sleep(0.01)
        
        if i % 200 == 0:
            print(f"   Step {i:4d}: reward={reward:7.3f}, "
                  f"dist_to_ball={info['distance_to_ball']:.3f}, "
                  f"robot_z={info['robot_position'][2]:.3f}, "
                  f"ball_z={info['ball_position'][2]:.3f}")
        
        if terminated:
            episode_count += 1
            print(f"\n⚠️  Episode {episode_count} 結束於 step {i}")
            if info.get('goal_scored'):
                print("   🎉 進球了！")
            else:
                print("   💀 機器人倒下了 (robot_z={:.3f})".format(info['robot_position'][2]))
            print(f"   累計獎勵: {episode_reward:.2f}")
            episode_reward = 0
            obs, info = env.reset()
            
            if episode_count >= 10:
                print("\n已完成 10 個 episode，結束測試")
                break
    
    env.close()
    print("\n✅ 測試完成！")