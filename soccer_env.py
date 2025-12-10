import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

class HumanoidSoccerEnv(gym.Env):
    """
    自定義人形機器人足球環境
    目標：讓機器人走向球並踢球
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        # TODO: 載入包含球和球門的 MuJoCo XML 模型
        # self.model = mujoco.MjModel.from_xml_path("humanoid_soccer.xml")
        # self.data = mujoco.MjData(self.model)
        
        # 暫時使用預設 humanoid（之後替換）
        self.base_env = gym.make("Humanoid-v5", render_mode=render_mode)
        
        # 定義觀察空間（原本 humanoid 觀察 + 球的位置）
        humanoid_obs_dim = self.base_env.observation_space.shape[0]
        ball_obs_dim = 6  # 球的 xyz 位置 + xyz 速度
        total_obs_dim = humanoid_obs_dim + ball_obs_dim
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float64
        )
        
        # 動作空間（和原本 humanoid 一樣）
        self.action_space = self.base_env.action_space
        
        # 球的初始位置
        self.ball_pos = np.array([1.0, 0.0, 0.1])  # 機器人前方 1 公尺
        self.ball_vel = np.array([0.0, 0.0, 0.0])
        
        # 球門位置
        self.goal_pos = np.array([5.0, 0.0, 0.0])  # 前方 5 公尺
        
        self.render_mode = render_mode
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        obs, info = self.base_env.reset(seed=seed)
        
        # 重置球的位置（加一點隨機性）
        self.ball_pos = np.array([
            1.0 + np.random.uniform(-0.2, 0.2),
            np.random.uniform(-0.3, 0.3),
            0.1
        ])
        self.ball_vel = np.zeros(3)
        
        # 組合觀察
        full_obs = self._get_obs(obs)
        
        return full_obs, info
    
    def step(self, action):
        # 執行動作
        obs, base_reward, terminated, truncated, info = self.base_env.step(action)
        
        # 計算自定義獎勵
        reward = self._compute_reward(obs, base_reward)
        
        # 組合觀察
        full_obs = self._get_obs(obs)
        
        return full_obs, reward, terminated, truncated, info
    
    def _get_obs(self, humanoid_obs):
        """組合 humanoid 觀察和球的資訊"""
        ball_obs = np.concatenate([self.ball_pos, self.ball_vel])
        return np.concatenate([humanoid_obs, ball_obs])
    
    def _compute_reward(self, obs, base_reward):
        """
        計算獎勵函數 - 這是最關鍵的部分！
        """
        reward = 0.0
        
        # 獲取機器人位置（假設在 obs 的前幾維）
        robot_pos = obs[:3]  # 需要根據實際 obs 結構調整
        
        # 1. 獎勵接近球
        dist_to_ball = np.linalg.norm(robot_pos[:2] - self.ball_pos[:2])
        reward += 1.0 * (1.0 / (1.0 + dist_to_ball))  # 越近獎勵越高
        
        # 2. 獎勵球移動向球門
        ball_to_goal_dist = np.linalg.norm(self.ball_pos[:2] - self.goal_pos[:2])
        reward += 0.5 * (5.0 - ball_to_goal_dist)  # 球越接近球門越好
        
        # 3. 大獎勵：球進門
        if ball_to_goal_dist < 0.5:
            reward += 100.0
        
        # 4. 懲罰跌倒（用 base_reward 的一部分判斷）
        # 如果 base_reward 很低，可能代表跌倒
        if base_reward < -1.0:
            reward -= 10.0
        
        # 5. 小獎勵：維持站立（來自原本的 humanoid reward）
        reward += 0.1 * base_reward
        
        return reward
    
    def render(self):
        return self.base_env.render()
    
    def close(self):
        self.base_env.close()