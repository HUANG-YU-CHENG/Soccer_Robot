# test_model.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# 載入環境（這次有 render_mode="human" 才會顯示視窗）
env = DummyVecEnv([lambda: gym.make("Humanoid-v5", render_mode="human")])
env = VecNormalize.load("vec_normalize.pkl", env)
env.training = False
env.norm_reward = False

# 載入模型
model = PPO.load("humanoid_walk_final")

# 執行並顯示
obs = env.reset()
for _ in range(10000):
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)

env.close()