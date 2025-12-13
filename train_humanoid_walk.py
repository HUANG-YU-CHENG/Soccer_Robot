import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
import torch

# 確認使用 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 建立多個並行環境（加速訓練）
n_envs = 8  # 4060 可以跑 4-8 個並行環境
env = make_vec_env("Humanoid-v5", n_envs=n_envs)

# 正規化觀察值和獎勵（很重要！）
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

# 建立 PPO 模型
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    device=device,
)

# 設定 checkpoint（每 10 萬步存一次）
checkpoint_callback = CheckpointCallback(
    save_freq=100000 // n_envs,
    save_path="./checkpoints/",
    name_prefix="humanoid_walk",
)

# 開始訓練
print("開始訓練...")
model.learn(
    total_timesteps=1_000_000,  # 先跑 100 萬步看看
    callback=checkpoint_callback,
    progress_bar=True,
)

# 儲存模型和正規化統計
model.save("humanoid_walk_final")
env.save("vec_normalize.pkl")

print("訓練完成！")