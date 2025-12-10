"""
train_soccer.py - 訓練人形機器人踢足球

使用 PPO 演算法訓練機器人：
1. 走向足球
2. 踢球向球門

使用方式：
    python train_soccer.py
"""

import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# 導入自定義環境
from humanoid_soccer_env import HumanoidSoccerEnv


def make_env(rank, seed=0):
    """
    建立環境的工廠函數（用於並行環境）
    """
    def _init():
        env = HumanoidSoccerEnv()
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def train():
    """主訓練函數"""
    
    # ==================== 設定 ====================
    
    # 訓練參數
    TOTAL_TIMESTEPS = 1_000_000     # 總訓練步數（可以調整）
    N_ENVS = 8                      # 並行環境數量（4060 建議 4-8）
    SAVE_FREQ = 100_000             # 每多少步存一次 checkpoint
    EVAL_FREQ = 50_000              # 每多少步評估一次
    
    # 資料夾設定
    LOG_DIR = "./logs/soccer/"
    CHECKPOINT_DIR = "./checkpoints/soccer/"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # 確認 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  使用裝置: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # ==================== 建立環境 ====================
    
    print(f"\n📦 建立 {N_ENVS} 個並行環境...")
    
    # 使用 SubprocVecEnv 進行真正的並行（比 DummyVecEnv 快）
    # 如果遇到問題，可以改用 DummyVecEnv
    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    
    # 正規化觀察值和獎勵（非常重要！）
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )
    
    print(f"   觀察空間: {env.observation_space.shape}")
    print(f"   動作空間: {env.action_space.shape}")
    
    # ==================== 建立模型 ====================
    
    print("\n🧠 建立 PPO 模型...")
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device=device,
        # PPO 超參數（可以調整）
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,         # 增加探索
        vf_coef=0.5,
        max_grad_norm=0.5,
        # 神經網路設定
        policy_kwargs={
            "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
        },
    )
    
    print(f"   模型參數量: {sum(p.numel() for p in model.policy.parameters()):,}")
    
    # ==================== 設定 Callbacks ====================
    
    # Checkpoint callback - 定期儲存模型
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ // N_ENVS,
        save_path=CHECKPOINT_DIR,
        name_prefix="soccer_humanoid",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    # 建立評估環境
    eval_env = DummyVecEnv([lambda: Monitor(HumanoidSoccerEnv())])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    # Eval callback - 定期評估並儲存最佳模型
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=CHECKPOINT_DIR,
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ // N_ENVS,
        n_eval_episodes=5,
        deterministic=True,
    )
    
    # ==================== 開始訓練 ====================
    
    print(f"\n🚀 開始訓練 {TOTAL_TIMESTEPS:,} 步...")
    print(f"   TensorBoard: tensorboard --logdir={LOG_DIR}")
    print(f"   Checkpoints: {CHECKPOINT_DIR}")
    print("-" * 50)
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n⚠️  訓練被中斷，正在儲存模型...")
    
    # ==================== 儲存最終模型 ====================
    
    print("\n💾 儲存最終模型...")
    model.save(os.path.join(CHECKPOINT_DIR, "soccer_humanoid_final"))
    env.save(os.path.join(CHECKPOINT_DIR, "vec_normalize.pkl"))
    
    print("✅ 訓練完成！")
    print(f"   模型位置: {CHECKPOINT_DIR}")
    
    env.close()
    eval_env.close()


if __name__ == "__main__":
    train()