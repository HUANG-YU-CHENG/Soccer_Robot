"""
train_soccer.py - 訓練人形機器人踢足球

使用 PPO 演算法訓練機器人：
1. 走向足球
2. 踢球向球門

使用方式：
    python train_soccer.py
    
    # 調整並行環境數量
    python train_soccer.py --n_envs 2
    
    # 調整訓練步數
    python train_soccer.py --timesteps 500000
"""

import os
import sys
import argparse
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

# 導入自定義環境
from humanoid_soccer_env import HumanoidSoccerEnv


class ProgressCallback(BaseCallback):
    """自定義進度回調，避免卡住問題"""
    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # 強制刷新輸出
            print(f"Step: {self.num_timesteps}", flush=True)
            sys.stdout.flush()
        return True


def make_env():
    """建立單一環境"""
    def _init():
        env = HumanoidSoccerEnv()
        env = Monitor(env)
        return env
    return _init


def train(n_envs=4, total_timesteps=1_000_000):
    """主訓練函數"""
    
    # ==================== 設定 ====================
    SAVE_FREQ = 50_000
    
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
    
    print(f"\n📦 建立 {n_envs} 個環境 (DummyVecEnv)...")
    
    # 使用 DummyVecEnv（Windows 較穩定）
    env = DummyVecEnv([make_env() for _ in range(n_envs)])
    
    # 正規化觀察值和獎勵
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
        # PPO 超參數
        learning_rate=3e-4,
        n_steps=1024,        # 減少以加快更新
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        # 神經網路設定
        policy_kwargs={
            "net_arch": dict(pi=[256, 256], vf=[256, 256]),
        },
    )
    
    print(f"   模型參數量: {sum(p.numel() for p in model.policy.parameters()):,}")
    
    # ==================== 設定 Callbacks ====================
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(SAVE_FREQ // n_envs, 1000),
        save_path=CHECKPOINT_DIR,
        name_prefix="soccer_humanoid",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    # 進度回調
    progress_callback = ProgressCallback(check_freq=5000)
    
    # ==================== 開始訓練 ====================
    
    print(f"\n🚀 開始訓練 {total_timesteps:,} 步...")
    print(f"   TensorBoard: tensorboard --logdir={LOG_DIR}")
    print(f"   Checkpoints: {CHECKPOINT_DIR}")
    print("-" * 50)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, progress_callback],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n⚠️  訓練被中斷，正在儲存模型...")
    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")
        print("正在儲存目前的模型...")
    
    # ==================== 儲存最終模型 ====================
    
    print("\n💾 儲存最終模型...")
    model.save(os.path.join(CHECKPOINT_DIR, "soccer_humanoid_final"))
    env.save(os.path.join(CHECKPOINT_DIR, "vec_normalize.pkl"))
    
    print("✅ 訓練完成！")
    print(f"   模型位置: {CHECKPOINT_DIR}")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="訓練足球機器人")
    parser.add_argument("--n_envs", type=int, default=2, help="並行環境數量 (預設: 2)")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="訓練步數 (預設: 1000000)")
    args = parser.parse_args()
    
    train(n_envs=args.n_envs, total_timesteps=args.timesteps)