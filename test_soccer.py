"""
test_soccer.py - 測試訓練好的足球機器人

使用方式：
    # 測試最終模型
    python test_soccer.py
    
    # 測試特定 checkpoint
    python test_soccer.py --model checkpoints/soccer/soccer_humanoid_500000_steps.zip
    
    # 測試隨機動作（不載入模型）
    python test_soccer.py --random
"""

import argparse
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from humanoid_soccer_env import HumanoidSoccerEnv


def test_model(model_path=None, vec_normalize_path=None, random_action=False, episodes=5):
    """
    測試模型
    
    Args:
        model_path: 模型檔案路徑
        vec_normalize_path: VecNormalize 統計檔案路徑
        random_action: 是否使用隨機動作（不載入模型）
        episodes: 測試回合數
    """
    
    # 建立環境
    print("📦 建立環境...")
    env = DummyVecEnv([lambda: HumanoidSoccerEnv(render_mode="human")])
    
    # 載入 VecNormalize
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        print(f"📊 載入正規化統計: {vec_normalize_path}")
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print("⚠️  未找到 VecNormalize 統計，使用原始觀察值")
    
    # 載入模型
    model = None
    if not random_action:
        if model_path and os.path.exists(model_path):
            print(f"🧠 載入模型: {model_path}")
            model = PPO.load(model_path)
        else:
            print("⚠️  未找到模型，使用隨機動作")
            random_action = True
    
    if random_action:
        print("🎲 使用隨機動作模式")
    
    # ==================== 開始測試 ====================
    
    print(f"\n🎮 開始測試 {episodes} 回合...")
    print("-" * 50)
    
    total_rewards = []
    goals_scored = 0
    
    for episode in range(episodes):
        obs = env.reset()
        episode_reward = 0
        step = 0
        done = False
        
        print(f"\n📍 Episode {episode + 1}")
        
        while not done:
            # 選擇動作
            if random_action:
                action = [env.action_space.sample()]
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            # 執行動作
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            step += 1
            
            # 每 100 步輸出一次資訊
            if step % 100 == 0:
                info_dict = info[0]
                print(f"   Step {step}: "
                      f"reward={reward[0]:.2f}, "
                      f"dist_to_ball={info_dict.get('distance_to_ball', 0):.2f}, "
                      f"ball_to_goal={info_dict.get('ball_to_goal_distance', 0):.2f}")
            
            # 檢查是否進球
            if info[0].get('goal_scored', False):
                print("   🎉 進球了！")
                goals_scored += 1
        
        total_rewards.append(episode_reward)
        print(f"   Episode {episode + 1} 結束: 總獎勵 = {episode_reward:.2f}, 步數 = {step}")
    
    # ==================== 統計結果 ====================
    
    print("\n" + "=" * 50)
    print("📊 測試結果統計")
    print("=" * 50)
    print(f"   測試回合數: {episodes}")
    print(f"   平均獎勵: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"   最高獎勵: {np.max(total_rewards):.2f}")
    print(f"   最低獎勵: {np.min(total_rewards):.2f}")
    print(f"   進球次數: {goals_scored} / {episodes}")
    print("=" * 50)
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description="測試足球機器人模型")
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/soccer/soccer_humanoid_final.zip",
        help="模型檔案路徑"
    )
    parser.add_argument(
        "--vec-normalize",
        type=str,
        default="checkpoints/soccer/vec_normalize.pkl",
        help="VecNormalize 統計檔案路徑"
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="使用隨機動作（不載入模型）"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="測試回合數"
    )
    
    args = parser.parse_args()
    
    test_model(
        model_path=args.model,
        vec_normalize_path=args.vec_normalize,
        random_action=args.random,
        episodes=args.episodes,
    )


if __name__ == "__main__":
    main()