"""One-off script: rebuild the plots/generalization test that were corrupted
by the PPO/A2C final-model folder collision (see training/pg_training.py),
using the real, independently trained A2C final model. Not part of the
regular training pipeline.
"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import A2C, DQN, PPO

from environment.custom_env import RedactionEnv, ShiftedRedactionEnv
from training.common import N_EVAL_EPISODES, convergence_episode, eval_sb3_model
from training.pg_training import PolicyNet, eval_reinforce_policy


def smooth(x, window=10):
    if len(x) < window:
        return x
    return pd.Series(x).rolling(window, min_periods=1).mean().values


def load_history(path):
    with open(path) as f:
        return json.load(f)


dqn_hist = load_history("models/dqn/final/history.json")
ppo_hist = load_history("models/ppo/final/history.json")
a2c_hist = load_history("models/a2c/final/history.json")
reinforce_hist = load_history("models/pg/reinforce_final/history.json")

dqn_final_rewards = dqn_hist["episode_rewards"]
ppo_final_rewards = ppo_hist["episode_rewards"]
a2c_final_rewards = a2c_hist["episode_rewards"]
reinforce_final_rewards = reinforce_hist["episode_reward"]

dqn_final_losses = dqn_hist["losses"]
ppo_final_entropies = ppo_hist["entropies"]
a2c_final_entropies = a2c_hist["entropies"]
reinforce_entropies = reinforce_hist["entropy"]

# --- cumulative reward curves (all four, subplots) ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
plot_data = [
    ("DQN", dqn_final_rewards, axes[0, 0]),
    ("REINFORCE", reinforce_final_rewards, axes[0, 1]),
    ("PPO", ppo_final_rewards, axes[1, 0]),
    ("A2C", a2c_final_rewards, axes[1, 1]),
]
for name, rewards, ax in plot_data:
    ax.plot(smooth(rewards), color="#1f77b4")
    ax.set_title(f"{name}: cumulative reward, smoothed")
    ax.set_xlabel("episode")
    ax.set_ylabel("episode reward")
fig.tight_layout()
fig.savefig("assets/cumulative_reward_curves.png", dpi=150)
plt.close(fig)

# --- entropy curves (REINFORCE, PPO, A2C) ---
plt.figure(figsize=(8, 4))
plt.plot(smooth(reinforce_entropies), label="REINFORCE")
plt.plot(smooth(ppo_final_entropies), label="PPO")
plt.plot(smooth(a2c_final_entropies), label="A2C")
plt.title("Policy entropy over training")
plt.xlabel("logged step or episode")
plt.ylabel("entropy")
plt.legend()
plt.tight_layout()
plt.savefig("assets/entropy_curves.png", dpi=150)
plt.close()

# --- convergence comparison ---
conv_data = {
    "DQN": convergence_episode(dqn_final_rewards),
    "REINFORCE": convergence_episode(reinforce_final_rewards),
    "PPO": convergence_episode(ppo_final_rewards),
    "A2C": convergence_episode(a2c_final_rewards),
}
plt.figure(figsize=(7, 4))
plt.bar(conv_data.keys(), conv_data.values(), color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
plt.title("Episodes to convergence (80% of final rolling mean reward)")
plt.ylabel("episode")
plt.tight_layout()
plt.savefig("assets/convergence_comparison.png", dpi=150)
plt.close()
print("convergence episodes:", conv_data)

# --- generalization test: real A2C model this time ---
final_dqn = DQN.load("models/dqn/final/best_model")
final_ppo = PPO.load("models/ppo/final/best_model")
final_a2c = A2C.load("models/a2c/final/best_model")

reinforce_obs_dim = RedactionEnv().observation_space.shape[0]
reinforce_n_actions = RedactionEnv().action_space.n
final_reinforce_policy = PolicyNet(reinforce_obs_dim, reinforce_n_actions, hidden=64)
final_reinforce_policy.load_state_dict(torch.load("models/pg/reinforce_final/best_model.pt"))

id_dqn, _ = eval_sb3_model(final_dqn, N_EVAL_EPISODES, seed=5000)
id_ppo, _ = eval_sb3_model(final_ppo, N_EVAL_EPISODES, seed=5001)
id_a2c, _ = eval_sb3_model(final_a2c, N_EVAL_EPISODES, seed=5002)
id_reinforce, _ = eval_reinforce_policy(final_reinforce_policy, N_EVAL_EPISODES, seed=5003)

shift_dqn, _ = eval_sb3_model(final_dqn, N_EVAL_EPISODES, seed=6000, env_cls=ShiftedRedactionEnv)
shift_ppo, _ = eval_sb3_model(final_ppo, N_EVAL_EPISODES, seed=6001, env_cls=ShiftedRedactionEnv)
shift_a2c, _ = eval_sb3_model(final_a2c, N_EVAL_EPISODES, seed=6002, env_cls=ShiftedRedactionEnv)
shift_reinforce, _ = eval_reinforce_policy(final_reinforce_policy, N_EVAL_EPISODES, seed=6003, env_cls=ShiftedRedactionEnv)

algos = ["DQN", "REINFORCE", "PPO", "A2C"]
in_dist = [id_dqn, id_reinforce, id_ppo, id_a2c]
shifted = [shift_dqn, shift_reinforce, shift_ppo, shift_a2c]

x = np.arange(len(algos))
width = 0.35
plt.figure(figsize=(8, 4.5))
plt.bar(x - width / 2, in_dist, width, label="training distribution")
plt.bar(x + width / 2, shifted, width, label="shifted, held-out distribution")
plt.xticks(x, algos)
plt.ylabel("mean episode reward")
plt.title("Generalization: training distribution vs shifted distribution")
plt.legend()
plt.tight_layout()
plt.savefig("assets/generalization_test.png", dpi=150)
plt.close()

gen_df = pd.DataFrame({"algorithm": algos, "in_distribution": in_dist, "shifted": shifted})
print(gen_df.to_markdown(index=False))
gen_df.to_csv("logs/generalization_test.csv", index=False)

ranked = sorted(zip(algos, in_dist), key=lambda kv: kv[1], reverse=True)
print("\nranked by in-distribution eval reward:")
for name, score in ranked:
    print(f"  {name:10s} {score:.3f}")
