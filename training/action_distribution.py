"""Evaluates each algorithm's best final model over many episodes and
reports what fraction of the time it picks each action, broken down by risk
tier. This is the quantitative evidence for "the agent explores all
possible actions, including edge cases" - rather than relying on a single
rendered episode as an anecdote.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import A2C, DQN, PPO

from environment.custom_env import ACTION_NAMES, RedactionEnv, risk_tier
from training.pg_training import PolicyNet, reinforce_predict

N_EPISODES = 100
TIERS = ["low", "mid", "high"]


def collect_action_counts(predict_fn, n_episodes=N_EPISODES, seed=9000):
    counts = {tier: {a: 0 for a in ACTION_NAMES} for tier in TIERS}
    env = RedactionEnv()

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        terminated = truncated = False
        while not (terminated or truncated):
            entity = info["entity"]
            action = predict_fn(obs)
            tier = risk_tier(entity["demographic_risk_score"])
            counts[tier][ACTION_NAMES[action]] += 1
            obs, _, terminated, truncated, info = env.step(action)

    return counts


def counts_to_frame(counts, algo_name):
    rows = []
    for tier in TIERS:
        total = sum(counts[tier].values())
        row = {"algorithm": algo_name, "risk_tier": tier, "n_entities": total}
        for a in ACTION_NAMES:
            row[a] = counts[tier][a] / total if total else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    dqn_model = DQN.load("models/dqn/final/best_model")
    ppo_model = PPO.load("models/ppo/final/best_model")
    a2c_model = A2C.load("models/a2c/final/best_model")

    obs_dim = RedactionEnv().observation_space.shape[0]
    n_actions = RedactionEnv().action_space.n
    reinforce_policy = PolicyNet(obs_dim, n_actions, hidden=64)
    reinforce_policy.load_state_dict(torch.load("models/pg/reinforce_final/best_model.pt"))

    predictors = {
        "DQN": lambda obs: int(dqn_model.predict(obs, deterministic=True)[0]),
        "PPO": lambda obs: int(ppo_model.predict(obs, deterministic=True)[0]),
        "A2C": lambda obs: int(a2c_model.predict(obs, deterministic=True)[0]),
        "REINFORCE": lambda obs: reinforce_predict(reinforce_policy, obs),
    }

    frames = []
    for algo_name, predict_fn in predictors.items():
        counts = collect_action_counts(predict_fn)
        frames.append(counts_to_frame(counts, algo_name))
        print(f"[{algo_name}] done")

    df = pd.concat(frames, ignore_index=True)
    df.to_csv("logs/action_distribution.csv", index=False)
    print(df.to_markdown(index=False))

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
    colors = {"FULL_REDACT": "#111", "PARTIAL_MASK": "#888", "SUBSTITUTE": "#3949ab", "NO_REDACT": "#2e7d32"}

    for ax, algo_name in zip(axes, predictors.keys()):
        sub = df[df["algorithm"] == algo_name].set_index("risk_tier").loc[TIERS]
        bottom = np.zeros(len(TIERS))
        for action in ACTION_NAMES:
            ax.bar(TIERS, sub[action], bottom=bottom, label=action, color=colors[action])
            bottom += sub[action].values
        ax.set_title(algo_name)
        ax.set_xlabel("risk tier")
        if ax is axes[0]:
            ax.set_ylabel("fraction of actions taken")

    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle(f"Action distribution by risk tier, per algorithm ({N_EPISODES} eval episodes)")
    fig.tight_layout()
    fig.savefig("assets/action_distribution.png", dpi=150)
    print("\nwrote logs/action_distribution.csv and assets/action_distribution.png")


if __name__ == "__main__":
    main()
