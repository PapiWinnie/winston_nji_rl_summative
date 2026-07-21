import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import A2C, PPO

from environment.custom_env import RedactionEnv
from training.common import (
    FINAL_TIMESTEPS,
    N_EVAL_EPISODES,
    SEED,
    SWEEP_TIMESTEPS,
    append_row_csv,
    convergence_episode,
    run_sb3_sweep,
    train_or_load_final_sb3,
)

BASE_DIR = "."


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.action_head = nn.Linear(hidden, n_actions)

    def forward(self, x):
        return self.action_head(self.body(x))


class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_reinforce(env, total_timesteps, learning_rate=3e-4, gamma=0.99,
                     use_baseline=True, hidden=64, seed=0, verbose=False, log_every=50):
    torch.manual_seed(seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = PolicyNet(obs_dim, n_actions, hidden)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    value_net = ValueNet(obs_dim, hidden) if use_baseline else None
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=learning_rate) if use_baseline else None

    history = {"episode_reward": [], "episode_length": [], "policy_loss": [], "entropy": []}
    steps_done, episode = 0, 0

    while steps_done < total_timesteps:
        obs, _ = env.reset(seed=seed + episode)
        log_probs, entropies, rewards, values = [], [], [], []
        terminated = truncated = False

        while not (terminated or truncated):
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = policy(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            if use_baseline:
                values.append(value_net(obs_t))
            obs, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            steps_done += 1

        returns, G = [], 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        log_probs_t = torch.cat(log_probs)
        entropy_t = torch.cat(entropies).mean()

        if use_baseline:
            values_t = torch.cat(values)
            advantages = returns - values_t.detach()
            value_loss = F.mse_loss(values_t, returns)
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()
        else:
            advantages = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = -(log_probs_t * advantages).mean() - 0.01 * entropy_t
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        history["episode_reward"].append(sum(rewards))
        history["episode_length"].append(len(rewards))
        history["policy_loss"].append(policy_loss.item())
        history["entropy"].append(entropy_t.item())
        episode += 1

        if verbose and episode % log_every == 0:
            recent_mean = np.mean(history["episode_reward"][-log_every:])
            print(f"episode {episode}, steps {steps_done}, mean reward (last {log_every}) {recent_mean:.3f}")

    return policy, history


def reinforce_predict(policy, obs):
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = policy(obs_t)
    return int(torch.argmax(logits, dim=-1).item())


def eval_reinforce_policy(policy, n_episodes, seed, env_cls=RedactionEnv):
    eval_env = env_cls(seed=seed)
    rewards = []
    for ep in range(n_episodes):
        obs, _ = eval_env.reset(seed=seed * 1000 + ep)
        done = trunc = False
        ep_r = 0.0
        while not (done or trunc):
            action = reinforce_predict(policy, obs)
            obs, r, done, trunc, _ = eval_env.step(action)
            ep_r += r
        rewards.append(ep_r)
    return float(np.mean(rewards)), rewards


def run_reinforce_sweep(configs, total_timesteps, n_eval_episodes, base_dir=BASE_DIR, seed=SEED):
    model_dir = f"{base_dir}/models/pg/reinforce_sweep"
    hist_dir = f"{base_dir}/logs/reinforce_sweep_histories"
    csv_path = f"{base_dir}/logs/reinforce_hparam_results.csv"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(hist_dir, exist_ok=True)

    done_df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()

    obs_dim = RedactionEnv().observation_space.shape[0]
    n_actions = RedactionEnv().action_space.n

    rows, histories, models = [], {}, {}

    for i, cfg in enumerate(configs):
        model_path = f"{model_dir}/run_{i}.pt"
        hist_path = f"{hist_dir}/run_{i}.json"
        already_done = (
            os.path.exists(model_path)
            and os.path.exists(hist_path)
            and not done_df.empty
            and i in done_df["run"].values
        )

        if already_done:
            print(f"[REINFORCE] run {i} found on disk, loading instead of retraining")
            policy = PolicyNet(obs_dim, n_actions, cfg.get("hidden", 64))
            policy.load_state_dict(torch.load(model_path))
            with open(hist_path) as f:
                hist = json.load(f)
            row = done_df[done_df["run"] == i].iloc[0].to_dict()
        else:
            print(f"[REINFORCE] run {i}/{len(configs) - 1} training")
            run_seed = seed + i
            env = RedactionEnv(seed=run_seed)
            t0 = time.time()
            policy, hist = train_reinforce(env, total_timesteps=total_timesteps, seed=run_seed, verbose=False, **cfg)
            train_time = time.time() - t0

            rewards = hist["episode_reward"]
            mean_eval_reward, _ = eval_reinforce_policy(policy, n_eval_episodes, seed=2000 + run_seed)
            final_mean = float(np.mean(rewards[-20:])) if len(rewards) >= 20 else float(np.mean(rewards))

            row = dict(cfg)
            row.update({
                "run": i,
                "mean_eval_reward": mean_eval_reward,
                "final_train_reward_mean20": final_mean,
                "convergence_episode": convergence_episode(rewards),
                "train_time_sec": round(train_time, 2),
            })

            torch.save(policy.state_dict(), model_path)
            with open(hist_path, "w") as f:
                json.dump(hist, f)
            append_row_csv(csv_path, row)

        rows.append(row)
        histories[i] = hist
        models[i] = policy

    df = pd.DataFrame(rows)
    best_idx = int(df["mean_eval_reward"].idxmax())
    return df, histories, models, best_idx


def train_or_load_final_reinforce(cfg, total_timesteps, base_dir=BASE_DIR, seed=SEED, force_retrain=False):
    final_dir = f"{base_dir}/models/pg/reinforce_final"
    os.makedirs(final_dir, exist_ok=True)
    model_path = f"{final_dir}/best_model.pt"
    hist_path = f"{final_dir}/history.json"

    obs_dim = RedactionEnv().observation_space.shape[0]
    n_actions = RedactionEnv().action_space.n

    if not force_retrain and os.path.exists(model_path) and os.path.exists(hist_path):
        print("[reinforce] final model already trained, loading from disk")
        policy = PolicyNet(obs_dim, n_actions, cfg.get("hidden", 64))
        policy.load_state_dict(torch.load(model_path))
        with open(hist_path) as f:
            hist = json.load(f)
        return policy, hist

    policy, hist = train_reinforce(RedactionEnv(seed=seed), total_timesteps=total_timesteps, seed=seed, verbose=True, log_every=100, **cfg)
    torch.save(policy.state_dict(), model_path)
    with open(hist_path, "w") as f:
        json.dump(hist, f)
    return policy, hist


# 12 runs each, varying at least 3 hyperparameters, matching the DQN sweep in
# training/dqn_training.py so all four algorithms get comparable tuning depth.
reinforce_configs = [
    dict(learning_rate=1e-3, gamma=0.99, use_baseline=True, hidden=64),
    dict(learning_rate=1e-4, gamma=0.99, use_baseline=True, hidden=64),
    dict(learning_rate=5e-4, gamma=0.95, use_baseline=True, hidden=64),
    dict(learning_rate=5e-4, gamma=0.999, use_baseline=True, hidden=64),
    dict(learning_rate=5e-4, gamma=0.99, use_baseline=False, hidden=64),
    dict(learning_rate=5e-4, gamma=0.99, use_baseline=True, hidden=32),
    dict(learning_rate=5e-4, gamma=0.99, use_baseline=True, hidden=128),
    dict(learning_rate=1e-3, gamma=0.95, use_baseline=False, hidden=32),
    dict(learning_rate=1e-3, gamma=0.999, use_baseline=True, hidden=128),
    dict(learning_rate=2e-4, gamma=0.97, use_baseline=True, hidden=64),
    dict(learning_rate=5e-4, gamma=0.90, use_baseline=True, hidden=64),
    dict(learning_rate=1e-3, gamma=0.99, use_baseline=True, hidden=128),
]

ppo_configs = [
    dict(learning_rate=3e-4, gamma=0.99, n_steps=128, batch_size=32, clip_range=0.2, ent_coef=0.0, gae_lambda=0.95),
    dict(learning_rate=1e-4, gamma=0.99, n_steps=128, batch_size=32, clip_range=0.2, ent_coef=0.0, gae_lambda=0.95),
    dict(learning_rate=1e-3, gamma=0.99, n_steps=128, batch_size=32, clip_range=0.2, ent_coef=0.0, gae_lambda=0.95),
    dict(learning_rate=3e-4, gamma=0.95, n_steps=128, batch_size=32, clip_range=0.2, ent_coef=0.0, gae_lambda=0.95),
    dict(learning_rate=3e-4, gamma=0.999, n_steps=128, batch_size=32, clip_range=0.2, ent_coef=0.0, gae_lambda=0.95),
    dict(learning_rate=3e-4, gamma=0.99, n_steps=64, batch_size=32, clip_range=0.2, ent_coef=0.0, gae_lambda=0.95),
    dict(learning_rate=3e-4, gamma=0.99, n_steps=256, batch_size=64, clip_range=0.2, ent_coef=0.0, gae_lambda=0.95),
    dict(learning_rate=3e-4, gamma=0.99, n_steps=128, batch_size=32, clip_range=0.1, ent_coef=0.0, gae_lambda=0.95),
    dict(learning_rate=3e-4, gamma=0.99, n_steps=128, batch_size=32, clip_range=0.3, ent_coef=0.0, gae_lambda=0.95),
    dict(learning_rate=3e-4, gamma=0.99, n_steps=128, batch_size=32, clip_range=0.2, ent_coef=0.01, gae_lambda=0.95),
    dict(learning_rate=3e-4, gamma=0.99, n_steps=128, batch_size=32, clip_range=0.2, ent_coef=0.05, gae_lambda=0.95),
    dict(learning_rate=3e-4, gamma=0.99, n_steps=128, batch_size=32, clip_range=0.2, ent_coef=0.0, gae_lambda=0.80),
]

a2c_configs = [
    dict(learning_rate=7e-4, gamma=0.99, n_steps=5, ent_coef=0.0, vf_coef=0.5),
    dict(learning_rate=1e-4, gamma=0.99, n_steps=5, ent_coef=0.0, vf_coef=0.5),
    dict(learning_rate=1e-3, gamma=0.99, n_steps=5, ent_coef=0.0, vf_coef=0.5),
    dict(learning_rate=7e-4, gamma=0.95, n_steps=5, ent_coef=0.0, vf_coef=0.5),
    dict(learning_rate=7e-4, gamma=0.999, n_steps=5, ent_coef=0.0, vf_coef=0.5),
    dict(learning_rate=7e-4, gamma=0.99, n_steps=8, ent_coef=0.0, vf_coef=0.5),
    dict(learning_rate=7e-4, gamma=0.99, n_steps=20, ent_coef=0.0, vf_coef=0.5),
    dict(learning_rate=7e-4, gamma=0.99, n_steps=5, ent_coef=0.01, vf_coef=0.5),
    dict(learning_rate=7e-4, gamma=0.99, n_steps=5, ent_coef=0.05, vf_coef=0.5),
    dict(learning_rate=7e-4, gamma=0.99, n_steps=5, ent_coef=0.0, vf_coef=0.25),
    dict(learning_rate=7e-4, gamma=0.99, n_steps=5, ent_coef=0.0, vf_coef=0.9),
    dict(learning_rate=7e-4, gamma=0.98, n_steps=10, ent_coef=0.02, vf_coef=0.5),
]


def main():
    reinforce_df, _, _, reinforce_best_idx = run_reinforce_sweep(reinforce_configs, SWEEP_TIMESTEPS, N_EVAL_EPISODES, BASE_DIR)
    print(reinforce_df.to_markdown(index=False))
    print(f"\nbest REINFORCE run: {reinforce_best_idx}, config {reinforce_configs[reinforce_best_idx]}")

    ppo_df, _, _, ppo_best_idx = run_sb3_sweep(PPO, "ppo", ppo_configs, SWEEP_TIMESTEPS, N_EVAL_EPISODES, BASE_DIR)
    print(ppo_df.to_markdown(index=False))
    print(f"\nbest PPO run: {ppo_best_idx}, config {ppo_configs[ppo_best_idx]}")

    a2c_df, _, _, a2c_best_idx = run_sb3_sweep(A2C, "a2c", a2c_configs, SWEEP_TIMESTEPS, N_EVAL_EPISODES, BASE_DIR)
    print(a2c_df.to_markdown(index=False))
    print(f"\nbest A2C run: {a2c_best_idx}, config {a2c_configs[a2c_best_idx]}")

    # Each algorithm gets its own final-model folder (models/ppo/final,
    # models/a2c/final) — the original notebook used the shared name "pg" for
    # both PPO's and A2C's final run, which meant A2C's call silently reloaded
    # PPO's already-saved model instead of training its own. Fixed here.
    train_or_load_final_sb3(PPO, "ppo", ppo_configs[ppo_best_idx], FINAL_TIMESTEPS, BASE_DIR)
    train_or_load_final_sb3(A2C, "a2c", a2c_configs[a2c_best_idx], FINAL_TIMESTEPS, BASE_DIR)
    train_or_load_final_reinforce(reinforce_configs[reinforce_best_idx], FINAL_TIMESTEPS, BASE_DIR)


if __name__ == "__main__":
    main()
