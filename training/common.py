import json
import os
import time

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from environment.custom_env import RedactionEnv

SEED = 42
SWEEP_TIMESTEPS = 8000
FINAL_TIMESTEPS = 25000
N_EVAL_EPISODES = 20


def append_row_csv(csv_path, row):
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)


class MetricsCallback(BaseCallback):
    """Pulls loss and entropy straight out of SB3's internal logger during
    training, so DQN/PPO/A2C runs can feed the same comparison plots as the
    custom REINFORCE loop."""

    def __init__(self):
        super().__init__()
        self.losses = []
        self.entropies = []

    def _on_step(self):
        name_to_value = self.model.logger.name_to_value
        if "train/loss" in name_to_value:
            self.losses.append(name_to_value["train/loss"])
        if "train/entropy_loss" in name_to_value:
            self.entropies.append(-name_to_value["train/entropy_loss"])
        return True


def eval_sb3_model(model, n_episodes, seed, env_cls=RedactionEnv):
    eval_env = env_cls(seed=seed)
    rewards = []
    for ep in range(n_episodes):
        obs, _ = eval_env.reset(seed=seed * 1000 + ep)
        done = trunc = False
        ep_r = 0.0
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, _ = eval_env.step(int(action))
            ep_r += r
        rewards.append(ep_r)
    return float(np.mean(rewards)), rewards


def eval_reinforce_policy(policy, reinforce_predict, n_episodes, seed, env_cls=RedactionEnv):
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


def convergence_episode(rewards, window=20):
    # First episode index where a rolling mean reward reaches 80% of its own
    # final value, a simple proxy for "roughly stopped improving."
    if len(rewards) < window:
        return len(rewards)
    final_mean = np.mean(rewards[-window:])
    threshold = 0.8 * final_mean
    roll = pd.Series(rewards).rolling(window).mean()
    hit = roll[roll >= threshold].index
    return int(hit[0]) if len(hit) > 0 else len(rewards)


def run_sb3_sweep(algo_cls, algo_name, configs, total_timesteps, n_eval_episodes, base_dir, seed=SEED):
    model_dir = f"{base_dir}/models/{algo_name.lower()}/sweep"
    hist_dir = f"{base_dir}/logs/{algo_name.lower()}_sweep_histories"
    csv_path = f"{base_dir}/logs/{algo_name.lower()}_hparam_results.csv"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(hist_dir, exist_ok=True)

    done_df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()

    rows, histories, models = [], {}, {}

    for i, cfg in enumerate(configs):
        model_path = f"{model_dir}/run_{i}"
        hist_path = f"{hist_dir}/run_{i}.json"
        already_done = (
            os.path.exists(model_path + ".zip")
            and os.path.exists(hist_path)
            and not done_df.empty
            and i in done_df["run"].values
        )

        if already_done:
            print(f"[{algo_name}] run {i} found on disk, loading instead of retraining")
            model = algo_cls.load(model_path)
            with open(hist_path) as f:
                hist = json.load(f)
            row = done_df[done_df["run"] == i].iloc[0].to_dict()
        else:
            print(f"[{algo_name}] run {i}/{len(configs) - 1} training")
            run_seed = seed + i
            env = Monitor(RedactionEnv(seed=run_seed))
            t0 = time.time()
            model = algo_cls("MlpPolicy", env, verbose=0, seed=run_seed, **cfg)
            cb = MetricsCallback()
            model.learn(total_timesteps=total_timesteps, callback=cb)
            train_time = time.time() - t0

            rewards = env.get_episode_rewards()
            mean_eval_reward, _ = eval_sb3_model(model, n_eval_episodes, seed=2000 + run_seed)
            final_mean = float(np.mean(rewards[-20:])) if len(rewards) >= 20 else float(np.mean(rewards))

            row = dict(cfg)
            row.update({
                "run": i,
                "mean_eval_reward": mean_eval_reward,
                "final_train_reward_mean20": final_mean,
                "convergence_episode": convergence_episode(rewards),
                "train_time_sec": round(train_time, 2),
            })
            hist = {"episode_rewards": rewards, "losses": cb.losses, "entropies": cb.entropies}

            model.save(model_path)
            with open(hist_path, "w") as f:
                json.dump(hist, f)
            append_row_csv(csv_path, row)

        rows.append(row)
        histories[i] = hist
        models[i] = model

    df = pd.DataFrame(rows)
    best_idx = int(df["mean_eval_reward"].idxmax())
    return df, histories, models, best_idx


def train_or_load_final_sb3(algo_cls, algo_name, cfg, total_timesteps, base_dir, seed=SEED, force_retrain=False):
    final_dir = f"{base_dir}/models/{algo_name}/final"
    os.makedirs(final_dir, exist_ok=True)
    model_path = f"{final_dir}/best_model"
    hist_path = f"{final_dir}/history.json"

    if not force_retrain and os.path.exists(model_path + ".zip") and os.path.exists(hist_path):
        print(f"[{algo_name}] final model already trained, loading from disk")
        model = algo_cls.load(model_path)
        with open(hist_path) as f:
            hist = json.load(f)
        return model, hist["episode_rewards"], hist["losses"], hist["entropies"]

    env = Monitor(RedactionEnv(seed=seed))
    cb = MetricsCallback()
    model = algo_cls("MlpPolicy", env, verbose=1, seed=seed, **cfg)
    model.learn(total_timesteps=total_timesteps, callback=cb)
    rewards = env.get_episode_rewards()

    model.save(model_path)
    with open(hist_path, "w") as f:
        json.dump({"episode_rewards": rewards, "losses": cb.losses, "entropies": cb.entropies}, f)

    return model, rewards, cb.losses, cb.entropies
