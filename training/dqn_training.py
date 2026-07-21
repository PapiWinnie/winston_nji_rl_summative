from stable_baselines3 import DQN

from training.common import FINAL_TIMESTEPS, N_EVAL_EPISODES, SWEEP_TIMESTEPS, run_sb3_sweep, train_or_load_final_sb3

BASE_DIR = "."

# 12 runs, varying learning rate, gamma, buffer size, batch size, exploration
# fraction, and target update interval across the sweep.
dqn_configs = [
    dict(learning_rate=1e-3, gamma=0.99, buffer_size=5000, batch_size=32, exploration_fraction=0.3, target_update_interval=250),
    dict(learning_rate=1e-4, gamma=0.99, buffer_size=5000, batch_size=32, exploration_fraction=0.3, target_update_interval=250),
    dict(learning_rate=5e-4, gamma=0.95, buffer_size=5000, batch_size=64, exploration_fraction=0.3, target_update_interval=250),
    dict(learning_rate=5e-4, gamma=0.999, buffer_size=5000, batch_size=64, exploration_fraction=0.3, target_update_interval=250),
    dict(learning_rate=5e-4, gamma=0.99, buffer_size=1000, batch_size=32, exploration_fraction=0.3, target_update_interval=250),
    dict(learning_rate=5e-4, gamma=0.99, buffer_size=20000, batch_size=32, exploration_fraction=0.3, target_update_interval=250),
    dict(learning_rate=5e-4, gamma=0.99, buffer_size=5000, batch_size=128, exploration_fraction=0.3, target_update_interval=250),
    dict(learning_rate=5e-4, gamma=0.99, buffer_size=5000, batch_size=32, exploration_fraction=0.1, target_update_interval=250),
    dict(learning_rate=5e-4, gamma=0.99, buffer_size=5000, batch_size=32, exploration_fraction=0.6, target_update_interval=250),
    dict(learning_rate=5e-4, gamma=0.99, buffer_size=5000, batch_size=32, exploration_fraction=0.3, target_update_interval=50),
    dict(learning_rate=5e-4, gamma=0.99, buffer_size=5000, batch_size=32, exploration_fraction=0.3, target_update_interval=1000),
    dict(learning_rate=1e-3, gamma=0.98, buffer_size=10000, batch_size=64, exploration_fraction=0.2, target_update_interval=500),
]


def main():
    dqn_df, _, _, dqn_best_idx = run_sb3_sweep(DQN, "dqn", dqn_configs, SWEEP_TIMESTEPS, N_EVAL_EPISODES, BASE_DIR)
    print(dqn_df.to_markdown(index=False))
    print(f"\nbest DQN run: {dqn_best_idx}, config {dqn_configs[dqn_best_idx]}")

    train_or_load_final_sb3(DQN, "dqn", dqn_configs[dqn_best_idx], FINAL_TIMESTEPS, BASE_DIR)


if __name__ == "__main__":
    main()
