# No BS: Adaptive Redaction Policy Agent (RL Summative)

A custom Gymnasium environment modeled on **No BS (No Bias Screening)** — a bilingual French/English tool that redacts demographic identifiers from job application documents in Yaounde's formal sector to reduce identity-based hiring bias. The RL agent walks a document's detected entities one at a time and chooses how aggressively to redact each one (`FULL_REDACT`, `PARTIAL_MASK`, `SUBSTITUTE`, `NO_REDACT`), trading off demographic-signal leakage against recruiter-utility loss.

Four algorithms are compared on the identical environment: **DQN**, **REINFORCE** (custom PyTorch implementation), **PPO**, and **A2C** (all via `stable_baselines3` except REINFORCE).

## Quickstart

```bash
uv sync
uv run main.py
```

This loads the best-performing agent (DQN) and plays it against a fresh synthetic document, printing verbose per-step terminal output and opening a live-updating HTML redaction viewer in your browser.

## Layout

- `environment/` — the `RedactionEnv` Gymnasium environment and the web-based document redaction renderer.
- `training/` — training/hyperparameter-sweep scripts for all four algorithms (`dqn_training.py`, `pg_training.py`, shared helpers in `common.py`).
- `models/`, `logs/`, `assets/` — trained models, hyperparameter sweep results (12 runs per algorithm), and the required comparison plots.
- `notebooks/` — the Google Colab notebook used to run the experiments on a T4 GPU.
- `main.py` — plays back the best-performing agent with live rendering, for the video demo.
