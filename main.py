import argparse
import time
import webbrowser
from pathlib import Path

from stable_baselines3 import DQN

from environment.custom_env import ACTION_NAMES, RedactionEnv
from environment.rendering import render_episode_html

MODEL_PATH = Path("models/dqn/final/best_model.zip")
RENDER_PATH = Path("assets/live_render.html")


def render_frame(trace, refresh_seconds):
    body = render_episode_html(trace) if trace else "<p>Waiting for first entity...</p>"
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="{refresh_seconds}">
<title>No BS - Redaction Agent</title></head>
<body style="font-family:monospace;padding:20px;background:#fff;">
<h2>No BS Redaction Agent &mdash; live episode (best DQN policy)</h2>
{body}
</body></html>"""
    RENDER_PATH.write_text(html, encoding="utf-8")


def run_episode(model, env, episode_num, step_delay, refresh_seconds):
    obs, info = env.reset()
    print(f"\n=== Episode {episode_num} | {len(env.document)} entities | language={env.document[0]['language']} ===")

    terminated = truncated = False
    cumulative_reward = 0.0
    step = 0

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        entity = info["entity"]

        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward
        step += 1

        print(
            f"  step {step:2d} | {entity['entity_type']:<10} ({entity['language']}) "
            f"risk={entity['demographic_risk_score']:.2f} conf={entity['ner_confidence']:.2f} "
            f"-> {ACTION_NAMES[action]:<12} reward={reward:+.2f} cum_reward={cumulative_reward:+.2f}"
        )

        render_frame(env.episode_trace, refresh_seconds)
        time.sleep(step_delay)

    print(f"Episode {episode_num} done: {step} entities processed, total reward = {cumulative_reward:.2f}")
    return cumulative_reward


def main():
    parser = argparse.ArgumentParser(description="Run the best-performing No BS redaction agent (DQN) and render it live.")
    parser.add_argument("--episodes", type=int, default=3, help="number of documents to process")
    parser.add_argument("--delay", type=float, default=1.5, help="seconds to pause between entities, for watchability")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="path to a trained SB3 model .zip")
    parser.add_argument("--no-browser", action="store_true", help="skip auto-opening the render in a browser")
    args = parser.parse_args()

    print(f"Loading best-performing agent from {args.model}")
    model = DQN.load(args.model)
    env = RedactionEnv()

    refresh_seconds = max(1, round(args.delay))
    render_frame([], refresh_seconds)
    if not args.no_browser:
        webbrowser.open(RENDER_PATH.resolve().as_uri())
        time.sleep(2)

    rewards = [
        run_episode(model, env, ep, args.delay, refresh_seconds)
        for ep in range(1, args.episodes + 1)
    ]

    print(f"\nAll episodes complete. Mean reward over {args.episodes} episodes: {sum(rewards) / len(rewards):.2f}")
    print(f"Final render available at: {RENDER_PATH.resolve()}")


if __name__ == "__main__":
    main()
