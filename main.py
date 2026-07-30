import argparse
import sys
import time
import webbrowser
from pathlib import Path

# Windows' default console codepage isn't UTF-8, so accented characters
# (French names/institutions) print as mangled bytes unless stdout is
# forced to encode as UTF-8 explicitly.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from stable_baselines3 import DQN

from environment.adapter import entities_from_nlp_pipeline
from environment.custom_env import ACTION_NAMES, COMPLETION_BONUS, REWARD, STEP_EFFICIENCY_BONUS, RedactionEnv, risk_tier
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


def run_real_text(model, env, entities, source_label, step_delay, refresh_seconds):
    """Same per-entity loop as run_episode, but driven by entities extracted
    from real text (see environment/adapter.py) instead of a synthetic
    document. There's no gym env to step through here, so the reward is
    computed directly from the same REWARD table env.step() uses - not a
    training signal, just showing what score this document would earn."""
    print(f"\n=== Processing '{source_label}' | {len(entities)} entities ===")

    cumulative_reward = 0.0
    trace = []

    for step, entity in enumerate(entities, start=1):
        obs = env._obs_for(entity)
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        tier = risk_tier(entity["demographic_risk_score"])
        reward = REWARD[tier][action] + STEP_EFFICIENCY_BONUS
        if step == len(entities):
            reward += COMPLETION_BONUS
        cumulative_reward += reward

        trace.append({
            "entity_type": entity["entity_type"], "language": entity["language"],
            "risk_score": entity["demographic_risk_score"], "confidence": entity["ner_confidence"],
            "risk_tier": tier, "action": ACTION_NAMES[action], "reward": reward,
            "surface_text": entity.get("surface_text"),
        })

        print(
            f"  step {step:2d} | {entity['entity_type']:<10} ({entity['language']}) "
            f"\"{entity.get('surface_text', '')}\" "
            f"risk={entity['demographic_risk_score']:.2f} conf={entity['ner_confidence']:.2f} "
            f"-> {ACTION_NAMES[action]:<12} reward={reward:+.2f} cum_reward={cumulative_reward:+.2f}"
        )

        render_frame(trace, refresh_seconds)
        time.sleep(step_delay)

    print(f"Done: {len(entities)} entities processed, total reward = {cumulative_reward:.2f}")
    return cumulative_reward


def main():
    parser = argparse.ArgumentParser(description="Run the best-performing No BS redaction agent (DQN) and render it live.")
    parser.add_argument("--episodes", type=int, default=3, help="number of documents to process")
    parser.add_argument("--delay", type=float, default=1.5, help="seconds to pause between entities, for watchability")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="path to a trained SB3 model .zip")
    parser.add_argument("--no-browser", action="store_true", help="skip auto-opening the render in a browser")
    parser.add_argument("--text-file", type=Path, default=None,
                         help="process a real document instead of synthetic ones, e.g. assets/sample_application.txt")
    parser.add_argument("--language", choices=["EN", "FR"], default="FR", help="language flag for --text-file mode")
    args = parser.parse_args()

    print(f"Loading best-performing agent from {args.model}")
    model = DQN.load(args.model)
    env = RedactionEnv()

    refresh_seconds = max(1, round(args.delay))
    render_frame([], refresh_seconds)
    if not args.no_browser:
        webbrowser.open(RENDER_PATH.resolve().as_uri())
        time.sleep(2)

    if args.text_file:
        text = args.text_file.read_text(encoding="utf-8")
        entities = entities_from_nlp_pipeline(text, language=args.language)
        run_real_text(model, env, entities, args.text_file.name, args.delay, refresh_seconds)
    else:
        rewards = [
            run_episode(model, env, ep, args.delay, refresh_seconds)
            for ep in range(1, args.episodes + 1)
        ]
        print(f"\nAll episodes complete. Mean reward over {args.episodes} episodes: {sum(rewards) / len(rewards):.2f}")

    print(f"Final render available at: {RENDER_PATH.resolve()}")


if __name__ == "__main__":
    main()
