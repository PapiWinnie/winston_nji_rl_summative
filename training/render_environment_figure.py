"""One-off script: render a static figure of the redaction viewer (matching
environment/rendering.py's color scheme) for the report's Environment
Description section, from a real episode trace.
"""
import json

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

RISK_COLORS = {"low": "#2e7d32", "mid": "#b8860b", "high": "#c62828"}
ACTION_STYLES = {
    "FULL_REDACT": ("#111", "#111", "[REDACTED]", "#f2f2f2"),
    "PARTIAL_MASK": ("#666", "#eee", None, "#222"),
    "SUBSTITUTE": ("#3949ab", "#e8eaf6", "[PLACEHOLDER]", "#1a1a2e"),
    "NO_REDACT": (None, "#fafafa", None, "#222"),
}

with open("assets/environment_visual_trace.json", encoding="utf-8") as f:
    trace = json.load(f)

n = len(trace)
fig, ax = plt.subplots(figsize=(2.1 * n, 3.2))
cum_reward = sum(s["reward"] for s in trace)

for i, step in enumerate(trace):
    border, bg, label_override, text_color = ACTION_STYLES[step["action"]]
    border = border or RISK_COLORS[step["risk_tier"]]
    label = label_override or step["entity_type"]

    box = FancyBboxPatch(
        (i * 2.0 + 0.1, 0.1), 1.8, 2.6,
        boxstyle="round,pad=0.05,rounding_size=0.1",
        linewidth=2, edgecolor=border, facecolor=bg,
    )
    ax.add_patch(box)

    cx = i * 2.0 + 1.0
    ax.text(cx, 2.35, label, ha="center", va="center", fontsize=10, fontweight="bold", color=text_color, family="monospace")
    ax.text(cx, 1.98, f'{step["entity_type"]} ({step["language"]})', ha="center", va="center", fontsize=8, color=text_color, alpha=0.8, family="monospace")
    surface = step.get("surface_text")
    if surface:
        shown = surface if len(surface) <= 20 else surface[:18] + "…"
        ax.text(cx, 1.65, f'"{shown}"', ha="center", va="center", fontsize=7.5, style="italic", color=text_color, family="monospace")
    ax.text(cx, 1.15, f'risk {step["risk_score"]:.2f}', ha="center", va="center", fontsize=8, color=RISK_COLORS[step["risk_tier"]], family="monospace")
    ax.text(cx, 0.90, f'conf {step["confidence"]:.2f}', ha="center", va="center", fontsize=8, color=RISK_COLORS[step["risk_tier"]], family="monospace")
    ax.text(cx, 0.45, step["action"], ha="center", va="center", fontsize=8.5, color=text_color, family="monospace")

ax.set_xlim(0, n * 2.0)
ax.set_ylim(0, 3.0)
ax.axis("off")
ax.set_title(
    f"No BS Redaction Agent — live episode (best DQN policy)   |   cumulative reward: {cum_reward:.2f}",
    fontsize=11, family="monospace", pad=12,
)
fig.tight_layout()
fig.savefig("assets/environment_visual.png", dpi=200, bbox_inches="tight", facecolor="white")
print("wrote assets/environment_visual.png")
