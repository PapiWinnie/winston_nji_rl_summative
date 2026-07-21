import json

RISK_COLORS = {"low": "#2e7d32", "mid": "#e6a817", "high": "#c62828"}

# (border color override, background color override, label override) per
# action. None means fall back to the risk-tier color / default background.
ACTION_STYLES = {
    "FULL_REDACT": ("#111", "#111", "[REDACTED]"),
    "PARTIAL_MASK": ("#666", "#eee", None),
    "SUBSTITUTE": ("#3949ab", "#e8eaf6", "[PLACEHOLDER]"),
    "NO_REDACT": (None, None, None),
}


def render_episode_html(episode_trace):
    chips = []
    cum_reward, risk_exposed, utility_kept = 0.0, 0.0, 0.0

    for step in episode_trace:
        cum_reward += step["reward"]
        if step["action"] == "NO_REDACT":
            risk_exposed += step["risk_score"]
            utility_kept += 1.0
        elif step["action"] == "PARTIAL_MASK":
            risk_exposed += step["risk_score"] * 0.4
            utility_kept += 0.5

        border, bg, label_override = ACTION_STYLES[step["action"]]
        border = border or RISK_COLORS[step["risk_tier"]]
        bg = bg or "#fafafa"
        label = label_override or step["entity_type"]

        chips.append(f'''
        <div style="display:inline-block;margin:4px;padding:8px 12px;
                    border:2px solid {border};border-radius:8px;background:{bg};
                    font-family:monospace;font-size:12px;min-width:110px;text-align:center;">
          <div style="font-weight:bold;">{label}</div>
          <div style="color:#555;">{step["entity_type"]} ({step["language"]})</div>
          <div style="color:{RISK_COLORS[step["risk_tier"]]};">risk {step["risk_score"]:.2f}, conf {step["confidence"]:.2f}</div>
          <div>{step["action"]}</div>
        </div>''')

    hud = f'''
    <div style="font-family:monospace;margin-bottom:10px;padding:8px;background:#f0f0f0;border-radius:6px;">
      <b>cumulative reward:</b> {cum_reward:.2f} &nbsp;|&nbsp;
      <b>risk exposed:</b> {risk_exposed:.2f} &nbsp;|&nbsp;
      <b>utility retained:</b> {utility_kept:.1f}/{len(episode_trace)}
    </div>'''

    return f'<div>{hud}<div>{"".join(chips)}</div></div>'


def render_episode_json(episode_trace):
    # Mirrors what the real No BS product would serialize for a frontend
    # after a redaction pass, directly usable as a mock API response.
    return json.dumps({"steps": episode_trace}, indent=2)
