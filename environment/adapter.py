"""Production-pipeline integration boundary.

`entities_from_nlp_pipeline` is the literal swap-in point between a real
No BS deployment and the trained RL redaction policy: it turns raw document
text into the exact entity-dict schema RedactionEnv consumes (entity_type,
language, ner_confidence, demographic_risk_score, position, total), the same
schema `generate_document` produces in custom_env.py.

The entity *extraction* here (finding emails/phones/names/addresses/
universities in real text via regex and keyword matching) is genuinely
functional, not a stub. What's still a placeholder is the demographic-risk
scoring: no annotated corpus or trained risk classifier exists yet (that's
future capstone work), so risk scores are drawn from the same synthetic
risk bands used in training. When the real fine-tuned XLM-RoBERTa NER and
risk classifier exist, only the internals of this function change - the
environment, the trained policy, and everything downstream of it are
untouched, since the schema this function returns is already what they
expect.
"""
import re

import numpy as np

from environment.custom_env import ACTION_NAMES, FR_RISK_BUMP, RISK_BANDS, risk_tier

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?:\+?237[\s.-]?)?6\d{2}[\s.-]?\d{2}[\s.-]?\d{2}[\s.-]?\d{2}\b")
NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2}\b")

# Small stand-in gazetteers for the entity types a regex alone can't reliably
# find. A real deployment would replace this whole module with a trained
# NER model; these keyword lists just prove the extraction step is real.
UNIVERSITY_KEYWORDS = ["university", "université", "polytechnique", "institute", "school", "ecole", "école"]
ADDRESS_KEYWORDS = ["street", "rue", "avenue", "quartier", "bastos", "tsinga", "district", "yaounde", "yaoundé"]


def _find_keyword_spans(text_lower, keywords):
    spans = []
    for kw in keywords:
        idx = text_lower.find(kw)
        if idx != -1:
            spans.append(idx)
    return spans


def entities_from_nlp_pipeline(doc_text, language="EN", rng=None):
    """Extract entities from real document text and score them.

    Returns a list of entity dicts in the same schema and order convention
    as `generate_document`, directly consumable by `RedactionEnv._obs_for`.
    """
    rng = rng if rng is not None else np.random.default_rng()
    lowered = doc_text.lower()

    spans = []
    spans += [("EMAIL", m.start(), 0.95) for m in EMAIL_RE.finditer(doc_text)]
    spans += [("PHONE", m.start(), 0.90) for m in PHONE_RE.finditer(doc_text)]
    spans += [("UNIVERSITY", i, 0.70) for i in _find_keyword_spans(lowered, UNIVERSITY_KEYWORDS)]
    spans += [("ADDRESS", i, 0.65) for i in _find_keyword_spans(lowered, ADDRESS_KEYWORDS)]
    spans += [("NAME", m.start(), 0.60) for m in NAME_RE.finditer(doc_text)]
    spans.sort(key=lambda s: s[1])

    total = len(spans)
    entities = []
    for position, (etype, _, confidence) in enumerate(spans):
        band = RISK_BANDS[etype]
        tier = rng.choice(["low", "mid", "high"], p=band["weights"])
        lo, hi = band[tier]
        risk = float(rng.uniform(lo, hi))
        if language == "FR":
            risk = float(np.clip(risk + FR_RISK_BUMP[etype], 0.0, 1.0))

        entities.append({
            "entity_type": etype, "language": language, "ner_confidence": confidence,
            "demographic_risk_score": risk, "position": position, "total": total,
        })
    return entities


def demo_on_real_text():
    """Feeds a real (made-up) bilingual application snippet through the
    extractor and the trained DQN policy, end to end, to prove the
    environment/policy boundary is production-pluggable."""
    from stable_baselines3 import DQN

    from environment.custom_env import RedactionEnv
    from environment.rendering import render_episode_html

    sample_text = (
        "Applicant Jean Paul Mbarga studied at the University of Yaounde I "
        "and currently resides in the Bastos quartier of Yaounde. "
        "Contact: jean.mbarga@example.cm or 677889900."
    )

    entities = entities_from_nlp_pipeline(sample_text, language="FR")
    print(f"extracted {len(entities)} entities from real text:\n")

    model = DQN.load("models/dqn/final/best_model")
    env = RedactionEnv()
    trace = []
    cumulative_reward = 0.0

    for entity in entities:
        obs = env._obs_for(entity)
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        tier = risk_tier(entity["demographic_risk_score"])
        reward = 0.0  # illustrative only; no env.step() call since there's no live document to advance
        trace.append({
            "entity_type": entity["entity_type"], "language": entity["language"],
            "risk_score": entity["demographic_risk_score"], "confidence": entity["ner_confidence"],
            "risk_tier": tier, "action": ACTION_NAMES[action], "reward": reward,
        })
        print(f"  {entity['entity_type']:<10} risk={entity['demographic_risk_score']:.2f} -> {ACTION_NAMES[action]}")

    with open("assets/real_text_demo.html", "w", encoding="utf-8") as f:
        f.write(render_episode_html(trace))
    print("\nwrote assets/real_text_demo.html")


if __name__ == "__main__":
    demo_on_real_text()
