import json

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

ENTITY_TYPES = ["NAME", "ADDRESS", "UNIVERSITY", "PHONE", "EMAIL"]
ENTITY_TYPE_IDX = {t: i for i, t in enumerate(ENTITY_TYPES)}

ACTION_NAMES = ["FULL_REDACT", "PARTIAL_MASK", "SUBSTITUTE", "NO_REDACT"]
FULL_REDACT, PARTIAL_MASK, SUBSTITUTE, NO_REDACT = 0, 1, 2, 3

# How often each entity type shows up in an application document. Names and
# universities dominate; phone and email are rarer and lower signal.
ENTITY_TYPE_WEIGHTS = {"NAME": 0.35, "UNIVERSITY": 0.25, "ADDRESS": 0.20, "PHONE": 0.12, "EMAIL": 0.08}

# Risk tiers grounded in the Bastos/Tsinga framing from the capstone: some
# neighbourhoods, name pools, and university tiers carry stronger regional or
# ethnic coding than others. Researcher-estimated pending real corpus collection.
RISK_BANDS = {
    "NAME": {"low": (0.05, 0.30), "mid": (0.30, 0.65), "high": (0.65, 0.98), "weights": (0.30, 0.40, 0.30)},
    "ADDRESS": {"low": (0.10, 0.35), "mid": (0.35, 0.70), "high": (0.70, 0.98), "weights": (0.20, 0.35, 0.45)},
    "UNIVERSITY": {"low": (0.05, 0.30), "mid": (0.30, 0.60), "high": (0.60, 0.90), "weights": (0.35, 0.40, 0.25)},
    "PHONE": {"low": (0.0, 0.20), "mid": (0.20, 0.40), "high": (0.40, 0.55), "weights": (0.75, 0.20, 0.05)},
    "EMAIL": {"low": (0.0, 0.15), "mid": (0.15, 0.35), "high": (0.35, 0.50), "weights": (0.80, 0.15, 0.05)},
}

# French-language documents draw a slightly higher risk floor for address and
# university: regional institution and neighbourhood naming reads as more
# ethnically legible in French administrative documents in this context.
FR_RISK_BUMP = {"ADDRESS": 0.08, "UNIVERSITY": 0.05, "NAME": 0.03, "PHONE": 0.0, "EMAIL": 0.0}

REWARD = {
    "high": {FULL_REDACT: 1.0, SUBSTITUTE: 1.0, PARTIAL_MASK: 0.3, NO_REDACT: -2.0},
    "mid": {PARTIAL_MASK: 0.8, FULL_REDACT: 0.2, SUBSTITUTE: 0.2, NO_REDACT: -0.5},
    "low": {NO_REDACT: 1.0, PARTIAL_MASK: -0.2, FULL_REDACT: -0.5, SUBSTITUTE: -0.5},
}
STEP_EFFICIENCY_BONUS = 0.05
COMPLETION_BONUS = 1.0


def risk_tier(risk_score):
    if risk_score >= 0.65:
        return "high"
    if risk_score >= 0.30:
        return "mid"
    return "low"


def sample_entity(rng, position, total, language):
    etype = rng.choice(ENTITY_TYPES, p=[ENTITY_TYPE_WEIGHTS[t] for t in ENTITY_TYPES])
    band = RISK_BANDS[etype]
    tier = rng.choice(["low", "mid", "high"], p=band["weights"])
    lo, hi = band[tier]
    risk = float(rng.uniform(lo, hi))
    if language == "FR":
        risk = float(np.clip(risk + FR_RISK_BUMP[etype], 0.0, 1.0))

    # NER confidence is noisy but correlated with a latent "clean detection"
    # factor rather than independent noise, since distinctive (higher-risk)
    # surface forms tend to be detected more confidently by a real NER model.
    clean_detection = rng.beta(5, 2)
    confidence = float(np.clip(clean_detection * 0.7 + risk * 0.3 + rng.normal(0, 0.05), 0.0, 1.0))

    return {
        "entity_type": etype, "language": language, "ner_confidence": confidence,
        "demographic_risk_score": risk, "position": position, "total": total,
    }


def generate_document(rng, min_entities=5, max_entities=15):
    language = "FR" if rng.random() < 0.6 else "EN"
    total = int(rng.integers(min_entities, max_entities + 1))
    return [sample_entity(rng, i, total, language) for i in range(total)]


class RedactionEnv(gym.Env):
    """One episode is one synthetic job application document. At each step
    the agent sees one entity and picks a redaction action for it."""

    metadata = {"render_modes": []}

    def __init__(self, min_entities=5, max_entities=15, seed=None):
        super().__init__()
        self.min_entities = min_entities
        self.max_entities = max_entities
        self.max_steps = max_entities

        self.action_space = spaces.Discrete(4)
        # [entity_type_onehot(5), language_flag(1), ner_confidence(1),
        #  demographic_risk_score(1), normalized_position(1), fraction_remaining(1)] -> (10,)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self.document = []
        self.pointer = 0
        self.episode_trace = []

    def _obs_for(self, entity):
        onehot = np.zeros(5, dtype=np.float32)
        onehot[ENTITY_TYPE_IDX[entity["entity_type"]]] = 1.0
        lang_flag = 1.0 if entity["language"] == "FR" else 0.0
        norm_pos = entity["position"] / max(entity["total"] - 1, 1)
        frac_remaining = (entity["total"] - entity["position"]) / entity["total"]
        return np.concatenate([
            onehot,
            np.array([lang_flag, entity["ner_confidence"], entity["demographic_risk_score"],
                      norm_pos, frac_remaining], dtype=np.float32),
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.document = generate_document(self._rng, self.min_entities, self.max_entities)
        self.pointer = 0
        self.episode_trace = []
        obs = self._obs_for(self.document[self.pointer])
        info = {"entity": dict(self.document[self.pointer])}
        return obs, info

    def step(self, action):
        entity = self.document[self.pointer]
        tier = risk_tier(entity["demographic_risk_score"])
        reward = REWARD[tier][action] + STEP_EFFICIENCY_BONUS

        self.episode_trace.append({
            "entity_type": entity["entity_type"], "language": entity["language"],
            "risk_score": entity["demographic_risk_score"], "confidence": entity["ner_confidence"],
            "risk_tier": tier, "action": ACTION_NAMES[action], "reward": reward,
        })

        self.pointer += 1
        terminated = self.pointer >= len(self.document)
        truncated = self.pointer >= self.max_steps and not terminated

        if terminated:
            reward += COMPLETION_BONUS
            self.episode_trace[-1]["reward"] = reward
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            info = {"entity": None, "episode_trace": self.episode_trace}
        else:
            next_entity = self.document[self.pointer]
            obs = self._obs_for(next_entity)
            info = {"entity": dict(next_entity)}

        return obs, reward, terminated, truncated, info

    def render(self):
        pass


class ShiftedRedactionEnv(RedactionEnv):
    """Held-out generalization distribution: heavier French skew and elevated
    address/university risk, simulating an unseen neighbourhood or
    institution pool. Used only for the generalization test."""

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        language = "FR" if self._rng.random() < 0.9 else "EN"
        total = int(self._rng.integers(self.min_entities, self.max_entities + 1))
        doc = [sample_entity(self._rng, i, total, language) for i in range(total)]
        for e in doc:
            if e["entity_type"] in ("ADDRESS", "UNIVERSITY"):
                e["demographic_risk_score"] = float(np.clip(e["demographic_risk_score"] + 0.1, 0.0, 1.0))

        self.document = doc
        self.pointer = 0
        self.episode_trace = []
        obs = self._obs_for(self.document[self.pointer])
        info = {"entity": dict(self.document[self.pointer])}
        return obs, info


register(id="RedactionEnv-v0", entry_point=lambda: RedactionEnv(), max_episode_steps=15)


def make_env(seed=None, shifted=False):
    def _init():
        cls = ShiftedRedactionEnv if shifted else RedactionEnv
        return cls(seed=seed)
    return _init
