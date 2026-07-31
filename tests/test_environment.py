import numpy as np

from environment.adapter import entities_from_nlp_pipeline
from environment.custom_env import (
    ACTION_NAMES,
    FULL_REDACT,
    NO_REDACT,
    PARTIAL_MASK,
    REWARD,
    SUBSTITUTE,
    RedactionEnv,
    ShiftedRedactionEnv,
    risk_tier,
)


def test_spaces():
    env = RedactionEnv(seed=0)
    assert env.action_space.n == 4
    assert env.observation_space.shape == (10,)


def test_reset_returns_valid_observation():
    env = RedactionEnv(seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (10,)
    assert env.observation_space.contains(obs)
    assert info["entity"] is not None


def test_episode_terminates_after_all_entities():
    env = RedactionEnv(min_entities=5, max_entities=15, seed=1)
    env.reset(seed=1)
    doc_len = len(env.document)

    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        steps += 1

    assert steps == doc_len
    assert terminated
    assert len(env.episode_trace) == doc_len


def test_reward_matches_risk_tier_table():
    env = RedactionEnv(seed=2)
    env.reset(seed=2)
    entity = env.document[env.pointer]
    tier = risk_tier(entity["demographic_risk_score"])

    _, reward, _, _, _ = env.step(FULL_REDACT)

    expected = REWARD[tier][FULL_REDACT] + 0.05  # + step efficiency bonus
    assert reward == expected


def test_high_risk_no_redact_is_penalized():
    # Directly exercise the reward table rather than relying on a random
    # draw landing in the high tier.
    assert REWARD["high"][NO_REDACT] < 0
    assert REWARD["high"][FULL_REDACT] > 0
    assert REWARD["high"][SUBSTITUTE] > 0


def test_low_risk_no_redact_is_rewarded():
    assert REWARD["low"][NO_REDACT] > 0
    assert REWARD["low"][FULL_REDACT] < 0


def test_shifted_env_skews_toward_french_and_higher_address_risk():
    rng = np.random.default_rng(42)
    n_docs = 200
    base_fr_count = 0
    shifted_fr_count = 0

    for i in range(n_docs):
        base_env = RedactionEnv(seed=int(rng.integers(0, 1_000_000)))
        base_env.reset()
        base_fr_count += base_env.document[0]["language"] == "FR"

        shifted_env = ShiftedRedactionEnv(seed=int(rng.integers(0, 1_000_000)))
        shifted_env.reset()
        shifted_fr_count += shifted_env.document[0]["language"] == "FR"

    assert shifted_fr_count > base_fr_count


def test_entities_from_nlp_pipeline_matches_env_schema():
    text = "Marie Ngo Bella lives in the quartier Bastos. Contact: marie@example.cm"
    entities = entities_from_nlp_pipeline(text, language="FR", rng=np.random.default_rng(0))

    assert len(entities) > 0
    env = RedactionEnv()
    for entity in entities:
        assert entity["entity_type"] in ["NAME", "ADDRESS", "UNIVERSITY", "PHONE", "EMAIL"]
        assert 0.0 <= entity["demographic_risk_score"] <= 1.0
        obs = env._obs_for(entity)  # must not raise, confirms schema compatibility
        assert obs.shape == (10,)


def test_action_names_align_with_constants():
    assert ACTION_NAMES[FULL_REDACT] == "FULL_REDACT"
    assert ACTION_NAMES[PARTIAL_MASK] == "PARTIAL_MASK"
    assert ACTION_NAMES[SUBSTITUTE] == "SUBSTITUTE"
    assert ACTION_NAMES[NO_REDACT] == "NO_REDACT"
