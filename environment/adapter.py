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

from environment.custom_env import FR_RISK_BUMP, RISK_BANDS

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"(?:\+?237[\s.-]?)?6\d{2}[\s.-]?\d{2}[\s.-]?\d{2}[\s.-]?\d{2}\b")
# [A-ZÀ-ÖØ-Þ]/[a-zà-öø-ÿ] rather than plain [A-Z]/[a-z] so accented French
# names (Émile, François, Bénédicte...) don't get cut off mid-word.
NAME_RE = re.compile(r"\b[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+(?:\s[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+){1,2}\b")

# Small stand-in gazetteers for the entity types a regex alone can't reliably
# find. A real deployment would replace this whole module with a trained
# NER model; these keyword lists just prove the extraction step is real.
# Deliberately excludes the city name itself (Yaounde/Yaounde) - it shows up
# inside institution names too ("Universite de Yaounde I"), and including it
# here would make one address mention get double-counted as a second entity.
UNIVERSITY_KEYWORDS = ["university", "université", "polytechnique", "institute", "institut", "school", "ecole", "école"]
ADDRESS_KEYWORDS = [
    "street", "rue", "avenue", "quartier", "neighbourhood", "neighborhood", "district",
    "bastos", "tsinga", "essos", "nlongkak", "mvog-ada", "ngousso", "biyem-assi",
]

# Keyword hits of the same type within this many characters of each other are
# treated as the same real-world mention (e.g. "quartier Bastos" shouldn't
# become two separate ADDRESS entities just because two keywords matched).
MERGE_WINDOW_CHARS = 60

# A bare "University"/"Université" keyword match doesn't name an actual
# institution - this extends the span to also capture a following
# "of X"/"de X" clause (e.g. "University of Buea", "Université de Douala"),
# so the surface text shown is the specific institution, not just the word
# "University" every time.
INSTITUTION_SUFFIX_RE = re.compile(
    r"[ ]+(?:of|de)[ ]+[A-ZÀ-ÖØ-Þ]\w*(?:[ ]+[A-ZÀ-ÖØ-Þ0-9]\w*)*"
)


def _extend_with_institution_name(doc_text, spans):
    extended = []
    for start, end in spans:
        m = INSTITUTION_SUFFIX_RE.match(doc_text, end)
        extended.append((start, m.end() if m else end))
    return extended


def _find_keyword_spans(text_lower, keywords):
    """Returns (start, end) character spans in the lowered text. Nearby hits
    of different keywords merge into one span (e.g. "quartier" + "bastos"
    both matching close together becomes one span covering "quartier
    Bastos"), so the surface text shown is the fuller phrase, not just
    whichever single keyword happened to be found first."""
    hits = sorted(
        (idx, idx + len(kw)) for kw in keywords for idx in [text_lower.find(kw)] if idx != -1
    )
    spans = []
    for start, end in hits:
        if spans and start - spans[-1][1] <= MERGE_WINDOW_CHARS:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end))
        else:
            spans.append((start, end))
    return spans


def entities_from_nlp_pipeline(doc_text, language="EN", rng=None):
    """Extract entities from real document text and score them.

    Returns a list of entity dicts in the same schema and order convention
    as `generate_document`, directly consumable by `RedactionEnv._obs_for`,
    plus a `surface_text` field holding the actual matched text - so a
    rendered episode can show what was found, not just its type.
    """
    rng = rng if rng is not None else np.random.default_rng()
    lowered = doc_text.lower()

    spans = []
    spans += [("EMAIL", m.start(), 0.95, m.group()) for m in EMAIL_RE.finditer(doc_text)]
    spans += [("PHONE", m.start(), 0.90, m.group()) for m in PHONE_RE.finditer(doc_text)]
    university_spans = _extend_with_institution_name(doc_text, _find_keyword_spans(lowered, UNIVERSITY_KEYWORDS))
    spans += [("UNIVERSITY", s, 0.70, doc_text[s:e]) for s, e in university_spans]
    spans += [("ADDRESS", s, 0.65, doc_text[s:e]) for s, e in _find_keyword_spans(lowered, ADDRESS_KEYWORDS)]
    spans += [("NAME", m.start(), 0.60, m.group()) for m in NAME_RE.finditer(doc_text)]
    spans.sort(key=lambda s: s[1])

    total = len(spans)
    entities = []
    for position, (etype, _, confidence, surface_text) in enumerate(spans):
        band = RISK_BANDS[etype]
        tier = rng.choice(["low", "mid", "high"], p=band["weights"])
        lo, hi = band[tier]
        risk = float(rng.uniform(lo, hi))
        if language == "FR":
            risk = float(np.clip(risk + FR_RISK_BUMP[etype], 0.0, 1.0))

        entities.append({
            "entity_type": etype, "language": language, "ner_confidence": confidence,
            "demographic_risk_score": risk, "position": position, "total": total,
            "surface_text": surface_text,
        })
    return entities
