"""Core function-based engine for SiagaBot.

This module intentionally does not read from the terminal or write to the
database.  It can therefore be reused by the CLI now and by a web adapter
later.
"""

from __future__ import annotations

import json
import random
import re
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
except ImportError:  # pragma: no cover - requirements install Sastrawi in production
    StemmerFactory = None  # type: ignore[assignment]


DEFAULT_THRESHOLD = 0.45

_STEMMER = StemmerFactory().create_stemmer() if StemmerFactory else None

# Keep negation and safety words. Removing these words could change the
# meaning of a medical complaint, for example "tidak berdarah".
STOPWORDS = {
    "aku",
    "saya",
    "kami",
    "kita",
    "anda",
    "kamu",
    "yang",
    "di",
    "ke",
    "dari",
    "dan",
    "atau",
    "dengan",
    "untuk",
    "pada",
    "ini",
    "itu",
    "nya",
    "tadi",
    "sudah",
    "lagi",
    "mau",
    "apa",
    "bagaimana",
    "bisa",
    "harus",
    "sebaiknya",
    "tolong",
    "dong",
    "ya",
    "sih",
}

# Body-location and context words are useful as context, but should not
# dominate intent matching because they appear in many different intents.
# The actual list can be extended from rules_and_metadata in the JSON file.
DEFAULT_LOW_INFORMATION_WORDS = {
    "kena",
    "terkena",
    "kaki",
    "tangan",
    "jari",
    "hidung",
    "mata",
    "kulit",
    "kepala",
    "lutut",
    "siku",
    "dahi",
    "badan",
    "telapak",
    "pergelangan",
    "engkel",
    "jempol",
    "wajah",
    "leher",
    "telinga",
    "kuping",
    "area",
    "bagian",
    "sisi",
    "tubuh",
}


def load_knowledge_map(file_path: str | Path) -> dict[str, Any]:
    """Load and return the JSON knowledge map."""

    path = Path(file_path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_text(
    text: str,
    normalization_map: dict[str, str] | None = None,
) -> str:
    """Normalize text into lowercase words without punctuation.

    Slang replacement is performed per token so that the same normalization
    can be applied to user input and knowledge-map patterns.
    """

    normalization_map = normalization_map or {}
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    words = text.split()
    words = [normalization_map.get(word, word) for word in words]
    return " ".join(words)


@lru_cache(maxsize=4096)
def stem_text(text: str) -> str:
    """Stem normalized Indonesian text when Sastrawi is available."""

    if _STEMMER is None:
        return text
    return _STEMMER.stem(text)


def _content_tokens(
    text: str,
    normalization_map: dict[str, str] | None = None,
    low_information_words: set[str] | None = None,
) -> list[str]:
    """Normalize, stem, and tokenize text for intent similarity."""

    normalized = normalize_text(text, normalization_map)
    stemmed = stem_text(normalized)
    return tokenize_text(stemmed, low_information_words)


def _match_tokens(
    text: str,
    normalization_map: dict[str, str] | None = None,
    low_information_words: set[str] | None = None,
) -> list[str]:
    """Return content tokens plus context words when content exists.

    Location words such as ``hidung`` and ``jari`` are weak evidence by
    themselves, but become useful alongside a symptom such as ``berdarah``
    or ``terjepit``. Keeping them here prevents generic patterns from making
    unrelated intents look identical.
    """

    low_information_words = (
        DEFAULT_LOW_INFORMATION_WORDS
        if low_information_words is None
        else low_information_words
    )
    normalized = normalize_text(text, normalization_map)
    stemmed_words = stem_text(normalized).split()
    content_tokens = tokenize_text(" ".join(stemmed_words), low_information_words)
    if not content_tokens:
        return []

    context_tokens = [
        word
        for word in stemmed_words
        if word in low_information_words and word not in STOPWORDS
    ]
    return content_tokens + context_tokens


def tokenize_text(
    text: str,
    low_information_words: set[str] | None = None,
) -> list[str]:
    """Split normalized text and remove non-essential stopwords."""

    low_information_words = (
        DEFAULT_LOW_INFORMATION_WORDS
        if low_information_words is None
        else low_information_words
    )
    return [
        word
        for word in text.split()
        if word not in STOPWORDS and word not in low_information_words
    ]


def jaccard_score(input_tokens: list[str], pattern_tokens: list[str]) -> float:
    """Calculate token-set Jaccard similarity."""

    input_set = set(input_tokens)
    pattern_set = set(pattern_tokens)
    union = input_set | pattern_set
    if not union:
        return 0.0
    return len(input_set & pattern_set) / len(union)


def fuzzy_score(input_text: str, pattern_text: str) -> float:
    """Calculate text similarity using Python's standard library."""

    if not input_text or not pattern_text:
        return 0.0
    return SequenceMatcher(None, input_text, pattern_text).ratio()


def coverage_score(input_tokens: list[str], pattern_tokens: list[str]) -> float:
    """Calculate how much of a pattern is covered by the input."""

    if not pattern_tokens:
        return 0.0
    return len(set(input_tokens) & set(pattern_tokens)) / len(set(pattern_tokens))


def similarity_components(
    input_text: str,
    pattern_text: str,
    normalization_map: dict[str, str] | None = None,
    low_information_words: set[str] | None = None,
) -> dict[str, float]:
    """Return each component and the configured combined score."""

    input_tokens = _match_tokens(
        input_text,
        normalization_map,
        low_information_words,
    )
    pattern_tokens = _match_tokens(
        pattern_text,
        normalization_map,
        low_information_words,
    )

    jaccard = jaccard_score(input_tokens, pattern_tokens)
    # Fuzzy matching also uses content tokens so low-information words do not
    # inflate the score through character-level similarity.
    fuzzy = fuzzy_score(" ".join(input_tokens), " ".join(pattern_tokens))
    coverage = coverage_score(input_tokens, pattern_tokens)
    score = (0.4 * jaccard) + (0.3 * fuzzy) + (0.3 * coverage)

    return {
        "jaccard": jaccard,
        "fuzzy": fuzzy,
        "coverage": coverage,
        "score": score,
    }


def similarity(
    input_text: str,
    pattern_text: str,
    normalization_map: dict[str, str] | None = None,
    low_information_words: set[str] | None = None,
) -> float:
    """Calculate and return only the combined similarity score."""

    return similarity_components(
        input_text,
        pattern_text,
        normalization_map,
        low_information_words,
    )["score"]


def _normalization_map(knowledge_map: dict[str, Any]) -> dict[str, str]:
    """Read the slang map from the knowledge map metadata."""

    metadata = knowledge_map.get("rules_and_metadata", {})
    return metadata.get("normalisasi", {})


def _threshold(knowledge_map: dict[str, Any]) -> float:
    """Read the configured threshold, with a safe default."""

    metadata = knowledge_map.get("rules_and_metadata", {})
    return float(metadata.get("threshold", DEFAULT_THRESHOLD))


def _low_information_words(knowledge_map: dict[str, Any]) -> set[str]:
    """Read and combine generic-word configuration from the knowledge map."""

    metadata = knowledge_map.get("rules_and_metadata", {})
    configured_words = metadata.get("low_information_words", [])
    return DEFAULT_LOW_INFORMATION_WORDS | set(configured_words)


def _intent_keyword_evidence(
    user_message: str,
    intent_tag: str | None,
    knowledge_map: dict[str, Any],
) -> float:
    """Return weighted evidence for an intent-specific keyword or phrase."""

    if not intent_tag:
        return 0.0

    metadata = knowledge_map.get("rules_and_metadata", {})
    keyword_groups = metadata.get("intent_keywords", {})
    configured_keywords = keyword_groups.get(intent_tag, {})
    if not configured_keywords:
        return 0.0

    normalization_map = _normalization_map(knowledge_map)
    low_information_words = _low_information_words(knowledge_map)
    input_tokens = set(
        _match_tokens(user_message, normalization_map, low_information_words)
    )
    matched_weight = 0.0

    for keyword, weight in configured_keywords.items():
        keyword_tokens = set(
            _match_tokens(keyword, normalization_map, low_information_words)
        )
        if keyword_tokens and keyword_tokens.issubset(input_tokens):
            matched_weight += float(weight)

    # A weight of four represents a strong, intent-specific phrase.  The cap
    # prevents long messages with repeated evidence from dominating ranking.
    return min(1.0, matched_weight / 4.0)


def _is_priority_match(
    user_message: str,
    pattern: str,
    normalization_map: dict[str, str],
    low_information_words: set[str],
) -> bool:
    """Return whether a priority pattern occurs in the normalized input."""

    normalized_input = normalize_text(user_message, normalization_map)
    normalized_pattern = normalize_text(pattern, normalization_map)
    input_tokens = set(tokenize_text(normalized_input, low_information_words))
    pattern_tokens = set(tokenize_text(normalized_pattern, low_information_words))
    return bool(pattern_tokens) and pattern_tokens.issubset(input_tokens)


def rank_intents(
    user_message: str,
    knowledge_map: dict[str, Any],
) -> list[dict[str, Any]]:
    """Rank intents and retain component scores for debugging and display."""

    normalization_map = _normalization_map(knowledge_map)
    low_information_words = _low_information_words(knowledge_map)
    threshold = _threshold(knowledge_map)
    rankings: list[dict[str, Any]] = []

    for intent in knowledge_map.get("intents", []):
        best_match: dict[str, Any] | None = None
        keyword_score = _intent_keyword_evidence(
            user_message,
            intent.get("tag"),
            knowledge_map,
        )

        for pattern in intent.get("patterns", []):
            components = similarity_components(
                user_message,
                pattern,
                normalization_map,
                low_information_words,
            )
            base_score = components["score"]
            # Keep the original similarity formula while adding a bounded,
            # intent-specific signal for paraphrases and informal wording.
            score = min(1.0, (0.60 * base_score) + (0.40 * keyword_score))
            priority = (
                intent.get("tag") == "darurat_umum"
                and _is_priority_match(
                    user_message,
                    pattern,
                    normalization_map,
                    low_information_words,
                )
            )
            candidate = {
                "intent": intent,
                "tag": intent.get("tag"),
                "matched_pattern": pattern,
                "pattern_length": len(
                    normalize_text(pattern, normalization_map).split()
                ),
                "jaccard": components["jaccard"],
                "fuzzy": components["fuzzy"],
                "coverage": components["coverage"],
                "score": score,
                "base_score": base_score,
                "keyword_score": keyword_score,
                "priority": priority,
                "matched": priority or score >= threshold,
            }

            if best_match is None or (
                candidate["score"],
                candidate["pattern_length"],
            ) > (
                best_match["score"],
                best_match["pattern_length"],
            ):
                best_match = candidate

            # A priority keyword must win even when the full input contains
            # additional words that lower its normal similarity score.
            if priority:
                best_match = candidate
                break

        if best_match is not None:
            rankings.append(best_match)

    rankings.sort(
        key=lambda item: (
            item["priority"],
            item["score"],
            item["pattern_length"],
        ),
        reverse=True,
    )
    return rankings


def find_best_intent(
    user_message: str,
    knowledge_map: dict[str, Any],
) -> dict[str, Any]:
    """Find the highest-scoring intent and apply the configured threshold."""

    rankings = rank_intents(user_message, knowledge_map)
    if not rankings:
        return {
            "intent": None,
            "score": 0.0,
            "matched_pattern": None,
            "matched": False,
            "rankings": [],
        }

    best_result = rankings[0].copy()
    best_result["rankings"] = rankings
    if not best_result["matched"]:
        best_result["intent"] = None
    return best_result


def _format_response(response: str, knowledge_map: dict[str, Any]) -> str:
    """Replace knowledge-map placeholders in a response."""

    bot_name = knowledge_map.get("bot_name", "SiagaBot")
    return response.replace("{nama_bot}", bot_name)


def choose_response(
    match_result: dict[str, Any],
    knowledge_map: dict[str, Any],
) -> str:
    """Choose one response from the matched intent or fallback responses."""

    intent = match_result.get("intent")
    if intent is not None and match_result.get("matched"):
        responses = intent.get("responses", [])
    else:
        responses = knowledge_map.get("fallback", [])

    if not responses:
        return "Maaf, saya belum dapat memberikan respons untuk pertanyaan tersebut."

    return _format_response(random.choice(responses), knowledge_map)


def process_message(
    user_message: str,
    knowledge_map: dict[str, Any],
) -> dict[str, Any]:
    """Process one user message without CLI or database side effects."""

    match_result = find_best_intent(user_message, knowledge_map)
    response = choose_response(match_result, knowledge_map)
    intent = match_result.get("intent")

    return {
        "user_message": user_message,
        "response": response,
        "intent": intent.get("tag") if intent else None,
        "score": match_result.get("score", 0.0),
        "matched_pattern": match_result.get("matched_pattern"),
        "similarity_components": {
            "jaccard": match_result.get("jaccard", 0.0),
            "fuzzy": match_result.get("fuzzy", 0.0),
            "coverage": match_result.get("coverage", 0.0),
        },
        "rankings": match_result.get("rankings", []),
    }
