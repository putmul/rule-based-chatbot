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
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLD = 0.45

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


def tokenize_text(text: str) -> list[str]:
    """Split normalized text and remove non-essential stopwords."""

    return [word for word in text.split() if word not in STOPWORDS]


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
) -> dict[str, float]:
    """Return each component and the configured combined score."""

    normalized_input = normalize_text(input_text, normalization_map)
    normalized_pattern = normalize_text(pattern_text, normalization_map)
    input_tokens = tokenize_text(normalized_input)
    pattern_tokens = tokenize_text(normalized_pattern)

    jaccard = jaccard_score(input_tokens, pattern_tokens)
    fuzzy = fuzzy_score(normalized_input, normalized_pattern)
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
) -> float:
    """Calculate and return only the combined similarity score."""

    return similarity_components(input_text, pattern_text, normalization_map)["score"]


def _normalization_map(knowledge_map: dict[str, Any]) -> dict[str, str]:
    """Read the slang map from the knowledge map metadata."""

    metadata = knowledge_map.get("rules_and_metadata", {})
    return metadata.get("normalisasi", {})


def _threshold(knowledge_map: dict[str, Any]) -> float:
    """Read the configured threshold, with a safe default."""

    metadata = knowledge_map.get("rules_and_metadata", {})
    return float(metadata.get("threshold", DEFAULT_THRESHOLD))


def _is_priority_match(
    user_message: str,
    pattern: str,
    normalization_map: dict[str, str],
) -> bool:
    """Return whether a priority pattern occurs in the normalized input."""

    normalized_input = normalize_text(user_message, normalization_map)
    normalized_pattern = normalize_text(pattern, normalization_map)
    input_tokens = set(tokenize_text(normalized_input))
    pattern_tokens = set(tokenize_text(normalized_pattern))
    return bool(pattern_tokens) and pattern_tokens.issubset(input_tokens)


def rank_intents(
    user_message: str,
    knowledge_map: dict[str, Any],
) -> list[dict[str, Any]]:
    """Rank intents and retain component scores for debugging and display."""

    normalization_map = _normalization_map(knowledge_map)
    threshold = _threshold(knowledge_map)
    rankings: list[dict[str, Any]] = []

    for intent in knowledge_map.get("intents", []):
        best_match: dict[str, Any] | None = None

        for pattern in intent.get("patterns", []):
            components = similarity_components(
                user_message,
                pattern,
                normalization_map,
            )
            priority = (
                intent.get("tag") == "darurat_umum"
                and _is_priority_match(user_message, pattern, normalization_map)
            )
            candidate = {
                "intent": intent,
                "tag": intent.get("tag"),
                "matched_pattern": pattern,
                "jaccard": components["jaccard"],
                "fuzzy": components["fuzzy"],
                "coverage": components["coverage"],
                "score": components["score"],
                "priority": priority,
                "matched": priority or components["score"] >= threshold,
            }

            if best_match is None or candidate["score"] > best_match["score"]:
                best_match = candidate

            # A priority keyword must win even when the full input contains
            # additional words that lower its normal similarity score.
            if priority:
                best_match = candidate
                break

        if best_match is not None:
            rankings.append(best_match)

    rankings.sort(
        key=lambda item: (item["priority"], item["score"]),
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
