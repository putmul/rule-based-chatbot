"""Mesin utama SiagaBot untuk mengenali intent dan memilih respons.

Modul ini hanya mengolah teks. Modul ini tidak membaca terminal dan tidak
menulis database, sehingga dapat digunakan kembali oleh CLI maupun web.
Alur sederhananya adalah: normalisasi teks -> hitung kemiripan -> pilih
intent terbaik -> pilih respons.
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
JACCARD_WEIGHT = 0.4
FUZZY_WEIGHT = 0.3
COVERAGE_WEIGHT = 0.3
BASE_SCORE_WEIGHT = 0.6
KEYWORD_SCORE_WEIGHT = 0.4

_STEMMER = StemmerFactory().create_stemmer() if StemmerFactory else None

# Kata negasi dan keselamatan tetap dipertahankan. Menghapusnya dapat
# mengubah arti keluhan medis, misalnya pada kalimat "tidak berdarah".
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

# Kata lokasi tubuh dan konteks berguna sebagai informasi tambahan, tetapi
# tidak boleh mendominasi pencocokan karena dapat muncul di banyak intent.
# Daftar ini dapat diperluas melalui rules_and_metadata pada file JSON.
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
    """Membaca file JSON dan mengembalikan basis pengetahuan chatbot."""

    path = Path(file_path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_text(
    text: str,
    normalization_map: dict[str, str] | None = None,
) -> str:
    """Menyeragamkan teks sebelum dibandingkan dengan pola intent.

    Teks diubah menjadi huruf kecil, tanda baca dihapus, lalu kata tidak baku
    diganti berdasarkan ``normalization_map``. Proses per kata membuat input
    user dan pola pada knowledge map diproses dengan cara yang sama.
    """

    normalization_map = normalization_map or {}
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    words = text.split()
    words = [normalization_map.get(word, word) for word in words]
    return " ".join(words)


@lru_cache(maxsize=4096)
def stem_text(text: str) -> str:
    """Mengubah kata ke bentuk dasar menggunakan Sastrawi jika tersedia."""

    if _STEMMER is None:
        return text
    return _STEMMER.stem(text)


def _content_tokens(
    text: str,
    normalization_map: dict[str, str] | None = None,
    low_information_words: set[str] | None = None,
) -> list[str]:
    """Menghasilkan token penting untuk perhitungan kemiripan intent."""

    normalized = normalize_text(text, normalization_map)
    stemmed = stem_text(normalized)
    return tokenize_text(stemmed, low_information_words)


def _match_tokens(
    text: str,
    normalization_map: dict[str, str] | None = None,
    low_information_words: set[str] | None = None,
) -> list[str]:
    """Menghasilkan token gejala ditambah token konteks yang relevan.

    Kata lokasi seperti ``hidung`` dan ``jari`` tidak cukup kuat jika berdiri
    sendiri, tetapi berguna jika muncul bersama gejala seperti ``berdarah``
    atau ``terjepit``.
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
    """Memecah teks dan membuang stopword serta kata yang terlalu umum."""

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
    """Menghitung kemiripan berdasarkan irisan himpunan token."""

    input_set = set(input_tokens)
    pattern_set = set(pattern_tokens)
    union = input_set | pattern_set
    if not union:
        return 0.0
    return len(input_set & pattern_set) / len(union)


def fuzzy_score(input_text: str, pattern_text: str) -> float:
    """Menghitung kemiripan susunan karakter dengan ``SequenceMatcher``."""

    if not input_text or not pattern_text:
        return 0.0
    return SequenceMatcher(None, input_text, pattern_text).ratio()


def coverage_score(input_tokens: list[str], pattern_tokens: list[str]) -> float:
    """Menghitung persentase token pola yang ditemukan pada input user."""

    if not pattern_tokens:
        return 0.0
    return len(set(input_tokens) & set(pattern_tokens)) / len(set(pattern_tokens))


def similarity_components(
    input_text: str,
    pattern_text: str,
    normalization_map: dict[str, str] | None = None,
    low_information_words: set[str] | None = None,
) -> dict[str, float]:
    """Menggabungkan tiga skor kemiripan menjadi satu skor dasar.

    Skor dasar menggunakan bobot Jaccard 40%, fuzzy 30%, dan coverage 30%.
    Komponen individual tetap dikembalikan agar dapat ditampilkan untuk
    debugging atau statistik.
    """

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
    # Fuzzy juga menggunakan token penting agar kata umum tidak menaikkan
    # skor secara tidak wajar hanya karena kemiripan karakter.
    fuzzy = fuzzy_score(" ".join(input_tokens), " ".join(pattern_tokens))
    coverage = coverage_score(input_tokens, pattern_tokens)
    score = (
        (JACCARD_WEIGHT * jaccard)
        + (FUZZY_WEIGHT * fuzzy)
        + (COVERAGE_WEIGHT * coverage)
    )

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
    """Mengembalikan hanya skor gabungan tanpa rincian komponennya."""

    return similarity_components(
        input_text,
        pattern_text,
        normalization_map,
        low_information_words,
    )["score"]


def _normalization_map(knowledge_map: dict[str, Any]) -> dict[str, str]:
    """Mengambil kamus normalisasi kata dari metadata knowledge map."""

    metadata = knowledge_map.get("rules_and_metadata", {})
    return metadata.get("normalisasi", {})


def _threshold(knowledge_map: dict[str, Any]) -> float:
    """Mengambil ambang minimal kecocokan dengan nilai default yang aman."""

    metadata = knowledge_map.get("rules_and_metadata", {})
    return float(metadata.get("threshold", DEFAULT_THRESHOLD))


def _low_information_words(knowledge_map: dict[str, Any]) -> set[str]:
    """Menggabungkan daftar kata umum bawaan dan konfigurasi knowledge map."""

    metadata = knowledge_map.get("rules_and_metadata", {})
    configured_words = metadata.get("low_information_words", [])
    return DEFAULT_LOW_INFORMATION_WORDS | set(configured_words)


def _intent_keyword_evidence(
    user_message: str,
    intent_tag: str | None,
    knowledge_map: dict[str, Any],
) -> float:
    """Menghitung bukti tambahan dari keyword khusus suatu intent.

    Keyword yang cocok menambah skor, tetapi nilainya dibatasi maksimal 1.0
    agar pesan panjang tidak otomatis mengalahkan semua intent lain.
    """

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

    # Bobot empat dianggap sebagai bukti kuat yang spesifik untuk intent.
    # Pembatas mencegah pesan panjang mendominasi hasil ranking.
    return min(1.0, matched_weight / 4.0)


def _is_priority_match(
    user_message: str,
    pattern: str,
    normalization_map: dict[str, str],
    low_information_words: set[str],
) -> bool:
    """Memeriksa apakah pola prioritas muncul lengkap pada input user."""

    normalized_input = normalize_text(user_message, normalization_map)
    normalized_pattern = normalize_text(pattern, normalization_map)
    input_tokens = set(tokenize_text(normalized_input, low_information_words))
    pattern_tokens = set(tokenize_text(normalized_pattern, low_information_words))
    return bool(pattern_tokens) and pattern_tokens.issubset(input_tokens)


def rank_intents(
    user_message: str,
    knowledge_map: dict[str, Any],
) -> list[dict[str, Any]]:
    """Mengurutkan semua intent dari kandidat terbaik hingga terendah.

    Setiap pola dibandingkan dengan input. Pola terbaik untuk satu intent
    disimpan, lalu seluruh kandidat diurutkan berdasarkan prioritas, skor,
    dan panjang pola. Rincian skor sengaja dipertahankan untuk debugging.
    """

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
            # Skor dasar dipadukan dengan keyword khusus agar variasi bahasa
            # informal tetap dapat dikenali tanpa mengabaikan kemiripan pola.
            score = min(
                1.0,
                (BASE_SCORE_WEIGHT * base_score)
                + (KEYWORD_SCORE_WEIGHT * keyword_score),
            )
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

            # Keyword prioritas harus tetap menang walaupun input memiliki
            # kata tambahan yang menurunkan skor kemiripan normalnya.
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
    """Memilih satu intent terbaik dan menerapkan ambang kecocokan."""

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
    """Mengganti placeholder respons, misalnya ``{nama_bot}``."""

    bot_name = knowledge_map.get("bot_name", "SiagaBot")
    return response.replace("{nama_bot}", bot_name)


def choose_response(
    match_result: dict[str, Any],
    knowledge_map: dict[str, Any],
) -> str:
    """Memilih respons acak dari intent cocok atau daftar fallback."""

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
    """Memproses satu pesan dari awal sampai menjadi respons chatbot.

    Fungsi ini hanya mengembalikan hasil pemrosesan. Penyimpanan database
    dilakukan oleh pemanggil seperti CLI atau aplikasi web.
    """

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
