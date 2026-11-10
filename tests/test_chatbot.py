import unittest
from pathlib import Path

from chatbot.chatbot import (
    find_best_intent,
    load_knowledge_map,
    normalize_text,
    process_message,
    rank_intents,
    similarity,
    similarity_components,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_MAP = load_knowledge_map(PROJECT_ROOT / "p3k_knowledge_map.json")


class ChatbotCoreTests(unittest.TestCase):
    def test_normalization_replaces_slang(self):
        result = normalize_text(
            "Aku GAK sengaja keinjak paku!",
            KNOWLEDGE_MAP["rules_and_metadata"]["normalisasi"],
        )
        self.assertEqual(result, "aku tidak sengaja tertusuk paku")

    def test_identical_pattern_has_high_similarity(self):
        score = similarity("kaki keseleo", "kaki terkilir")
        self.assertGreaterEqual(score, 0.45)

    def test_similarity_components_are_returned(self):
        components = similarity_components("kaki keseleo", "kaki terkilir")
        self.assertEqual(
            set(components),
            {"jaccard", "fuzzy", "coverage", "score"},
        )
        self.assertAlmostEqual(
            components["score"],
            (0.4 * components["jaccard"])
            + (0.3 * components["fuzzy"])
            + (0.3 * components["coverage"]),
        )

    def test_returns_top_five_rankings(self):
        rankings = rank_intents("kaki keseleo", KNOWLEDGE_MAP)
        self.assertGreaterEqual(len(rankings), 5)
        self.assertEqual(rankings[0]["tag"], "keseleo")
        self.assertGreaterEqual(rankings[0]["score"], rankings[1]["score"])

    def test_finds_cut_intent(self):
        result = find_best_intent("Jari saya kena cutter", KNOWLEDGE_MAP)
        self.assertEqual(result["intent"]["tag"], "luka_sayat")
        self.assertTrue(result["matched"])

    def test_emergency_intent_has_priority(self):
        result = find_best_intent("Orang itu pingsan", KNOWLEDGE_MAP)
        self.assertEqual(result["intent"]["tag"], "darurat_umum")
        self.assertTrue(result["rankings"][0]["priority"])

    def test_unknown_input_uses_fallback(self):
        result = process_message("berapa harga sepatu hari ini", KNOWLEDGE_MAP)
        self.assertIsNone(result["intent"])
        self.assertGreater(len(result["response"]), 0)


if __name__ == "__main__":
    unittest.main()
