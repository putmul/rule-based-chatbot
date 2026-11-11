"""Evaluate the chatbot against the CSV dataset in ``evaluasi``."""

from __future__ import annotations

import csv
import unittest
from collections import defaultdict
from pathlib import Path

from chatbot.chatbot import load_knowledge_map, process_message


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "evaluasi" / "evaluasi_p3k.csv"
KNOWLEDGE_MAP_PATH = PROJECT_ROOT / "p3k_knowledge_map.json"


def load_evaluation_rows() -> list[dict[str, str]]:
    """Load and validate evaluation examples from the CSV file."""

    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        expected_columns = {"kalimat", "intent"}
        if set(reader.fieldnames or []) != expected_columns:
            raise ValueError(
                f"Kolom CSV harus {sorted(expected_columns)}, "
                f"diterima: {reader.fieldnames}"
            )

        rows = []
        for line_number, row in enumerate(reader, start=2):
            sentence = (row.get("kalimat") or "").strip()
            expected_intent = (row.get("intent") or "").strip()
            if not sentence or not expected_intent:
                raise ValueError(f"Data kosong pada baris CSV {line_number}")
            rows.append({"kalimat": sentence, "intent": expected_intent})
        return rows


class CsvEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.knowledge_map = load_knowledge_map(KNOWLEDGE_MAP_PATH)
        cls.rows = load_evaluation_rows()

    def test_csv_contains_evaluation_examples(self) -> None:
        self.assertGreater(len(self.rows), 0, "CSV evaluasi tidak boleh kosong")

    def test_all_csv_examples_match_expected_intent(self) -> None:
        results: list[dict[str, object]] = []

        for row_number, row in enumerate(self.rows, start=2):
            result = process_message(row["kalimat"], self.knowledge_map)
            predicted_intent = result.get("intent")
            results.append(
                {
                    "row_number": row_number,
                    "sentence": row["kalimat"],
                    "expected": row["intent"],
                    "predicted": predicted_intent,
                    "score": result.get("score", 0.0),
                }
            )

        summary: dict[str, dict[str, int]] = defaultdict(
            lambda: {"total": 0, "correct": 0}
        )
        failures = []
        for item in results:
            expected = str(item["expected"])
            summary[expected]["total"] += 1
            if item["expected"] == item["predicted"]:
                summary[expected]["correct"] += 1
            else:
                failures.append(item)

        print("\nHASIL EVALUASI CSV")
        print("=" * 78)
        print(f"{'Intent':<32} {'Total':>7} {'Benar':>7} {'Salah':>7} {'Akurasi':>10}")
        print("-" * 78)
        for intent, counts in summary.items():
            incorrect = counts["total"] - counts["correct"]
            accuracy = counts["correct"] / counts["total"] * 100
            print(
                f"{intent:<32} {counts['total']:>7} {counts['correct']:>7} "
                f"{incorrect:>7} {accuracy:>9.2f}%"
            )

        total = len(results)
        correct = total - len(failures)
        print("-" * 78)
        print(f"{'TOTAL':<32} {total:>7} {correct:>7} {len(failures):>7} {correct / total * 100:>9.2f}%")

        if failures:
            print("\nCONTOH PREDIKSI SALAH (maksimal 3)")
            print("=" * 78)
            for item in failures[:3]:
                print(
                    f"{item['row_number']}. {item['sentence']}\n"
                    f"   expected={item['expected']}, predicted={item['predicted']}, "
                    f"score={float(item['score']):.4f}"
                )

        self.assertEqual(
            len(failures),
            0,
            f"Ditemukan {len(failures)} prediksi intent yang tidak sesuai.",
        )


if __name__ == "__main__":
    unittest.main()
