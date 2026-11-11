"""Evaluate intent detection against representative P3K messages.

Run from the project root with:

    python -m unittest tests.test_intent_accuracy -v

The test prints a per-intent accuracy table and lists every misclassified
message. The test fails when at least one message is assigned to the wrong
intent.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chatbot.chatbot import find_best_intent, load_knowledge_map  # noqa: E402


KNOWLEDGE_MAP = load_knowledge_map(PROJECT_ROOT / "p3k_knowledge_map.json")


MESSAGES: dict[str, list[str]] = {
    "luka_lecet": [
        "jatuh dari sepeda, dengkul sama siku tergores kerikil",
        "Setelah kulit tergesek aspal, perlu diberi antiseptik atau cukup dibersihkan saja?",
        "pas lari pagi tadi aku kepleset di jalan paving, telapak kaki dan lengan kiriku perih banget kayak kulitnya kebuka, enaknya diapain ya",
        "Anak saya terjatuh dari skuter dan kulit lututnya terkelupas.",
        "tumit kgores trotoar, kulitnya ngelupas dkit",
    ],
    "luka_sayat": [
        "pas bukain kaleng sarden tanganku kena tutup kalengnya, berdarah",
        "Bagaimana cara menghentikan perdarahan dari luka iris di jari?",
        "Kemarin lagi beresin pecahan piring di dapur, ga sengaja jempol kaki saya tergores serpihannya sampai keluar darah, harus dibersihin gimana?",
        "Anak saya terkena ujung kertas tajam dan jarinya terbelah tipis.",
        "tangan ke gores ujung seng, ada garis merah panjang",
    ],
    "luka_bakar_ringan": [
        "lengan kena uap rice cooker pas buka tutup, merah",
        "Bolehkah mengoleskan odol pada kulit yang terbakar setrika?",
        "tadi pas bikin kopi, air termosnya tumpah ke paha aku, langsung kerasa panas banget dan kulitnya jadi merah-merah",
        "Adik saya menyentuh kompor yang masih menyala dan jari telunjuknya melepuh.",
        "kejiret api korek, jari kebakar dkit",
    ],
    "tertusuk_benda": [
        "kena duri mawar pas metik bunga, masuk ke ujung jari",
        "Bagaimana cara mencabut duri kecil yang menancap di telapak kaki?",
        "waktu beresin gudang, kakiku kena ujung paku tripleks yang menonjol, lubangnya kecil tapi kerasa dalam",
        "Anak saya menginjak peniti yang tergeletak di lantai kamar.",
        "tusuk gigi nancap ke tangan, gmn ya",
    ],
    "memar_benjol": [
        "dengkul biru abis nabrak kursi",
        "Apa bedanya kompres dingin dan hangat untuk bekas benturan?",
        "tadi ke gym terus barbelnya jatuh kena kaki, bagian atas kakinya sekarang ungu kehitaman dan agak nyeri",
        "Bahu saya terbentur kusen pintu dan kini tampak lebam kebiruan.",
        "pipi kejedot stang sepeda, benjol dkt",
    ],
    "keseleo": [
        "pergelangan tangan keplintir waktu angkat galon, sekarang ngilu",
        "Berapa lama biasanya kaki terkilir bisa sembuh?",
        "kemarin turun dari motor kakiku salah pijak ke lubang, kakinya langsung miring ke samping dan sekarang bengkak di sisi luar mata kaki",
        "Teman saya keseleo saat naik gunung dan sulit menumpu berat badan di kakinya.",
        "kaki kemeleset trs ngilu, kykny kseleo",
    ],
    "mimisan": [
        "hidungku berdarah setelah bersin kencang",
        "Perlu nggak menyumbat hidung pakai tisu saat mimisan?",
        "anak saya yang umur 6 tahun bangun tidur tiba-tiba hidungnya berdarah, bantalnya sampai ada noda merah, apa yang harus saya lakukan sekarang?",
        "Rekan kerja saya mengalami pendarahan hidung ringan akibat udara AC yang kering.",
        "idung berdarah lg, tolong dong",
    ],
    "jari_terjepit": [
        "jari tengah kepencet jendela geser",
        "Kuku yang menghitam setelah terjepit bisa copot sendiri nggak?",
        "tadi lagi pasang rolling door, tiba-tiba turun dan jariku kejepit di sela-selanya, sakit banget dan mulai bengkak",
        "Keponakan saya menjepit jarinya di engsel lemari dan kukunya membiru.",
        "jempol ke jepit tang, ngilu bgt",
    ],
    "kelilipan": [
        "mata kemasukan asap bakar sampah, perih dan berair",
        "Apa yang harus saya lakukan bila ada kotoran kecil masuk ke mata saat bersih-bersih rumah?",
        "pas lagi nyapu halaman tadi angin kenceng, mata kananku kena butiran tanah, sekarang merah dan nggak berhenti ngeluarin air",
        "Anak saya mengeluh matanya tidak nyaman karena terkena serbuk gergaji.",
        "klilipan mulu, ada pasir dikit di mata",
    ],
    "tersedak": [
        "aku kesedak biji semangka, batuk terus",
        "Gimana cara menolong anak balita yang tersedak mainan kecil?",
        "lagi makan bakso bareng ibu, tiba-tiba dia megang lehernya dan wajahnya panik, ga bisa batuk",
        "Rekan kerja saya tersedak kerupuk saat makan siang dan terbatuk-batuk.",
        "keslek kacang, tenggorokan gatel batuk2",
    ],
    "benda_masuk_telinga_hidung": [
        "nyamuk masuk ke telinga, bunyinya berdengung terus",
        "Bagaimana cara aman mengeluarkan manik-manik dari lubang hidung balita?",
        "tadi adikku main lego terus dia bilang ada yang masuk ke hidungnya dan sekarang hidung kirinya kayak mampet",
        "Biji jagung masuk ke liang telinga anak saya saat bermain di kebun.",
        "ada kapas ketinggalan di kuping habis dibersihin",
    ],
    "kena_bahan_kimia_rumah": [
        "kena cairan karbol pel di kaki, perih",
        "Apa yang harus dilakukan jika cairan pembersih toilet tumpah mengenai kulit?",
        "tadi lagi ganti oli motor, tangan sama lengan bajuku kena tumpahan oli dan bensin, kulitnya jadi merah dan gatal",
        "Anak saya menelan sedikit deterjen cair yang tertinggal di gelas.",
        "keminum pembersih kaca dkit, gmn nih",
    ],
}


def evaluate_messages() -> list[dict[str, Any]]:
    """Return one classification result for every expected message."""

    results: list[dict[str, Any]] = []
    for expected_intent, messages in MESSAGES.items():
        for message in messages:
            result = find_best_intent(message, KNOWLEDGE_MAP)
            predicted_intent = (
                result["intent"].get("tag") if result.get("intent") else None
            )
            results.append(
                {
                    "expected": expected_intent,
                    "message": message,
                    "predicted": predicted_intent,
                    "score": result.get("score", 0.0),
                    "matched_pattern": result.get("matched_pattern"),
                    "correct": predicted_intent == expected_intent,
                }
            )
    return results


def _percentage(correct: int, total: int) -> str:
    return f"{(correct / total * 100):.2f}%" if total else "0.00%"


def print_report(results: list[dict[str, Any]]) -> None:
    """Print an easy-to-scan accuracy summary and misclassified messages."""

    print("\nRINGKASAN AKURASI PENENTUAN INTENT")
    print("=" * 72)
    print(f"{'Intent':<30} {'Total':>7} {'Benar':>7} {'Salah':>7} {'Akurasi':>10}")
    print("-" * 72)

    for expected_intent in MESSAGES:
        intent_results = [
            result for result in results if result["expected"] == expected_intent
        ]
        total = len(intent_results)
        correct = sum(result["correct"] for result in intent_results)
        print(
            f"{expected_intent:<30} {total:>7} {correct:>7} "
            f"{total - correct:>7} {_percentage(correct, total):>10}"
        )

    total = len(results)
    correct = sum(result["correct"] for result in results)
    print("-" * 72)
    print(
        f"{'TOTAL':<30} {total:>7} {correct:>7} {total - correct:>7} "
        f"{_percentage(correct, total):>10}"
    )

    wrong_results = [result for result in results if not result["correct"]]
    print("\nPESAN DENGAN HASIL SALAH")
    print("=" * 72)
    if not wrong_results:
        print("Tidak ada. Semua pesan diklasifikasikan dengan benar.")
        return

    for number, result in enumerate(wrong_results, start=1):
        print(f"{number}. Pesan: {result['message']}")
        print(f"   Seharusnya : {result['expected']}")
        print(f"   Hasil       : {result['predicted'] or 'tidak terdeteksi'}")
        print(f"   Skor        : {result['score']:.4f}")
        print(f"   Pola cocok  : {result['matched_pattern']!r}")


class IntentAccuracyTests(unittest.TestCase):
    def test_all_messages_match_expected_intents(self) -> None:
        results = evaluate_messages()
        print_report(results)

        wrong_results = [result for result in results if not result["correct"]]
        details = "\n".join(
            f"- {result['message']!r}: expected={result['expected']}, "
            f"predicted={result['predicted']!r}"
            for result in wrong_results
        )
        self.assertFalse(
            wrong_results,
            f"{len(wrong_results)} pesan salah diklasifikasikan:\n{details}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
