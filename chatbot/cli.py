"""Command-line interface for SiagaBot."""

from __future__ import annotations

import uuid
import sys
from pathlib import Path

from .chatbot import load_knowledge_map, process_message
from .database import close_connection, create_connection, save_chat_history


PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_MAP_PATH = PROJECT_ROOT / "p3k_knowledge_map.json"


def _configure_console() -> None:
    """Keep emoji and Indonesian text printable on Windows consoles."""

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _print_similarity_details(result: dict) -> None:
    """Display similarity components and the five highest-ranked intents."""

    rankings = result.get("rankings", [])[:5]
    if not rankings:
        return

    print("\n[Similarity - Top 5 Ranking]")
    for position, ranking in enumerate(rankings, start=1):
        priority_label = " [PRIORITAS]" if ranking.get("priority") else ""
        print(
            f"{position}. {ranking['tag']}{priority_label} | "
            f"score={ranking['score']:.4f} | "
            f"Jaccard={ranking['jaccard']:.4f} | "
            f"Fuzzy={ranking['fuzzy']:.4f} | "
            f"Coverage={ranking['coverage']:.4f}"
        )
        print(f"   Pattern: {ranking['matched_pattern']}")


def run_cli() -> None:
    """Run the interactive SiagaBot CLI."""

    _configure_console()
    knowledge_map = load_knowledge_map(KNOWLEDGE_MAP_PATH)
    session_id = uuid.uuid4().hex
    connection = None

    try:
        try:
            connection = create_connection()
            print("[Database] Histori chat aktif.")
        except Exception as error:
            print(f"[Database] Histori tidak aktif: {error}")

        print(f"{knowledge_map.get('bot_name', 'SiagaBot')}: ")
        print(knowledge_map.get("greeting", "Halo, saya SiagaBot."))
        print("Ketik 'keluar' untuk mengakhiri sesi.\n")

        while True:
            try:
                user_message = input("Anda: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSesi diakhiri.")
                break

            if user_message.lower() in {"keluar", "exit", "quit"}:
                print("Sesi diakhiri.")
                break

            if not user_message:
                continue

            result = process_message(user_message, knowledge_map)
            _print_similarity_details(result)
            print(f"{knowledge_map.get('bot_name', 'SiagaBot')}: {result['response']}\n")

            if connection is not None:
                try:
                    save_chat_history(connection, session_id, result)
                except Exception as error:
                    print(f"[Database] Gagal menyimpan histori: {error}")
                    close_connection(connection)
                    connection = None
    finally:
        close_connection(connection)


if __name__ == "__main__":
    run_cli()
