"""MariaDB helpers for storing SiagaBot chat history."""

from __future__ import annotations

import os
from typing import Any


def _load_environment() -> None:
    """Load .env when python-dotenv is installed."""

    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def create_connection() -> Any:
    """Create a MariaDB connection using environment variables."""

    _load_environment()

    try:
        import mysql.connector
    except ImportError as error:
        raise RuntimeError(
            "mysql-connector-python belum terpasang. "
            "Jalankan: pip install -r requirements.txt"
        ) from error

    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "3306")),
        database=os.getenv("DB_NAME", "siagabot"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
    )


def save_chat_history(
    connection: Any,
    session_id: str,
    result: dict[str, Any],
) -> None:
    """Save one chatbot exchange using a parameterized query."""

    query = """
        INSERT INTO chat_history
            (session_id, user_message, bot_response, matched_intent, similarity_score)
        VALUES (%s, %s, %s, %s, %s)
    """
    cursor = connection.cursor()
    try:
        cursor.execute(
            query,
            (
                session_id,
                result["user_message"],
                result["response"],
                result.get("intent"),
                result.get("score", 0.0),
            ),
        )
        connection.commit()
    finally:
        cursor.close()


def close_connection(connection: Any) -> None:
    """Close a database connection safely."""

    if connection is not None and connection.is_connected():
        connection.close()


def get_statistics_summary(connection: Any) -> dict[str, Any]:
    """Return aggregate chat statistics for the dashboard."""

    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(
            """
            SELECT
                COUNT(*) AS total,
                COALESCE(SUM(matched_intent IS NOT NULL), 0) AS recognized,
                COALESCE(SUM(matched_intent IS NULL), 0) AS unknown
            FROM chat_history
            """
        )
        summary = cursor.fetchone() or {"total": 0, "recognized": 0, "unknown": 0}

        cursor.execute(
            """
            SELECT matched_intent AS intent, COUNT(*) AS total
            FROM chat_history
            WHERE matched_intent IS NOT NULL
            GROUP BY matched_intent
            ORDER BY total DESC, intent ASC
            LIMIT 1
            """
        )
        most_common = cursor.fetchone()
        summary["most_common_intent"] = most_common["intent"] if most_common else None
        return {key: int(value) if isinstance(value, (int, float)) else value for key, value in summary.items()}
    finally:
        cursor.close()


def get_intent_statistics(connection: Any) -> list[dict[str, Any]]:
    """Return counts and percentages for every intent, including unknown input."""

    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(
            """
            SELECT COALESCE(matched_intent, 'Tidak dikenali') AS intent, COUNT(*) AS total
            FROM chat_history
            GROUP BY matched_intent
            ORDER BY total DESC, intent ASC
            """
        )
        rows = cursor.fetchall()
        total = sum(int(row["total"]) for row in rows)
        return [
            {
                "intent": row["intent"],
                "total": int(row["total"]),
                "percentage": round((int(row["total"]) / total * 100) if total else 0, 2),
            }
            for row in rows
        ]
    finally:
        cursor.close()


def get_unknown_questions(connection: Any, limit: int = 100) -> list[dict[str, Any]]:
    """Return recent questions that did not match an intent."""

    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(
            """
            SELECT id, user_message, similarity_score, created_at
            FROM chat_history
            WHERE matched_intent IS NULL
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        return cursor.fetchall()
    finally:
        cursor.close()
