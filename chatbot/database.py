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
