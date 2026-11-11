"""Flask web interface for SiagaBot."""

from __future__ import annotations

import uuid
import os
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request, session

from chatbot.chatbot import load_knowledge_map, process_message
from chatbot.database import (
    close_connection,
    create_connection,
    get_intent_statistics,
    get_statistics_summary,
    get_unknown_questions,
    save_chat_history,
)

PROJECT_ROOT = Path(__file__).resolve().parent
KNOWLEDGE_MAP = load_knowledge_map(PROJECT_ROOT / "p3k_knowledge_map.json")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "siagabot-development-key")


def _session_id() -> str:
    if "session_id" not in session:
        session["session_id"] = uuid.uuid4().hex
    return session["session_id"]


def _database_call(callback):
    connection = None
    try:
        connection = create_connection()
        callback(connection)
        return True
    except Exception:
        return None
    finally:
        close_connection(connection)


@app.get("/")
def chat_page():
    return render_template("chat.html", knowledge_map=KNOWLEDGE_MAP)


@app.get("/statistics")
def statistics_page():
    return render_template("statistics.html", bot_name=KNOWLEDGE_MAP.get("bot_name", "SiagaBot"))


@app.post("/api/chat")
def chat_api():
    payload = request.get_json(silent=True) or {}
    user_message = str(payload.get("message", "")).strip()
    if not user_message:
        return jsonify({"error": "Pertanyaan tidak boleh kosong."}), 400

    result = process_message(user_message, KNOWLEDGE_MAP)
    stored = _database_call(lambda connection: save_chat_history(connection, _session_id(), result))
    return jsonify({
        "user_message": result["user_message"],
        "response": result["response"],
        "intent": result.get("intent"),
        "score": result.get("score", 0.0),
        "matched": result.get("intent") is not None,
        "history_saved": stored is not None,
    })


@app.get("/api/statistics")
def statistics_api():
    connection = None
    try:
        connection = create_connection()
        unknown = get_unknown_questions(connection)
        for item in unknown:
            if isinstance(item.get("created_at"), datetime):
                item["created_at"] = item["created_at"].isoformat(sep=" ")
            if item.get("similarity_score") is not None:
                item["similarity_score"] = float(item["similarity_score"])
        return jsonify({
            "summary": get_statistics_summary(connection),
            "intents": get_intent_statistics(connection),
            "unknown_questions": unknown,
        })
    except Exception as error:
        return jsonify({"error": f"Statistik belum tersedia: {error}"}), 503
    finally:
        close_connection(connection)


if __name__ == "__main__":
    app.run(debug=True)
