import sqlite3
import json
from pathlib import Path

DB_PATH = Path(__file__).parent / "sales_calls.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_name TEXT NOT NULL,
                started_at TEXT NOT NULL,
                duration_seconds INTEGER NOT NULL,
                transcript TEXT NOT NULL,
                summary TEXT,
                requirements TEXT,
                objections TEXT,
                next_steps TEXT,
                status TEXT NOT NULL,
                audio_files TEXT
            )
        """)


def save_call(data):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO calls (
                customer_name, started_at, duration_seconds,
                transcript, summary, requirements, objections,
                next_steps, status, audio_files
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get("customer_name", "Guest"),
            data.get("started_at", ""),
            int(data.get("duration_seconds", 0)),
            data.get("transcript", ""),
            data.get("summary", ""),
            data.get("requirements", ""),
            data.get("objections", ""),
            data.get("next_steps", ""),
            data.get("status", "Follow-up Required"),
            json.dumps(data.get("audio_files", [])),
        ))


def get_calls():
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM calls ORDER BY id DESC"
        ).fetchall()
        return [dict(row) for row in rows]
