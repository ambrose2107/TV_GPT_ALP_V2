"""Durable storage for the latest optimizer snapshot.

Uses PostgreSQL when DATABASE_URL is configured (Render PostgreSQL). SQLite
remains a local-development fallback; it is not durable on ephemeral hosts.
"""
import json
import os

SNAPSHOT_KEY = "optimizer_last_snapshot_v1"
_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS optimizer_snapshots (
    snapshot_key TEXT PRIMARY KEY,
    payload JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""


def _database_url():
    url = (os.environ.get("DATABASE_URL") or "").strip()
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url


def _postgres_connection():
    # Import lazily so local SQLite-only development can still run if the
    # optional driver is not installed in a developer environment.
    import psycopg
    return psycopg.connect(_database_url(), connect_timeout=5)


def save_optimizer_snapshot(encoded_json):
    """Store a JSON snapshot; raise on configured-Postgres failures."""
    parsed = json.loads(encoded_json)
    url = _database_url()
    if url:
        with _postgres_connection() as conn:
            conn.execute(_TABLE_SQL)
            conn.execute(
                """
                INSERT INTO optimizer_snapshots (snapshot_key, payload, updated_at)
                VALUES (%s, %s::jsonb, NOW())
                ON CONFLICT (snapshot_key) DO UPDATE
                SET payload = EXCLUDED.payload, updated_at = NOW()
                """,
                (SNAPSHOT_KEY, json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)),
            )
        return

    # SQLite fallback is useful locally, but Render's ephemeral filesystem
    # means it is not a substitute for configuring DATABASE_URL.
    from core.database import set_setting
    set_setting(SNAPSHOT_KEY, encoded_json)


def load_optimizer_snapshot():
    """Return the saved snapshot as a JSON string, or None when absent."""
    url = _database_url()
    if url:
        with _postgres_connection() as conn:
            conn.execute(_TABLE_SQL)
            row = conn.execute(
                "SELECT payload FROM optimizer_snapshots WHERE snapshot_key = %s",
                (SNAPSHOT_KEY,),
            ).fetchone()
            if row:
                payload = row[0]
                return payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

        # One-time best-effort migration from an existing SQLite snapshot
        # helps preserve a run saved before DATABASE_URL was configured.
        try:
            from core.database import get_setting
            legacy = get_setting(SNAPSHOT_KEY)
            if legacy:
                save_optimizer_snapshot(legacy)
                return legacy
        except Exception:
            pass
        return None

    from core.database import get_setting
    return get_setting(SNAPSHOT_KEY)
