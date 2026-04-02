"""
db.py — SQLite schema + database manager for FORENIX AI
Creates and manages the forensic_logs.sqlite database.
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional


DB_PATH = os.environ.get("FORENIX_DB", "forensic_logs.sqlite")


# ─────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT NOT NULL,
    path        TEXT NOT NULL,
    fps         REAL,
    width       INTEGER,
    height      INTEGER,
    duration_s  REAL,
    ingested_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS detections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id    INTEGER NOT NULL REFERENCES videos(id),
    camera_id   TEXT    NOT NULL DEFAULT 'CAM-01',
    frame_num   INTEGER NOT NULL,
    timestamp_s REAL    NOT NULL,
    wall_time   TEXT    NOT NULL,
    object      TEXT    NOT NULL,
    confidence  REAL    NOT NULL,
    bbox_x      INTEGER NOT NULL,
    bbox_y      INTEGER NOT NULL,
    bbox_w      INTEGER NOT NULL,
    bbox_h      INTEGER NOT NULL,
    flagged     INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_object      ON detections(object);
CREATE INDEX IF NOT EXISTS idx_video       ON detections(video_id);
CREATE INDEX IF NOT EXISTS idx_confidence  ON detections(confidence);
CREATE INDEX IF NOT EXISTS idx_flagged     ON detections(flagged);
CREATE INDEX IF NOT EXISTS idx_wall_time   ON detections(wall_time);
"""


# ─────────────────────────────────────────────
# Connection helper
# ─────────────────────────────────────────────

def get_conn(path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(path: str = DB_PATH) -> None:
    """Create the database and all tables if they don't exist."""
    with get_conn(path) as conn:
        conn.executescript(SCHEMA)
    print(f"[db] Database ready at: {path}")


# ─────────────────────────────────────────────
# Write helpers
# ─────────────────────────────────────────────

def insert_video(conn: sqlite3.Connection, filename: str, path: str,
                 fps: float, width: int, height: int,
                 duration_s: float) -> int:
    cur = conn.execute(
        """INSERT INTO videos (filename, path, fps, width, height, duration_s)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (filename, path, fps, width, height, duration_s)
    )
    conn.commit()
    return cur.lastrowid


def insert_detection_batch(conn: sqlite3.Connection,
                           rows: list[dict]) -> None:
    """Bulk insert a list of detection dicts. Much faster than row-by-row."""
    conn.executemany(
        """INSERT INTO detections
           (video_id, camera_id, frame_num, timestamp_s, wall_time,
            object, confidence, bbox_x, bbox_y, bbox_w, bbox_h, flagged)
           VALUES
           (:video_id, :camera_id, :frame_num, :timestamp_s, :wall_time,
            :object, :confidence, :bbox_x, :bbox_y, :bbox_w, :bbox_h, :flagged)""",
        rows
    )
    conn.commit()


# ─────────────────────────────────────────────
# Query helpers
# ─────────────────────────────────────────────

def search(
    conn: sqlite3.Connection,
    *,
    object_class: Optional[str] = None,
    camera_id: Optional[str] = None,
    min_confidence: float = 0.0,
    flagged_only: bool = False,
    video_id: Optional[int] = None,
    limit: int = 500,
) -> list[sqlite3.Row]:
    """
    Flexible search over detections.

    Examples
    --------
    search(conn, object_class='person')
    search(conn, object_class='car', min_confidence=0.8)
    search(conn, flagged_only=True)
    search(conn, camera_id='CAM-02', min_confidence=0.7)
    """
    sql = "SELECT * FROM detections WHERE confidence >= ?"
    params: list = [min_confidence]

    if object_class:
        sql += " AND object = ?"
        params.append(object_class.lower())
    if camera_id:
        sql += " AND camera_id = ?"
        params.append(camera_id)
    if flagged_only:
        sql += " AND flagged = 1"
    if video_id is not None:
        sql += " AND video_id = ?"
        params.append(video_id)

    sql += " ORDER BY wall_time DESC LIMIT ?"
    params.append(limit)

    return conn.execute(sql, params).fetchall()


def summary(conn: sqlite3.Connection) -> dict:
    """Return aggregate stats for the current database."""
    row = conn.execute("""
        SELECT
            COUNT(*)                          AS total,
            COUNT(DISTINCT object)            AS distinct_objects,
            COUNT(DISTINCT camera_id)         AS cameras,
            ROUND(AVG(confidence), 4)         AS avg_conf,
            SUM(flagged)                      AS flagged_count
        FROM detections
    """).fetchone()
    by_class = conn.execute("""
        SELECT object, COUNT(*) AS cnt
        FROM detections
        GROUP BY object
        ORDER BY cnt DESC
    """).fetchall()
    return {
        "total": row["total"],
        "distinct_objects": row["distinct_objects"],
        "cameras": row["cameras"],
        "avg_confidence": row["avg_conf"],
        "flagged": row["flagged_count"],
        "by_class": {r["object"]: r["cnt"] for r in by_class},
    }
