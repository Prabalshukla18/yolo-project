"""
search.py — Interactive forensic search interface for FORENIX AI.

Provides both a Python API and a command-line interface to query
the forensic_logs SQLite database.

Usage (CLI)
-----------
    python search.py person
    python search.py car --min-conf 0.8
    python search.py person --camera CAM-02 --flagged
    python search.py --summary
    python search.py person --export results.csv

Usage (API)
-----------
    from search import ForensicSearch
    fs = ForensicSearch()
    results = fs.query("person", min_confidence=0.8)
    fs.print_table(results)
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from typing import Optional

from db import get_conn, init_db, search, summary, DB_PATH


# ─────────────────────────────────────────────
# ForensicSearch API
# ─────────────────────────────────────────────

class ForensicSearch:
    """High-level search API over the forensic detections database."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn = get_conn(db_path)

    def query(
        self,
        object_class: Optional[str] = None,
        *,
        camera_id: Optional[str] = None,
        min_confidence: float = 0.0,
        flagged_only: bool = False,
        video_id: Optional[int] = None,
        limit: int = 500,
    ) -> list:
        """
        Search the forensic log.

        Parameters
        ----------
        object_class    : Filter by detected class ('person', 'car', etc.)
        camera_id       : Filter by camera ('CAM-01', 'CAM-02', …)
        min_confidence  : Minimum confidence score (0.0–1.0)
        flagged_only    : Return only events marked as suspicious
        video_id        : Restrict to a specific video ingestion
        limit           : Max rows returned

        Returns
        -------
        List of sqlite3.Row objects (accessible as dicts or by column name).
        """
        return search(
            self.conn,
            object_class=object_class,
            camera_id=camera_id,
            min_confidence=min_confidence,
            flagged_only=flagged_only,
            video_id=video_id,
            limit=limit,
        )

    def get_summary(self) -> dict:
        return summary(self.conn)

    # ── display helpers ───────────────────────

    def print_table(self, rows: list, max_rows: int = 50) -> None:
        if not rows:
            print("  (no results)")
            return

        cols = ["id", "camera_id", "frame_num", "timestamp_s",
                "wall_time", "object", "confidence", "flagged"]
        widths = {"id": 6, "camera_id": 8, "frame_num": 8,
                  "timestamp_s": 10, "wall_time": 26,
                  "object": 12, "confidence": 10, "flagged": 7}

        # Header
        header = "  ".join(c.upper().ljust(widths[c]) for c in cols)
        print("\n" + "─" * len(header))
        print(header)
        print("─" * len(header))

        for i, row in enumerate(rows):
            if i >= max_rows:
                print(f"  … {len(rows) - max_rows} more rows (use --limit to adjust)")
                break
            flag_str = "⚠  YES" if row["flagged"] else "   no"
            line = (
    f"  {str(row['id']).ljust(6)}"
    f"  {str(row['camera_id']).ljust(8)}"
    f"  {str(row['frame_num']).ljust(8)}"
    f"  {format(row['timestamp_s'], '.2f') + 's':<10}"
    f"  {str(row['wall_time'])[:24].ljust(26)}"
    f"  {str(row['object']).ljust(12)}"
    f"  {format(row['confidence'], '.4f'):<10}"
    f"  {flag_str}"
)
            print(line)

        print("─" * len(header))
        print(f"  {len(rows)} row(s) returned\n")

    def print_summary(self) -> None:
        s = self.get_summary()
        print("\n╔══════════════════════════════════════╗")
        print("║   FORENIX AI — Database Summary      ║")
        print("╠══════════════════════════════════════╣")
        print(f"║  Total detections : {s['total']:>16,} ║")
        print(f"║  Distinct objects : {s['distinct_objects']:>16} ║")
        print(f"║  Active cameras   : {s['cameras']:>16} ║")
        print(f"║  Avg confidence   : {s['avg_confidence']:>16.4f} ║")
        print(f"║  Flagged events   : {s['flagged']:>16,} ║")
        print("╠══════════════════════════════════════╣")
        print("║  By object class                     ║")
        for obj, cnt in s["by_class"].items():
            bar = "█" * min(20, cnt // max(1, s["total"] // 20))
            print(f"║  {obj:<12} {cnt:>6}  {bar:<20} ║")
        print("╚══════════════════════════════════════╝\n")

    def export_csv(self, rows: list, path: str) -> None:
        if not rows:
            print("  Nothing to export.")
            return
        keys = rows[0].keys()
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows([dict(r) for r in rows])
        print(f"  Exported {len(rows)} rows → {path}")

    def close(self):
        self.conn.close()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="forenix-search",
        description="FORENIX AI — Forensic CCTV log search interface",
    )
    p.add_argument("object", nargs="?",
                   help="Object class to search (person, car, truck, …)")
    p.add_argument("--camera", "-c", metavar="CAM",
                   help="Filter by camera ID (e.g. CAM-01)")
    p.add_argument("--min-conf", type=float, default=0.0, metavar="FLOAT",
                   help="Minimum confidence threshold (default: 0.0)")
    p.add_argument("--flagged", action="store_true",
                   help="Return only flagged/suspicious events")
    p.add_argument("--video-id", type=int, metavar="ID",
                   help="Restrict to a specific video_id")
    p.add_argument("--limit", type=int, default=500, metavar="N",
                   help="Max rows to return (default: 500)")
    p.add_argument("--export", metavar="FILE.csv",
                   help="Export results to CSV file")
    p.add_argument("--summary", action="store_true",
                   help="Show database summary statistics")
    p.add_argument("--db", default=DB_PATH, metavar="PATH",
                   help=f"SQLite database path (default: {DB_PATH})")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    fs = ForensicSearch(args.db)

    if args.summary:
        fs.print_summary()
        fs.close()
        return

    rows = fs.query(
        object_class=args.object,
        camera_id=args.camera,
        min_confidence=args.min_conf,
        flagged_only=args.flagged,
        video_id=args.video_id,
        limit=args.limit,
    )

    fs.print_table(rows)

    if args.export:
        fs.export_csv(rows, args.export)

    fs.close()


if __name__ == "__main__":
    main()
