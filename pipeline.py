"""
pipeline.py — Full FORENIX AI pipeline orchestrator.

Ties together:
  1. Video input validation
  2. Frame extraction  (OpenCV via extractor.py)
  3. YOLOv8 detection  (Ultralytics via detector.py)
  4. Structured log generation
  5. SQLite storage     (db.py)
  6. Summary report

Usage (CLI)
-----------
    python pipeline.py footage/cam01.mp4
    python pipeline.py footage/cam01.mp4 --camera CAM-02 --sample-every 5
    python pipeline.py footage/cam01.mp4 --model yolov8x.pt --conf 0.6
    python pipeline.py footage/cam01.mp4 --keyframes --export report.csv

Usage (API)
-----------
    from pipeline import run_pipeline
    run_pipeline("footage/cam01.mp4", camera_id="CAM-01", sample_every_n=10)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Optional

from db import get_conn, init_db, insert_video, insert_detection_batch, summary
from extractor import FrameExtractor
from detector import YOLODetector
from search import ForensicSearch

# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────

BATCH_SIZE = 32   # frames buffered before flushing to DB


def run_pipeline(
    video_path: str,
    *,
    camera_id: str = "CAM-01",
    model: str = "yolov8n.pt",
    confidence: float = 0.50,
    iou: float = 0.45,
    sample_every_n: int = 1,
    sample_every_s: Optional[float] = None,
    keyframes_only: bool = False,
    resize_to: Optional[tuple[int, int]] = None,
    start_time: Optional[datetime] = None,
    db_path: str = "forensic_logs.sqlite",
    verbose: bool = True,
) -> dict:
    """
    Run the complete forensic analysis pipeline on a video file.

    Returns the summary statistics dict from the database.
    """

    t0 = time.perf_counter()
    _log = print if verbose else lambda *a, **kw: None

    # ── Step 1: DB init ───────────────────────
    _log("\n[1/6] Initialising database …")
    init_db(db_path)
    conn = get_conn(db_path)

    # ── Step 2: Frame extractor ───────────────
    _log("[2/6] Opening video …")
    extractor = FrameExtractor(
        video_path,
        sample_every_n=sample_every_n,
        sample_every_s=sample_every_s,
        keyframes_only=keyframes_only,
        resize_to=resize_to,
        start_time=start_time,
    )
    vm = extractor.video_meta
    _log(f"      {vm.filename}  {vm.width}×{vm.height}  "
         f"{vm.fps:.2f} fps  {vm.duration_s:.1f}s")

    # Register video in DB
    video_id = insert_video(
        conn, vm.filename, vm.path,
        vm.fps, vm.width, vm.height, vm.duration_s,
    )
    _log(f"      Registered as video_id={video_id}")

    # ── Step 3: YOLOv8 detector ───────────────
    _log("[3/6] Loading YOLOv8 model …")
    detector = YOLODetector(
        model=model,
        confidence=confidence,
        iou=iou,
    )

    # ── Step 4 + 5: Extract → Detect → Store ─
    _log("[4/6] Extracting frames + running inference …")
    _log("[5/6] Writing structured logs to SQLite …")

    total_frames = 0
    total_detections = 0
    batch_frames: list[tuple] = []
    flush_rows: list[dict] = []

    for meta, frame in extractor:
        total_frames += 1
        batch_frames.append((meta, frame))

        if len(batch_frames) >= BATCH_SIZE:
            rows = detector.detect_batch(batch_frames,
                                         video_id=video_id,
                                         camera_id=camera_id)
            flush_rows.extend(rows)
            batch_frames = []

            if len(flush_rows) >= BATCH_SIZE * 4:
                insert_detection_batch(conn, flush_rows)
                total_detections += len(flush_rows)
                flush_rows = []

        if verbose and total_frames % 50 == 0:
            elapsed = time.perf_counter() - t0
            fps_proc = total_frames / elapsed if elapsed else 0
            print(f"      frame {total_frames:>6}  "
                  f"detections so far: {total_detections + len(flush_rows):>6}  "
                  f"speed: {fps_proc:.1f} fps", end="\r")

    # Flush remaining frames
    if batch_frames:
        rows = detector.detect_batch(batch_frames,
                                     video_id=video_id,
                                     camera_id=camera_id)
        flush_rows.extend(rows)
    if flush_rows:
        insert_detection_batch(conn, flush_rows)
        total_detections += len(flush_rows)

    if verbose:
        print()  # newline after \r progress

    # ── Step 6: Summary ───────────────────────
    _log("[6/6] Building summary …")
    stats = summary(conn)
    conn.close()

    elapsed = time.perf_counter() - t0
    _log(f"\n{'═'*46}")
    _log(f"  FORENIX AI — Pipeline complete")
    _log(f"{'─'*46}")
    _log(f"  Frames processed   : {total_frames:>12,}")
    _log(f"  Total detections   : {total_detections:>12,}")
    _log(f"  Flagged events     : {stats['flagged']:>12,}")
    _log(f"  Avg confidence     : {stats['avg_confidence']:>12.4f}")
    _log(f"  Elapsed            : {elapsed:>11.1f}s")
    _log(f"  Processing speed   : {total_frames/elapsed:.1f} fps")
    _log(f"  Database           : {db_path}")
    _log(f"{'═'*46}\n")

    return stats


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="forenix",
        description="FORENIX AI — CCTV Video Forensic Analysis Pipeline",
    )
    p.add_argument("video", help="Path to the input video file")
    p.add_argument("--camera", default="CAM-01", metavar="ID",
                   help="Camera identifier tag (default: CAM-01)")
    p.add_argument("--model", default="yolov8n.pt", metavar="FILE",
                   help="YOLOv8 weights file (default: yolov8n.pt)")
    p.add_argument("--conf", type=float, default=0.50, metavar="FLOAT",
                   help="Detection confidence threshold (default: 0.50)")
    p.add_argument("--iou", type=float, default=0.45, metavar="FLOAT",
                   help="NMS IoU threshold (default: 0.45)")
    p.add_argument("--sample-every", type=int, default=1, metavar="N",
                   help="Sample every Nth frame (default: 1 = all frames)")
    p.add_argument("--sample-secs", type=float, metavar="S",
                   help="Sample every S seconds (overrides --sample-every)")
    p.add_argument("--keyframes", action="store_true",
                   help="Use scene-change detection instead of fixed sampling")
    p.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"),
                   help="Resize frames before inference (e.g. --resize 640 480)")
    p.add_argument("--db", default="forensic_logs.sqlite", metavar="PATH",
                   help="SQLite output path (default: forensic_logs.sqlite)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress progress output")
    p.add_argument("--export", metavar="FILE.csv",
                   help="Export results to CSV after analysis")
    p.add_argument("--query", metavar="OBJECT",
                   help="Run a quick search after pipeline completes")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    resize = tuple(args.resize) if args.resize else None

    run_pipeline(
        args.video,
        camera_id=args.camera,
        model=args.model,
        confidence=args.conf,
        iou=args.iou,
        sample_every_n=args.sample_every,
        sample_every_s=args.sample_secs,
        keyframes_only=args.keyframes,
        resize_to=resize,
        db_path=args.db,
        verbose=not args.quiet,
    )

    if args.query or args.export:
        fs = ForensicSearch(args.db)
        rows = fs.query(object_class=args.query)
        if args.query:
            fs.print_table(rows)
        if args.export:
            fs.export_csv(rows, args.export)
        fs.close()


if __name__ == "__main__":
    main()