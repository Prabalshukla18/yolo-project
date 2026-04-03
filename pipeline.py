"""
pipeline.py — FORENIX AI pipeline orchestrator  (OPTIMISED v2)

Key speed improvements over v1
───────────────────────────────
1. Producer-consumer threading  — OpenCV decode and YOLO inference run on
   separate threads via a bounded queue; no more stall between GPU calls.
2. Auto-resize before inference — 4K → 640 px (configurable) for ~20-40×
   fewer pixels; accuracy loss is negligible for surveillance.
3. True YOLO batch inference    — frames grouped into GPU batches (default 8)
   so the model processes multiple images per forward pass.
4. Smarter sampling defaults    — sample_every_n=2 (process every other frame);
   at 60 fps this halves work with near-zero forensic impact.
5. Bulk SQLite writes           — unchanged from v1 (already good).
"""

from __future__ import annotations

import argparse, os, queue, sys, threading, time
from datetime import datetime
from typing import Optional

from db import get_conn, init_db, insert_video, insert_detection_batch, summary
from extractor import FrameExtractor
from detector import YOLODetector
from search import ForensicSearch

DEFAULT_INFER_SIZE   = 640
DEFAULT_BATCH_SIZE   = 8
DEFAULT_QUEUE_DEPTH  = 32
DEFAULT_SAMPLE_EVERY = 2
DB_FLUSH_EVERY       = 256

_SENTINEL = object()


def run_pipeline(
    video_path: str,
    *,
    camera_id: str = "CAM-01",
    model: str = "yolov8n.pt",
    confidence: float = 0.50,
    iou: float = 0.45,
    sample_every_n: int = DEFAULT_SAMPLE_EVERY,
    sample_every_s: Optional[float] = None,
    keyframes_only: bool = False,
    infer_size: int = DEFAULT_INFER_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    start_time: Optional[datetime] = None,
    db_path: str = "forensic_logs.sqlite",
    verbose: bool = True,
) -> dict:
    t0 = time.perf_counter()
    _log = print if verbose else lambda *a, **kw: None

    _log("\n[1/6] Initialising database …")
    init_db(db_path)
    conn = get_conn(db_path)

    _log("[2/6] Opening video …")
    extractor = FrameExtractor(
        video_path,
        sample_every_n=sample_every_n,
        sample_every_s=sample_every_s,
        keyframes_only=keyframes_only,
        start_time=start_time,
    )
    vm = extractor.video_meta
    _log(f"      {vm.filename}  {vm.width}×{vm.height}  {vm.fps:.2f} fps  {vm.duration_s:.1f}s")
    _log(f"      Sampling: every {sample_every_n} frame(s)  → ~{extractor.frames_count()} frames")
    _log(f"      Inference size: {infer_size}px  batch: {batch_size}")

    video_id = insert_video(conn, vm.filename, vm.path, vm.fps, vm.width, vm.height, vm.duration_s)
    _log(f"      Registered as video_id={video_id}")

    _log("[3/6] Loading YOLOv8 model …")
    detector = YOLODetector(model=model, confidence=confidence, iou=iou, imgsz=infer_size)

    _log("[4/6] Extracting frames + running inference …")
    _log("[5/6] Writing structured logs to SQLite …")

    frame_q: queue.Queue = queue.Queue(maxsize=DEFAULT_QUEUE_DEPTH)
    counters = {"frames": 0, "detections": 0}

    def producer():
        try:
            for meta, frame in extractor:
                frame_q.put((meta, frame))
        finally:
            frame_q.put(_SENTINEL)

    prod_thread = threading.Thread(target=producer, daemon=True)
    prod_thread.start()

    flush_buf: list[dict] = []
    batch: list[tuple] = []

    def flush_to_db():
        nonlocal flush_buf
        if flush_buf:
            insert_detection_batch(conn, flush_buf)
            counters["detections"] += len(flush_buf)
            flush_buf = []

    def run_batch():
        nonlocal batch
        if not batch:
            return
        rows = detector.detect_batch(batch, video_id=video_id, camera_id=camera_id)
        flush_buf.extend(rows)
        counters["frames"] += len(batch)
        batch = []
        if len(flush_buf) >= DB_FLUSH_EVERY:
            flush_to_db()

    while True:
        item = frame_q.get()
        if item is _SENTINEL:
            break
        meta, frame = item
        batch.append((meta, frame))
        if len(batch) >= batch_size:
            run_batch()

        if verbose and counters["frames"] > 0 and counters["frames"] % 50 == 0:
            elapsed = time.perf_counter() - t0
            fps_proc = counters["frames"] / elapsed if elapsed else 0
            print(f"      frame {counters['frames']:>6}  "
                  f"detections: {counters['detections'] + len(flush_buf):>6}  "
                  f"speed: {fps_proc:.1f} fps", end="\r")

    run_batch()
    flush_to_db()
    if verbose:
        print()
    prod_thread.join()

    _log("[6/6] Building summary …")
    stats = summary(conn)
    conn.close()

    elapsed = time.perf_counter() - t0
    fps_proc = counters["frames"] / elapsed if elapsed else 0

    _log(f"\n{'═'*46}")
    _log(f"  FORENIX AI — Pipeline complete  [v2 optimised]")
    _log(f"{'─'*46}")
    _log(f"  Frames processed   : {counters['frames']:>12,}")
    _log(f"  Total detections   : {counters['detections']:>12,}")
    _log(f"  Flagged events     : {stats['flagged']:>12,}")
    _log(f"  Avg confidence     : {stats['avg_confidence']:>12.4f}")
    _log(f"  Elapsed            : {elapsed:>11.1f}s")
    _log(f"  Processing speed   : {fps_proc:.1f} fps")
    _log(f"  Database           : {db_path}")
    _log(f"{'═'*46}\n")
    return stats


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="forenix",
        description="FORENIX AI — Optimised CCTV Forensic Analysis Pipeline v2")
    p.add_argument("video")
    p.add_argument("--camera", default="CAM-01")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.50)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--sample-every", type=int, default=DEFAULT_SAMPLE_EVERY)
    p.add_argument("--sample-secs", type=float)
    p.add_argument("--keyframes", action="store_true")
    p.add_argument("--infer-size", type=int, default=DEFAULT_INFER_SIZE)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--db", default="forensic_logs.sqlite")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--export", metavar="FILE.csv")
    p.add_argument("--query", metavar="OBJECT")
    return p


def main():
    args = build_parser().parse_args()
    if not os.path.exists(args.video):
        print(f"Error: video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)
    run_pipeline(
        args.video, camera_id=args.camera, model=args.model,
        confidence=args.conf, iou=args.iou, sample_every_n=args.sample_every,
        sample_every_s=args.sample_secs, keyframes_only=args.keyframes,
        infer_size=args.infer_size, batch_size=args.batch_size,
        db_path=args.db, verbose=not args.quiet,
    )
    if args.query or args.export:
        fs = ForensicSearch(args.db)
        rows = fs.query(object_class=args.query)
        if args.query: fs.print_table(rows)
        if args.export: fs.export_csv(rows, args.export)
        fs.close()


if __name__ == "__main__":
    main()
